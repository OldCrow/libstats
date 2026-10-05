#include "libstats/distributions/beta.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"  // beta_i, inverse_beta_i, lbeta, digamma
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

static double requirePositive(double v, const char* name) {
    if (v <= 0.0 || !std::isfinite(v)) {
        throw std::invalid_argument(std::string(name) + " must be a positive finite number");
    }
    return v;
}

BetaDistribution::BetaDistribution(double alpha, double beta)
    : DistributionBase(),
      alpha_(requirePositive(alpha, "Alpha (shape1)")),
      beta_(requirePositive(beta, "Beta (shape2)")) {
    updateCacheUnsafe();
}

BetaDistribution::BetaDistribution(const BetaDistribution& other) : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    alphaMinus1_ = other.alphaMinus1_;
    betaMinus1_ = other.betaMinus1_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    mode_ = other.mode_;
    isUniform_ = other.isUniform_;
    isSymmetric_ = other.isSymmetric_;
    isUnimodal_ = other.isUnimodal_;
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

BetaDistribution& BetaDistribution::operator=(const BetaDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        alphaMinus1_ = other.alphaMinus1_;
        betaMinus1_ = other.betaMinus1_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        mode_ = other.mode_;
        isUniform_ = other.isUniform_;
        isSymmetric_ = other.isSymmetric_;
        isUnimodal_ = other.isUnimodal_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicAlpha_.store(alpha_, std::memory_order_release);
        atomicBeta_.store(beta_, std::memory_order_release);
    }
    return *this;
}

BetaDistribution::BetaDistribution(BetaDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    alphaMinus1_ = other.alphaMinus1_;
    betaMinus1_ = other.betaMinus1_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    mode_ = other.mode_;
    isUniform_ = other.isUniform_;
    isSymmetric_ = other.isSymmetric_;
    isUnimodal_ = other.isUnimodal_;
    other.alpha_ = detail::ONE;
    other.beta_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

BetaDistribution& BetaDistribution::operator=(BetaDistribution&& other) noexcept {
    if (this != &other) {
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        alphaMinus1_ = other.alphaMinus1_;
        betaMinus1_ = other.betaMinus1_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        mode_ = other.mode_;
        isUniform_ = other.isUniform_;
        isSymmetric_ = other.isSymmetric_;
        isUnimodal_ = other.isUnimodal_;
        other.alpha_ = detail::ONE;
        other.beta_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicAlpha_.store(alpha_, std::memory_order_release);
        atomicBeta_.store(beta_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

BetaDistribution BetaDistribution::createUnchecked(double alpha, double beta) noexcept {
    return BetaDistribution(alpha, beta, true);
}

BetaDistribution::BetaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept
    : DistributionBase(), alpha_(alpha), beta_(beta) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

void BetaDistribution::setAlpha(double alpha) {
    validateParameters(alpha, getBeta());
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void BetaDistribution::setBeta(double beta) {
    validateParameters(getAlpha(), beta);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void BetaDistribution::setParameters(double alpha, double beta) {
    validateParameters(alpha, beta);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

VoidResult BetaDistribution::trySetAlpha(double alpha) noexcept {
    auto v = validateBetaParameters(alpha, getBeta());
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult BetaDistribution::trySetBeta(double beta) noexcept {
    auto v = validateBetaParameters(getAlpha(), beta);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult BetaDistribution::trySetParameters(double alpha, double beta) noexcept {
    auto v = validateBetaParameters(alpha, beta);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult BetaDistribution::validateCurrentParameters() const noexcept {
    return validateBetaParameters(getAlpha(), getBeta());
}

//==============================================================================
// 3. STATISTICAL MOMENTS
//==============================================================================

double BetaDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mean_;
}

double BetaDistribution::getVariance() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return variance_;
}

double BetaDistribution::getSkewness() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (alpha_ <= detail::ZERO_DOUBLE || beta_ <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double ab = alpha_ + beta_;
    return detail::TWO * (beta_ - alpha_) * std::sqrt(ab + detail::ONE) /
           ((ab + detail::TWO) * std::sqrt(alpha_ * beta_));
}

double BetaDistribution::getKurtosis() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double ab = alpha_ + beta_;
    const double num = 6.0 * ((alpha_ - beta_) * (alpha_ - beta_) * (ab + detail::ONE) -
                              alpha_ * beta_ * (ab + detail::TWO));
    const double den = alpha_ * beta_ * (ab + detail::TWO) * (ab + 3.0);
    return num / den;
}

double BetaDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mode_;
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

namespace {
// From both shapes 20 the direct log density lnc + (α − 1)·log x + (β − 1)·log(1 − x) cancels
// terms of size (α + β)·log 2 in lnc = −log B(α, β) against the two logs: 1.8e-11 relative in the
// pdf, 2.2e-14 at Beta(50, 60) x = 0.4. There it is the Stirling-form incomplete-beta prefactor
// log[x^α (1 − x)^β / B(α, β)], which has no large terms, less log x and log(1 − x). One large
// shape alone does not cancel (lnc and the log terms then differ in size), so the switch takes
// both, as log_beta_prefactor's does.
[[nodiscard]] inline bool betaStirlingShapes(double a, double b) noexcept {
    return a >= detail::STIRLING_PREFACTOR_SHAPE && b >= detail::STIRLING_PREFACTOR_SHAPE;
}

// log density at x in (0, 1); shape_constant = beta_prefactor_constant(a, b), read only in
// Stirling's form.
[[nodiscard]] inline double betaLogDensity(double x, double a, double b, double lnc, double am1,
                                           double bm1, double shape_constant) noexcept {
    if (!betaStirlingShapes(a, b))
        return lnc + am1 * std::log(x) + bm1 * std::log1p(-x);
    return detail::log_beta_prefactor(x, a, b, shape_constant) - std::log(x) - std::log1p(-x);
}

[[nodiscard]] inline double betaShapeConstant(double a, double b) noexcept {
    return betaStirlingShapes(a, b) ? detail::beta_prefactor_constant(a, b) : 0.0;
}

// betaLogDensity's Stirling branch over a batch, in the scalar path's order: a·log1pmx(u) +
// b·log1pmx(v) + C − log x − log(1 − x), u = (x − x₀)/x₀, v = (x₀ − x)/(1 − x₀), x₀ = a/(a + b).
// log1pmx_series covers |u|, |v| < ½ branch-free; beyond, from x itself as the scalar path.
// Inputs outside (0, 1) come out meaningless; the caller's fixup overwrites them.
void stirlingBetaLogDensityBatch(const double* x, double* out, std::size_t count, double a,
                                 double b) noexcept {
    constexpr std::size_t kBlock = 256;
    alignas(64) double log_x[kBlock];
    alignas(64) double log_1mx[kBlock];
    std::size_t far_index[kBlock];
    const double sum = a + b;
    const double x0 = a / sum;
    const double one_minus_x0 = b / sum;
    const double shape_constant = detail::beta_prefactor_constant(a, b);
    const double log_one_minus_x0 = std::log(one_minus_x0);
    for (std::size_t start = 0; start < count; start += kBlock) {
        const std::size_t n = std::min(kBlock, count - start);
        const double* xb = x + start;
        double* ob = out + start;
        arch::simd::VectorOps::vector_log(xb, log_x, n);
        for (std::size_t i = 0; i < n; ++i)
            log_1mx[i] = -xb[i];
        arch::simd::VectorOps::vector_log1p(log_1mx, log_1mx, n);
        for (std::size_t i = 0; i < n; ++i)
            ob[i] = a * detail::log1pmx_series((xb[i] - x0) / x0) +
                    b * detail::log1pmx_series((x0 - xb[i]) / one_minus_x0);
        // Beyond the series, log(1 + u) and log(1 + v) from x itself, as log_beta_prefactor
        // forms them (the rounded u, v ≈ −1 have lost x, or 1 − x). Gathered first, as in Gamma's
        // batch: a guarded log in one loop is called for every
        // lane (log is pure on Darwin, so the compiler hoists it), 2x the batch at Beta(25, 25).
        std::size_t far = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double u = (xb[i] - x0) / x0;
            const double v = (x0 - xb[i]) / one_minus_x0;
            if (std::fabs(u) >= detail::LOG1PMX_SERIES_LIMIT ||
                std::fabs(v) >= detail::LOG1PMX_SERIES_LIMIT)
                far_index[far++] = i;
        }
        for (std::size_t k = 0; k < far; ++k) {
            const std::size_t i = far_index[k];
            const double u = (xb[i] - x0) / x0;
            const double v = (x0 - xb[i]) / one_minus_x0;
            const double lu = std::fabs(u) < detail::LOG1PMX_SERIES_LIMIT
                                  ? detail::log1pmx_series(u)
                                  : std::log(xb[i] / x0) - u;
            const double lv = std::fabs(v) < detail::LOG1PMX_SERIES_LIMIT
                                  ? detail::log1pmx_series(v)
                                  : (std::log1p(-xb[i]) - log_one_minus_x0) - v;
            ob[i] = a * lu + b * lv;
        }
        for (std::size_t i = 0; i < n; ++i)
            ob[i] = ob[i] + shape_constant - log_x[i] - log_1mx[i];
    }
}
}  // namespace

double BetaDistribution::getProbability(double x) const {
    // Snapshot cached fields under the appropriate lock; no re-acquire = no TOCTOU gap.
    double a, b, lnc, am1, bm1;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
            if (!cache_valid_)
                updateCacheUnsafe();
            a = alpha_;
            b = beta_;
            lnc = logNormConst_;
            am1 = alphaMinus1_;
            bm1 = betaMinus1_;
        } else {
            a = alpha_;
            b = beta_;
            lnc = logNormConst_;
            am1 = alphaMinus1_;
            bm1 = betaMinus1_;
        }
    }
    if (x <= detail::ZERO_DOUBLE || x >= detail::ONE) {
        // Boundary: PDF = 0 for α,β > 1; ∞ for α or β < 1 (return +inf); 1 for α or β = 1
        if (x < detail::ZERO_DOUBLE || x > detail::ONE)
            return detail::ZERO_DOUBLE;
        // x = 0 or x = 1: handle carefully
        if (x == detail::ZERO_DOUBLE) {
            if (a > detail::ONE)
                return detail::ZERO_DOUBLE;
            if (a == detail::ONE)  // exactly: just below 1 the limit is +inf
                return std::exp(lnc);
            return std::numeric_limits<double>::infinity();
        }
        // x = 1
        if (b > detail::ONE)
            return detail::ZERO_DOUBLE;
        if (b == detail::ONE)  // exactly: just below 1 the limit is +inf
            return std::exp(lnc);
        return std::numeric_limits<double>::infinity();
    }
    return std::exp(betaLogDensity(x, a, b, lnc, am1, bm1, betaShapeConstant(a, b)));
}

double BetaDistribution::getLogProbability(double x) const {
    // Snapshot cached fields under the appropriate lock; no re-acquire = no TOCTOU gap.
    double a, b, lnc, am1, bm1;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
            if (!cache_valid_)
                updateCacheUnsafe();
            a = alpha_;
            b = beta_;
            lnc = logNormConst_;
            am1 = alphaMinus1_;
            bm1 = betaMinus1_;
        } else {
            a = alpha_;
            b = beta_;
            lnc = logNormConst_;
            am1 = alphaMinus1_;
            bm1 = betaMinus1_;
        }
    }
    if (x < detail::ZERO_DOUBLE || x > detail::ONE) {
        return -std::numeric_limits<double>::infinity();
    }
    if (x == detail::ZERO_DOUBLE) {
        if (a > detail::ONE)
            return -std::numeric_limits<double>::infinity();
        if (a == detail::ONE)  // exactly: just below 1 the limit is +inf
            return lnc;
        return std::numeric_limits<double>::infinity();
    }
    if (x == detail::ONE) {
        if (b > detail::ONE)
            return -std::numeric_limits<double>::infinity();
        if (b == detail::ONE)  // exactly: just below 1 the limit is +inf
            return lnc;
        return std::numeric_limits<double>::infinity();
    }
    return betaLogDensity(x, a, b, lnc, am1, bm1, betaShapeConstant(a, b));
}

double BetaDistribution::getCumulativeProbability(double x) const {
    if (x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (x >= detail::ONE)
        return detail::ONE;
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double a = alpha_, b = beta_;
    lock.unlock();
    return detail::beta_i(x, a, b);
}

double BetaDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return detail::ONE;
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double a = alpha_, b = beta_;
    lock.unlock();
    return detail::inverse_beta_i(p, a, b);
}

double BetaDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double a = alpha_, b = beta_;
    lock.unlock();

    // X ~ Gamma(α, 1), Y ~ Gamma(β, 1) → X/(X+Y) ~ Beta(α, β)
    std::gamma_distribution<double> gamma_a(a, detail::ONE);
    std::gamma_distribution<double> gamma_b(b, detail::ONE);
    const double x = gamma_a(rng);
    const double y = gamma_b(rng);
    const double sum = x + y;
    if (sum <= detail::ZERO_DOUBLE)
        return detail::HALF;  // numerical safety
    return x / sum;
}

std::vector<double> BetaDistribution::sample(std::mt19937& rng, size_t n) const {
    // Read parameters once under a single lock to avoid n lock acquisitions.
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double a = alpha_, b = beta_;
    lock.unlock();

    std::gamma_distribution<double> gamma_a(a, detail::ONE);
    std::gamma_distribution<double> gamma_b(b, detail::ONE);
    std::vector<double> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const double x = gamma_a(rng);
        const double y = gamma_b(rng);
        const double sum = x + y;
        samples.push_back(sum <= detail::ZERO_DOUBLE ? detail::HALF : x / sum);
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void BetaDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    for (double v : values) {
        if (!std::isfinite(v) || v <= detail::ZERO_DOUBLE || v >= detail::ONE) {
            throw std::invalid_argument(
                "All values must be finite and strictly in (0, 1) for Beta MLE");
        }
    }

    const double n = static_cast<double>(values.size());

    // Precompute log(x) and log(1-x) sums
    double log_x_sum = 0.0, log_1mx_sum = 0.0;
    double sum = 0.0, sum_sq = 0.0;
    for (double v : values) {
        log_x_sum += std::log(v);
        log_1mx_sum += std::log1p(-v);
        sum += v;
        sum_sq += v * v;
    }
    const double mean_x = sum / n;
    const double var_x = sum_sq / n - mean_x * mean_x;

    // Method-of-moments initial estimates
    double alpha_est = 2.0, beta_est = 2.0;
    if (var_x > detail::ZERO_DOUBLE && var_x < mean_x * (detail::ONE - mean_x)) {
        const double c = mean_x * (detail::ONE - mean_x) / var_x - detail::ONE;
        if (c > detail::ZERO_DOUBLE) {
            alpha_est = std::max(0.1, mean_x * c);
            beta_est = std::max(0.1, (detail::ONE - mean_x) * c);
        }
    }

    // Newton-Raphson on the two-dimensional score system:
    //   dL/dalpha = n*[psi(alpha) - psi(alpha+beta)] - sum(log(x)) = 0
    //   dL/dbeta  = n*[psi(beta)  - psi(alpha+beta)] - sum(log(1-x)) = 0
    const double log_x_bar = log_x_sum / n;
    const double log_1mx_bar = log_1mx_sum / n;

    const int max_iter = 100;
    const double tol = 1e-8;
    double alpha_cur = alpha_est, beta_cur = beta_est;

    for (int iter = 0; iter < max_iter; ++iter) {
        const double ab = alpha_cur + beta_cur;
        const double psi_a = detail::digamma(alpha_cur);
        const double psi_b = detail::digamma(beta_cur);
        const double psi_ab = detail::digamma(ab);

        const double sa = psi_a - psi_ab - log_x_bar;
        const double sb = psi_b - psi_ab - log_1mx_bar;

        if (std::abs(sa) < tol && std::abs(sb) < tol)
            break;

        // Diagonal Newton step (FIT-3): use detail::trigamma() directly instead of
        // finite-differencing digamma (which required 6 digamma calls per step).
        // Exact derivatives give quadratic Newton convergence.
        const double tpsi_a = detail::trigamma(alpha_cur);
        const double tpsi_b = detail::trigamma(beta_cur);
        const double tpsi_ab = detail::trigamma(ab);

        // 2×2 Jacobian (negated Hessian of log-likelihood per observation):
        // J = [[tpsi_a - tpsi_ab, -tpsi_ab],
        //      [-tpsi_ab,          tpsi_b - tpsi_ab]]
        const double Jaa = tpsi_a - tpsi_ab;
        const double Jbb = tpsi_b - tpsi_ab;
        const double Jab = -tpsi_ab;
        const double det = Jaa * Jbb - Jab * Jab;

        if (std::abs(det) < 1e-15)
            break;

        // Newton step: [Δα, Δβ] = J^{-1} * [sa, sb]
        const double delta_a = (Jbb * sa - Jab * sb) / det;
        const double delta_b = (Jaa * sb - Jab * sa) / det;

        alpha_cur = std::max(0.01, alpha_cur - delta_a);
        beta_cur = std::max(0.01, beta_cur - delta_b);

        if (std::abs(delta_a) < tol && std::abs(delta_b) < tol)
            break;
    }

    setParameters(alpha_cur, beta_cur);
}

void BetaDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                        std::vector<BetaDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void BetaDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = detail::ONE;
    beta_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string BetaDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "BetaDistribution(alpha=" << alpha_ << ", beta=" << beta_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double BetaDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // H = lbeta(α,β) - (α-1)*ψ(α) - (β-1)*ψ(β) + (α+β-2)*ψ(α+β)
    const double ab = alpha_ + beta_;
    return detail::lbeta(alpha_, beta_) - alphaMinus1_ * detail::digamma(alpha_) -
           betaMinus1_ * detail::digamma(beta_) + (ab - detail::TWO) * detail::digamma(ab);
}

//==============================================================================
// 13–14. BATCH OPERATIONS
//==============================================================================

void BetaDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                      const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const BetaDistribution& dist, double value) { return dist.getProbability(value); },
        [](const BetaDistribution& dist, const double* vals, double* res, size_t count) {
            double lnc, am1, bm1;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                am1 = dist.alphaMinus1_;
                bm1 = dist.betaMinus1_;
            });
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, lnc, am1, bm1);
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, am1, bm1;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                am1 = dist.alphaMinus1_;
                bm1 = dist.betaMinus1_;
            });
            constexpr std::size_t CHUNK = 1024;
            ParallelUtils::parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                   lnc, am1, bm1);
            });
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, am1, bm1;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                am1 = dist.alphaMinus1_;
                bm1 = dist.betaMinus1_;
            });
            constexpr std::size_t CHUNK = 1024;
            pool.parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                   lnc, am1, bm1);
            });
            pool.waitForAll();
        });
}

void BetaDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                         const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const BetaDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const BetaDistribution& dist, const double* vals, double* res, size_t count) {
            double lnc, am1, bm1;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                am1 = dist.alphaMinus1_;
                bm1 = dist.betaMinus1_;
            });
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, lnc, am1, bm1);
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, am1, bm1;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                am1 = dist.alphaMinus1_;
                bm1 = dist.betaMinus1_;
            });
            constexpr std::size_t CHUNK = 1024;
            ParallelUtils::parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                      lnc, am1, bm1);
            });
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, am1, bm1;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                am1 = dist.alphaMinus1_;
                bm1 = dist.betaMinus1_;
            });
            constexpr std::size_t CHUNK = 1024;
            pool.parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                      lnc, am1, bm1);
            });
            pool.waitForAll();
        });
}

void BetaDistribution::getCumulativeProbability(std::span<const double> values,
                                                std::span<double> results,
                                                const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const BetaDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const BetaDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double a = dist.alpha_, b = dist.beta_;
            lock.unlock();
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, a, b);
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            // Acquire cache once; hoist lgamma prefix for the batch.
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double a = dist.alpha_, b = dist.beta_;
            lock.unlock();
            const double log_prefix = detail::beta_prefactor_constant(a, b);
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= 0.0)
                        res[i] = 0.0;
                    else if (x >= 1.0)
                        res[i] = 1.0;
                    else
                        res[i] = detail::beta_i(x, a, b, log_prefix);
                });
            } else {
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals.data(), res.data(), count, a, b);
            }
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double a = dist.alpha_, b = dist.beta_;
            lock.unlock();
            const double log_prefix = detail::beta_prefactor_constant(a, b);
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= 0.0)
                    res[i] = 0.0;
                else if (x >= 1.0)
                    res[i] = 1.0;
                else
                    res[i] = detail::beta_i(x, a, b, log_prefix);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool BetaDistribution::operator==(const BetaDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(alpha_ - other.alpha_) <= detail::DEFAULT_TOLERANCE &&
           std::abs(beta_ - other.beta_) <= detail::DEFAULT_TOLERANCE;
}

bool BetaDistribution::operator!=(const BetaDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const BetaDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, BetaDistribution& dist) {
    std::string line;

    // Expected format: "BetaDistribution(alpha=<value>, beta=<value>)"
    // Read the entire line to handle spaces in the format

    // Skip leading whitespace and read the entire formatted string
    if (!std::getline(is, line)) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Trim leading whitespace
    size_t start = line.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    line = line.substr(start);

    if (!line.starts_with("BetaDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t a_pos = line.find("alpha=");
    const size_t comma = line.find(",", a_pos);
    const size_t b_pos = line.find("beta=");
    const size_t close = line.find(")", b_pos);
    if (a_pos == std::string::npos || comma == std::string::npos || b_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double a = std::stod(line.substr(a_pos + 6, comma - a_pos - 6));
        const double b = std::stod(line.substr(b_pos + 5, close - b_pos - 5));
        auto result = dist.trySetParameters(a, b);
        if (result.isError())
            is.setstate(std::ios::failbit);
    } catch (...) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// Log-space pipeline for PDF and LogPDF.
// vector_log for log(x), betaLog1mx for log(1-x), + one aligned temp.
// Scalar fixup for x <= 0 or x >= 1 (delegates to single-value method).
//
// LogPDF (7 steps):
//   Step 1: temp    = log(x)                  [vector_log(values, temp)]
//   Step 2: temp    = (α-1)*log(x)             [scalar_multiply]
//   Steps 3-4: results = log(1-x)              [betaLog1mx: 1-x and vector_log up to
//                                               |β-1| = 16, vector_log1p(-x) beyond]
//   Step 5: results = (β-1)*log(1-x)           [scalar_multiply]
//   Step 6: results = (α-1)log(x)+(β-1)log(1-x) [vector_add(temp, results)]
//   Step 7: results += log_norm_const          [scalar_add]
//
// PDF: steps 1-7 then vector_exp.
// CDF architecture: detail::beta_i (regularized incomplete beta) is evaluated
//   per element via a continued-fraction algorithm. The convergence rate varies
//   with (x, alpha, beta): some inputs converge in a few iterations, others
//   require many more. This data-dependent iteration count prevents SIMD —
//   there is no fixed-length uniform operation sequence to vectorize.
//   Contrast with PDF/LogPDF above, which reduce to a fixed 8-step arithmetic
//   + log/exp pipeline that maps directly to SIMD and achieves 3–5x speedup.
//   Vectorizing beta_i correctly would require a fixed-iteration polynomial
//   approximation, which is outside the scope of this library.
//==============================================================================

namespace {
// results = log(1 − x) over the batch. Up to |β − 1| = 16, as 1 − x then vector_log: 1 − x's
// rounding (up to ε/2 for x < ½), which β − 1 multiplies, stays within the law there, and this is
// the cheaper form. Past it, vector_log1p(−x), relative-accurate as the scalar log1p is: formed
// the first way the error reached 1e-11 at β = 1e5. Measured on Kaby Lake (2026-10-04),
// vector_log1p costs ~1.2x the first form at small β and halves the large-shape batch, which
// used to fall back to the scalar loop.
void betaLog1mx(const double* values, double* results, std::size_t count,
                double beta_minus_one) noexcept {
    if (std::fabs(beta_minus_one) <= 16.0) {
        arch::simd::VectorOps::scalar_add(values, -detail::ONE, results, count);        // x-1
        arch::simd::VectorOps::scalar_multiply(results, -detail::ONE, results, count);  // 1-x
        arch::simd::VectorOps::vector_log(results, results, count);                     // log(1-x)
    } else {
        arch::simd::VectorOps::scalar_multiply(values, -detail::ONE, results, count);  // -x
        arch::simd::VectorOps::vector_log1p(results, results, count);                  // log1p(-x)
    }
}
}  // namespace

void BetaDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                     std::size_t count, double log_norm_const,
                                                     double alpha_minus_one,
                                                     double beta_minus_one) const noexcept {
    // SIMD at every shape: betaLog1mx forms log(1 − x) accurately for any β − 1, and from both
    // shapes 20 the Stirling form replaces the cancelling direct one (betaLogDensity).
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);
    const double a = alpha_minus_one + detail::ONE;  // exact for shapes >= 1, where it is read
    const double b = beta_minus_one + detail::ONE;
    const double shape_constant = betaShapeConstant(a, b);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= 0.0 || x >= 1.0) {
                results[i] = getProbability(x);
            } else {
                results[i] = std::exp(betaLogDensity(x, a, b, log_norm_const, alpha_minus_one,
                                                     beta_minus_one, shape_constant));
            }
        }
        return;
    }

    if (betaStirlingShapes(a, b)) {
        stirlingBetaLogDensityBatch(values, results, count, a, b);
    } else {
        std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

        // Step 1-2: temp = (α-1)*log(x)
        arch::simd::VectorOps::vector_log(values, temp.data(), count);
        arch::simd::VectorOps::scalar_multiply(temp.data(), alpha_minus_one, temp.data(), count);

        // Step 3-6: results = (β-1)*log(1-x)
        betaLog1mx(values, results, count, beta_minus_one);
        arch::simd::VectorOps::scalar_multiply(results, beta_minus_one, results, count);

        // Step 7: results = (α-1)*log(x) + (β-1)*log(1-x)
        arch::simd::VectorOps::vector_add(temp.data(), results, results, count);

        // Step 8: results += log_norm_const
        arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);
    }

    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: x <= 0 or x >= 1 (NaN/Inf from log(0) or log of negative)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0 || values[i] >= 1.0) {
            results[i] = getProbability(values[i]);
        }
    }
}

void BetaDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                        std::size_t count, double log_norm_const,
                                                        double alpha_minus_one,
                                                        double beta_minus_one) const noexcept {
    // SIMD at every shape: betaLog1mx forms log(1 − x) accurately for any β − 1, and from both
    // shapes 20 the Stirling form replaces the cancelling direct one (betaLogDensity).
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);
    const double a = alpha_minus_one + detail::ONE;  // exact for shapes >= 1, where it is read
    const double b = beta_minus_one + detail::ONE;
    const double shape_constant = betaShapeConstant(a, b);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= 0.0 || x >= 1.0) {
                results[i] = getLogProbability(x);
            } else {
                results[i] = betaLogDensity(x, a, b, log_norm_const, alpha_minus_one,
                                            beta_minus_one, shape_constant);
            }
        }
        return;
    }

    if (betaStirlingShapes(a, b)) {
        stirlingBetaLogDensityBatch(values, results, count, a, b);
    } else {
        std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

        // Step 1-2: temp = (α-1)*log(x)
        arch::simd::VectorOps::vector_log(values, temp.data(), count);
        arch::simd::VectorOps::scalar_multiply(temp.data(), alpha_minus_one, temp.data(), count);

        // Step 3-6: results = (β-1)*log(1-x)
        betaLog1mx(values, results, count, beta_minus_one);
        arch::simd::VectorOps::scalar_multiply(results, beta_minus_one, results, count);

        // Step 7-8: full LogPDF
        arch::simd::VectorOps::vector_add(temp.data(), results, results, count);
        arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);
    }

    // Fixup: x <= 0 or x >= 1
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0 || values[i] >= 1.0) {
            results[i] = getLogProbability(values[i]);
        }
    }
}

void BetaDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                               double* results, std::size_t count,
                                                               double alpha,
                                                               double beta) const noexcept {
    // Scalar per element. See section 18 header for why beta_i cannot be
    // vectorized without replacing it with a fixed-iteration approximation.
    // Hoist the lgamma prefix: lgamma(a+b) - lgamma(a) - lgamma(b) is constant
    // for fixed (alpha, beta), saving 3 lgamma calls per element.
    const double log_prefix = detail::beta_prefactor_constant(alpha, beta);
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x <= detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (x >= detail::ONE) {
            results[i] = detail::ONE;
        } else {
            results[i] = detail::beta_i(x, alpha, beta, log_prefix);
        }
    }
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void BetaDistribution::updateCacheUnsafe() const noexcept {
    alphaMinus1_ = alpha_ - detail::ONE;
    betaMinus1_ = beta_ - detail::ONE;
    logNormConst_ = -detail::lbeta(alpha_, beta_);

    const double ab = alpha_ + beta_;
    mean_ = alpha_ / ab;
    variance_ = alpha_ * beta_ / (ab * ab * (ab + detail::ONE));

    // Mode
    if (alpha_ > detail::ONE && beta_ > detail::ONE) {
        mode_ = (alpha_ - detail::ONE) / (ab - detail::TWO);
        isUnimodal_ = true;
    } else if (alpha_ <= detail::ONE && beta_ > detail::ONE) {
        mode_ = detail::ZERO_DOUBLE;
        isUnimodal_ = false;
    } else if (alpha_ > detail::ONE && beta_ <= detail::ONE) {
        mode_ = detail::ONE;
        isUnimodal_ = false;
    } else {
        mode_ = std::numeric_limits<double>::quiet_NaN();  // U-shaped or undefined
        isUnimodal_ = false;
    }

    isUniform_ = (std::abs(alpha_ - detail::ONE) <= detail::DEFAULT_TOLERANCE &&
                  std::abs(beta_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
    isSymmetric_ = (std::abs(alpha_ - beta_) <= detail::DEFAULT_TOLERANCE);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

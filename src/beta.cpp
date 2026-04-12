#include "libstats/distributions/beta.h"

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"  // beta_i, inverse_beta_i, lbeta, digamma
#include "libstats/core/validation.h"

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
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
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
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);

        bool success = false;
        try {
            std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
            if (std::try_lock(lock1, lock2) == -1) {
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
                atomicAlpha_.store(alpha_, std::memory_order_release);
                atomicBeta_.store(beta_, std::memory_order_release);
                success = true;
            }
        } catch (...) {
        }

        if (!success) {
            alpha_ = other.alpha_;
            beta_ = other.beta_;
            other.alpha_ = detail::ONE;
            other.beta_ = detail::ONE;
            cache_valid_ = false;
            other.cache_valid_ = false;
            atomicAlpha_.store(alpha_, std::memory_order_release);
            atomicBeta_.store(beta_, std::memory_order_release);
        }
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
    return VoidResult::ok(true);
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
    return VoidResult::ok(true);
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
    return VoidResult::ok(true);
}

VoidResult BetaDistribution::validateCurrentParameters() const noexcept {
    return validateBetaParameters(getAlpha(), getBeta());
}

//==============================================================================
// 3. STATISTICAL MOMENTS
//==============================================================================

double BetaDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mean_;
}

double BetaDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return variance_;
}

double BetaDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (alpha_ <= detail::ZERO_DOUBLE || beta_ <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double ab = alpha_ + beta_;
    return detail::TWO * (beta_ - alpha_) * std::sqrt(ab + detail::ONE) /
           ((ab + detail::TWO) * std::sqrt(alpha_ * beta_));
}

double BetaDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double ab = alpha_ + beta_;
    const double num = 6.0 * ((alpha_ - beta_) * (alpha_ - beta_) * (ab + detail::ONE) -
                              alpha_ * beta_ * (ab + detail::TWO));
    const double den = alpha_ * beta_ * (ab + detail::TWO) * (ab + 3.0);
    return num / den;
}

double BetaDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mode_;
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double BetaDistribution::getProbability(double x) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (x <= detail::ZERO_DOUBLE || x >= detail::ONE) {
        // Boundary: PDF = 0 for α,β > 1; ∞ for α or β < 1 (return +inf); 1 for α or β = 1
        if (x < detail::ZERO_DOUBLE || x > detail::ONE)
            return detail::ZERO_DOUBLE;
        // x = 0 or x = 1: handle carefully
        if (x == detail::ZERO_DOUBLE) {
            if (alpha_ > detail::ONE)
                return detail::ZERO_DOUBLE;
            if (std::abs(alpha_ - detail::ONE) <= detail::DEFAULT_TOLERANCE)
                return std::exp(logNormConst_);
            return std::numeric_limits<double>::infinity();
        }
        // x = 1
        if (beta_ > detail::ONE)
            return detail::ZERO_DOUBLE;
        if (std::abs(beta_ - detail::ONE) <= detail::DEFAULT_TOLERANCE)
            return std::exp(logNormConst_);
        return std::numeric_limits<double>::infinity();
    }
    return std::exp(logNormConst_ + alphaMinus1_ * std::log(x) +
                    betaMinus1_ * std::log(detail::ONE - x));
}

double BetaDistribution::getLogProbability(double x) const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (x < detail::ZERO_DOUBLE || x > detail::ONE) {
        return -std::numeric_limits<double>::infinity();
    }
    if (x == detail::ZERO_DOUBLE) {
        if (alpha_ > detail::ONE)
            return -std::numeric_limits<double>::infinity();
        if (std::abs(alpha_ - detail::ONE) <= detail::DEFAULT_TOLERANCE)
            return logNormConst_;
        return std::numeric_limits<double>::infinity();
    }
    if (x == detail::ONE) {
        if (beta_ > detail::ONE)
            return -std::numeric_limits<double>::infinity();
        if (std::abs(beta_ - detail::ONE) <= detail::DEFAULT_TOLERANCE)
            return logNormConst_;
        return std::numeric_limits<double>::infinity();
    }
    return logNormConst_ + alphaMinus1_ * std::log(x) + betaMinus1_ * std::log(detail::ONE - x);
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
    std::vector<double> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(sample(rng));
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
        log_1mx_sum += std::log(detail::ONE - v);
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

        // Diagonal Newton step: use trigamma for the Jacobian diagonal
        // trigamma(x) ≈ (digamma(x+h) - digamma(x-h)) / (2h) — numerical derivative
        const double h = 1e-4;
        const double tpsi_a =
            (detail::digamma(alpha_cur + h) - detail::digamma(alpha_cur - h)) / (detail::TWO * h);
        const double tpsi_b =
            (detail::digamma(beta_cur + h) - detail::digamma(beta_cur - h)) / (detail::TWO * h);
        const double tpsi_ab =
            (detail::digamma(ab + h) - detail::digamma(ab - h)) / (detail::TWO * h);

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

void BetaDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = detail::ONE;
    beta_ = detail::ONE;
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
        *this, values, results, hint, detail::DistributionTraits<BetaDistribution>::distType(),
        detail::OperationType::PDF,
        [](const BetaDistribution& dist, double value) { return dist.getProbability(value); },
        [](const BetaDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, lnc, am1, bm1);
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            // Chunk the batch so each parallel task uses the SIMD pipeline
            // (vector_log / vector_exp) instead of per-element scalar math.
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    dist.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                       lnc, am1, bm1);
                });
            } else {
                dist.getProbabilityBatchUnsafeImpl(vals.data(), res.data(), count, lnc, am1, bm1);
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
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
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
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
                dist.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                   lnc, am1, bm1);
            });
        });
}

void BetaDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                         const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<BetaDistribution>::distType(),
        detail::OperationType::LOG_PDF,
        [](const BetaDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const BetaDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, lnc, am1, bm1);
        },
        [](const BetaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Span size mismatch");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            // Chunk the batch so each parallel task uses the SIMD pipeline
            // (vector_log) instead of per-element scalar math.
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    dist.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start,
                                                          len, lnc, am1, bm1);
                });
            } else {
                dist.getLogProbabilityBatchUnsafeImpl(vals.data(), res.data(), count, lnc, am1,
                                                      bm1);
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
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
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
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    const_cast<BetaDistribution&>(dist).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double am1 = dist.alphaMinus1_;
            const double bm1 = dist.betaMinus1_;
            lock.unlock();
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
                dist.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                      lnc, am1, bm1);
            });
        });
}

void BetaDistribution::getCumulativeProbability(std::span<const double> values,
                                                std::span<double> results,
                                                const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<BetaDistribution>::distType(),
        detail::OperationType::CDF,
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
            const double log_prefix = detail::lgamma(a + b) - detail::lgamma(a) - detail::lgamma(b);
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
            const double log_prefix = detail::lgamma(a + b) - detail::lgamma(a) - detail::lgamma(b);
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= 0.0)
                    res[i] = 0.0;
                else if (x >= 1.0)
                    res[i] = 1.0;
                else
                    res[i] = detail::beta_i(x, a, b, log_prefix);
            });
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
            const double log_prefix = detail::lgamma(a + b) - detail::lgamma(a) - detail::lgamma(b);
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= 0.0)
                    res[i] = 0.0;
                else if (x >= 1.0)
                    res[i] = 1.0;
                else
                    res[i] = detail::beta_i(x, a, b, log_prefix);
            });
        });
}

static detail::PerformanceHint betaStrategyToHint(detail::Strategy strategy) noexcept {
    detail::PerformanceHint hint;
    switch (strategy) {
        case detail::Strategy::SCALAR:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
            break;
        case detail::Strategy::VECTORIZED:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
            break;
        case detail::Strategy::PARALLEL:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
            break;
        case detail::Strategy::WORK_STEALING:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT;
            break;
    }
    return hint;
}

void BetaDistribution::getProbabilityWithStrategy(std::span<const double> values,
                                                  std::span<double> results,
                                                  detail::Strategy strategy) const {
    getProbability(values, results, betaStrategyToHint(strategy));
}

void BetaDistribution::getLogProbabilityWithStrategy(std::span<const double> values,
                                                     std::span<double> results,
                                                     detail::Strategy strategy) const {
    getLogProbability(values, results, betaStrategyToHint(strategy));
}

void BetaDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                                            std::span<double> results,
                                                            detail::Strategy strategy) const {
    getCumulativeProbability(values, results, betaStrategyToHint(strategy));
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
    std::string token;
    is >> token;
    if (token.find("BetaDistribution(") != 0) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t a_pos = token.find("alpha=");
    const size_t comma = token.find(",", a_pos);
    const size_t b_pos = token.find("beta=");
    const size_t close = token.find(")", b_pos);
    if (a_pos == std::string::npos || comma == std::string::npos || b_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double a = std::stod(token.substr(a_pos + 6, comma - a_pos - 6));
        const double b = std::stod(token.substr(b_pos + 5, close - b_pos - 5));
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
// Two vector_log calls (one for log(x), one for log(1-x)) + one aligned temp.
// Scalar fixup for x <= 0 or x >= 1 (delegates to single-value method).
//
// LogPDF (8 steps):
//   Step 1: temp    = log(x)                  [vector_log(values, temp)]
//   Step 2: temp    = (α-1)*log(x)             [scalar_multiply]
//   Step 3: results = x-1                      [scalar_add(values, -1)]
//   Step 4: results = 1-x                      [scalar_multiply(results, -1)]
//   Step 5: results = log(1-x)                 [vector_log]
//   Step 6: results = (β-1)*log(1-x)           [scalar_multiply]
//   Step 7: results = (α-1)log(x)+(β-1)log(1-x) [vector_add(temp, results)]
//   Step 8: results += log_norm_const          [scalar_add]
//
// PDF: steps 1-8 then vector_exp.
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

void BetaDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                     std::size_t count, double log_norm_const,
                                                     double alpha_minus_one,
                                                     double beta_minus_one) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= 0.0 || x >= 1.0) {
                results[i] = getProbability(x);
            } else {
                results[i] = std::exp(log_norm_const + alpha_minus_one * std::log(x) +
                                      beta_minus_one * std::log(detail::ONE - x));
            }
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1-2: temp = (α-1)*log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    arch::simd::VectorOps::scalar_multiply(temp.data(), alpha_minus_one, temp.data(), count);

    // Step 3-6: results = (β-1)*log(1-x)
    arch::simd::VectorOps::scalar_add(values, -detail::ONE, results, count);        // x-1
    arch::simd::VectorOps::scalar_multiply(results, -detail::ONE, results, count);  // 1-x
    arch::simd::VectorOps::vector_log(results, results, count);                     // log(1-x)
    arch::simd::VectorOps::scalar_multiply(results, beta_minus_one, results, count);

    // Step 7: results = (α-1)*log(x) + (β-1)*log(1-x)
    arch::simd::VectorOps::vector_add(temp.data(), results, results, count);

    // Step 8: results += log_norm_const
    arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);

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
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= 0.0 || x >= 1.0) {
                results[i] = getLogProbability(x);
            } else {
                results[i] = log_norm_const + alpha_minus_one * std::log(x) +
                             beta_minus_one * std::log(detail::ONE - x);
            }
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1-2: temp = (α-1)*log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    arch::simd::VectorOps::scalar_multiply(temp.data(), alpha_minus_one, temp.data(), count);

    // Step 3-6: results = (β-1)*log(1-x)
    arch::simd::VectorOps::scalar_add(values, -detail::ONE, results, count);
    arch::simd::VectorOps::scalar_multiply(results, -detail::ONE, results, count);
    arch::simd::VectorOps::vector_log(results, results, count);
    arch::simd::VectorOps::scalar_multiply(results, beta_minus_one, results, count);

    // Step 7-8: full LogPDF
    arch::simd::VectorOps::vector_add(temp.data(), results, results, count);
    arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);

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
    const double log_prefix =
        detail::lgamma(alpha + beta) - detail::lgamma(alpha) - detail::lgamma(beta);
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

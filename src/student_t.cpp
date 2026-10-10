#include "libstats/distributions/student_t.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"  // provides detail::digamma, detail::t_cdf, detail::inverse_t_cdf
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stats {

namespace {
// PARALLEL and WORK_STEALING run the batch kernel over slices of this many elements (#191).
constexpr std::size_t kBatchSlice = 1024;
}  // namespace

namespace {
// log(1 + t²/ν) for the Student-t kernel (#159). Past |t| ~ 1e154 t² overflows and the direct form
// collapses the density to 0 / −inf; there log(1 + t²/ν) = 2·log|t| + log(1/ν) + log1p(ν/t²), and
// the last term is below double resolution.
[[nodiscard]] inline double log1p_t2_over_nu(double t, double inv_nu) noexcept {
    const double u = t * t * inv_nu;
    return std::isfinite(u) ? std::log1p(u) : 2.0 * std::log(std::fabs(t)) + std::log(inv_nu);
}

// D(ν) = ψ((ν+1)/2) − ψ(ν/2) − 1/ν, the digamma part of the ν score, and D'(ν). The two digammas
// agree to 1/ν and D ~ 1/(2ν²), so the direct form loses about 2·log10(ν) digits: 4e-8 relative at
// ν = 1e4, nothing left by 1e8. From ν = 100 use the asymptotic series
// ψ(z + ½) − ψ(z) = Σ_k (2 − 2^{1−k})·B_k / (k·z^k), z = ν/2; its first omitted term is below
// 1e-17 relative there, and it meets the direct form to 3e-12 at the switch.
struct TScoreDigammaPart {
    double d;
    double dd;
};
[[nodiscard]] inline TScoreDigammaPart tScoreDigammaPart(double nu) noexcept {
    if (nu >= 100.0) {
        const double w = 1.0 / (nu * nu);
        return {w * (0.5 + w * (-0.25 + w * (0.5 + w * (-17.0 / 8.0 + w * 15.5)))),
                -w / nu * (1.0 + w * (-1.0 + w * (3.0 + w * (-17.0 + w * 155.0))))};
    }
    return {
        detail::digamma((nu + 1.0) * 0.5) - detail::digamma(nu * 0.5) - 1.0 / nu,
        0.5 * (detail::trigamma((nu + 1.0) * 0.5) - detail::trigamma(nu * 0.5)) + 1.0 / (nu * nu)};
}

// log(1 + t²/ν) − r with r = t²/(ν + t²). Both terms are ~t²/ν and the difference ~r²/2, so for
// small r sum the series −log(1 − r) − r = Σ_{k≥2} r^k/k (1 − r = 1/(1 + t²/ν)), all terms
// positive; its first omitted term is below 1e-21 relative at r < 1e-3.
[[nodiscard]] inline double log1pMinusRatio(double t, double inv_nu, double r) noexcept {
    if (r < 1e-3) {
        return r * r *
               (1.0 / 2 +
                r * (1.0 / 3 +
                     r * (1.0 / 4 + r * (1.0 / 5 + r * (1.0 / 6 + r * (1.0 / 7 + r / 8))))));
    }
    return log1p_t2_over_nu(t, inv_nu) - r;
}

// The SIMD pipeline forms x² and returns 0 / −inf where it overflows; redo those lanes through the
// overflow-safe form, deciding from the input, not the result.
[[nodiscard]] inline bool kernelOverflows(double t, double inv_nu) noexcept {
    return std::isfinite(t) && !std::isfinite(t * t * inv_nu);
}

// The SIMD pipeline takes vector_log(1 + x²/ν), whose absolute error of ~ε the density multiplies
// by (ν + 1)/2: 1e-10 relative at ν = 1e6. Past (ν + 1)/2 = 16 the batch takes the scalar log1p
// loop instead.
[[nodiscard]] inline bool simdLogIsAccurate(double neg_half_nu_plus_one) noexcept {
    return neg_half_nu_plus_one >= -16.0;
}
}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

static double requireValidNu(double nu) {
    if (nu <= 0.0 || !std::isfinite(nu)) {
        throw std::invalid_argument("Degrees of freedom nu must be a positive finite number");
    }
    return nu;
}

StudentTDistribution::StudentTDistribution(double nu)
    : DistributionBase(), nu_(requireValidNu(nu)) {
    updateCacheUnsafe();
}

StudentTDistribution::StudentTDistribution(const StudentTDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    nu_ = other.nu_;
    halfNu_ = other.halfNu_;
    halfNuPlusOne_ = other.halfNuPlusOne_;
    negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
    invNu_ = other.invNu_;
    logNormConst_ = other.logNormConst_;
    variance_ = other.variance_;
    kurtosis_ = other.kurtosis_;
    isCauchy_ = other.isCauchy_;
    isMeanDefined_ = other.isMeanDefined_;
    isVarianceDefined_ = other.isVarianceDefined_;
    atomicNu_.store(nu_, std::memory_order_release);
}

StudentTDistribution& StudentTDistribution::operator=(const StudentTDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        nu_ = other.nu_;
        halfNu_ = other.halfNu_;
        halfNuPlusOne_ = other.halfNuPlusOne_;
        negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
        invNu_ = other.invNu_;
        logNormConst_ = other.logNormConst_;
        variance_ = other.variance_;
        kurtosis_ = other.kurtosis_;
        isCauchy_ = other.isCauchy_;
        isMeanDefined_ = other.isMeanDefined_;
        isVarianceDefined_ = other.isVarianceDefined_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicNu_.store(nu_, std::memory_order_release);
    }
    return *this;
}

StudentTDistribution::StudentTDistribution(StudentTDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    nu_ = other.nu_;
    halfNu_ = other.halfNu_;
    halfNuPlusOne_ = other.halfNuPlusOne_;
    negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
    invNu_ = other.invNu_;
    logNormConst_ = other.logNormConst_;
    variance_ = other.variance_;
    kurtosis_ = other.kurtosis_;
    isCauchy_ = other.isCauchy_;
    isMeanDefined_ = other.isMeanDefined_;
    isVarianceDefined_ = other.isVarianceDefined_;
    other.nu_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicNu_.store(nu_, std::memory_order_release);
}

StudentTDistribution& StudentTDistribution::operator=(StudentTDistribution&& other) noexcept {
    if (this != &other) {
        // Both locks, as copy-assignment takes, in std::lock order; the source is written, so
        // exclusively (#184).
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        nu_ = other.nu_;
        halfNu_ = other.halfNu_;
        halfNuPlusOne_ = other.halfNuPlusOne_;
        negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
        invNu_ = other.invNu_;
        logNormConst_ = other.logNormConst_;
        variance_ = other.variance_;
        kurtosis_ = other.kurtosis_;
        isCauchy_ = other.isCauchy_;
        isMeanDefined_ = other.isMeanDefined_;
        isVarianceDefined_ = other.isVarianceDefined_;
        other.nu_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicNu_.store(nu_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

StudentTDistribution StudentTDistribution::createUnchecked(double nu) noexcept {
    return StudentTDistribution(nu, true);
}

StudentTDistribution::StudentTDistribution(double nu, bool /*bypassValidation*/) noexcept
    : DistributionBase(), nu_(nu) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

void StudentTDistribution::setNu(double nu) {
    validateParameters(nu);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    nu_ = nu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

VoidResult StudentTDistribution::trySetNu(double nu) noexcept {
    auto validation = validateStudentTParameters(nu);
    if (validation.isError()) {
        return validation;
    }
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    nu_ = nu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult StudentTDistribution::validateCurrentParameters() const noexcept {
    return validateStudentTParameters(getNu());
}

//==============================================================================
// 3. STATISTICAL MOMENTS
//==============================================================================

double StudentTDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!isMeanDefined_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return detail::ZERO_DOUBLE;
}

double StudentTDistribution::getVariance() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return variance_;
}

double StudentTDistribution::getSkewness() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (nu_ <= 3.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return detail::ZERO_DOUBLE;
}

double StudentTDistribution::getKurtosis() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return kurtosis_;
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double StudentTDistribution::getProbability(double x) const {
    double lnc, nhnpo, inv_nu;
    withCacheSnapshot([&] {
        lnc = logNormConst_;
        nhnpo = negHalfNuPlusOne_;
        inv_nu = invNu_;
    });
    return std::exp(lnc + nhnpo * log1p_t2_over_nu(x, inv_nu));
}

double StudentTDistribution::getLogProbability(double x) const {
    double lnc, nhnpo, inv_nu;
    withCacheSnapshot([&] {
        lnc = logNormConst_;
        nhnpo = negHalfNuPlusOne_;
        inv_nu = invNu_;
    });
    return lnc + nhnpo * log1p_t2_over_nu(x, inv_nu);
}

double StudentTDistribution::getCumulativeProbability(double x) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_nu = nu_;
    lock.unlock();
    return detail::t_cdf(x, cached_nu);
}

double StudentTDistribution::getQuantile(double p) const {
    if (std::isnan(p))
        return std::numeric_limits<double>::quiet_NaN();  // NaN in, NaN out (AR D3)
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_nu = nu_;
    lock.unlock();
    return detail::inverse_t_cdf(p, cached_nu);
}

double StudentTDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_half_nu = halfNu_;
    const double cached_nu = nu_;
    lock.unlock();

    // t = Z / sqrt(chi2 / nu)  where chi2 ~ Gamma(nu/2, 2) = chi-squared(nu)
    std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);
    std::gamma_distribution<double> gamma_gen(cached_half_nu, detail::TWO);

    const double z = normal(rng);
    const double chi2 = gamma_gen(rng);
    return z / std::sqrt(chi2 / cached_nu);
}

std::vector<double> StudentTDistribution::sample(std::mt19937& rng, size_t n) const {
    // Read parameters once to avoid n lock acquisitions.
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_half_nu = halfNu_;
    const double cached_nu = nu_;
    lock.unlock();

    std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);
    std::gamma_distribution<double> gamma_gen(cached_half_nu, detail::TWO);
    std::vector<double> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const double z = normal(rng);
        const double chi2 = gamma_gen(rng);
        samples.push_back(z / std::sqrt(chi2 / cached_nu));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void StudentTDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    for (double v : values) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("Data must contain only finite values");
        }
    }

    const double n = static_cast<double>(values.size());

    // Upper bound, the Gaussian limit. In θ = 1/ν the MLE is asymptotically N(θ, 1/(3.5 n)) near
    // θ = 0, so n data resolve ν only up to ~sqrt(3.5 n); the old bound of 1000 was reached by
    // Gaussian data at any n and cut off estimates the data supported (n ≳ 1e6). 1e8 is past any
    // ν a sample held in memory can tell from a Gaussian, and t(1e8) differs from N(0, 1) by ~1e-8.
    constexpr double NU_MAX = 1e8;

    // Initial estimate: method of moments using sample kurtosis.
    // Excess kurtosis = 6/(nu-4) for nu>4, so nu = 4 + 6/kurtosis.
    // For nu <= 4, or when sample kurtosis is unavailable, start at nu=5.
    // Clamp the initial estimate to keep the optimizer in a region with
    // meaningful gradient — starting above ~100 risks flat-tail divergence.
    double nu_est = 5.0;
    if (values.size() >= 4) {
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
        double m2 = 0.0, m4 = 0.0;
        for (double v : values) {
            double d = v - mean;
            double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n;
        m4 /= n;
        if (m2 > detail::ZERO_DOUBLE) {
            double excess_kurt = m4 / (m2 * m2) - 3.0;
            if (excess_kurt > detail::ZERO_DOUBLE) {
                double nu_from_kurt = 4.0 + 6.0 / excess_kurt;
                if (nu_from_kurt > detail::ONE && std::isfinite(nu_from_kurt)) {
                    nu_est = std::min(nu_from_kurt, 100.0);
                }
            }
        }
    }

    // Newton-Raphson on the score equation S(nu) = 0:
    //   S(nu) = n*[psi((nu+1)/2) - psi(nu/2) - 1/nu]
    //           - sum(log(1 + xi^2/nu))
    //           + ((nu+1)/nu) * sum(xi^2 / (nu + xi^2))
    //
    // Exact derivative (replaces 3-point finite differences which cost 3n digamma evals/step):
    //   S'(nu) = n/2 * [psi'((nu+1)/2) - psi'(nu/2)] + n/nu^2
    //            - (1/nu^2) * sum(xi^2 * (xi^2 - nu) / (nu + xi^2)^2)   [data term]
    // = 2 trigamma calls + 1 additional data pass.
    // Beta already uses the same exact-derivative pattern; StudentT converges in fewer steps.

    // Solved as a bracketed Newton in u = log ν on [log NU_MIN, log NU_MAX]: S > 0 raises the
    // lower end, S < 0 lowers the upper, a Newton step (g'(u) = ν·S'(ν)) is taken only where it
    // is a descent step that stays inside the bracket, a bisection otherwise. The plain step in
    // ν this replaced was capped upward but not downward, so from the ν = 5 start on data with
    // ν ≤ 2 the first step overshot to the 0.1 floor and stopped there (F1).
    constexpr double NU_MIN = 0.1;
    const int max_iter = 100;
    const double tol = 1e-8;
    // S and S' at ν. The data term −log(1 + xi²/ν) + ((ν+1)/ν)·r is formed as r/ν − (log(1 + xi²/ν)
    // − r), and the digamma part through tScoreDigammaPart, so that S keeps its relative accuracy
    // where it falls as ~1/ν²: per observation S ≈ (1 + 2xi² − xi⁴)/(2ν²) at large ν.
    const auto score = [&values, n](double nu, double& s, double& ds) {
        const TScoreDigammaPart dig = tScoreDigammaPart(nu);
        const double inv_nu = detail::ONE / nu;
        s = n * dig.d;
        ds = n * dig.dd;
        // r = xi²/(ν + xi²), formed as 1/(1 + ν/xi²) so that xi² = 0 gives 0 and an overflowed
        // xi² gives 1; then (xi² − ν)/(ν + xi²) = 2r − 1. The log term is overflow-safe (#159).
        for (double xi : values) {
            const double r = detail::ONE / (detail::ONE + nu / (xi * xi));
            s += r * inv_nu - log1pMinusRatio(xi, inv_nu, r);
            ds -= r * (detail::TWO * r - detail::ONE) * inv_nu * inv_nu;
        }
    };

    // Gaussian limit: where S(NU_MAX) ≥ 0 the likelihood still rises at the bound (the data are no
    // heavier-tailed than a Gaussian), and the fit returns NU_MAX itself.
    double s = 0.0, ds = 0.0;
    score(NU_MAX, s, ds);
    if (!(s < detail::ZERO_DOUBLE)) {
        setNu(NU_MAX);
        return;
    }

    double nu = nu_est;
    double lo = std::log(NU_MIN), hi = std::log(NU_MAX);

    for (int iter = 0; iter < max_iter; ++iter) {
        score(nu, s, ds);

        // S falls as ~1/ν² at large ν, so an unscaled |S| < tol·n is met anywhere past ν ~ 1e4;
        // test ν²·S there (the score in θ = 1/ν).
        if (std::abs(s) * std::max(detail::ONE, nu * nu) < tol * n)
            break;

        const double u = std::log(nu);
        if (s > detail::ZERO_DOUBLE)
            lo = u;
        else
            hi = u;
        if (hi - lo < tol)
            break;
        // g(u) = S(eᵘ), g'(u) = ν·S'(ν); Newton where S is falling, inside the bracket.
        double next = u - s / (nu * ds);
        if (!(ds < detail::ZERO_DOUBLE) || !(next > lo && next < hi))
            next = detail::HALF * (lo + hi);
        const double step = next - u;
        nu = std::exp(next);
        if (std::abs(step) < tol)
            break;
    }

    setNu(std::clamp(nu, NU_MIN, NU_MAX));
}

void StudentTDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<StudentTDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void StudentTDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    nu_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string StudentTDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "StudentTDistribution(nu=" << nu_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double StudentTDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // H = (nu+1)/2 * [psi((nu+1)/2) - psi(nu/2)] + log(sqrt(nu)*B(nu/2, 1/2))
    // where B is the beta function.
    // Simplified: H = halfNuPlusOne_*(psi(halfNuPlusOne_) - psi(halfNu_)) + lbeta(halfNu_, 0.5)
    //                 + 0.5*log(nu)
    const double psi_plus = detail::digamma(halfNuPlusOne_);
    const double psi_half = detail::digamma(halfNu_);
    return halfNuPlusOne_ * (psi_plus - psi_half) + detail::lbeta(halfNu_, detail::HALF) +
           detail::HALF * std::log(nu_);
}

//==============================================================================
// 13–14. SMART AUTO-DISPATCH AND EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

void StudentTDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const StudentTDistribution& dist, double value) { return dist.getProbability(value); },
        [](const StudentTDistribution& dist, const double* vals, double* res, size_t count) {
            double lnc, nhnpo, inv_nu;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                nhnpo = dist.negHalfNuPlusOne_;
                inv_nu = dist.invNu_;
            });
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, lnc, nhnpo, inv_nu);
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, nhnpo, inv_nu;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                nhnpo = dist.negHalfNuPlusOne_;
                inv_nu = dist.invNu_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    dist.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                       lnc, nhnpo, inv_nu);
                });
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, nhnpo, inv_nu;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                nhnpo = dist.negHalfNuPlusOne_;
                inv_nu = dist.invNu_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                dist.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                   lnc, nhnpo, inv_nu);
            });
        });
}

void StudentTDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const StudentTDistribution& dist, double value) {
            return dist.getLogProbability(value);
        },
        [](const StudentTDistribution& dist, const double* vals, double* res, size_t count) {
            double lnc, nhnpo, inv_nu;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                nhnpo = dist.negHalfNuPlusOne_;
                inv_nu = dist.invNu_;
            });
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, lnc, nhnpo, inv_nu);
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, nhnpo, inv_nu;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                nhnpo = dist.negHalfNuPlusOne_;
                inv_nu = dist.invNu_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    dist.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start,
                                                          len, lnc, nhnpo, inv_nu);
                });
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double lnc, nhnpo, inv_nu;
            dist.withCacheSnapshot([&] {
                lnc = dist.logNormConst_;
                nhnpo = dist.negHalfNuPlusOne_;
                inv_nu = dist.invNu_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                dist.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                      lnc, nhnpo, inv_nu);
            });
        });
}

void StudentTDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const StudentTDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const StudentTDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_nu);
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            // Slices of the batch kernel; this lambda was a serial loop (#176).
            constexpr std::size_t CHUNK = 1024;
            ParallelUtils::parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start,
                                                             res.data() + start, len, cached_nu);
            });
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start,
                                                             res.data() + start, len, cached_nu);
            });
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool StudentTDistribution::operator==(const StudentTDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(nu_ - other.nu_) <= detail::DEFAULT_TOLERANCE;
}

bool StudentTDistribution::operator!=(const StudentTDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const StudentTDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, StudentTDistribution& dist) {
    std::string token;
    double nu;

    is >> token;
    if (!token.starts_with("StudentTDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t nu_pos = token.find("nu=");
    if (nu_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t close = token.find(")", nu_pos);
    if (close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        nu = detail::parse_double(token.substr(nu_pos + 3, close - nu_pos - 3));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }
    auto result = dist.trySetNu(nu);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// Log-space pipeline for PDF and LogPDF.
// No out-of-support fixup: Student's t is defined on all of ℝ.
//
// LogPDF:
//   Step 1: results = x²                     (vector_multiply)
//   Step 2: results = x²/ν                   (scalar_multiply by inv_nu)
//   Step 3: results = 1 + x²/ν              (scalar_add 1)
//   Step 4: results = log(1 + x²/ν)         (vector_log)
//   Step 5: results = −(ν+1)/2 · log(...)   (scalar_multiply by neg_half_nu_plus_one)
//   Step 6: results += log_norm_const        (scalar_add)
//
// PDF: steps 1–6 then vector_exp.
//
// CDF architecture: detail::t_cdf delegates to regularized incomplete beta
//   (beta_i) via the identity t_cdf(t, ν) = 1 - 0.5·I_{x(t)}(ν/2, 1/2).
//   beta_i uses a continued-fraction algorithm whose iteration count varies
//   per input: the same fundamental constraint as Gamma CDF (see gamma.cpp
//   section 18). PDF and LogPDF use a fixed 6-step pipeline and achieve
//   4–8x SIMD speedup; this CDF path is scalar for the same reason.
//==============================================================================

void StudentTDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count, double log_norm_const,
                                                         double neg_half_nu_plus_one,
                                                         double inv_nu) const noexcept {
    const bool use_simd =
        arch::simd::SIMDPolicy::shouldUseSIMD(count) && simdLogIsAccurate(neg_half_nu_plus_one);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            results[i] = std::exp(log_norm_const +
                                  neg_half_nu_plus_one * log1p_t2_over_nu(values[i], inv_nu));
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = x²/ν
    arch::simd::VectorOps::scalar_multiply(results, inv_nu, results, count);
    // Step 3: results = 1 + x²/ν
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    // Step 4: results = log(1 + x²/ν)
    arch::simd::VectorOps::vector_log(results, results, count);
    // Step 5: results = −(ν+1)/2 · log(1 + x²/ν)
    arch::simd::VectorOps::scalar_multiply(results, neg_half_nu_plus_one, results, count);
    // Step 6: results += log_norm_const → full LogPDF
    arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);
    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 7: lanes where x²/ν overflowed came out 0 (#159).
    for (std::size_t i = 0; i < count; ++i) {
        if (kernelOverflows(values[i], inv_nu))
            results[i] = std::exp(log_norm_const +
                                  neg_half_nu_plus_one * log1p_t2_over_nu(values[i], inv_nu));
    }
}

void StudentTDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                            std::size_t count,
                                                            double log_norm_const,
                                                            double neg_half_nu_plus_one,
                                                            double inv_nu) const noexcept {
    const bool use_simd =
        arch::simd::SIMDPolicy::shouldUseSIMD(count) && simdLogIsAccurate(neg_half_nu_plus_one);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            results[i] =
                log_norm_const + neg_half_nu_plus_one * log1p_t2_over_nu(values[i], inv_nu);
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = x²/ν
    arch::simd::VectorOps::scalar_multiply(results, inv_nu, results, count);
    // Step 3: results = 1 + x²/ν
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    // Step 4: results = log(1 + x²/ν)
    arch::simd::VectorOps::vector_log(results, results, count);
    // Step 5: results = −(ν+1)/2 · log(1 + x²/ν)
    arch::simd::VectorOps::scalar_multiply(results, neg_half_nu_plus_one, results, count);
    // Step 6: results += log_norm_const → full LogPDF
    arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);
    // Step 7: lanes where x²/ν overflowed came out −inf (#159).
    for (std::size_t i = 0; i < count; ++i) {
        if (kernelOverflows(values[i], inv_nu))
            results[i] =
                log_norm_const + neg_half_nu_plus_one * log1p_t2_over_nu(values[i], inv_nu);
    }
}

void StudentTDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                   double* results,
                                                                   std::size_t count,
                                                                   double nu) const noexcept {
    // Scalar per element. See section 18 header for the explanation.
    const double lbeta_a_half = detail::lbeta(detail::HALF * nu, detail::HALF);
    for (std::size_t i = 0; i < count; ++i) {
        results[i] = detail::t_cdf(values[i], nu, lbeta_a_half);
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

// computeDigamma removed: promoted to detail::digamma in math_utils.
//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void StudentTDistribution::updateCacheUnsafe() const noexcept {
    halfNu_ = nu_ * detail::HALF;
    halfNuPlusOne_ = (nu_ + detail::ONE) * detail::HALF;
    negHalfNuPlusOne_ = -halfNuPlusOne_;
    invNu_ = detail::ONE / nu_;

    // log normalization constant: lgamma((ν+1)/2) − 0.5·log(ν·π) − lgamma(ν/2)
    // lgamma((ν+1)/2) − lgamma(ν/2) − ½·log(νπ) = −log B(ν/2, ½) − ½·log ν; lbeta forms it without
    // the lgamma difference, which was 2e-10 relative off at ν = 1e6.
    logNormConst_ = -detail::lbeta(halfNu_, detail::HALF) - detail::HALF * std::log(nu_);

    // Moments (conditional on ν)
    if (nu_ > detail::TWO) {
        variance_ = nu_ / (nu_ - detail::TWO);
    } else if (nu_ > detail::ONE) {
        variance_ = std::numeric_limits<double>::infinity();
    } else {
        variance_ = std::numeric_limits<double>::quiet_NaN();
    }

    if (nu_ > 4.0) {
        kurtosis_ = 6.0 / (nu_ - 4.0);
    } else {
        kurtosis_ = std::numeric_limits<double>::quiet_NaN();
    }

    // Optimization flags
    isCauchy_ = (std::abs(nu_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
    isMeanDefined_ = (nu_ > detail::ONE);
    isVarianceDefined_ = (nu_ > detail::TWO);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicNu_.store(nu_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

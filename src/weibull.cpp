#include "libstats/distributions/weibull.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;
using stats::detail::validateNonNegativeParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

WeibullDistribution::WeibullDistribution(double shape, double scale)
    : DistributionBase(), shape_(shape), scale_(scale) {
    validateParameters(shape, scale);
    updateCacheUnsafe();
}

WeibullDistribution::WeibullDistribution(const WeibullDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    shape_ = other.shape_;
    scale_ = other.scale_;
    logShape_ = other.logShape_;
    logScale_ = other.logScale_;
    shapeMinus1_ = other.shapeMinus1_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    isExponential_ = other.isExponential_;
    atomicShape_.store(shape_, std::memory_order_release);
    atomicScale_.store(scale_, std::memory_order_release);
}

WeibullDistribution& WeibullDistribution::operator=(const WeibullDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        shape_ = other.shape_;
        scale_ = other.scale_;
        logShape_ = other.logShape_;
        logScale_ = other.logScale_;
        shapeMinus1_ = other.shapeMinus1_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        isExponential_ = other.isExponential_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicShape_.store(shape_, std::memory_order_release);
        atomicScale_.store(scale_, std::memory_order_release);
    }
    return *this;
}

WeibullDistribution::WeibullDistribution(WeibullDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    shape_ = other.shape_;
    scale_ = other.scale_;
    logShape_ = other.logShape_;
    logScale_ = other.logScale_;
    shapeMinus1_ = other.shapeMinus1_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    isExponential_ = other.isExponential_;
    other.shape_ = detail::ONE;
    other.scale_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicShape_.store(shape_, std::memory_order_release);
    atomicScale_.store(scale_, std::memory_order_release);
}

WeibullDistribution& WeibullDistribution::operator=(WeibullDistribution&& other) noexcept {
    if (this != &other) {

        shape_ = other.shape_;
        scale_ = other.scale_;
        logShape_ = other.logShape_;
        logScale_ = other.logScale_;
        shapeMinus1_ = other.shapeMinus1_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        isExponential_ = other.isExponential_;
        other.shape_ = detail::ONE;
        other.scale_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicShape_.store(shape_, std::memory_order_release);
        atomicScale_.store(scale_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

WeibullDistribution WeibullDistribution::createUnchecked(double shape, double scale) noexcept {
    return WeibullDistribution(shape, scale, true);
}

WeibullDistribution::WeibullDistribution(double shape, double scale,
                                         bool /*bypassValidation*/) noexcept
    : DistributionBase(), shape_(shape), scale_(scale) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void WeibullDistribution::setShape(double shape) {
    // Acquire lock first, then validate against scale_ member to eliminate
    // the TOCTOU window where a concurrent setScale() could change scale_
    // between validateParameters() and the unique_lock acquisition.
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(shape, scale_);
    shape_ = shape;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void WeibullDistribution::setScale(double scale) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(shape_, scale);
    scale_ = scale;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void WeibullDistribution::setParameters(double shape, double scale) {
    validateParameters(shape, scale);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    shape_ = shape;
    scale_ = scale;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double WeibullDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return mean_;
}

double WeibullDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return variance_;
}

double WeibullDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // Skewness = [Γ(1+3/k)λ³ − 3μσ² − μ³] / σ³  where σ² = variance_
    // More efficiently via standardized moments:
    const double g1 = std::exp(detail::lgamma(detail::ONE + detail::ONE / shape_));
    const double g2 = std::exp(detail::lgamma(detail::ONE + detail::TWO / shape_));
    const double g3 = std::exp(detail::lgamma(detail::ONE + detail::THREE / shape_));
    const double var_raw = g2 - g1 * g1;  // λ-normalised variance
    if (var_raw <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    const double num = g3 - detail::THREE * g1 * g2 + detail::TWO * g1 * g1 * g1;
    return scale_ * scale_ * scale_ * num / (variance_ * std::sqrt(variance_));
}

double WeibullDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double g1 = std::exp(detail::lgamma(detail::ONE + detail::ONE / shape_));
    const double g2 = std::exp(detail::lgamma(detail::ONE + detail::TWO / shape_));
    const double g3 = std::exp(detail::lgamma(detail::ONE + detail::THREE / shape_));
    const double g4 = std::exp(detail::lgamma(detail::ONE + detail::FOUR / shape_));
    const double var_raw = g2 - g1 * g1;
    if (var_raw <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    // Raw fourth central moment (λ-normalised) / normalised-variance² − 3
    const double m4 = (g4 - detail::FOUR * g3 * g1 + detail::SIX * g2 * g1 * g1 -
                       detail::THREE * g1 * g1 * g1 * g1);
    return scale_ * scale_ * scale_ * scale_ * m4 / (variance_ * variance_) - detail::THREE;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult WeibullDistribution::trySetShape(double shape) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateWeibullParameters(shape, scale_);
    if (v.isError())
        return v;
    shape_ = shape;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult WeibullDistribution::trySetScale(double scale) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateWeibullParameters(shape_, scale);
    if (v.isError())
        return v;
    scale_ = scale;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult WeibullDistribution::trySetParameters(double shape, double scale) noexcept {
    auto v = validateWeibullParameters(shape, scale);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    shape_ = shape;
    scale_ = scale;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult WeibullDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateWeibullParameters(shape_, scale_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double WeibullDistribution::getProbability(double x) const {
    if (x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // At x = 0: PDF = 1/λ for k=1; 0 for k>1; +∞ for k<1
    // getLogProbability correctly returns +inf / -inf; getProbability must match.
    if (x == detail::ZERO_DOUBLE) {
        if (std::abs(shape_ - detail::ONE) <= detail::DEFAULT_TOLERANCE)
            return detail::ONE / scale_;
        return (shape_ > detail::ONE) ? detail::ZERO_DOUBLE
                                      : std::numeric_limits<double>::infinity();
    }
    return std::exp(getLogProbability(x));
}

double WeibullDistribution::getLogProbability(double x) const noexcept {
    if (x < detail::ZERO_DOUBLE)
        return detail::NEGATIVE_INFINITY;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (x == detail::ZERO_DOUBLE) {
        // k=1: log(1/λ); k>1: −∞; k<1: +∞
        if (std::abs(shape_ - detail::ONE) <= detail::DEFAULT_TOLERANCE)
            return -logScale_;
        return (shape_ > detail::ONE) ? detail::NEGATIVE_INFINITY
                                      : std::numeric_limits<double>::infinity();
    }
    const double log_x = std::log(x);
    const double z = log_x - logScale_;         // log(x/λ)
    const double power = std::exp(shape_ * z);  // (x/λ)^k
    return logNormConst_ + shapeMinus1_ * log_x - power;
}

double WeibullDistribution::getCumulativeProbability(double x) const {
    if (x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return detail::ONE - std::exp(-std::exp(shape_ * (std::log(x) - logScale_)));
}

double WeibullDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p >= detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1) for Weibull distribution");
    }
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // Q(p) = λ · (−log(1−p))^(1/k)
    return scale_ * std::pow(-std::log(detail::ONE - p), detail::ONE / shape_);
}

double WeibullDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double k = shape_, lam = scale_;
    lock.unlock();
    std::weibull_distribution<double> dist(k, lam);
    return dist(rng);
}

std::vector<double> WeibullDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double k = shape_, lam = scale_;
    lock.unlock();

    std::weibull_distribution<double> dist(k, lam);
    for (size_t i = 0; i < n; ++i)
        samples.push_back(dist(rng));
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

namespace {

/// Method-of-moments seed for Weibull k using coefficient-of-variation.
/// Returns a positive estimate of k; 1.0 as fallback.
double weibull_mom_k_init(double mean, double variance) noexcept {
    if (mean <= detail::ZERO || variance <= detail::ZERO)
        return detail::ONE;
    const double cv = std::sqrt(variance) / mean;
    double k_est;
    if (cv < 0.2)
        k_est = detail::ONE / (cv * cv * detail::SIX);
    else if (cv < detail::ONE)
        k_est = std::pow(1.2 / cv, 1.086);  // Thoman-Bain-Antle approximation
    else
        k_est = detail::ONE / cv;
    return std::max(detail::MIN_DISTRIBUTION_PARAMETER,
                    std::min(k_est, detail::MAX_DISTRIBUTION_PARAMETER));
}

/// Newton–Raphson on the Weibull profile score for k.
///
/// Profile score: g(k) = E_k[log x] − 1/k − s̄ = 0
///   where E_k[log x] = Σ(xᵢ^k · log xᵢ) / Σ(xᵢ^k) and s̄ = mean(log xᵢ).
/// Derivative: g'(k) = Var_k[log x] + 1/k² > 0 always (Newton always converges).
/// After convergence: λ̂ = (Σxᵢ^k / n)^(1/k).
///
/// @return {k, lambda} pair
std::pair<double, double> weibull_mle_newton(const std::vector<double>& log_x,
                                             const std::vector<double>& log_x2, double n,
                                             double s_bar, double k_init) noexcept {
    const std::size_t sz = log_x.size();
    double k = k_init;

    for (int iter = 0; iter < 100; ++iter) {
        double s0 = detail::ZERO_DOUBLE;  // Σ xᵢ^k
        double s1 = detail::ZERO_DOUBLE;  // Σ xᵢ^k · log xᵢ
        double s2 = detail::ZERO_DOUBLE;  // Σ xᵢ^k · (log xᵢ)²
        for (std::size_t i = 0; i < sz; ++i) {
            const double xk = std::exp(k * log_x[i]);
            s0 += xk;
            s1 += xk * log_x[i];
            s2 += xk * log_x2[i];
        }
        if (s0 <= detail::ZERO)
            break;

        const double c1 = s1 / s0;             // E_k[log x]
        const double var = s2 / s0 - c1 * c1;  // Var_k[log x] ≥ 0
        const double g = c1 - detail::ONE / k - s_bar;
        const double gp = var + detail::ONE / (k * k);  // always > 0
        if (gp <= detail::ZERO)
            break;

        const double dk = g / gp;
        k -= dk;
        if (k <= detail::ZERO)
            k = detail::MIN_DISTRIBUTION_PARAMETER;
        if (std::fabs(dk) < 1e-11 * k)
            break;
    }

    // λ̂ = (Σxᵢ^k / n)^(1/k)
    double s0f = detail::ZERO_DOUBLE;
    for (std::size_t i = 0; i < sz; ++i)
        s0f += std::exp(k * log_x[i]);
    const double lambda =
        (s0f > detail::ZERO && n > detail::ZERO) ? std::exp(std::log(s0f / n) / k) : detail::ONE;
    return {k, lambda};
}

}  // anonymous namespace

void WeibullDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    std::vector<double> log_x, log_x2;
    log_x.reserve(values.size());
    log_x2.reserve(values.size());

    double sum = detail::ZERO_DOUBLE;
    double sum_sq = detail::ZERO_DOUBLE;
    double sum_log = detail::ZERO_DOUBLE;
    std::size_t count = 0;

    for (double v : values) {
        if (v <= detail::ZERO_DOUBLE || !std::isfinite(v)) {
            throw std::invalid_argument(
                "Weibull distribution requires strictly positive finite values");
        }
        ++count;
        sum += v;
        sum_sq += v * v;
        const double l = std::log(v);
        sum_log += l;
        log_x.push_back(l);
        log_x2.push_back(l * l);
    }

    const double n = static_cast<double>(count);
    const double mean = sum / n;
    const double var = sum_sq / n - mean * mean;
    const double s_bar = sum_log / n;
    const double k_init = weibull_mom_k_init(mean, var);

    const auto [k, lambda] = weibull_mle_newton(log_x, log_x2, n, s_bar, k_init);

    if (std::isfinite(k) && std::isfinite(lambda) && k > detail::ZERO && lambda > detail::ZERO &&
        k < detail::MAX_DISTRIBUTION_PARAMETER && lambda < detail::MAX_DISTRIBUTION_PARAMETER) {
        setParameters(k, lambda);
    } else {
        throw std::runtime_error("Weibull MLE did not converge to a valid estimate");
    }
}

void WeibullDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                           std::vector<WeibullDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void WeibullDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    shape_ = detail::ONE;
    scale_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);  // NEW-TS-4
    updateCacheUnsafe();
}

std::string WeibullDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "WeibullDistribution(shape=" << shape_ << ",scale=" << scale_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double WeibullDistribution::getShapeAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicShape_.load(std::memory_order_acquire);
    }
    return getShape();
}

double WeibullDistribution::getScaleAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicScale_.load(std::memory_order_acquire);
    }
    return getScale();
}

double WeibullDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (shape_ <= detail::ONE)
        return detail::ZERO_DOUBLE;
    return scale_ * std::pow((shape_ - detail::ONE) / shape_, detail::ONE / shape_);
}

double WeibullDistribution::getMedian() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // λ·(ln 2)^(1/k)
    return scale_ * std::pow(detail::LN2, detail::ONE / shape_);
}

double WeibullDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = γ·(1 − 1/k) + log(λ/k) + 1
    return detail::EULER_MASCHERONI * (detail::ONE - detail::ONE / shape_) + logScale_ - logShape_ +
           detail::ONE;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void WeibullDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                         const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const WeibullDistribution& d, double x) { return d.getProbability(x); },
        [](const WeibullDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.shape_, ls = d.logScale_, km1 = d.shapeMinus1_,
                         lnc = d.logNormConst_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, k, ls, km1, lnc);
        },
        [](const WeibullDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.shape_, ls = d.logScale_, km1 = d.shapeMinus1_,
                         lnc = d.logNormConst_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        return;
                    }
                    if (x == detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        return;
                    }
                    const double z = std::log(x) - ls;
                    res[i] = std::exp(lnc + km1 * std::log(x) - std::exp(k * z));
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        continue;
                    }
                    if (x == detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        continue;
                    }
                    const double z = std::log(x) - ls;
                    res[i] = std::exp(lnc + km1 * std::log(x) - std::exp(k * z));
                }
            }
        },
        [](const WeibullDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.shape_, ls = d.logScale_, km1 = d.shapeMinus1_,
                         lnc = d.logNormConst_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                    return;
                }
                const double z = std::log(x) - ls;
                res[i] = std::exp(lnc + km1 * std::log(x) - std::exp(k * z));
            });
            pool.waitForAll();
        });
}

void WeibullDistribution::getLogProbability(std::span<const double> values,
                                            std::span<double> results,
                                            const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const WeibullDistribution& d, double x) { return d.getLogProbability(x); },
        [](const WeibullDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.shape_, ls = d.logScale_, km1 = d.shapeMinus1_,
                         lnc = d.logNormConst_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, k, ls, km1, lnc);
        },
        [](const WeibullDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.shape_, ls = d.logScale_, km1 = d.shapeMinus1_,
                         lnc = d.logNormConst_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        return;
                    }
                    if (x == detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        return;
                    }
                    const double z = std::log(x) - ls;
                    res[i] = lnc + km1 * std::log(x) - std::exp(k * z);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        continue;
                    }
                    const double z = std::log(x) - ls;
                    res[i] = lnc + km1 * std::log(x) - std::exp(k * z);
                }
            }
        },
        [](const WeibullDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.shape_, ls = d.logScale_, km1 = d.shapeMinus1_,
                         lnc = d.logNormConst_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                    return;
                }
                const double z = std::log(x) - ls;
                res[i] = lnc + km1 * std::log(x) - std::exp(k * z);
            });
            pool.waitForAll();
        });
}

void WeibullDistribution::getCumulativeProbability(std::span<const double> values,
                                                   std::span<double> results,
                                                   const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const WeibullDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const WeibullDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double ls = d.logScale_, k = d.shape_;
            lock.unlock();
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, ls, k);
        },
        [](const WeibullDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double ls = d.logScale_, k = d.shape_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE)
                                 ? detail::ZERO_DOUBLE
                                 : detail::ONE - std::exp(-std::exp(k * (std::log(x) - ls)));
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE)
                                 ? detail::ZERO_DOUBLE
                                 : detail::ONE - std::exp(-std::exp(k * (std::log(x) - ls)));
                }
            }
        },
        [](const WeibullDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<WeibullDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double ls = d.logScale_, k = d.shape_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE)
                             ? detail::ZERO_DOUBLE
                             : detail::ONE - std::exp(-std::exp(k * (std::log(x) - ls)));
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool WeibullDistribution::operator==(const WeibullDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(shape_ - other.shape_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(scale_ - other.scale_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool WeibullDistribution::operator!=(const WeibullDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const WeibullDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, WeibullDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("WeibullDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t sh_pos = token.find("shape=");
    const size_t comma = token.find(",", sh_pos);
    const size_t sc_pos = token.find("scale=");
    const size_t close = token.find(")", sc_pos);
    if (sh_pos == std::string::npos || comma == std::string::npos || sc_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double sh = std::stod(token.substr(sh_pos + 6, comma - sh_pos - 6));
        const double sc = std::stod(token.substr(sc_pos + 6, close - sc_pos - 6));
        auto result = d.trySetParameters(sh, sc);
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
// Compute+fixup pattern: SIMD runs over the full array assuming x > 0.
// A scalar fixup pass at the end handles x <= 0.
//
// LogPDF (8 steps, one temp buffer):
//   temp    = log(x)                      [vector_log]
//   results = z = temp − log(λ)           [scalar_add(temp, −logScale_, results)]
//   results = k·z                         [scalar_multiply(k_)]
//   results = exp(k·z) = (x/λ)^k         [vector_exp]
//   results = −(x/λ)^k                   [scalar_multiply(−1)]
//   temp    = (k−1)·z                     [scalar_multiply(temp, shapeMinus1_)]
//   results += (k−1)·z                   [vector_add]
//   results += logNormConst_              [scalar_add]
//
// PDF: append vector_exp.
//
// CDF (8 steps, no temp buffer):
//   results = log(x)                      [vector_log]
//   results = log(x/λ)                    [scalar_add(−logScale_)]
//   results = k·log(x/λ)                  [scalar_multiply(k_)]
//   results = (x/λ)^k                     [vector_exp]
//   results = −(x/λ)^k                   [scalar_multiply(−1)]
//   results = exp(−(x/λ)^k)              [vector_exp]
//   results = −exp(...)                   [scalar_multiply(−1)]
//   results = 1 − exp(−(x/λ)^k)          [scalar_add(1)]
//==============================================================================

void WeibullDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_shape,
    double cached_log_scale, double cached_shape_minus1,
    double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            const double lx = std::log(x);
            const double z = lx - cached_log_scale;
            results[i] = std::exp(cached_log_norm_const + cached_shape_minus1 * lx -
                                  std::exp(cached_shape * z));
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    // Step 2: results = log(x) − log(λ)  (= z)
    arch::simd::VectorOps::scalar_add(temp.data(), -cached_log_scale, results, count);
    // Step 3: results = k·z
    arch::simd::VectorOps::scalar_multiply(results, cached_shape, results, count);
    // Step 4: results = exp(k·z) = (x/λ)^k
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 5: results = −(x/λ)^k
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    // Step 6: temp = (k−1)·log(x)   [temp still holds log(x) from step 1]
    arch::simd::VectorOps::scalar_multiply(temp.data(), cached_shape_minus1, temp.data(), count);
    // Step 7: results = (k−1)·log(x) − (x/λ)^k
    arch::simd::VectorOps::vector_add(results, temp.data(), results, count);
    // Step 8: results += logNormConst_
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);
    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: x <= 0 is outside support; PDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

void WeibullDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_shape,
    double cached_log_scale, double cached_shape_minus1,
    double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
                continue;
            }
            const double lx = std::log(x);
            const double z = lx - cached_log_scale;
            results[i] =
                cached_log_norm_const + cached_shape_minus1 * lx - std::exp(cached_shape * z);
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    // Step 2: results = log(x) − log(λ)  (= z)
    arch::simd::VectorOps::scalar_add(temp.data(), -cached_log_scale, results, count);
    // Step 3: results = k·z
    arch::simd::VectorOps::scalar_multiply(results, cached_shape, results, count);
    // Step 4: results = exp(k·z) = (x/λ)^k
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 5: results = −(x/λ)^k
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    // Step 6: temp = (k−1)·log(x)
    arch::simd::VectorOps::scalar_multiply(temp.data(), cached_shape_minus1, temp.data(), count);
    // Step 7: results = (k−1)·log(x) − (x/λ)^k
    arch::simd::VectorOps::vector_add(results, temp.data(), results, count);
    // Step 8: results += logNormConst_
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);

    // Fixup: x <= 0 is outside support; LogPDF = −∞.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void WeibullDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_log_scale,
    double cached_shape) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            results[i] =
                detail::ONE - std::exp(-std::exp(cached_shape * (std::log(x) - cached_log_scale)));
        }
        return;
    }

    // Step 1: results = log(x)
    arch::simd::VectorOps::vector_log(values, results, count);
    // Step 2: results = log(x/λ) = log(x) − log(λ)
    arch::simd::VectorOps::scalar_add(results, -cached_log_scale, results, count);
    // Step 3: results = k·log(x/λ)
    arch::simd::VectorOps::scalar_multiply(results, cached_shape, results, count);
    // Step 4: results = (x/λ)^k
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 5: results = −(x/λ)^k
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    // Step 6: results = exp(−(x/λ)^k)
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 7: results = −exp(−(x/λ)^k)
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    // Step 8: results = 1 − exp(−(x/λ)^k)
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);

    // Fixup: x <= 0 is outside support; CDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void WeibullDistribution::updateCacheUnsafe() const noexcept {
    logShape_ = std::log(shape_);
    logScale_ = std::log(scale_);
    shapeMinus1_ = shape_ - detail::ONE;
    // logNormConst = log(k) − k·log(λ) = logShape_ − shape_·logScale_
    logNormConst_ = logShape_ - shape_ * logScale_;

    // Mean: λ·Γ(1 + 1/k)
    const double g1 = std::exp(std::lgamma(detail::ONE + detail::ONE / shape_));
    mean_ = scale_ * g1;

    // Variance: λ²·[Γ(1 + 2/k) − Γ(1 + 1/k)²]
    const double g2 = std::exp(std::lgamma(detail::ONE + detail::TWO / shape_));
    variance_ = scale_ * scale_ * (g2 - g1 * g1);

    isExponential_ = (std::fabs(shape_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicShape_.store(shape_, std::memory_order_release);
    atomicScale_.store(scale_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

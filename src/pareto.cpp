#include "libstats/distributions/pareto.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

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

ParetoDistribution::ParetoDistribution(double scale, double alpha)
    : DistributionBase(), scale_(scale), alpha_(alpha) {
    validateParameters(scale, alpha);
    updateCacheUnsafe();
}

ParetoDistribution::ParetoDistribution(const ParetoDistribution& other) : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    scale_ = other.scale_;
    alpha_ = other.alpha_;
    logScale_ = other.logScale_;
    logAlpha_ = other.logAlpha_;
    negAlphaPlusOne_ = other.negAlphaPlusOne_;
    logNormConst_ = other.logNormConst_;
    negAlpha_ = other.negAlpha_;
    invAlpha_ = other.invAlpha_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    atomicScale_.store(scale_, std::memory_order_release);
    atomicAlpha_.store(alpha_, std::memory_order_release);
}

ParetoDistribution& ParetoDistribution::operator=(const ParetoDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        scale_ = other.scale_;
        alpha_ = other.alpha_;
        logScale_ = other.logScale_;
        logAlpha_ = other.logAlpha_;
        negAlphaPlusOne_ = other.negAlphaPlusOne_;
        logNormConst_ = other.logNormConst_;
        negAlpha_ = other.negAlpha_;
        invAlpha_ = other.invAlpha_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicScale_.store(scale_, std::memory_order_release);
        atomicAlpha_.store(alpha_, std::memory_order_release);
    }
    return *this;
}

ParetoDistribution::ParetoDistribution(ParetoDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    scale_ = other.scale_;
    alpha_ = other.alpha_;
    logScale_ = other.logScale_;
    logAlpha_ = other.logAlpha_;
    negAlphaPlusOne_ = other.negAlphaPlusOne_;
    logNormConst_ = other.logNormConst_;
    negAlpha_ = other.negAlpha_;
    invAlpha_ = other.invAlpha_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    other.scale_ = detail::ONE;
    other.alpha_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicScale_.store(scale_, std::memory_order_release);
    atomicAlpha_.store(alpha_, std::memory_order_release);
}

ParetoDistribution& ParetoDistribution::operator=(ParetoDistribution&& other) noexcept {
    if (this != &other) {
        scale_ = other.scale_;
        alpha_ = other.alpha_;
        logScale_ = other.logScale_;
        logAlpha_ = other.logAlpha_;
        negAlphaPlusOne_ = other.negAlphaPlusOne_;
        logNormConst_ = other.logNormConst_;
        negAlpha_ = other.negAlpha_;
        invAlpha_ = other.invAlpha_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        other.scale_ = detail::ONE;
        other.alpha_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicScale_.store(scale_, std::memory_order_release);
        atomicAlpha_.store(alpha_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

ParetoDistribution ParetoDistribution::createUnchecked(double scale, double alpha) noexcept {
    return ParetoDistribution(scale, alpha, true);
}

ParetoDistribution::ParetoDistribution(double scale, double alpha,
                                       bool /*bypassValidation*/) noexcept
    : DistributionBase(), scale_(scale), alpha_(alpha) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void ParetoDistribution::setScale(double scale) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(scale, alpha_);
    scale_ = scale;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void ParetoDistribution::setAlpha(double alpha) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(scale_, alpha);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void ParetoDistribution::setParameters(double scale, double alpha) {
    validateParameters(scale, alpha);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    scale_ = scale;
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double ParetoDistribution::getMean() const {
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

double ParetoDistribution::getVariance() const {
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

double ParetoDistribution::getSkewness() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (alpha_ <= detail::THREE)
        return std::numeric_limits<double>::infinity();
    // 2(1+α)/(α−3) * sqrt((α−2)/α)
    return detail::TWO * (detail::ONE + alpha_) / (alpha_ - detail::THREE) *
           std::sqrt((alpha_ - detail::TWO) / alpha_);
}

double ParetoDistribution::getKurtosis() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (alpha_ <= detail::FOUR)
        return std::numeric_limits<double>::infinity();
    // 6(α³+α²−6α−2) / (α(α−3)(α−4))
    const double a2 = alpha_ * alpha_;
    const double a3 = a2 * alpha_;
    return detail::SIX * (a3 + a2 - detail::SIX * alpha_ - detail::TWO) /
           (alpha_ * (alpha_ - detail::THREE) * (alpha_ - detail::FOUR));
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult ParetoDistribution::trySetScale(double scale) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateParetoParameters(scale, alpha_);
    if (v.isError())
        return v;
    scale_ = scale;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult ParetoDistribution::trySetAlpha(double alpha) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateParetoParameters(scale_, alpha);
    if (v.isError())
        return v;
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult ParetoDistribution::trySetParameters(double scale, double alpha) noexcept {
    auto v = validateParetoParameters(scale, alpha);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    scale_ = scale;
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult ParetoDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateParetoParameters(scale_, alpha_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double ParetoDistribution::getProbability(double x) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (x < scale_)
        return detail::ZERO_DOUBLE;
    return std::exp(getLogProbability(x));
}

double ParetoDistribution::getLogProbability(double x) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (x < scale_)
        return detail::NEGATIVE_INFINITY;
    return logNormConst_ + negAlphaPlusOne_ * std::log(x);
}

double ParetoDistribution::getCumulativeProbability(double x) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (x < scale_)
        return detail::ZERO_DOUBLE;
    // 1 − (x_m/x)^α = 1 − exp(α·log(x_m/x))
    return detail::ONE - std::pow(scale_ / x, alpha_);
}

double ParetoDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p >= detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1) for Pareto distribution");
    }
    if (p == detail::ZERO_DOUBLE)
        return scale_;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // x_m * (1-p)^(-1/α) = x_m / (1-p)^(1/α)
    return scale_ * std::pow(detail::ONE - p, -invAlpha_);
}

double ParetoDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double cached_scale = scale_;
    const double cached_inv_alpha = invAlpha_;
    lock.unlock();

    // Inverse CDF method: x_m / U^(1/α) where U ~ Uniform(0, 1)
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    return cached_scale * std::pow(uniform(rng), -cached_inv_alpha);
}

std::vector<double> ParetoDistribution::sample(std::mt19937& rng, size_t n) const {
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
    const double cached_scale = scale_;
    const double cached_inv_alpha = invAlpha_;
    lock.unlock();

    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(cached_scale * std::pow(uniform(rng), -cached_inv_alpha));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void ParetoDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    for (double v : values) {
        if (v <= detail::ZERO_DOUBLE || !std::isfinite(v)) {
            throw std::invalid_argument(
                "Pareto distribution requires strictly positive finite values");
        }
    }

    // Closed-form MLE: scale_hat = min(xᵢ), alpha_hat = n / Σ log(xᵢ/scale_hat).
    const double scale_hat = *std::min_element(values.begin(), values.end());
    const double n = static_cast<double>(values.size());

    double sum_log_ratio = detail::ZERO_DOUBLE;
    for (double v : values) {
        sum_log_ratio += std::log(v / scale_hat);
    }

    if (sum_log_ratio <= detail::ZERO_DOUBLE) {
        // All values identical to scale_hat — α is undefined, use α=1 as fallback.
        setParameters(scale_hat, detail::ONE);
    } else {
        setParameters(scale_hat, n / sum_log_ratio);
    }
}

void ParetoDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                          std::vector<ParetoDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void ParetoDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    scale_ = detail::ONE;
    alpha_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string ParetoDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "ParetoDistribution(scale=" << scale_ << ",alpha=" << alpha_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double ParetoDistribution::getScaleAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicScale_.load(std::memory_order_acquire);
    }
    return getScale();
}

double ParetoDistribution::getAlphaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicAlpha_.load(std::memory_order_acquire);
    }
    return getAlpha();
}

double ParetoDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return scale_;  // Mode is always at the lower boundary x_m.
}

double ParetoDistribution::getMedian() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // x_m * 2^(1/α)
    return scale_ * std::exp(detail::LN2 * invAlpha_);
}

double ParetoDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = log(x_m/α) + 1 + 1/α = log(x_m) - log(α) + 1 + 1/α
    return logScale_ - logAlpha_ + detail::ONE + invAlpha_;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void ParetoDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                        const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const ParetoDistribution& d, double x) { return d.getProbability(x); },
        [](const ParetoDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, neg_ap1 = d.negAlphaPlusOne_, lnc = d.logNormConst_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, scale, neg_ap1, lnc);
        },
        [](const ParetoDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, lnc = d.logNormConst_, neg_ap1 = d.negAlphaPlusOne_;
            lock.unlock();

            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] =
                        (x < scale) ? detail::ZERO_DOUBLE : std::exp(lnc + neg_ap1 * std::log(x));
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] =
                        (x < scale) ? detail::ZERO_DOUBLE : std::exp(lnc + neg_ap1 * std::log(x));
                }
            }
        },
        [](const ParetoDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, lnc = d.logNormConst_, neg_ap1 = d.negAlphaPlusOne_;
            lock.unlock();

            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x < scale) ? detail::ZERO_DOUBLE : std::exp(lnc + neg_ap1 * std::log(x));
            });
            pool.waitForAll();
        });
}

void ParetoDistribution::getLogProbability(std::span<const double> values,
                                           std::span<double> results,
                                           const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const ParetoDistribution& d, double x) { return d.getLogProbability(x); },
        [](const ParetoDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, neg_ap1 = d.negAlphaPlusOne_, lnc = d.logNormConst_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, scale, neg_ap1, lnc);
        },
        [](const ParetoDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, lnc = d.logNormConst_, neg_ap1 = d.negAlphaPlusOne_;
            lock.unlock();

            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x < scale) ? detail::NEGATIVE_INFINITY : lnc + neg_ap1 * std::log(x);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x < scale) ? detail::NEGATIVE_INFINITY : lnc + neg_ap1 * std::log(x);
                }
            }
        },
        [](const ParetoDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, lnc = d.logNormConst_, neg_ap1 = d.negAlphaPlusOne_;
            lock.unlock();

            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x < scale) ? detail::NEGATIVE_INFINITY : lnc + neg_ap1 * std::log(x);
            });
            pool.waitForAll();
        });
}

void ParetoDistribution::getCumulativeProbability(std::span<const double> values,
                                                  std::span<double> results,
                                                  const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const ParetoDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const ParetoDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_;
            const double log_scale = d.logScale_;
            const double neg_alpha = d.negAlpha_;
            lock.unlock();
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, scale, log_scale,
                                                      neg_alpha);
        },
        [](const ParetoDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, alpha = d.alpha_;
            lock.unlock();

            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x < scale) ? detail::ZERO_DOUBLE
                                         : detail::ONE - std::pow(scale / x, alpha);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x < scale) ? detail::ZERO_DOUBLE
                                         : detail::ONE - std::pow(scale / x, alpha);
                }
            }
        },
        [](const ParetoDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<ParetoDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double scale = d.scale_, alpha = d.alpha_;
            lock.unlock();

            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] =
                    (x < scale) ? detail::ZERO_DOUBLE : detail::ONE - std::pow(scale / x, alpha);
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

bool ParetoDistribution::operator==(const ParetoDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(scale_ - other.scale_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(alpha_ - other.alpha_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool ParetoDistribution::operator!=(const ParetoDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const ParetoDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, ParetoDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("ParetoDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t sc_pos = token.find("scale=");
    const size_t comma = token.find(",", sc_pos);
    const size_t al_pos = token.find("alpha=");
    const size_t close = token.find(")", al_pos);
    if (sc_pos == std::string::npos || comma == std::string::npos || al_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double sc = std::stod(token.substr(sc_pos + 6, comma - sc_pos - 6));
        const double al = std::stod(token.substr(al_pos + 6, close - al_pos - 6));
        auto result = d.trySetParameters(sc, al);
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
// LogPDF pipeline (three steps — the simplest in the library):
//   results = log(x)              [vector_log]
//   results *= −(α+1)             [scalar_multiply]
//   results += log_norm_const     [scalar_add]
//
// PDF: append vector_exp.
//
// CDF pipeline (six steps, no temp buffer):
//   results = log(x)              [vector_log]
//   results -= log(x_m)           [scalar_add(-log_scale)]
//   results *= -α                 [scalar_multiply(neg_alpha_)]
//     → α·(log(x_m) - log(x)) = α·log(x_m/x)
//   results = exp(...)            [vector_exp]   → (x_m/x)^α
//   results *= -1                 [scalar_multiply(-1)]
//   results += 1                  [scalar_add(1)]
//==============================================================================

void ParetoDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_scale,
    double cached_neg_alpha_p1, double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < cached_scale) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                results[i] = std::exp(cached_log_norm_const + cached_neg_alpha_p1 * std::log(x));
            }
        }
        return;
    }

    // Step 1: results = log(x)
    arch::simd::VectorOps::vector_log(values, results, count);
    // Step 2: results = −(α+1)·log(x)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_alpha_p1, results, count);
    // Step 3: results += log_norm_const
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);
    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: x < scale is outside support; PDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < cached_scale)
            results[i] = detail::ZERO_DOUBLE;
    }
}

void ParetoDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_scale,
    double cached_neg_alpha_p1, double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < cached_scale) {
                results[i] = detail::NEGATIVE_INFINITY;
            } else {
                results[i] = cached_log_norm_const + cached_neg_alpha_p1 * std::log(x);
            }
        }
        return;
    }

    // Step 1: results = log(x)
    arch::simd::VectorOps::vector_log(values, results, count);
    // Step 2: results = −(α+1)·log(x)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_alpha_p1, results, count);
    // Step 3: results += log_norm_const
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);

    // Fixup: x < scale is outside support; LogPDF = −∞.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < cached_scale)
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void ParetoDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_scale,
    double cached_log_scale, double cached_neg_alpha) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use pow(scale/x, alpha) = exp(alpha*log(scale/x)) for consistency
        const double cached_alpha = -cached_neg_alpha;  // alpha is stored as negAlpha_
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < cached_scale) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                results[i] =
                    detail::ONE - std::exp(cached_alpha * (cached_log_scale - std::log(x)));
            }
        }
        return;
    }

    // Step 1: results = log(x)
    arch::simd::VectorOps::vector_log(values, results, count);
    // Step 2: results = log(x) − log(x_m)   (= log(x/x_m))
    arch::simd::VectorOps::scalar_add(results, -cached_log_scale, results, count);
    // Step 3: results = −α·log(x/x_m) = α·log(x_m/x)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_alpha, results, count);
    // Step 4: results = (x_m/x)^α
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 5: results = −(x_m/x)^α
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    // Step 6: results = 1 − (x_m/x)^α
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);

    // Fixup: x < scale is outside support; CDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < cached_scale)
            results[i] = detail::ZERO_DOUBLE;
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void ParetoDistribution::updateCacheUnsafe() const noexcept {
    logScale_ = std::log(scale_);
    logAlpha_ = std::log(alpha_);
    negAlphaPlusOne_ = -(alpha_ + detail::ONE);
    logNormConst_ = logAlpha_ + alpha_ * logScale_;
    negAlpha_ = -alpha_;
    invAlpha_ = detail::ONE / alpha_;

    // Mean: α·x_m/(α−1) for α > 1; +∞ otherwise.
    mean_ = (alpha_ > detail::ONE) ? alpha_ * scale_ / (alpha_ - detail::ONE)
                                   : std::numeric_limits<double>::infinity();

    // Variance: x_m²·α/((α−1)²·(α−2)) for α > 2; +∞ otherwise.
    if (alpha_ > detail::TWO) {
        const double am1 = alpha_ - detail::ONE;
        variance_ = scale_ * scale_ * alpha_ / (am1 * am1 * (alpha_ - detail::TWO));
    } else {
        variance_ = std::numeric_limits<double>::infinity();
    }

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicScale_.store(scale_, std::memory_order_release);
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

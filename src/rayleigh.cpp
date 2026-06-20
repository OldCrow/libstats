#include "libstats/distributions/rayleigh.h"
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;
using stats::detail::validateNonNegativeParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"
#include "libstats/core/validation.h"

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

RayleighDistribution::RayleighDistribution(double sigma) : DistributionBase(), sigma_(sigma) {
    validateParameters(sigma);
    updateCacheUnsafe();
}

RayleighDistribution::RayleighDistribution(const RayleighDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    sigma_ = other.sigma_;
    logSigma_ = other.logSigma_;
    negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    atomicSigma_.store(sigma_, std::memory_order_release);
}

RayleighDistribution& RayleighDistribution::operator=(const RayleighDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        sigma_ = other.sigma_;
        logSigma_ = other.logSigma_;
        negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicSigma_.store(sigma_, std::memory_order_release);
    }
    return *this;
}

RayleighDistribution::RayleighDistribution(RayleighDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    sigma_ = other.sigma_;
    logSigma_ = other.logSigma_;
    negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    other.sigma_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
}

RayleighDistribution& RayleighDistribution::operator=(RayleighDistribution&& other) noexcept {
    if (this != &other) {

        sigma_ = other.sigma_;
        logSigma_ = other.logSigma_;
        negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        other.sigma_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicSigma_.store(sigma_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

RayleighDistribution RayleighDistribution::createUnchecked(double sigma) noexcept {
    return RayleighDistribution(sigma, true);
}

RayleighDistribution::RayleighDistribution(double sigma, bool /*bypassValidation*/) noexcept
    : DistributionBase(), sigma_(sigma) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void RayleighDistribution::setSigma(double sigma) {
    validateParameters(sigma);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void RayleighDistribution::setParameters(double sigma) {
    setSigma(sigma);
}

double RayleighDistribution::getMean() const noexcept {
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

double RayleighDistribution::getVariance() const noexcept {
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

double RayleighDistribution::getSkewness() const noexcept {
    // Skewness is a constant for Rayleigh: 2√π(π−3) / (4−π)^(3/2)
    const double four_minus_pi = detail::FOUR - detail::PI;
    return detail::TWO * detail::SQRT_PI * (detail::PI - detail::THREE) /
           (four_minus_pi * std::sqrt(four_minus_pi));
}

double RayleighDistribution::getKurtosis() const noexcept {
    // Excess kurtosis: −(6π²−24π+16) / (4−π)²
    const double four_minus_pi = detail::FOUR - detail::PI;
    return -(detail::SIX * detail::PI * detail::PI - 24.0 * detail::PI + 16.0) /
           (four_minus_pi * four_minus_pi);
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult RayleighDistribution::trySetSigma(double sigma) noexcept {
    auto v = validateRayleighParameters(sigma);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult RayleighDistribution::trySetParameters(double sigma) noexcept {
    return trySetSigma(sigma);
}

VoidResult RayleighDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateRayleighParameters(sigma_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double RayleighDistribution::getProbability(double x) const {
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
    return std::exp(getLogProbability(x));
}

double RayleighDistribution::getLogProbability(double x) const noexcept {
    if (x <= detail::ZERO_DOUBLE)
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
    // log(x) + logNormConst_ + negHalfInvSigmaSquared_ * x²
    return std::log(x) + logNormConst_ + negHalfInvSigmaSquared_ * x * x;
}

double RayleighDistribution::getCumulativeProbability(double x) const {
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
    return detail::ONE - std::exp(negHalfInvSigmaSquared_ * x * x);
}

double RayleighDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p >= detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1) for Rayleigh distribution");
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
    // σ·√(−2·log(1−p))
    return sigma_ * std::sqrt(-detail::TWO * std::log(detail::ONE - p));
}

double RayleighDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double s = sigma_;
    lock.unlock();
    // Inverse-CDF: x = σ·√(−2·log(U)), U ~ Uniform(0,1)
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    return s * std::sqrt(-detail::TWO * std::log(uniform(rng)));
}

std::vector<double> RayleighDistribution::sample(std::mt19937& rng, size_t n) const {
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
    const double s = sigma_;
    lock.unlock();

    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(s * std::sqrt(-detail::TWO * std::log(uniform(rng))));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void RayleighDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    double sum_sq = detail::ZERO_DOUBLE;
    for (double v : values) {
        if (v <= detail::ZERO_DOUBLE || !std::isfinite(v)) {
            throw std::invalid_argument(
                "Rayleigh distribution requires strictly positive finite values");
        }
        sum_sq += v * v;
    }

    // σ̂ = √(Σxᵢ²/(2n))
    const double sigma_hat = std::sqrt(sum_sq / (detail::TWO * static_cast<double>(values.size())));
    if (std::isfinite(sigma_hat) && sigma_hat > detail::ZERO) {
        setSigma(sigma_hat);
    } else {
        reset();
    }
}

void RayleighDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<RayleighDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void RayleighDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string RayleighDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "RayleighDistribution(sigma=" << sigma_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double RayleighDistribution::getSigmaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicSigma_.load(std::memory_order_acquire);
    }
    return getSigma();
}

double RayleighDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return sigma_;  // Mode is always σ.
}

double RayleighDistribution::getMedian() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // σ·√(2·ln 2)
    return sigma_ * std::sqrt(detail::TWO * detail::LN2);
}

double RayleighDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = 1 + log(σ/√2) + γ/2 = 1 + log(σ) − ½·log(2) + γ/2
    return detail::ONE + logSigma_ - detail::HALF * detail::LN2 +
           detail::HALF * detail::EULER_MASCHERONI;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void RayleighDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const RayleighDistribution& d, double x) { return d.getProbability(x); },
        [](const RayleighDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_, lnc = d.logNormConst_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, nhis, lnc);
        },
        [](const RayleighDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_, lnc = d.logNormConst_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE)
                                 ? detail::ZERO_DOUBLE
                                 : std::exp(std::log(x) + lnc + nhis * x * x);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE)
                                 ? detail::ZERO_DOUBLE
                                 : std::exp(std::log(x) + lnc + nhis * x * x);
                }
            }
        },
        [](const RayleighDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_, lnc = d.logNormConst_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE) ? detail::ZERO_DOUBLE
                                                    : std::exp(std::log(x) + lnc + nhis * x * x);
            });
            pool.waitForAll();
        });
}

void RayleighDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const RayleighDistribution& d, double x) { return d.getLogProbability(x); },
        [](const RayleighDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_, lnc = d.logNormConst_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, nhis, lnc);
        },
        [](const RayleighDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_, lnc = d.logNormConst_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE) ? detail::NEGATIVE_INFINITY
                                                        : std::log(x) + lnc + nhis * x * x;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE) ? detail::NEGATIVE_INFINITY
                                                        : std::log(x) + lnc + nhis * x * x;
                }
            }
        },
        [](const RayleighDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_, lnc = d.logNormConst_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE) ? detail::NEGATIVE_INFINITY
                                                    : std::log(x) + lnc + nhis * x * x;
            });
            pool.waitForAll();
        });
}

void RayleighDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const RayleighDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const RayleighDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_;
            lock.unlock();
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, nhis);
        },
        [](const RayleighDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE) ? detail::ZERO_DOUBLE
                                                        : detail::ONE - std::exp(nhis * x * x);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE) ? detail::ZERO_DOUBLE
                                                        : detail::ONE - std::exp(nhis * x * x);
                }
            }
        },
        [](const RayleighDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<RayleighDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double nhis = d.negHalfInvSigmaSquared_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE) ? detail::ZERO_DOUBLE
                                                    : detail::ONE - std::exp(nhis * x * x);
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

bool RayleighDistribution::operator==(const RayleighDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(sigma_ - other.sigma_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool RayleighDistribution::operator!=(const RayleighDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const RayleighDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, RayleighDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("RayleighDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t sg_pos = token.find("sigma=");
    const size_t close = token.find(")", sg_pos);
    if (sg_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double sg = std::stod(token.substr(sg_pos + 6, close - sg_pos - 6));
        auto result = d.trySetSigma(sg);
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
// Why standalone rather than delegating to WeibullDistribution(k=2, λ=σ√2):
// Rayleigh's CDF and LogPDF depend only on x², eliminating the log(x/λ) term
// that drives Weibull's 8-step pipeline (two transcendentals, one temp buffer).
// The result is a 5-step LogPDF and a 5-step CDF, both with smaller
// temporary-buffer requirements and fewer dependency chains.
//
// LogPDF (5 steps, one temp buffer):
//   temp    = x²                         [vector_multiply(values, values)]
//   temp    = −x²/(2σ²)                  [scalar_multiply(neg_half_inv_sigma2_)]
//   results = log(x)                     [vector_log]
//   results += temp                      [vector_add(temp, results)]
//   results += log_norm_const_           [scalar_add(−2·log(σ))]
// PDF: append vector_exp.
//
// CDF (5 steps, no temp buffer):
//   results = x²                         [vector_multiply(values, values)]
//   results = −x²/(2σ²)                  [scalar_multiply(neg_half_inv_sigma2_)]
//   results = exp(−x²/(2σ²))             [vector_exp]
//   results = −exp(...)                  [scalar_multiply(−1)]
//   results = 1 − exp(−x²/(2σ²))         [scalar_add(1)]
// Contrast with Weibull CDF which also needs 8 steps even though k=2 is
// a special case, because the generic pipeline works on log(x/λ), not x².
//==============================================================================

void RayleighDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_neg_half_inv_sigma2,
    double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            results[i] =
                std::exp(std::log(x) + cached_log_norm_const + cached_neg_half_inv_sigma2 * x * x);
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = x²
    arch::simd::VectorOps::vector_multiply(values, values, temp.data(), count);
    // Step 2: temp = −x²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(temp.data(), cached_neg_half_inv_sigma2, temp.data(),
                                           count);
    // Step 3: results = log(x)
    arch::simd::VectorOps::vector_log(values, results, count);
    // Step 4: results = log(x) + (−x²/(2σ²))
    arch::simd::VectorOps::vector_add(results, temp.data(), results, count);
    // Step 5: results += logNormConst_
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);
    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: x ≤ 0 is outside support; PDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

void RayleighDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_neg_half_inv_sigma2,
    double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
                continue;
            }
            results[i] = std::log(x) + cached_log_norm_const + cached_neg_half_inv_sigma2 * x * x;
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = x²
    arch::simd::VectorOps::vector_multiply(values, values, temp.data(), count);
    // Step 2: temp = −x²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(temp.data(), cached_neg_half_inv_sigma2, temp.data(),
                                           count);
    // Step 3: results = log(x)
    arch::simd::VectorOps::vector_log(values, results, count);
    // Step 4: results = log(x) + (−x²/(2σ²))
    arch::simd::VectorOps::vector_add(results, temp.data(), results, count);
    // Step 5: results += logNormConst_
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);

    // Fixup: x ≤ 0 is outside support; LogPDF = −∞.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void RayleighDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count,
    double cached_neg_half_inv_sigma2) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            results[i] = detail::ONE - std::exp(cached_neg_half_inv_sigma2 * x * x);
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = −x²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_half_inv_sigma2, results, count);
    // Step 3: results = exp(−x²/(2σ²))
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 4: results = −exp(...)
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    // Step 5: results = 1 − exp(−x²/(2σ²))
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);

    // Fixup: x ≤ 0 is outside support; CDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void RayleighDistribution::updateCacheUnsafe() const noexcept {
    logSigma_ = std::log(sigma_);
    const double inv_sigma2 = detail::ONE / (sigma_ * sigma_);
    negHalfInvSigmaSquared_ = -detail::HALF * inv_sigma2;
    logNormConst_ = -detail::TWO * logSigma_;  // = log(1/σ²)

    // Mean: σ·√(π/2) = σ·SQRT_PI·INV_SQRT_2
    mean_ = sigma_ * detail::SQRT_PI * detail::INV_SQRT_2;

    // Variance: σ²·(4−π)/2
    variance_ = sigma_ * sigma_ * (detail::FOUR - detail::PI) * detail::HALF;

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

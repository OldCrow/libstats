#include "libstats/distributions/laplace.h"
#include "libstats/common/distribution_impl_common.h"
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

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

LaplaceDistribution::LaplaceDistribution(double mu, double b)
    : DistributionBase(), mu_(mu), b_(b) {
    validateParameters(mu, b);
    updateCacheUnsafe();
}

LaplaceDistribution::LaplaceDistribution(const LaplaceDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_         = other.mu_;
    b_          = other.b_;
    neg_inv_b_  = other.neg_inv_b_;
    neg_log2b_  = other.neg_log2b_;
    half_inv_b_ = other.half_inv_b_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicB_.store(b_,   std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

LaplaceDistribution& LaplaceDistribution::operator=(const LaplaceDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_         = other.mu_;
        b_          = other.b_;
        neg_inv_b_  = other.neg_inv_b_;
        neg_log2b_  = other.neg_log2b_;
        half_inv_b_ = other.half_inv_b_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicB_.store(b_,   std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

LaplaceDistribution::LaplaceDistribution(LaplaceDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mu_         = other.mu_;
    b_          = other.b_;
    neg_inv_b_  = other.neg_inv_b_;
    neg_log2b_  = other.neg_log2b_;
    half_inv_b_ = other.half_inv_b_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.b_  = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicB_.store(b_,   std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
}

LaplaceDistribution& LaplaceDistribution::operator=(LaplaceDistribution&& other) noexcept {
    if (this != &other) {
        mu_         = other.mu_;
        b_          = other.b_;
        neg_inv_b_  = other.neg_inv_b_;
        neg_log2b_  = other.neg_log2b_;
        half_inv_b_ = other.half_inv_b_;
        other.mu_ = detail::ZERO_DOUBLE;
        other.b_  = detail::ONE;
        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicB_.store(b_,   std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        other.atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

LaplaceDistribution LaplaceDistribution::createUnchecked(double mu, double b) noexcept {
    return LaplaceDistribution(mu, b, true);
}

LaplaceDistribution::LaplaceDistribution(double mu, double b,
                                         bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), b_(b) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

double LaplaceDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicMu_.load(std::memory_order_acquire);
    return getMu();
}

double LaplaceDistribution::getBAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicB_.load(std::memory_order_acquire);
    return getB();
}

void LaplaceDistribution::setMu(double mu) {
    validateParameters(mu, b_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void LaplaceDistribution::setB(double b) {
    validateParameters(mu_, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void LaplaceDistribution::setParameters(double mu, double b) {
    validateParameters(mu, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    b_  = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult LaplaceDistribution::trySetMu(double mu) noexcept {
    auto v = validateLaplaceParameters(mu, b_);
    if (v.isError()) return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult LaplaceDistribution::trySetB(double b) noexcept {
    auto v = validateLaplaceParameters(mu_, b);
    if (v.isError()) return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult LaplaceDistribution::trySetParameters(double mu, double b) noexcept {
    auto v = validateLaplaceParameters(mu, b);
    if (v.isError()) return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    b_  = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult LaplaceDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateLaplaceParameters(mu_, b_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double LaplaceDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // PDF = exp(LogPDF) = exp(neg_log2b_ - |x-mu_|/b_)
    return std::exp(neg_log2b_ + neg_inv_b_ * std::fabs(x - mu_));
}

double LaplaceDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // LogPDF = neg_log2b_ + neg_inv_b_ * |x - mu_|
    return neg_log2b_ + neg_inv_b_ * std::fabs(x - mu_);
}

double LaplaceDistribution::getCumulativeProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return (x > 0) ? detail::ONE : detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }

    const double d = x - mu_;
    if (d <= detail::ZERO_DOUBLE) {
        // x <= mu: CDF = 0.5 * exp((x-mu)/b) = 0.5 * exp(d/b)
        return detail::HALF * std::exp(d / b_);
    } else {
        // x > mu: CDF = 1 - 0.5 * exp(-(x-mu)/b) = 1 - 0.5 * exp(-d/b)
        return detail::ONE - detail::HALF * std::exp(-d / b_);
    }
}

double LaplaceDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return -std::numeric_limits<double>::infinity();
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }

    if (p == detail::HALF)
        return mu_;

    // Quantile: mu + b * sign(p-0.5) * log(1 - 2|p-0.5|)
    // For p < 0.5: mu + b * log(2p)
    // For p > 0.5: mu - b * log(2*(1-p))
    if (p < detail::HALF) {
        return mu_ + b_ * std::log(detail::TWO * p);
    } else {
        return mu_ - b_ * std::log(detail::TWO * (detail::ONE - p));
    }
}

double LaplaceDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double m = mu_, bv = b_;
    lock.unlock();

    // Inverse CDF method: U ~ Uniform(0,1), return quantile(U)
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(),
                                                   detail::ONE);
    const double u = uniform(rng);
    if (u < detail::HALF) {
        return m + bv * std::log(detail::TWO * u);
    } else {
        return m - bv * std::log(detail::TWO * (detail::ONE - u));
    }
}

std::vector<double> LaplaceDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double m = mu_, bv = b_;
    lock.unlock();

    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(),
                                                   detail::ONE);
    for (size_t i = 0; i < n; ++i) {
        const double u = uniform(rng);
        if (u < detail::HALF) {
            samples.push_back(m + bv * std::log(detail::TWO * u));
        } else {
            samples.push_back(m - bv * std::log(detail::TWO * (detail::ONE - u)));
        }
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void LaplaceDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    for (double v : values) {
        if (!std::isfinite(v))
            throw std::invalid_argument("All values must be finite for Laplace MLE");
    }

    const std::size_t n = values.size();

    // μ̂ = median: sort a copy, take middle element (or average of two middle)
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    double mu_hat;
    if (n % 2 == 1) {
        mu_hat = sorted[n / 2];
    } else {
        mu_hat = (sorted[n / 2 - 1] + sorted[n / 2]) * detail::HALF;
    }

    // b̂ = (1/n) Σ|xᵢ − μ̂|  (mean absolute deviation from median)
    double mad = detail::ZERO_DOUBLE;
    for (double v : values) {
        mad += std::fabs(v - mu_hat);
    }
    const double b_hat = mad / static_cast<double>(n);

    // Guard against degenerate b (e.g. all values identical → MAD = 0).
    // Clamp rather than throw so callers get a valid (if extreme) distribution.
    if (!std::isfinite(b_hat) || b_hat <= detail::ZERO_DOUBLE)
        throw std::invalid_argument("Laplace MLE produced a degenerate scale estimate (b ≤ 0); data may be constant or contain non-finite values");

    setParameters(mu_hat, b_hat);
}

void LaplaceDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                           std::vector<LaplaceDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void LaplaceDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    b_  = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string LaplaceDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "LaplaceDistribution(mu=" << mu_ << ",b=" << b_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double LaplaceDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = 1 + log(2b) = 1 - neg_log2b_
    return detail::ONE - neg_log2b_;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void LaplaceDistribution::getProbability(std::span<const double> values,
                                         std::span<double> results,
                                         const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        // Scalar element
        [](const LaplaceDistribution& d, double x) { return d.getProbability(x); },
        // SIMD vectorised batch
        [](const LaplaceDistribution& d, const double* vals, double* res, std::size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, nib = d.neg_inv_b_, nlb = d.neg_log2b_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, m, nib, nlb);
        },
        // Parallel fallback
        [](const LaplaceDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0) return;
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, nib = d.neg_inv_b_, nlb = d.neg_log2b_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = std::isfinite(x)
                                 ? std::exp(nlb + nib * std::fabs(x - m))
                                 : detail::ZERO_DOUBLE;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = std::isfinite(x)
                                 ? std::exp(nlb + nib * std::fabs(x - m))
                                 : detail::ZERO_DOUBLE;
                }
            }
        },
        // Work-stealing
        [](const LaplaceDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, nib = d.neg_inv_b_, nlb = d.neg_log2b_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = std::isfinite(x) ? std::exp(nlb + nib * std::fabs(x - m))
                                          : detail::ZERO_DOUBLE;
            });
            pool.waitForAll();
        });
}

void LaplaceDistribution::getLogProbability(std::span<const double> values,
                                            std::span<double> results,
                                            const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const LaplaceDistribution& d, double x) { return d.getLogProbability(x); },
        [](const LaplaceDistribution& d, const double* vals, double* res, std::size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, nib = d.neg_inv_b_, nlb = d.neg_log2b_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, m, nib, nlb);
        },
        [](const LaplaceDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0) return;
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, nib = d.neg_inv_b_, nlb = d.neg_log2b_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = std::isfinite(x) ? nlb + nib * std::fabs(x - m)
                                              : detail::NEGATIVE_INFINITY;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = std::isfinite(x) ? nlb + nib * std::fabs(x - m)
                                              : detail::NEGATIVE_INFINITY;
                }
            }
        },
        [](const LaplaceDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, nib = d.neg_inv_b_, nlb = d.neg_log2b_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = std::isfinite(x) ? nlb + nib * std::fabs(x - m)
                                          : detail::NEGATIVE_INFINITY;
            });
            pool.waitForAll();
        });
}

void LaplaceDistribution::getCumulativeProbability(std::span<const double> values,
                                                   std::span<double> results,
                                                   const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const LaplaceDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const LaplaceDistribution& d, const double* vals, double* res, std::size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, hib = d.half_inv_b_;
            lock.unlock();
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, m, hib);
        },
        [](const LaplaceDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0) return;
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, hib = d.half_inv_b_;
            lock.unlock();
            const double inv_b = detail::TWO * hib;  // 1/b = 2*(0.5/b)
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (!std::isfinite(x)) { res[i] = (x > 0) ? detail::ONE : detail::ZERO_DOUBLE; return; }
                    const double dv = x - m;
                    res[i] = (dv <= detail::ZERO_DOUBLE)
                                 ? detail::HALF * std::exp(dv * inv_b)
                                 : detail::ONE - detail::HALF * std::exp(-dv * inv_b);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (!std::isfinite(x)) { res[i] = (x > 0) ? detail::ONE : detail::ZERO_DOUBLE; continue; }
                    // hib = 0.5/b, so 1/b = 2*hib
                    const double dv = x - m;
                    const double inv_b = detail::TWO * hib;
                    res[i] = (dv <= detail::ZERO_DOUBLE)
                                 ? detail::HALF * std::exp(dv * inv_b)
                                 : detail::ONE - detail::HALF * std::exp(-dv * inv_b);
                }
            }
        },
        [](const LaplaceDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LaplaceDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double m = d.mu_, hib = d.half_inv_b_;
            lock.unlock();
            const double inv_b = detail::TWO * hib;
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (!std::isfinite(x)) { res[i] = (x > 0) ? detail::ONE : detail::ZERO_DOUBLE; return; }
                const double dv = x - m;
                res[i] = (dv <= detail::ZERO_DOUBLE)
                             ? detail::HALF * std::exp(dv * inv_b)
                             : detail::ONE - detail::HALF * std::exp(-dv * inv_b);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool LaplaceDistribution::operator==(const LaplaceDistribution& other) const {
    if (this == &other) return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) <= detail::DEFAULT_TOLERANCE &&
           std::fabs(b_  - other.b_)  <= detail::DEFAULT_TOLERANCE;
}

bool LaplaceDistribution::operator!=(const LaplaceDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const LaplaceDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, LaplaceDistribution& dist) {
    std::string token;
    double mu, b;

    is >> token;
    if (!token.starts_with("LaplaceDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t mu_pos = token.find("mu=");
    const size_t b_pos  = token.find(",b=");
    const size_t close  = token.find(")", b_pos != std::string::npos ? b_pos : 0);

    if (mu_pos == std::string::npos || b_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        mu = std::stod(token.substr(mu_pos + 3, b_pos - mu_pos - 3));
        b  = std::stod(token.substr(b_pos  + 3, close - b_pos - 3));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(mu, b);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void LaplaceDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count,
    double cached_mu, double cached_neg_inv_b,
    double cached_neg_log2b) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    // Allocate aligned temp buffer
    std::vector<double, arch::simd::aligned_allocator<double>> tmp(count);

    // Step 1: tmp = x - mu
    VectorOps::scalar_add(values, -cached_mu, tmp.data(), count);

    // Step 2: tmp = |x - mu|  (scalar fabs loop; compilers auto-vectorise this)
    for (std::size_t i = 0; i < count; ++i)
        tmp[i] = std::fabs(tmp[i]);

    // Step 3: tmp = -|x - mu| / b
    VectorOps::scalar_multiply(tmp.data(), cached_neg_inv_b, tmp.data(), count);

    // Step 4: results = tmp + neg_log2b  = -log(2b) - |x-mu|/b = LogPDF
    VectorOps::scalar_add(tmp.data(), cached_neg_log2b, results, count);

    // Fixup: non-finite inputs → -inf
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void LaplaceDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count,
    double cached_mu, double cached_neg_inv_b,
    double cached_neg_log2b) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    // Steps 1-4: compute LogPDF into results
    getLogProbabilityBatchUnsafeImpl(values, results, count,
                                     cached_mu, cached_neg_inv_b, cached_neg_log2b);

    // Step 5: results = exp(LogPDF)
    VectorOps::vector_exp(results, results, count);

    // Fixup: non-finite inputs → 0
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
            results[i] = detail::ZERO_DOUBLE;
    }
}

void LaplaceDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count,
    double cached_mu, double cached_half_inv_b) const noexcept {
    // CDF: piecewise (signed branch prevents a clean SIMD pipeline).
    // half_inv_b = 0.5/b, so 1/b = 2 * half_inv_b.
    const double inv_b = detail::TWO * cached_half_inv_b;
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (!std::isfinite(x)) {
            results[i] = (x > 0) ? detail::ONE : detail::ZERO_DOUBLE;
            continue;
        }
        const double dv = x - cached_mu;
        if (dv <= detail::ZERO_DOUBLE) {
            results[i] = detail::HALF * std::exp(dv * inv_b);
        } else {
            results[i] = detail::ONE - detail::HALF * std::exp(-dv * inv_b);
        }
    }
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void LaplaceDistribution::updateCacheUnsafe() const noexcept {
    neg_inv_b_  = -detail::ONE / b_;
    neg_log2b_  = -std::log(detail::TWO * b_);
    half_inv_b_ = detail::HALF / b_;

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicB_.store(b_,   std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

#include "libstats/distributions/exponential.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/common/cpu_detection_fwd.h"  // CPU feature queries (lightweight)
#include "libstats/core/log_space_ops.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"
// Note: parallel execution included through distribution base inheritance
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"  // For DispatchUtils::autoDispatch

#include <algorithm>
#include <cmath>
#include <execution>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <sstream>
#include <vector>
#define LIBSTATS_ALL_OF(range, predicate) std::ranges::all_of((range), (predicate))
#define LIBSTATS_SORT(range) std::ranges::sort((range))

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

ExponentialDistribution::ExponentialDistribution(double lambda)
    : DistributionBase(), lambda_(lambda) {
    validateParameters(lambda);
    // Cache will be updated on first use
}

ExponentialDistribution::ExponentialDistribution(const ExponentialDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    // Cache will be updated on first use
}

ExponentialDistribution& ExponentialDistribution::operator=(const ExponentialDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);

        // Copy parameters (don't call base class operator= to avoid deadlock)
        lambda_ = other.lambda_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

ExponentialDistribution::ExponentialDistribution(ExponentialDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    lambda_ = other.lambda_;
    other.lambda_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

ExponentialDistribution& ExponentialDistribution::operator=(
    ExponentialDistribution&& other) noexcept {
    if (this != &other) {
        lambda_ = other.lambda_;
        other.lambda_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. SAFE FACTORY METHODS
//==============================================================================

// Note: Safe factory methods are implemented inline in the header for performance

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void ExponentialDistribution::setLambda(double lambda) {
    validateParameters(lambda);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

double ExponentialDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    return invLambda_;
}

double ExponentialDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    return invLambdaSquared_;
}

double ExponentialDistribution::getScale() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    return invLambda_;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult ExponentialDistribution::trySetLambda(double lambda) noexcept {
    auto validation = validateExponentialParameters(lambda);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult ExponentialDistribution::trySetParameters(double lambda) noexcept {
    auto validation = validateExponentialParameters(lambda);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double ExponentialDistribution::getProbability(double x) const {
    // Return 0 for negative values
    if (x < detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return std::exp(-x);
    }

    // General case: f(x) = λ * exp(-λx)
    return lambda_ * std::exp(negLambda_ * x);
}

double ExponentialDistribution::getLogProbability(double x) const noexcept {
    // Return -∞ for negative values
    if (x < detail::ZERO_DOUBLE) {
        return detail::NEGATIVE_INFINITY;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -x;
    }

    // General case: log(f(x)) = log(λ) - λx
    return logLambda_ + negLambda_ * x;
}

double ExponentialDistribution::getCumulativeProbability(double x) const {
    // Return 0 for negative values
    if (x < detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return detail::ONE - std::exp(-x);
    }

    // General case: F(x) = 1 - exp(-λx)
    return detail::ONE - std::exp(negLambda_ * x);
}

double ExponentialDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -std::log(detail::ONE - p);
    }

    // General case: F^(-1)(p) = -ln(1-p)/λ
    return -std::log(detail::ONE - p) * invLambda_;
}

double ExponentialDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Use high-quality uniform distribution
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);

    double u = uniform(rng);

    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -std::log(u);
    }

    // General case: inverse transform sampling
    // X = -ln(U)/λ where U ~ Uniform(0,1)
    return -std::log(u) * invLambda_;
}

std::vector<double> ExponentialDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Cache parameters for thread-safe batch generation
    const bool cached_is_unit_rate = isUnitRate_;
    const double cached_inv_lambda = invLambda_;

    lock.unlock();  // Release lock before generation

    // Use high-quality uniform distribution for batch generation
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);

    // Generate samples using inverse transform method: X = -ln(U)/λ
    for (size_t i = 0; i < n; ++i) {
        double u = uniform(rng);

        if (cached_is_unit_rate) {
            // Fast path for unit exponential (λ = 1)
            samples.push_back(-std::log(u));
        } else {
            // General case: X = -ln(U)/λ
            samples.push_back(-std::log(u) * cached_inv_lambda);
        }
    }

    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void ExponentialDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    // C++20 best practices: Use ranges and views for safe validation
    // Check for non-positive values using ranges algorithms
    if (!LIBSTATS_ALL_OF(values, [](double value) { return value > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument("Exponential distribution requires positive values");
    }

    // Calculate mean using standard accumulate (following Gaussian pattern)
    const double sum = std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE);
    const double sample_mean = sum / static_cast<double>(values.size());

    if (sample_mean <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Sample mean must be positive for exponential distribution");
    }

    // Set parameters (this will validate and invalidate cache)
    setLambda(detail::ONE / sample_mean);
}

void ExponentialDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                               std::vector<ExponentialDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void ExponentialDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string ExponentialDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "ExponentialDistribution(lambda=" << lambda_ << ")";
    return oss.str();
}

//==============================================================================
// 7. ADVANCED STATISTICAL METHODS
//==============================================================================

//==============================================================================
// 8. GOODNESS-OF-FIT TESTS
//==============================================================================

//==============================================================================
// 9. CROSS-VALIDATION METHODS
//==============================================================================

//==============================================================================
// 10. INFORMATION CRITERIA
//==============================================================================

//==============================================================================
// 11. BOOTSTRAP METHODS
//==============================================================================

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

// Utility methods moved from header for PIMPL optimization - no longer inline
double ExponentialDistribution::getLambdaAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicLambda_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getLambda();
}

double ExponentialDistribution::getSkewness() const noexcept {
    return 2.0;  // Exponential distribution is always right-skewed
}

double ExponentialDistribution::getKurtosis() const noexcept {
    return 6.0;  // Exponential distribution has high kurtosis
}

int ExponentialDistribution::getNumParameters() const noexcept {
    return 1;
}

std::string ExponentialDistribution::getDistributionName() const {
    return "Exponential";
}

bool ExponentialDistribution::isDiscrete() const noexcept {
    return false;
}

double ExponentialDistribution::getSupportLowerBound() const noexcept {
    return 0.0;
}

double ExponentialDistribution::getSupportUpperBound() const noexcept {
    return std::numeric_limits<double>::infinity();
}

VoidResult ExponentialDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateExponentialParameters(lambda_);
}

double ExponentialDistribution::getHalfLife() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::log(2.0) / lambda_;
}

bool ExponentialDistribution::isMemoryless() const noexcept {
    return true;
}

double ExponentialDistribution::getMedian() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return detail::LN2 / lambda_;  // Use precomputed ln(2)
}

double ExponentialDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return detail::ONE - std::log(lambda_);
}

double ExponentialDistribution::getMode() const noexcept {
    return detail::ZERO_DOUBLE;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void ExponentialDistribution::getProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const ExponentialDistribution& dist, double value) {
            return dist.getProbability(value);
        },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();

            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals,
           std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = std::exp(-x);
                    } else {
                        res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = std::exp(-x);
                    } else {
                        res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                    }
                }
            }
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = std::exp(-x);
                } else {
                    res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                }
            });
        });
}

void ExponentialDistribution::getLogProbability(std::span<const double> values,
                                                std::span<double> results,
                                                const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const ExponentialDistribution& dist, double value) {
            return dist.getLogProbability(value);
        },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();

            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_log_lambda,
                                                  cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals,
           std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_rate) {
                        res[i] = -x;
                    } else {
                        res[i] = cached_log_lambda + cached_neg_lambda * x;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_rate) {
                        res[i] = -x;
                    } else {
                        res[i] = cached_log_lambda + cached_neg_lambda * x;
                    }
                }
            }
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else if (cached_is_unit_rate) {
                    res[i] = -x;
                } else {
                    res[i] = cached_log_lambda + cached_neg_lambda * x;
                }
            });
        });
}

void ExponentialDistribution::getCumulativeProbability(std::span<const double> values,
                                                       std::span<double> results,
                                                       const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const ExponentialDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();

            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals,
           std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = detail::ONE - std::exp(-x);
                    } else {
                        res[i] = detail::ONE - std::exp(cached_neg_lambda * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = detail::ONE - std::exp(-x);
                    } else {
                        res[i] = detail::ONE - std::exp(cached_neg_lambda * x);
                    }
                }
            }
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = detail::ONE - std::exp(-x);
                } else {
                    res[i] = detail::ONE - std::exp(cached_neg_lambda * x);
                }
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (POWER USER INTERFACE)
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool ExponentialDistribution::operator==(const ExponentialDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    return std::abs(lambda_ - other.lambda_) <= detail::DEFAULT_TOLERANCE;
}

//==============================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const ExponentialDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, ExponentialDistribution& distribution) {
    std::string token;
    double lambda;

    // Expected format: "ExponentialDistribution(lambda=<value>)"
    // We'll parse this step by step

    // Skip whitespace and read the first part
    is >> token;
    if (!token.starts_with("ExponentialDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract lambda value
    if (token.find("lambda=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t lambda_pos = token.find("lambda=") + 7;
    size_t close_paren = token.find(")", lambda_pos);
    if (close_paren == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string lambda_str = token.substr(lambda_pos, close_paren - lambda_pos);
        lambda = std::stod(lambda_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Validate and set parameter using the safe API
    auto result = distribution.trySetParameters(lambda);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==============================================================================
// 17. PRIVATE FACTORY IMPLEMENTATION METHODS
//==============================================================================

// Note: Private factory implementation methods are currently inline in the header
// This section exists for standardization and documentation purposes

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// These three methods are the computational core of Exponential batch ops.
// Called after the public API validates inputs and extracts cached parameters
// under a read lock. Raw pointers avoid span bounds-checking overhead.
//
// Architecture: compute+fixup pattern
//   SIMD runs over the full array assuming in-support inputs (x >= 0), then a
//   scalar pass zeroes/neginfs any negative elements. Callers of exponential
//   distributions rarely pass x < 0, so the fixup loop is almost always a
//   no-op in practice. This keeps the hot SIMD path branch-free.
//
//   PDF:    results = cached_neg_lambda * x → exp → * cached_lambda
//             fixup: x < 0 → 0
//   LogPDF: results = cached_neg_lambda * x → + cached_log_lambda
//             fixup: x < 0 → -inf   (purely affine — highest SIMD gain)
//   CDF:    results = cached_neg_lambda * x → exp → negate → + 1
//             fixup: x < 0 → 0
//
// See GaussianDistribution::getProbabilityBatchUnsafeImpl for the in-place
// workspace convention and SIMDPolicy threshold rationale.
//==============================================================================

void ExponentialDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, size_t count, double cached_lambda,
    double cached_neg_lambda) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (std::abs(cached_lambda - detail::ONE) <= detail::DEFAULT_TOLERANCE) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        }
        return;
    }

    // Step 1: results = cached_neg_lambda * x  (= -λx)
    arch::simd::VectorOps::scalar_multiply(values, cached_neg_lambda, results, count);
    // Step 2: results = exp(-λx)
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 3: results = λ · exp(-λx)
    arch::simd::VectorOps::scalar_multiply(results, cached_lambda, results, count);
    // Fixup: x < 0 is outside support; PDF = 0.
    for (size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, size_t count, double cached_log_lambda,
    double cached_neg_lambda) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
            } else if (std::abs(cached_log_lambda - detail::ZERO_DOUBLE) <=
                       detail::DEFAULT_TOLERANCE) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        }
        return;
    }

    // Purely affine: log(λ) - λx. No transcendentals — maximum SIMD throughput.
    // Step 1: results = cached_neg_lambda * x  (= -λx)
    arch::simd::VectorOps::scalar_multiply(values, cached_neg_lambda, results, count);
    // Step 2: results = log(λ) + (-λx)  (= log(λ) - λx)
    arch::simd::VectorOps::scalar_add(results, cached_log_lambda, results, count);
    // Fixup: x < 0 is outside support; LogPDF = -inf.
    for (size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::NEGATIVE_INFINITY;
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, size_t count, double cached_neg_lambda) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (std::abs(cached_neg_lambda + detail::ONE) <= detail::DEFAULT_TOLERANCE) {
                results[i] = detail::ONE - std::exp(-x);
            } else {
                results[i] = detail::ONE - std::exp(cached_neg_lambda * x);
            }
        }
        return;
    }

    // Step 1: results = cached_neg_lambda * x  (= -λx)
    arch::simd::VectorOps::scalar_multiply(values, cached_neg_lambda, results, count);
    // Step 2: results = exp(-λx)
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 3: results = -(exp(-λx))  then  results = 1 - exp(-λx)
    arch::simd::VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    // Fixup: x < 0 is outside support; CDF = 0.
    for (size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

// Note: Private computational methods are implemented inline in the header for performance
// This section exists for standardization and documentation purposes

//==============================================================================
// 20. PRIVATE UTILITY METHODS
//==============================================================================

// Note: Private utility methods are implemented inline in the header for performance
// This section exists for standardization and documentation purposes

//==============================================================================
// 21. DISTRIBUTION PARAMETERS
//==============================================================================

// Note: Distribution parameters are declared in the header as private member variables
// This section exists for standardization and documentation purposes

//==============================================================================
// 22. PERFORMANCE CACHE
//==============================================================================

// Note: Performance cache variables are declared in the header as mutable private members
// This section exists for standardization and documentation purposes

//==============================================================================
// 23. OPTIMIZATION FLAGS
//==============================================================================

// Note: Optimization flags are declared in the header as private member variables
// This section exists for standardization and documentation purposes

//==============================================================================
// 24. SPECIALIZED CACHES
//==============================================================================

// Note: Specialized caches are declared in the header as private member variables
// This section exists for standardization and documentation purposes

}  // namespace stats

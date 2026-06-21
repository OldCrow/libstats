#include "libstats/distributions/poisson.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/core/math_constants.h"
#include "libstats/core/parallel_batch_fit.h"
#include "libstats/core/statistical_constants.h"

// Core functionality - lightweight headers
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/log_space_ops.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/safety.h"

// Platform headers - use forward declarations where available
#include "libstats/common/cpu_detection_fwd.h"  // Lightweight CPU detection
// Note: parallel_execution.h is transitively included via dispatch_utils.h
#include "libstats/common/simd_policy_fwd.h"  // Lightweight SIMD policy
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h

#include <algorithm>
#include <any>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTORS
//==============================================================================

PoissonDistribution::PoissonDistribution(double lambda) : DistributionBase(), lambda_(lambda) {
    validateParameters(lambda);
    // Cache will be updated on first use
}

PoissonDistribution::PoissonDistribution(const PoissonDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    // Cache will be updated on first use
}

PoissonDistribution& PoissonDistribution::operator=(const PoissonDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);

        // Copy parameters
        lambda_ = other.lambda_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

PoissonDistribution::PoissonDistribution(PoissonDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    lambda_ = other.lambda_;
    other.lambda_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

PoissonDistribution& PoissonDistribution::operator=(PoissonDistribution&& other) noexcept {
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

//==========================================================================
// 2. SAFE FACTORY METHODS (Exception-free construction)
//==========================================================================

// Note: All methods in this section currently implemented inline in the header
// This section maintained for template compliance

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void PoissonDistribution::setLambda(double lambda) {
    validateParameters(lambda);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void PoissonDistribution::setParameters(double lambda) {
    validateParameters(lambda);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

double PoissonDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_;
}

double PoissonDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_;
}

double PoissonDistribution::getSkewness() const noexcept {
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

    return detail::ONE / sqrtLambda_;  // 1/√λ
}

double PoissonDistribution::getKurtosis() const noexcept {
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

    return invLambda_;  // 1/λ (excess kurtosis)
}

double PoissonDistribution::getLambdaAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicLambda_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getLambda();
}

inline int PoissonDistribution::getNumParameters() const noexcept {
    return 1;
}

inline std::string PoissonDistribution::getDistributionName() const {
    return "Poisson";
}

inline bool PoissonDistribution::isDiscrete() const noexcept {
    return true;
}

inline double PoissonDistribution::getSupportLowerBound() const noexcept {
    return 0.0;
}

inline double PoissonDistribution::getSupportUpperBound() const noexcept {
    return std::numeric_limits<double>::infinity();
}

double PoissonDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::floor(lambda_);
}

//==============================================================================
// 4. RESULT-BASED SETTERS (C++20 Best Practice: Complex implementations in .cpp)
//==============================================================================

VoidResult PoissonDistribution::trySetLambda(double lambda) noexcept {
    auto validation = validatePoissonParameters(lambda);
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

VoidResult PoissonDistribution::trySetParameters(double lambda) noexcept {
    auto validation = validatePoissonParameters(lambda);
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

inline VoidResult PoissonDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validatePoissonParameters(lambda_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double PoissonDistribution::getProbability(double x) const {
    if (x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x))
        return detail::ZERO_DOUBLE;

    return getProbabilityExact(k);
}

double PoissonDistribution::getLogProbability(double x) const noexcept {
    if (x < detail::ZERO_DOUBLE)
        return detail::MIN_LOG_PROBABILITY;

    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x))
        return detail::MIN_LOG_PROBABILITY;

    return getLogProbabilityExact(k);
}

double PoissonDistribution::getCumulativeProbability(double x) const {
    if (x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x))
        return detail::ONE;

    return getCumulativeProbabilityExact(k);
}

double PoissonDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0,1]");
    }

    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    // Snapshot lambda_ under a shared lock to prevent a data race with
    // concurrent setLambda() / trySetLambda() calls (NEW-TS-2).
    double local_lambda;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        local_lambda = lambda_;
    }

    // Use bracketing search for quantile. MC-4: use a wide integer bound to
    // avoid overflow while expanding for large lambda.
    std::int64_t lower = 0;
    std::int64_t upper = static_cast<std::int64_t>(std::ceil(
        local_lambda + detail::QUANTILE_UPPER_BOUND_MULTIPLIER * std::sqrt(local_lambda)));
    upper = std::max<std::int64_t>(upper, 1);

    // Expand upper bound if necessary
    constexpr std::int64_t kMaxQuantileSearch =
        static_cast<std::int64_t>(std::numeric_limits<int>::max());
    while (upper < kMaxQuantileSearch &&
           getCumulativeProbabilityExact(static_cast<int>(upper)) < p) {
        lower = upper;
        upper = std::min(upper * 2, kMaxQuantileSearch);
    }
    if (getCumulativeProbabilityExact(static_cast<int>(upper)) < p) {
        return static_cast<double>(upper);
    }

    // Binary search
    while (upper - lower > 1) {
        const std::int64_t mid = lower + (upper - lower) / detail::TWO_INT;
        if (getCumulativeProbabilityExact(static_cast<int>(mid)) < p) {
            lower = mid;
        } else {
            upper = mid;
        }
    }

    return static_cast<double>(upper);
}

double PoissonDistribution::sample(std::mt19937& rng) const {
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

    const double cached_lambda = lambda_;
    const bool cached_is_small = isSmallLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;

    lock.unlock();  // Release lock before generation

    if (cached_is_small) {
        // Knuth's algorithm for small lambda
        double L = cached_exp_neg_lambda;
        int k = 0;
        double p = detail::ONE;

        std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);

        do {
            k++;
            p *= uniform(rng);
        } while (p > L);

        return static_cast<double>(k - 1);
    } else {
        // For large lambda, delegate to the standard library's Poisson sampler
        // which uses an exact algorithm (e.g. Atkinson's PA or similar) rather
        // than the biased normal-approximation-plus-rounding path (MC-15).
        std::poisson_distribution<int> dist(cached_lambda);
        return static_cast<double>(dist(rng));
    }
}

std::vector<double> PoissonDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    // Get cached parameters for efficiency
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

    const double cached_lambda = lambda_;
    const bool cached_is_small = isSmallLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    lock.unlock();

    // Generate batch samples using the appropriate method
    if (cached_is_small) {
        // Knuth's algorithm for small lambda - optimized for batch
        double L = cached_exp_neg_lambda;
        std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);

        for (size_t i = 0; i < n; ++i) {
            int k = 0;
            double p = detail::ONE;

            do {
                k++;
                p *= uniform(rng);
            } while (p > L);

            samples.push_back(static_cast<double>(k - 1));
        }
    } else {
        // Exact large-lambda path via std::poisson_distribution (MC-15).
        std::poisson_distribution<int> dist(cached_lambda);
        for (size_t i = 0; i < n; ++i)
            samples.push_back(static_cast<double>(dist(rng)));
    }

    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void PoissonDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit to empty data");
    }

    // Check minimum data points for reliable fitting
    if (values.size() < detail::MIN_DATA_POINTS_FOR_CHI_SQUARE) {  // Minimum data points for
                                                                   // reliable fitting
        throw std::invalid_argument("Insufficient data points for reliable Poisson fitting");
    }

    // Validate that all values are non-negative (count data)
    for (double value : values) {
        if (value < detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }

    // For Poisson distribution, MLE gives λ = sample mean
    double sum = std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE);
    double sample_mean = sum / static_cast<double>(values.size());

    // Ensure fitted lambda is positive
    if (sample_mean <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Sample mean must be positive for Poisson distribution");
    }

    // Set the new parameter
    setLambda(sample_mean);
}

void PoissonDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                           std::vector<PoissonDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void PoissonDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string PoissonDistribution::toString() const {
    std::ostringstream oss;
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    oss << "Poisson(λ=" << lambda_ << ")";
    return oss.str();
}

//==========================================================================
// 7. ADVANCED STATISTICAL METHODS
//==========================================================================

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

std::vector<int> PoissonDistribution::sampleIntegers(std::mt19937& rng, std::size_t count) const {
    std::vector<int> samples;
    samples.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        samples.push_back(static_cast<int>(sample(rng)));
    }

    return samples;
}

double PoissonDistribution::getProbabilityExact(int k) const {
    if (k < 0)
        return detail::ZERO_DOUBLE;

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

    if (isSmallLambda_) {
        return computePMFSmall(k);
    } else {
        return computePMFLarge(k);
    }
}

double PoissonDistribution::getLogProbabilityExact(int k) const noexcept {
    if (k < 0)
        return detail::MIN_LOG_PROBABILITY;

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

    return computeLogPMF(k);
}

double PoissonDistribution::getCumulativeProbabilityExact(int k) const {
    if (k < 0)
        return detail::ZERO_DOUBLE;

    return computeCDF(k);
}

bool PoissonDistribution::canUseNormalApproximation() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_ > detail::NORMAL_APPROXIMATION_THRESHOLD;  // Rule of thumb: λ > threshold for
                                                              // reasonable normal approximation
}

double PoissonDistribution::getMedian() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // For Poisson distribution, median ≈ λ + 1/3 - 0.02/λ for large λ
    // For small λ, use numerical approximation via quantile function
    if (lambda_ > 10.0) {
        return lambda_ + (1.0 / 3.0) - (0.02 / lambda_);
    } else {
        // Use quantile function for more accurate median calculation for small λ
        return getQuantile(0.5);
    }
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH METHODS
//==============================================================================

void PoissonDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                         const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const PoissonDistribution& dist, double value) { return dist.getProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_lambda = dist.lambda_;
            const double cached_log_lambda = dist.logLambda_;
            const double cached_exp_neg_lambda = dist.expNegLambda_;
            lock.unlock();

            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda, cached_log_lambda,
                                               cached_exp_neg_lambda);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_lambda = dist.lambda_;
            const double cached_log_lambda = dist.logLambda_;
            const double cached_exp_neg_lambda = dist.expNegLambda_;
            [[maybe_unused]] const bool cached_is_small_lambda = dist.isSmallLambda_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        return;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::ZERO_DOUBLE;
                        return;
                    }

                    // Compute PMF using cached parameters
                    if (k == 0) {
                        res[i] = cached_exp_neg_lambda;
                    } else if (cached_lambda < detail::SMALL_LAMBDA_THRESHOLD &&
                               k < static_cast<int>(PoissonDistribution::FACTORIAL_CACHE.size())) {
                        res[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda /
                                 PoissonDistribution::FACTORIAL_CACHE[static_cast<std::size_t>(k)];
                    } else {
                        double log_result = k * cached_log_lambda - cached_lambda -
                                            PoissonDistribution::logFactorial(k);
                        res[i] = std::exp(log_result);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        continue;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::ZERO_DOUBLE;
                        continue;
                    }

                    if (k == 0) {
                        res[i] = cached_exp_neg_lambda;
                    } else if (cached_lambda < detail::SMALL_LAMBDA_THRESHOLD &&
                               k < static_cast<int>(PoissonDistribution::FACTORIAL_CACHE.size())) {
                        res[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda /
                                 PoissonDistribution::FACTORIAL_CACHE[static_cast<std::size_t>(k)];
                    } else {
                        double log_result = k * cached_log_lambda - cached_lambda -
                                            PoissonDistribution::logFactorial(k);
                        res[i] = std::exp(log_result);
                    }
                }
            }
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res,
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
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_lambda = dist.lambda_;
            const double cached_log_lambda = dist.logLambda_;
            const double cached_exp_neg_lambda = dist.expNegLambda_;
            [[maybe_unused]] const bool cached_is_small_lambda = dist.isSmallLambda_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (vals[i] < detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                    return;
                }

                int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                if (!PoissonDistribution::isValidCount(vals[i])) {
                    res[i] = detail::ZERO_DOUBLE;
                    return;
                }

                if (k == 0) {
                    res[i] = cached_exp_neg_lambda;
                } else if (cached_lambda < detail::SMALL_LAMBDA_THRESHOLD &&
                           k < static_cast<int>(PoissonDistribution::FACTORIAL_CACHE.size())) {
                    res[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda /
                             PoissonDistribution::FACTORIAL_CACHE[static_cast<std::size_t>(k)];
                } else {
                    double log_result = k * cached_log_lambda - cached_lambda -
                                        PoissonDistribution::logFactorial(k);
                    res[i] = std::exp(log_result);
                }
            });
        });
}

void PoissonDistribution::getLogProbability(std::span<const double> values,
                                            std::span<double> results,
                                            const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const PoissonDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_lambda = dist.lambda_;
            const double cached_log_lambda = dist.logLambda_;
            lock.unlock();

            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda,
                                                  cached_log_lambda);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_lambda = dist.lambda_;
            const double cached_log_lambda = dist.logLambda_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::MIN_LOG_PROBABILITY;
                        return;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::MIN_LOG_PROBABILITY;
                        return;
                    }

                    // log P(X = k) = k * log(λ) - λ - log(k!)
                    res[i] = k * cached_log_lambda - cached_lambda -
                             PoissonDistribution::logFactorial(k);
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::MIN_LOG_PROBABILITY;
                        continue;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::MIN_LOG_PROBABILITY;
                        continue;
                    }

                    // log P(X = k) = k * log(λ) - λ - log(k!)
                    res[i] = k * cached_log_lambda - cached_lambda -
                             PoissonDistribution::logFactorial(k);
                }
            }
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res,
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
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_lambda = dist.lambda_;
            const double cached_log_lambda = dist.logLambda_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (vals[i] < detail::ZERO_DOUBLE) {
                    res[i] = detail::MIN_LOG_PROBABILITY;
                    return;
                }

                int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                if (!PoissonDistribution::isValidCount(vals[i])) {
                    res[i] = detail::MIN_LOG_PROBABILITY;
                    return;
                }

                // log P(X = k) = k * log(λ) - λ - log(k!)
                res[i] =
                    k * cached_log_lambda - cached_lambda - PoissonDistribution::logFactorial(k);
            });
        });
}

void PoissonDistribution::getCumulativeProbability(std::span<const double> values,
                                                   std::span<double> results,
                                                   const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const PoissonDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            const double cached_lambda = dist.lambda_;
            lock.unlock();

            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_lambda = dist.lambda_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        return;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::ONE;
                        return;
                    }

                    // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
                    res[i] = detail::gamma_q(k + 1, cached_lambda);
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                        continue;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::ONE;
                        continue;
                    }

                    // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
                    res[i] = detail::gamma_q(k + 1, cached_lambda);
                }
            }
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res,
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
                    const_cast<PoissonDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_lambda = dist.lambda_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (vals[i] < detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                    return;
                }

                int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                if (!PoissonDistribution::isValidCount(vals[i])) {
                    res[i] = detail::ONE;
                    return;
                }

                // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
                res[i] = detail::gamma_q(k + 1, cached_lambda);
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool PoissonDistribution::operator==(const PoissonDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    return std::abs(lambda_ - other.lambda_) <= detail::DEFAULT_TOLERANCE;
}

bool PoissonDistribution::operator!=(const PoissonDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const PoissonDistribution& dist) {
    os << dist.toString();
    return os;
}

std::istream& operator>>(std::istream& is, PoissonDistribution& distribution) {
    std::string token;
    double lambda;

    // Expected format: "Poisson(λ=<value>)"
    // We'll parse this step by step

    // Skip whitespace and read the first part
    is >> token;
    if (!token.starts_with("Poisson(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract λ value
    if (token.find("λ=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t lambda_pos = token.find("λ=") + 2;
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

    // Validate and set parameters using the safe API
    auto result = distribution.trySetParameters(lambda);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==========================================================================
// 17. PRIVATE FACTORY METHODS
//==========================================================================

PoissonDistribution PoissonDistribution::createUnchecked(double lambda) noexcept {
    PoissonDistribution dist(lambda, true);  // bypass validation
    return dist;
}

PoissonDistribution::PoissonDistribution(double lambda, bool /*bypassValidation*/) noexcept
    : DistributionBase(), lambda_(lambda) {
    // Cache will be updated on first use
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void PoissonDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                        std::size_t count, double lambda,
                                                        double log_lambda,
                                                        double exp_neg_lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }

            int k = roundToNonNegativeInt(values[i]);
            if (!isValidCount(values[i])) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }

            if (k == 0) {
                results[i] = exp_neg_lambda;
            } else if (lambda < detail::SMALL_LAMBDA_THRESHOLD &&
                       k < static_cast<int>(FACTORIAL_CACHE.size())) {
                results[i] = std::pow(lambda, k) * exp_neg_lambda /
                             FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                double log_result = k * log_lambda - lambda - logFactorial(k);
                results[i] = std::exp(log_result);
            }
        }
        return;
    }

    // If SIMD is enabled but no vectorized implementation available, fall back to scalar
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }

        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }

        if (k == 0) {
            results[i] = exp_neg_lambda;
        } else if (lambda < detail::SMALL_LAMBDA_THRESHOLD &&
                   k < static_cast<int>(FACTORIAL_CACHE.size())) {
            results[i] =
                std::pow(lambda, k) * exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
        } else {
            double log_result = k * log_lambda - lambda - logFactorial(k);
            results[i] = std::exp(log_result);
        }
    }
}

void PoissonDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                           std::size_t count, double lambda,
                                                           double log_lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < detail::ZERO_DOUBLE) {
                results[i] = detail::MIN_LOG_PROBABILITY;
                continue;
            }

            int k = roundToNonNegativeInt(values[i]);
            if (!isValidCount(values[i])) {
                results[i] = detail::MIN_LOG_PROBABILITY;
                continue;
            }

            results[i] = k * log_lambda - lambda - logFactorial(k);
        }
        return;
    }

    // If SIMD is enabled but no vectorized implementation available, fall back to scalar
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::MIN_LOG_PROBABILITY;
            continue;
        }

        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = detail::MIN_LOG_PROBABILITY;
            continue;
        }

        results[i] = k * log_lambda - lambda - logFactorial(k);
    }
}

void PoissonDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                  double* results,
                                                                  std::size_t count,
                                                                  double lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }

            int k = roundToNonNegativeInt(values[i]);
            if (!isValidCount(values[i])) {
                results[i] = detail::ONE;
                continue;
            }

            results[i] = detail::gamma_q(k + 1, lambda);
        }
        return;
    }

    // If SIMD is enabled but no vectorized implementation available, fall back to scalar
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }

        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = detail::ONE;
            continue;
        }

        results[i] = detail::gamma_q(k + 1, lambda);
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

// Implementation moved from header - NOT inline due to complexity
void PoissonDistribution::updateCacheUnsafe() const noexcept {
    // Primary calculations - compute once, reuse multiple times
    logLambda_ = std::log(lambda_);
    expNegLambda_ = std::exp(-lambda_);
    sqrtLambda_ = std::sqrt(lambda_);
    invLambda_ = detail::ONE / lambda_;

    // Stirling's approximation for log(Γ(λ+1)) = log(λ!)
    logGammaLambdaPlus1_ = std::lgamma(lambda_ + detail::ONE);

    // Optimization flags
    isSmallLambda_ = (lambda_ < detail::SMALL_LAMBDA_THRESHOLD);
    isLargeLambda_ = (lambda_ > detail::HUNDRED);
    isVeryLargeLambda_ = (lambda_ > detail::THOUSAND);
    isIntegerLambda_ = (std::abs(lambda_ - std::round(lambda_)) <= detail::DEFAULT_TOLERANCE);
    isTinyLambda_ = (lambda_ < detail::TENTH);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);

    // Update atomic parameters for lock-free access
    atomicLambda_.store(lambda_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

// Static validation method moved from header for better compile times
void PoissonDistribution::validateParameters(double lambda) {
    if (std::isnan(lambda) || std::isinf(lambda) || lambda <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");
    }
    if (lambda > detail::MAX_POISSON_LAMBDA) {
        throw std::invalid_argument("Lambda too large for accurate Poisson computation");
    }
}

double PoissonDistribution::computePMFSmall(int k) const noexcept {
    // Direct computation for small lambda
    if (k < static_cast<int>(FACTORIAL_CACHE.size())) {
        // Use cached factorial
        return std::pow(lambda_, k) * expNegLambda_ / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
    } else {
        // Use log-space computation
        return std::exp(computeLogPMF(k));
    }
}

double PoissonDistribution::computePMFLarge(int k) const noexcept {
    // Use Stirling's approximation or normal approximation for large lambda
    if (isVeryLargeLambda_ && std::abs(k - lambda_) < detail::THREE * sqrtLambda_) {
        // Normal approximation with continuity correction
        double z = (k + detail::HALF - lambda_) / sqrtLambda_;
        return std::exp(detail::NEG_HALF * z * z) / (sqrtLambda_ * detail::SQRT_2PI);
    } else {
        // Use log-space computation
        return std::exp(computeLogPMF(k));
    }
}

double PoissonDistribution::computeLogPMF(int k) const noexcept {
    // log P(X = k) = k * log(λ) - λ - log(k!)
    double log_factorial_k = logFactorial(k);
    return k * logLambda_ - lambda_ - log_factorial_k;
}

double PoissonDistribution::computeCDF(int k) const noexcept {
    // Use regularized incomplete gamma function: P(X <= k) = Q(k+1, λ)
    // where Q(a,x) is the regularized upper incomplete gamma function
    return detail::gamma_q(k + 1, lambda_);
}

double PoissonDistribution::factorial(int n) noexcept {
    if (n < 0)
        return detail::ZERO_DOUBLE;
    if (n < static_cast<int>(FACTORIAL_CACHE.size())) {
        return FACTORIAL_CACHE[static_cast<std::size_t>(n)];
    }

    // Use Stirling's approximation for large n
    if (n > 170)
        return std::numeric_limits<double>::infinity();  // Overflow

    double result = detail::ONE;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double PoissonDistribution::logFactorial(int n) noexcept {
    if (n < 0)
        return detail::MIN_LOG_PROBABILITY;
    if (n == 0 || n == 1)
        return detail::ZERO_DOUBLE;

    if (n < static_cast<int>(FACTORIAL_CACHE.size())) {
        return std::log(FACTORIAL_CACHE[static_cast<std::size_t>(n)]);
    }

    // Use Stirling's approximation: log(n!) ≈ n*log(n) - n + 0.5*log(2πn)
    return std::lgamma(n + detail::ONE);
}

//==========================================================================
// 20. PRIVATE UTILITY METHODS
//==========================================================================

// Static utility methods moved from header for better compile times
inline int PoissonDistribution::roundToNonNegativeInt(double x) noexcept {
    if (x < 0.0)
        return 0;
    return static_cast<int>(std::round(x));
}

inline bool PoissonDistribution::isValidCount(double x) noexcept {
    return (x >= 0.0 && x <= static_cast<double>(INT_MAX));
}

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

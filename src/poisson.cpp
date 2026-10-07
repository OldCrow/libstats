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
#include <string_view>
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
        // Both locks, as copy-assignment takes, in std::lock order; the source is written, so
        // exclusively (#184).
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
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

double PoissonDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_;
}

double PoissonDistribution::getVariance() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_;
}

double PoissonDistribution::getSkewness() const {
    double sqrtL;
    withCacheSnapshot([&] { sqrtL = sqrtLambda_; });
    return detail::ONE / sqrtL;
}

double PoissonDistribution::getKurtosis() const {
    double invL;
    withCacheSnapshot([&] { invL = invLambda_; });
    return invL;
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

inline bool PoissonDistribution::isDiscrete() const noexcept {
    return true;
}

inline double PoissonDistribution::getSupportLowerBound() const noexcept {
    return 0.0;
}

inline double PoissonDistribution::getSupportUpperBound() const noexcept {
    return std::numeric_limits<double>::infinity();
}

double PoissonDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::floor(lambda_);
}

double PoissonDistribution::getEntropy() const {
    // Read lambda_ under a shared lock — no derived cache needed.
    double lambda;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        lambda = lambda_;
    }

    // Stirling-based asymptotic: H ≈ ½ log(2πeλ) − 1/(12λ)
    // Used for λ > 20: the approximation error is < 0.001 nats at λ=20, and the
    // exact-summation loop below fires its early-exit guard at k=0 for λ ≳ 34.5
    // (because P(0) = e^{-λ} < 1e-15 there), returning 0 incorrectly.
    if (lambda > 20.0) {
        return detail::HALF_LN_2PI + detail::HALF * std::log(lambda) + detail::HALF -
               detail::ONE / (12.0 * lambda);
    }

    // Exact: H = −Σ P(k) · log P(k),  where log P(k) = k·log(λ) − λ − log(k!)
    // Safe for λ ≤ 20: P(0) = e^{-20} ≈ 2e-9 > 1e-15, so the loop runs correctly.
    const double log_lambda = std::log(lambda);
    double H = 0.0;
    double log_k_fact = 0.0;  // log(0!) = 0
    const int K_max = static_cast<int>(lambda + 10.0 * std::sqrt(lambda) + 20.0);

    for (int k = 0; k <= K_max; ++k) {
        const double log_p = static_cast<double>(k) * log_lambda - lambda - log_k_fact;
        const double p = std::exp(log_p);
        if (p < 1e-15)
            break;
        H -= p * log_p;
        // Advance: log((k+1)!) = log(k!) + log(k+1)
        log_k_fact += std::log(static_cast<double>(k + 1));
    }
    return H;
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

VoidResult PoissonDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validatePoissonParameters(lambda_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double PoissonDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x))
        return detail::ZERO_DOUBLE;

    return getProbabilityExact(k);
}

double PoissonDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE)
        return detail::NEGATIVE_INFINITY;

    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x))
        return detail::NEGATIVE_INFINITY;

    return getLogProbabilityExact(k);
}

double PoissonDistribution::getCumulativeProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
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

    // Smallest k with CDF(k) >= p, searched outward from the normal
    // approximation: two or three CDF evaluations. The count is capped at
    // INT_MAX. The CDF reads the same λ copy as the guess, not the live member.
    const double stddev = std::sqrt(local_lambda);
    const std::int64_t k = detail::discrete_quantile_search(
        [local_lambda](std::int64_t i) {
            return i < 0 ? detail::ZERO_DOUBLE : computeCDF(static_cast<int>(i), local_lambda);
        },
        p, 0, static_cast<std::int64_t>(std::numeric_limits<int>::max()),
        detail::discrete_quantile_guess(p, local_lambda, stddev, detail::ONE / stddev));
    return static_cast<double>(k);
}

double PoissonDistribution::sample(std::mt19937& rng) const {
    double cached_lambda, cached_exp_neg_lambda;
    bool cached_is_small;
    withCacheSnapshot([&] {
        cached_lambda = lambda_;
        cached_is_small = isSmallLambda_;
        cached_exp_neg_lambda = expNegLambda_;
    });
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

    double cached_lambda, cached_exp_neg_lambda;
    bool cached_is_small;
    withCacheSnapshot([&] {
        cached_lambda = lambda_;
        cached_is_small = isSmallLambda_;
        cached_exp_neg_lambda = expNegLambda_;
    });
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

    bool is_small;
    double lambda, exp_neg_lambda;
    withCacheSnapshot([&] {
        is_small = isSmallLambda_;
        lambda = lambda_;
        exp_neg_lambda = expNegLambda_;
    });
    return is_small ? computePMFSmall(k, lambda, exp_neg_lambda) : computePMFLarge(k, lambda);
}

double PoissonDistribution::getLogProbabilityExact(int k) const noexcept {
    if (k < 0)
        return detail::NEGATIVE_INFINITY;

    double lambda;
    withCacheSnapshot([&] { lambda = lambda_; });
    return computeLogPMF(k, lambda);
}

double PoissonDistribution::getCumulativeProbabilityExact(int k) const {
    if (k < 0)
        return detail::ZERO_DOUBLE;

    double lambda;
    withCacheSnapshot([&] { lambda = lambda_; });
    return computeCDF(k, lambda);
}

bool PoissonDistribution::canUseNormalApproximation() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_ > detail::NORMAL_APPROXIMATION_THRESHOLD;  // Rule of thumb: λ > threshold for
                                                              // reasonable normal approximation
}

double PoissonDistribution::getMedian() const {
    double lambda;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        lambda = lambda_;
    }
    // For Poisson distribution, median ≈ λ + 1/3 - 0.02/λ for large λ
    // For small λ, use numerical approximation via quantile function. getQuantile takes the
    // shared lock itself: calling it under ours locked the shared_mutex recursively (undefined,
    // and a deadlock with SRWLOCK once a writer queues in between).
    if (lambda > 10.0) {
        return lambda + (1.0 / 3.0) - (0.02 / lambda);
    }
    return getQuantile(0.5);
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
            double lam, loglam, enl;
            dist.withCacheSnapshot([&] {
                lam = dist.lambda_;
                loglam = dist.logLambda_;
                enl = dist.expNegLambda_;
            });
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, lam, loglam, enl);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            double cached_lambda, cached_exp_neg_lambda;
            dist.withCacheSnapshot([&] {
                cached_lambda = dist.lambda_;
                cached_exp_neg_lambda = dist.expNegLambda_;
            });
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                        res[i] = vals[i];
                        return;
                    }
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
                        double log_result =
                            detail::poisson_log_pmf(static_cast<double>(k), cached_lambda);
                        res[i] = std::exp(log_result);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                        res[i] = vals[i];
                        continue;
                    }
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
                        double log_result =
                            detail::poisson_log_pmf(static_cast<double>(k), cached_lambda);
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

            double cached_lambda, cached_exp_neg_lambda;
            dist.withCacheSnapshot([&] {
                cached_lambda = dist.lambda_;
                cached_exp_neg_lambda = dist.expNegLambda_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                    res[i] = vals[i];
                    return;
                }
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
                    double log_result =
                        detail::poisson_log_pmf(static_cast<double>(k), cached_lambda);
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
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    dist.updateCacheUnsafe();
                // Snapshot while unique_lock is still held — eliminates TOCTOU gap.
                const double cached_lambda = dist.lambda_;
                const double cached_log_lambda = dist.logLambda_;
                dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda,
                                                      cached_log_lambda);
                return;
            }
            // Cache hit — snapshot under shared_lock.
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

            // Snapshot parameters under the appropriate lock to avoid TOCTOU.
            double cached_lambda;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_lambda = dist.lambda_;
                } else {
                    cached_lambda = dist.lambda_;
                }
            }

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                        res[i] = vals[i];
                        return;
                    }
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        return;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        return;
                    }

                    // log P(X = k) = k * log(λ) - λ - log(k!), via detail::poisson_log_pmf (#172)
                    res[i] = detail::poisson_log_pmf(static_cast<double>(k), cached_lambda);
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                        res[i] = vals[i];
                        continue;
                    }
                    if (vals[i] < detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        continue;
                    }

                    int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                    if (!PoissonDistribution::isValidCount(vals[i])) {
                        res[i] = detail::NEGATIVE_INFINITY;
                        continue;
                    }

                    // log P(X = k) = k * log(λ) - λ - log(k!), via detail::poisson_log_pmf (#172)
                    res[i] = detail::poisson_log_pmf(static_cast<double>(k), cached_lambda);
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

            // Snapshot parameters under the appropriate lock to avoid TOCTOU.
            double cached_lambda;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_lambda = dist.lambda_;
                } else {
                    cached_lambda = dist.lambda_;
                }
            }

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                    res[i] = vals[i];
                    return;
                }
                if (vals[i] < detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                    return;
                }

                int k = PoissonDistribution::roundToNonNegativeInt(vals[i]);
                if (!PoissonDistribution::isValidCount(vals[i])) {
                    res[i] = detail::NEGATIVE_INFINITY;
                    return;
                }

                // log P(X = k) = k * log(λ) - λ - log(k!), via detail::poisson_log_pmf (#172)
                res[i] = detail::poisson_log_pmf(static_cast<double>(k), cached_lambda);
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
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    dist.updateCacheUnsafe();
                // Snapshot while unique_lock is still held — eliminates TOCTOU gap.
                const double cached_lambda = dist.lambda_;
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda);
                return;
            }
            // Cache hit — snapshot under shared_lock.
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

            // Snapshot parameters under the appropriate lock to avoid TOCTOU.
            double cached_lambda;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_lambda = dist.lambda_;
                } else {
                    cached_lambda = dist.lambda_;
                }
            }

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                        res[i] = vals[i];
                        return;
                    }
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
                    if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                        res[i] = vals[i];
                        continue;
                    }
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

            // Snapshot parameters under the appropriate lock to avoid TOCTOU.
            double cached_lambda;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_lambda = dist.lambda_;
                } else {
                    cached_lambda = dist.lambda_;
                }
            }

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (std::isnan(vals[i])) {  // NaN propagates, as on the scalar path
                    res[i] = vals[i];
                    return;
                }
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
    // d == d would lock the same shared_mutex twice from one thread (undefined).
    if (this == &other)
        return true;
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

    // Extract λ value. "λ" is two bytes in UTF-8, so the tag's length is its size(), not 2 (#187).
    constexpr std::string_view kLambdaTag = "λ=";
    const size_t tag_pos = token.find(kLambdaTag);
    if (tag_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t lambda_pos = tag_pos + kLambdaTag.size();
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
                                                        [[maybe_unused]] double log_lambda,
                                                        double exp_neg_lambda) const noexcept {
    // SIMD deferred: lgamma prevents vectorization of the PMF kernel.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            results[i] = values[i];
            continue;
        }
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
            double log_result = detail::poisson_log_pmf(static_cast<double>(k), lambda);
            results[i] = std::exp(log_result);
        }
    }
}

void PoissonDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double lambda,
    [[maybe_unused]] double log_lambda) const noexcept {
    // SIMD deferred: lgamma prevents vectorization of the PMF kernel.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            results[i] = values[i];
            continue;
        }
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::NEGATIVE_INFINITY;
            continue;
        }

        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = detail::NEGATIVE_INFINITY;
            continue;
        }

        results[i] = detail::poisson_log_pmf(static_cast<double>(k), lambda);
    }
}

void PoissonDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                  double* results,
                                                                  std::size_t count,
                                                                  double lambda) const noexcept {
    // SIMD deferred: lgamma prevents vectorization of the PMF kernel.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            results[i] = values[i];
            continue;
        }
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

double PoissonDistribution::computePMFSmall(int k, double lambda, double exp_neg_lambda) noexcept {
    // Direct computation for small lambda
    if (k < static_cast<int>(FACTORIAL_CACHE.size())) {
        // Use cached factorial
        return std::pow(lambda, k) * exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
    } else {
        // Use log-space computation
        return std::exp(computeLogPMF(k, lambda));
    }
}

double PoissonDistribution::computePMFLarge(int k, double lambda) noexcept {
    // Log-space, as the batch path computes it. A normal approximation (with continuity correction)
    // used to stand in within 3σ of a very large λ: 4e-3 relative off at λ = 1e5.
    return std::exp(computeLogPMF(k, lambda));
}

double PoissonDistribution::computeLogPMF(int k, double lambda) noexcept {
    // log P(X = k) = k * log(λ) - λ - log(k!), formed without cancellation at large k or λ (#172)
    return detail::poisson_log_pmf(static_cast<double>(k), lambda);
}

double PoissonDistribution::computeCDF(int k, double lambda) noexcept {
    // Use regularized incomplete gamma function: P(X <= k) = Q(k+1, λ)
    // where Q(a,x) is the regularized upper incomplete gamma function
    return detail::gamma_q(k + 1, lambda);
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
    // EDGE-1: guard NaN and +inf before casting — static_cast<int>(NaN) is UB.
    // #167: also guard finite x above the int range (1e10, 1e300): the cast is UB
    // there too. isValidCount bounds x so round(x) fits in int; callers still test
    // isValidCount(x) themselves to pick the out-of-support result.
    if (!isValidCount(x))
        return 0;
    return static_cast<int>(std::round(x));
}

inline bool PoissonDistribution::isValidCount(double x) noexcept {
    // EDGE-2: `x <= INT_MAX` accepts 2147483647.5, which std::round sends to 2^31 and
    // overflows int. Bound at INT_MAX - 1 = 2147483646 instead: round(x) <= INT_MAX - 1, safe
    // to cast. (INT_MAX itself is exact in double; the half-up rounding is the hazard.)
    constexpr double kMaxSafeCount = static_cast<double>(std::numeric_limits<int>::max() - 1);
    return (std::isfinite(x) && x >= 0.0 && x <= kMaxSafeCount);
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

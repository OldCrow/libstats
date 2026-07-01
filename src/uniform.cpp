#include "libstats/distributions/uniform.h"

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
#include "libstats/core/statistical_constants.h"

// Platform headers - use forward declarations where available
#include "libstats/common/cpu_detection_fwd.h"  // Lightweight CPU detection
// Note: parallel_execution.h is transitively included via dispatch_utils.h
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTORS
//==============================================================================

UniformDistribution::UniformDistribution(double a, double b) : DistributionBase(), a_(a), b_(b) {
    validateParameters(a, b);
    // Cache will be updated on first use
}

UniformDistribution::UniformDistribution(const UniformDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    a_ = other.a_;
    b_ = other.b_;
    // Cache will be updated on first use
}

UniformDistribution& UniformDistribution::operator=(const UniformDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock

        // Copy parameters (don't call base class operator= to avoid deadlock)
        a_ = other.a_;
        b_ = other.b_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

UniformDistribution::UniformDistribution(UniformDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    a_ = other.a_;
    b_ = other.b_;
    other.a_ = detail::ZERO_DOUBLE;
    other.b_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

UniformDistribution& UniformDistribution::operator=(UniformDistribution&& other) noexcept {
    if (this != &other) {
        a_ = other.a_;
        b_ = other.b_;
        other.a_ = detail::ZERO_DOUBLE;
        other.b_ = detail::ONE;

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

void UniformDistribution::setLowerBound(double a) {
    // Copy current upper bound for validation (thread-safe)
    double currentB;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentB = b_;
    }

    // Validate parameters outside of any lock
    validateParameters(a, currentB);

    // Set parameter under lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // CRITICAL: Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

void UniformDistribution::setUpperBound(double b) {
    // Copy current lower bound for validation (thread-safe)
    double currentA;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentA = a_;
    }

    // Validate parameters outside of any lock
    validateParameters(currentA, b);

    // Set parameter under lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // CRITICAL: Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

void UniformDistribution::setBounds(double a, double b) {
    validateParameters(a, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // CRITICAL: Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

void UniformDistribution::setParameters(double a, double b) {
    validateParameters(a, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // CRITICAL: Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

double UniformDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        return midpoint_;  // snapshot + early return under unique_lock
    }
    return midpoint_;
}

double UniformDistribution::getVariance() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        return variance_;  // snapshot + early return under unique_lock
    }
    return variance_;
}

double UniformDistribution::getWidth() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        return width_;  // snapshot + early return under unique_lock
    }
    return width_;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult UniformDistribution::trySetLowerBound(double a) noexcept {
    double currentB;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentB = b_;
    }

    auto validation = validateUniformParameters(a, currentB);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult UniformDistribution::trySetUpperBound(double b) noexcept {
    double currentA;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentA = a_;
    }

    auto validation = validateUniformParameters(currentA, b);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult UniformDistribution::trySetParameters(double a, double b) noexcept {
    auto validation = validateUniformParameters(a, b);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double UniformDistribution::getProbability(double x) const {
    // Ensure cache is valid once before using - using the same pattern as other methods
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        // Snapshot while unique_lock is still held.
        const double lo = a_, hi = b_;
        const bool is_unit = isUnitInterval_;
        const double inv_w = invWidth_;
        if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
        if (x < lo || x > hi) return detail::ZERO_DOUBLE;
        if (is_unit) return detail::ONE;
        return inv_w;
    }

    // NaN must propagate before the bounds checks (both comparisons are false for NaN,
    // so NaN would silently pass through and return the density constant).
    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();

    // Check if x is within the support [a, b]
    if (x < a_ || x > b_) {
        return detail::ZERO_DOUBLE;
    }

    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return detail::ONE;
    }

    // General case: PDF = 1/(b-a) for x in [a,b]
    return invWidth_;
}

double UniformDistribution::getLogProbability(double x) const {
    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        // Snapshot while unique_lock is still held.
        const double lo = a_, hi = b_;
        const bool is_unit = isUnitInterval_;
        const double w = width_;
        if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();
        if (x < lo || x > hi) return detail::NEGATIVE_INFINITY;
        if (is_unit) return detail::ZERO_DOUBLE;
        return -std::log(w);
    }

    if (std::isnan(x)) return std::numeric_limits<double>::quiet_NaN();

    // Check if x is within the support [a, b]
    if (x < a_ || x > b_) {
        return detail::NEGATIVE_INFINITY;
    }

    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return detail::ZERO_DOUBLE;  // log(1) = 0
    }

    // General case: log(PDF) = log(1/(b-a)) = -log(b-a)
    return -std::log(width_);
}

double UniformDistribution::getCumulativeProbability(double x) const {
    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        // Snapshot while unique_lock is still held.
        const double lo = a_, hi = b_;
        const bool is_unit = isUnitInterval_;
        const double inv_w = invWidth_;
        if (x < lo) return detail::ZERO_DOUBLE;
        if (x > hi) return detail::ONE;
        if (is_unit) return x;
        return (x - lo) * inv_w;
    }

    // CDF for uniform distribution
    if (x < a_) {
        return detail::ZERO_DOUBLE;
    }
    if (x > b_) {
        return detail::ONE;
    }

    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return x;  // CDF(x) = x for U(0,1)
    }

    // General case: CDF(x) = (x-a)/(b-a)
    return (x - a_) * invWidth_;
}

double UniformDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE) {
        return a_;
    }
    if (p == detail::ONE) {
        return b_;
    }

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        // Snapshot while unique_lock is still held.
        const bool is_unit = isUnitInterval_;
        const double lo = a_, w = width_;
        if (is_unit) return p;
        return lo + p * w;
    }

    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return p;  // Quantile(p) = p for U(0,1)
    }

    // General case: Quantile(p) = a + p*(b-a)
    return a_ + p * width_;
}

double UniformDistribution::sample(std::mt19937& rng) const {
    // Snapshot parameters under the appropriate lock to avoid TOCTOU.
    bool cached_is_unit_interval;
    double cached_a, cached_width;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
            if (!cache_valid_) updateCacheUnsafe();
            cached_is_unit_interval = isUnitInterval_;
            cached_a = a_;
            cached_width = width_;
        } else {
            cached_is_unit_interval = isUnitInterval_;
            cached_a = a_;
            cached_width = width_;
        }
    }

    // Use high-quality uniform distribution
    std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);

    double u = uniform(rng);

    // Fast path for unit interval [0,1]
    if (cached_is_unit_interval) {
        return u;
    }

    // General case: linear transformation X = a + (b-a)*U
    return cached_a + cached_width * u;
}

std::vector<double> UniformDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    std::uniform_real_distribution<double> dist(detail::ZERO_DOUBLE, detail::ONE);

    // Snapshot cached fields; no re-acquire = no TOCTOU gap.
    double cached_a, cached_width;
    bool cached_is_unit_interval;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
            if (!cache_valid_) updateCacheUnsafe();
            cached_a = a_;
            cached_width = width_;
            cached_is_unit_interval = isUnitInterval_;
        } else {
            cached_a = a_;
            cached_width = width_;
            cached_is_unit_interval = isUnitInterval_;
        }
    }

    // Generate batch samples using linear transformation
    for (size_t i = 0; i < n; ++i) {
        double u = dist(rng);
        if (cached_is_unit_interval) {
            samples.push_back(u);
        } else {
            samples.push_back(cached_a + u * cached_width);
        }
    }

    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void UniformDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    // Reject non-finite values before calling minmax_element (UB with NaN)
    for (double v : values) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("Uniform fit requires all values to be finite");
        }
    }

    // For uniform distribution, use sample minimum and maximum
    const auto minmax = std::minmax_element(values.begin(), values.end());
    const double sample_min = *minmax.first;
    const double sample_max = *minmax.second;

    // Check for degenerate case
    if (sample_min >= sample_max) {
        throw std::invalid_argument("All values are identical - cannot fit uniform distribution");
    }

    // Add small margin to ensure all sample points are within bounds
    const double margin = (sample_max - sample_min) * detail::DEFAULT_TOLERANCE;
    const double fitted_a = sample_min - margin;
    const double fitted_b = sample_max + margin;

    // Set parameters (this will validate and invalidate cache)
    setBounds(fitted_a, fitted_b);
}

void UniformDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                           std::vector<UniformDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void UniformDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = detail::ZERO_DOUBLE;
    b_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string UniformDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "UniformDistribution(a=" << a_ << ", b=" << b_ << ")";
    return oss.str();
}

//==============================================================================
// 7. ADVANCED STATISTICAL METHODS
//==============================================================================

std::tuple<double, double, bool> UniformDistribution::uniformityTest(
    const std::vector<double>& data, double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (significance_level <= detail::ZERO_DOUBLE || significance_level >= detail::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    const size_t n = data.size();
    const double sample_min = *std::min_element(data.begin(), data.end());
    const double sample_max = *std::max_element(data.begin(), data.end());
    const double sample_range = sample_max - sample_min;

    if (sample_range == detail::ZERO_DOUBLE) {
        // All data points are identical - not uniform
        return {std::numeric_limits<double>::infinity(), detail::ZERO_DOUBLE, false};
    }

    // Use range/variance ratio test
    // For uniform distribution: Var = Range²/12
    // Test statistic: T = 12 * Var / Range²
    // Should be close to 1 for uniform data

    double sample_variance = detail::ZERO_DOUBLE;
    double sample_mean = detail::ZERO_DOUBLE;
    for (double x : data) {
        sample_mean += x;
    }
    sample_mean /= static_cast<double>(n);

    for (double x : data) {
        sample_variance += (x - sample_mean) * (x - sample_mean);
    }
    sample_variance /= static_cast<double>(n - 1);

    const double expected_variance = sample_range * sample_range / 12.0;
    const double test_statistic = sample_variance / expected_variance;

    // For large n, this approximately follows a known distribution
    // Simplified p-value calculation
    const double p_value =
        detail::TWO * std::min(test_statistic, detail::TWO - test_statistic);  // Symmetric around 1

    const bool uniformity_is_valid = (p_value > significance_level);

    return {test_statistic, p_value, uniformity_is_valid};
}

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
//==========================================================================

//==========================================================================
// 9. CROSS-VALIDATION METHODS
//==========================================================================

//==========================================================================
// 10. INFORMATION CRITERIA
//==========================================================================

//==========================================================================
// 11. BOOTSTRAP METHODS
//==========================================================================

//==========================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==========================================================================

// Inline utility methods moved from header for PIMPL optimization
// These retain 'inline' hint to allow compiler optimization while reducing header bloat

double UniformDistribution::getRange() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return b_ - a_;  // Direct subtraction is most efficient
}

bool UniformDistribution::contains(double x) const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return x >= a_ && x <= b_;
}

double UniformDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::log(b_ - a_);  // ln(range)
}

bool UniformDistribution::isUnitInterval() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (std::abs(a_ - detail::ZERO_DOUBLE) <= detail::DEFAULT_TOLERANCE) &&
           (std::abs(b_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
}

bool UniformDistribution::isSymmetricAroundZero() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::abs(a_ + b_) <= detail::DEFAULT_TOLERANCE;
}

double UniformDistribution::getMedian() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (a_ + b_) / 2.0;
}

double UniformDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (a_ + b_) / 2.0;
}

double UniformDistribution::getMidpoint() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (a_ + b_) * detail::HALF;  // Multiplication is faster than division
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS (New Simplified API)
//==============================================================================

void UniformDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                         const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const UniformDistribution& dist, double value) { return dist.getProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            // Use the unsafe implementation directly since batch methods were removed
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    dist.updateCacheUnsafe();
                }
                // Snapshot under unique_lock — eliminates TOCTOU gap.
                const double cached_a = dist.a_;
                const double cached_b = dist.b_;
                const double cached_inv_width = dist.invWidth_;
                dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                   cached_inv_width);
                return;
            }
            // Cache hit — snapshot under shared_lock.
            const double cached_a = dist.a_;
            const double cached_b = dist.b_;
            const double cached_inv_width = dist.invWidth_;
            lock.unlock();
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                               cached_inv_width);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cached fields; no re-acquire = no TOCTOU gap.
            double cached_a, cached_b, cached_inv_width;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_) {
                        dist.updateCacheUnsafe();
                    }
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                } else {
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                }
            }

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] =
                        (x >= cached_a && x <= cached_b) ? cached_inv_width : detail::ZERO_DOUBLE;
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] =
                        (x >= cached_a && x <= cached_b) ? cached_inv_width : detail::ZERO_DOUBLE;
                }
            }
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cached fields; no re-acquire = no TOCTOU gap.
            double cached_a, cached_b, cached_inv_width;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_) {
                        dist.updateCacheUnsafe();
                    }
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                } else {
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                }
            }

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x >= cached_a && x <= cached_b) ? cached_inv_width : detail::ZERO_DOUBLE;
            });
        });
}

void UniformDistribution::getLogProbability(std::span<const double> values,
                                            std::span<double> results,
                                            const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const UniformDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            // Use the unsafe implementation directly since batch methods were removed
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    dist.updateCacheUnsafe();
                }
                // Snapshot under unique_lock — eliminates TOCTOU gap.
                const double cached_a = dist.a_;
                const double cached_b = dist.b_;
                const double cached_log_inv_width = -std::log(dist.width_);
                dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                      cached_log_inv_width);
                return;
            }
            // Cache hit — snapshot under shared_lock.
            const double cached_a = dist.a_;
            const double cached_b = dist.b_;
            const double cached_log_inv_width = -std::log(dist.width_);
            lock.unlock();
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                  cached_log_inv_width);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cached fields; no re-acquire = no TOCTOU gap.
            double cached_a, cached_b, cached_log_inv_width;
            bool cached_is_unit_interval;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_) {
                        dist.updateCacheUnsafe();
                    }
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_log_inv_width = -std::log(dist.width_);
                    cached_is_unit_interval = dist.isUnitInterval_;
                } else {
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_log_inv_width = -std::log(dist.width_);
                    cached_is_unit_interval = dist.isUnitInterval_;
                }
            }

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < cached_a || x > cached_b) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_interval) {
                        res[i] = detail::ZERO_DOUBLE;  // log(1) = 0
                    } else {
                        res[i] = cached_log_inv_width;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < cached_a || x > cached_b) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_interval) {
                        res[i] = detail::ZERO_DOUBLE;  // log(1) = 0
                    } else {
                        res[i] = cached_log_inv_width;
                    }
                }
            }
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cached fields; no re-acquire = no TOCTOU gap.
            double cached_a, cached_b, cached_log_inv_width;
            bool cached_is_unit_interval;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_) {
                        dist.updateCacheUnsafe();
                    }
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_log_inv_width = -std::log(dist.width_);
                    cached_is_unit_interval = dist.isUnitInterval_;
                } else {
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_log_inv_width = -std::log(dist.width_);
                    cached_is_unit_interval = dist.isUnitInterval_;
                }
            }

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < cached_a || x > cached_b) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else if (cached_is_unit_interval) {
                    res[i] = detail::ZERO_DOUBLE;  // log(1) = 0
                } else {
                    res[i] = cached_log_inv_width;
                }
            });
        });
}

void UniformDistribution::getCumulativeProbability(std::span<const double> values,
                                                   std::span<double> results,
                                                   const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const UniformDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            // Use the unsafe implementation directly since batch methods were removed
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    dist.updateCacheUnsafe();
                }
                // Snapshot under unique_lock — eliminates TOCTOU gap.
                const double cached_a = dist.a_;
                const double cached_b = dist.b_;
                const double cached_inv_width = dist.invWidth_;
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                             cached_inv_width);
                return;
            }
            // Cache hit — snapshot under shared_lock.
            const double cached_a = dist.a_;
            const double cached_b = dist.b_;
            const double cached_inv_width = dist.invWidth_;
            lock.unlock();
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                         cached_inv_width);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cached fields; no re-acquire = no TOCTOU gap.
            double cached_a, cached_b, cached_inv_width;
            bool cached_is_unit_interval;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_) {
                        dist.updateCacheUnsafe();
                    }
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                    cached_is_unit_interval = dist.isUnitInterval_;
                } else {
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                    cached_is_unit_interval = dist.isUnitInterval_;
                }
            }

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < cached_a) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (x > cached_b) {
                        res[i] = detail::ONE;
                    } else if (cached_is_unit_interval) {
                        res[i] = x;  // CDF(x) = x for U(0,1)
                    } else {
                        res[i] = (x - cached_a) * cached_inv_width;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < cached_a) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (x > cached_b) {
                        res[i] = detail::ONE;
                    } else if (cached_is_unit_interval) {
                        res[i] = x;  // CDF(x) = x for U(0,1)
                    } else {
                        res[i] = (x - cached_a) * cached_inv_width;
                    }
                }
            }
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cached fields; no re-acquire = no TOCTOU gap.
            double cached_a, cached_b, cached_inv_width;
            bool cached_is_unit_interval;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_) {
                        dist.updateCacheUnsafe();
                    }
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                    cached_is_unit_interval = dist.isUnitInterval_;
                } else {
                    cached_a = dist.a_;
                    cached_b = dist.b_;
                    cached_inv_width = dist.invWidth_;
                    cached_is_unit_interval = dist.isUnitInterval_;
                }
            }

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < cached_a) {
                    res[i] = detail::ZERO_DOUBLE;
                } else if (x > cached_b) {
                    res[i] = detail::ONE;
                } else if (cached_is_unit_interval) {
                    res[i] = x;  // CDF(x) = x for U(0,1)
                } else {
                    res[i] = (x - cached_a) * cached_inv_width;
                }
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool UniformDistribution::operator==(const UniformDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    return std::abs(a_ - other.a_) <= detail::DEFAULT_TOLERANCE &&
           std::abs(b_ - other.b_) <= detail::DEFAULT_TOLERANCE;
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const UniformDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, UniformDistribution& distribution) {
    std::string token;
    double a, b;

    // Expected format: "UniformDistribution(a=<value>, b=<value>)"
    // We'll parse this step by step

    // Skip whitespace and read the first part
    is >> token;
    if (!token.starts_with("UniformDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract 'a' value
    if (token.find("a=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t a_pos = token.find("a=") + 2;
    size_t comma_pos = token.find(",", a_pos);
    if (comma_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string a_str = token.substr(a_pos, comma_pos - a_pos);
        a = std::stod(a_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract 'b' value
    if (token.find("b=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t b_pos = token.find("b=") + 2;
    size_t close_paren = token.find(")", b_pos);
    if (close_paren == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string b_str = token.substr(b_pos, close_paren - b_pos);
        b = std::stod(b_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Validate and set parameters using the safe API
    auto result = distribution.trySetParameters(a, b);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==========================================================================
// 17. PRIVATE FACTORY METHODS
//==========================================================================

// Note: All methods in this section currently implemented inline in the header
// This section maintained for template compliance

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION USING VECTOROPS
//
// PDF and LogPDF batch kernels remain scalar: the computation is a constant
// fill guarded by a bounds check, and there is no arithmetic to vectorize
// without a blend/select primitive in VectorOps.
//
// CDF is partially vectorized. The interior is a linear function
// (x - a) * inv_width that maps cleanly to scalar_add + scalar_multiply.
// A scalar fixup pass then clamps values outside [a, b]. The unit-interval
// special case collapses to the same formula (a=0, inv_width=1), so the
// two-branch structure in the old SIMD path has been removed.
//==============================================================================

void UniformDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                        std::size_t count, double a, double b,
                                                        double inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: For uniform distribution, computation is extremely simple (just bounds checking)
        // so SIMD rarely provides benefits, but we use centralized policy for consistency
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            // Use exclusion check (x < a || x > b) for CPU efficiency:
            // - Short-circuits on first true condition (common for out-of-support values)
            // - Matches scalar implementation exactly for consistency
            results[i] = (x < a || x > b) ? detail::ZERO_DOUBLE : inv_width;
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation if possible
    // Note: For uniform distribution, vectorization typically doesn't provide significant benefits
    // due to the simple nature of bounds checking, but we implement for consistency
    // In practice, this will mostly fall back to scalar due to the nature of the operation

    // Use scalar implementation even when SIMD is available because uniform distribution
    // operations are not amenable to vectorization (primarily branching logic)
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        // Use exclusion check (x < a || x > b) for CPU efficiency:
        // - Short-circuits on first true condition (common for out-of-support values)
        // - Matches scalar implementation exactly for consistency
        results[i] = (x < a || x > b) ? detail::ZERO_DOUBLE : inv_width;
    }
}

void UniformDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                           std::size_t count, double a, double b,
                                                           double log_inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: For uniform distribution, computation is extremely simple (just bounds checking)
        // so SIMD rarely provides benefits, but we use centralized policy for consistency
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            // Use exclusion check (x < a || x > b) for consistency with scalar and PDF SIMD:
            // - Matches boundary conditions exactly
            // - Short-circuits efficiently for out-of-support values
            results[i] = (x < a || x > b) ? detail::NEGATIVE_INFINITY : log_inv_width;
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation if possible
    // Note: For uniform distribution, vectorization typically doesn't provide significant benefits
    // due to the simple nature of bounds checking, but we implement for consistency
    // In practice, this will mostly fall back to scalar due to the nature of the operation

    // Use scalar implementation even when SIMD is available because uniform distribution
    // operations are not amenable to vectorization (primarily branching logic)
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        // Use exclusion check (x < a || x > b) for consistency with scalar and PDF SIMD:
        // - Matches boundary conditions exactly
        // - Short-circuits efficiently for out-of-support values
        results[i] = (x < a || x > b) ? detail::NEGATIVE_INFINITY : log_inv_width;
    }
}

void UniformDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                  double* results,
                                                                  std::size_t count, double a,
                                                                  double b,
                                                                  double inv_width) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Scalar path for small arrays.
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < a) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (x > b) {
                results[i] = detail::ONE;
            } else {
                results[i] = (x - a) * inv_width;
            }
        }
        return;
    }

    // Vectorize the linear interior (x-a)*inv_width across the full batch,
    // then a scalar pass clamps the boundary values.
    // The unit-interval case (a=0, inv_width=1) is handled by the same formula.
    // Step 1: results = values - a
    arch::simd::VectorOps::scalar_add(values, -a, results, count);
    // Step 2: results = (values - a) * inv_width
    arch::simd::VectorOps::scalar_multiply(results, inv_width, results, count);
    // Clamp fixup: boundaries are 0 below a and 1 above b.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < a) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (values[i] > b) {
            results[i] = detail::ONE;
        }
    }
}

//==========================================================================
// 19. PRIVATE COMPUTATIONAL METHODS (if needed)
//==========================================================================

// Note: All methods in this section currently implemented inline in the header
// This section maintained for template compliance

//==========================================================================
// 20. PRIVATE UTILITY METHODS (if needed)
//==========================================================================

// For Uniform distribution, internal helper methods are minimal.
// Additional data processing utilities, validation helpers, or
// formatting utilities would be placed here if needed in future versions.

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

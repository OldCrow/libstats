#include "libstats/distributions/gaussian.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/common/cpu_detection_fwd.h"       // CPU feature queries (lightweight)
#include "libstats/common/platform_constants_fwd.h"  // Parallel thresholds (lightweight)
#include "libstats/common/simd_policy_fwd.h"         // SIMD policy decisions (lightweight)
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/parallel_batch_fit.h"
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h
#include "libstats/core/debug_flags.h"  // for kDebugOptimizations
#include "libstats/core/safety.h"
// Note: simd.h still included in implementation files that actually use SIMD operations

#include <algorithm>
#include <cmath>
#include <numeric>
#include <ranges>
#include <sstream>  // for toString()
#include <span>
#include <vector>

namespace stats {

//==============================================================================
// COMPLEX METHODS (Implementation in .cpp per C++20 best practices)
//==============================================================================

// Note: Simple statistical moments (getMean, getVariance, getSkewness, getKurtosis)
// are implemented inline in the header for optimal performance since they are
// trivial calculations or constants for the Gaussian distribution. Methods involving
// complex logic or thread safety lock ordering are implemented in the .cpp file

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTORS
//==============================================================================

GaussianDistribution::GaussianDistribution(double mean, double standardDeviation)
    : DistributionBase(), mean_(mean), standardDeviation_(standardDeviation) {
    validateParameters(mean, standardDeviation);
}

GaussianDistribution::GaussianDistribution(const GaussianDistribution& other)
    : DistributionBase() {  // Default-construct base; we copy all state under the lock below.
    // Acquire the source lock BEFORE reading any field from other, including
    // cache state. The previous DistributionBase(other) copy ran without the
    // lock, leaving a window where a concurrent fit() or invalidateCache() on
    // other could race with the base-class field reads.
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
    // Propagate cache validity so the copy does not unnecessarily recompute.
    cache_valid_ = other.cache_valid_;
    cacheValidAtomic_.store(other.cacheValidAtomic_.load(std::memory_order_acquire),
                            std::memory_order_release);
}

GaussianDistribution& GaussianDistribution::operator=(const GaussianDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);

        // Copy parameters (don't call base class operator= to avoid deadlock)
        mean_ = other.mean_;
        standardDeviation_ = other.standardDeviation_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

GaussianDistribution::GaussianDistribution(GaussianDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
    other.mean_ = detail::ZERO_DOUBLE;
    other.standardDeviation_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
}

GaussianDistribution& GaussianDistribution::operator=(GaussianDistribution&& other) noexcept {
    if (this != &other) {
        mean_ = other.mean_;
        standardDeviation_ = other.standardDeviation_;
        other.mean_ = detail::ZERO_DOUBLE;
        other.standardDeviation_ = detail::ONE;

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

void GaussianDistribution::setMean(double mean) {
    // Acquire unique lock before reading standardDeviation_ for validation so
    // the read and write are in the same critical section (NEW-TS-3).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(mean, standardDeviation_);
    mean_ = mean;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GaussianDistribution::setStandardDeviation(double stdDev) {
    // Acquire unique lock before reading mean_ for validation (NEW-TS-3).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(mean_, stdDev);
    standardDeviation_ = stdDev;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

// Parameter setters with validation (for existing exception-based API)
void GaussianDistribution::setParameters(double mean, double standardDeviation) {
    validateParameters(mean, standardDeviation);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = mean;
    standardDeviation_ = standardDeviation;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==============================================================================
// 4. RESULT-BASED SETTERS (C++20 Best Practice: Complex implementations in .cpp)
//==============================================================================

VoidResult GaussianDistribution::trySetMean(double mean) noexcept {
    // Acquire unique lock before reading standardDeviation_ for validation (NEW-TS-3).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto validation = validateGaussianParameters(mean, standardDeviation_);
    if (validation.isError()) {
        return validation;
    }
    mean_ = mean;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    return VoidResult::ok({});
}

VoidResult GaussianDistribution::trySetStandardDeviation(double stdDev) noexcept {
    // Acquire unique lock before reading mean_ for validation (NEW-TS-3).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto validation = validateGaussianParameters(mean_, stdDev);
    if (validation.isError()) {
        return validation;
    }
    standardDeviation_ = stdDev;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    return VoidResult::ok({});
}

VoidResult GaussianDistribution::trySetParameters(double mean, double standardDeviation) noexcept {
    auto validation = validateGaussianParameters(mean, standardDeviation);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = mean;
    standardDeviation_ = standardDeviation;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult GaussianDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateGaussianParameters(mean_, standardDeviation_);
}

// Simple getters for constant values - moved from header to reduce compilation load
double GaussianDistribution::getSkewness() const {
    return detail::ZERO_DOUBLE;  // Gaussian distribution is symmetric
}

double GaussianDistribution::getKurtosis() const {
    return detail::ZERO_DOUBLE;  // Gaussian has zero excess kurtosis
}

int GaussianDistribution::getNumParameters() const noexcept {
    return 2;  // Mean and standard deviation
}

bool GaussianDistribution::isDiscrete() const noexcept {
    return false;  // Gaussian is continuous
}

double GaussianDistribution::getSupportLowerBound() const noexcept {
    return -std::numeric_limits<double>::infinity();
}

double GaussianDistribution::getSupportUpperBound() const noexcept {
    return std::numeric_limits<double>::infinity();
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//
// SAFETY DOCUMENTATION FOR DEVELOPERS:
//
// This section contains core probability methods that compute PDF, CDF, etc,
// and they ensure cache validity by using shared locks and are thread-safe.
//
// Key Methods:
// - getProbability()
// - getLogProbability()
// - getCumulativeProbability()
//==============================================================================

double GaussianDistribution::getProbability(double x) const {
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

    // Fast path for standard normal
    if (isStandardNormal_) {
        const double sq_diff = x * x;
        return detail::INV_SQRT_2PI * std::exp(detail::NEG_HALF * sq_diff);
    }

    // General case
    const double diff = x - mean_;
    const double sq_diff = diff * diff;
    return normalizationConstant_ * std::exp(negHalfSigmaSquaredInv_ * sq_diff);
}

double GaussianDistribution::getLogProbability(double x) const {
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

    // Fast path for standard normal - direct computation (no log-sum-exp needed here)
    if (isStandardNormal_) {
        const double sq_diff = x * x;
        return detail::NEG_HALF_LN_2PI + detail::NEG_HALF * sq_diff;
    }

    // General case - direct computation for Gaussian log-PDF
    const double diff = x - mean_;
    const double sq_diff = diff * diff;
    return detail::NEG_HALF_LN_2PI - logStandardDeviation_ + negHalfSigmaSquaredInv_ * sq_diff;
}

double GaussianDistribution::getCumulativeProbability(double x) const {
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

    // Fast path for standard normal
    if (isStandardNormal_) {
        return detail::HALF * (detail::ONE + std::erf(x * detail::INV_SQRT_2));
    }

    // General case
    const double normalized = (x - mean_) / sigmaSqrt2_;
    return detail::HALF * (detail::ONE + std::erf(normalized));
}

double GaussianDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE)
        return -std::numeric_limits<double>::infinity();
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);

    if (p == detail::HALF) {
        return mean_;  // Median equals mean for normal distribution
    }

    // Use inverse error function for standard normal quantile
    // For standard normal: quantile = sqrt(2) * erfinv(2p - 1)
    // For general normal: quantile = mean + sigma * sqrt(2) * erfinv(2p - 1)

    const double erf_input = detail::TWO * p - detail::ONE;
    double z = detail::erf_inv(erf_input);
    return mean_ + standardDeviation_ * detail::SQRT_2 * z;
}

double GaussianDistribution::sample(std::mt19937& rng) const {
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

    // Optimized Box-Muller transform with enhanced numerical stability
    static thread_local bool has_spare = false;
    static thread_local double spare;

    if (has_spare) {
        has_spare = false;
        return mean_ + standardDeviation_ * spare;
    }

    has_spare = true;

    // Use high-quality uniform distribution
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);

    double u1, u2;
    double magnitude, angle;

    do {
        u1 = uniform(rng);
        u2 = uniform(rng);

        // Box-Muller transformation
        magnitude = std::sqrt(detail::NEG_TWO * std::log(u1));
        angle = detail::TWO_PI * u2;

        // Check for numerical validity
        if (std::isfinite(magnitude) && std::isfinite(angle)) {
            break;
        }
    } while (true);

    spare = magnitude * std::sin(angle);
    double z = magnitude * std::cos(angle);

    return mean_ + standardDeviation_ * z;
}

std::vector<double> GaussianDistribution::sample(std::mt19937& rng, size_t n) const {
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

    // Cache parameters for batch generation
    const double cached_mu = mean_;
    const double cached_sigma = standardDeviation_;
    const bool cached_is_standard = isStandardNormal_;

    lock.unlock();  // Release lock before generation

    std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);

    // Efficient batch Box-Muller: generate samples in pairs
    const size_t pairs = n / 2;
    const bool has_odd = (n % 2) == 1;

    for (size_t i = 0; i < pairs; ++i) {
        // Generate two independent uniform random variables
        double u1 = uniform(rng);
        double u2 = uniform(rng);

        // Ensure u1 is not zero to avoid log(0)
        while (u1 <= std::numeric_limits<double>::min()) {
            u1 = uniform(rng);
        }

        // Box-Muller transformation
        const double magnitude = std::sqrt(detail::NEG_TWO * std::log(u1));
        const double angle = detail::TWO_PI * u2;

        const double z1 = magnitude * std::cos(angle);
        const double z2 = magnitude * std::sin(angle);

        // Transform to desired distribution parameters
        if (cached_is_standard) {
            samples.push_back(z1);
            samples.push_back(z2);
        } else {
            samples.push_back(cached_mu + cached_sigma * z1);
            samples.push_back(cached_mu + cached_sigma * z2);
        }
    }

    // Handle odd number of samples - generate one more using single Box-Muller
    if (has_odd) {
        double u1 = uniform(rng);
        double u2 = uniform(rng);

        while (u1 <= std::numeric_limits<double>::min()) {
            u1 = uniform(rng);
        }

        const double magnitude = std::sqrt(detail::NEG_TWO * std::log(u1));
        const double angle = detail::TWO_PI * u2;
        const double z = magnitude * std::cos(angle);

        if (cached_is_standard) {
            samples.push_back(z);
        } else {
            samples.push_back(cached_mu + cached_sigma * z);
        }
    }

    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void GaussianDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit to empty data");
    }

    // Check minimum data points for reliable fitting
    if (values.size() < detail::MIN_DATA_POINTS_FOR_FITTING) {
        throw std::invalid_argument("Insufficient data points for reliable Gaussian fitting");
    }

    const std::size_t n = values.size();

    // Use parallel execution for large datasets
    double running_mean, sample_variance;

    if (arch::should_use_distribution_parallel(n)) {
        // Parallel Welford's algorithm using chunked computation
        const std::size_t grain_size = arch::get_adaptive_grain_size(2, n);  // Mixed operations
        const std::size_t num_chunks = (n + grain_size - 1) / grain_size;

        // Storage for partial results from each chunk
        std::vector<double> chunk_means(num_chunks);
        std::vector<double> chunk_m2s(num_chunks);
        std::vector<std::size_t> chunk_counts(num_chunks);

        // Phase 1: Compute partial statistics in parallel chunks
        // Create indices for parallel processing
        std::vector<std::size_t> chunk_indices(num_chunks);
        std::iota(chunk_indices.begin(), chunk_indices.end(), 0);

        arch::safe_for_each(chunk_indices.begin(), chunk_indices.end(), [&](std::size_t chunk_idx) {
            const std::size_t start_idx = chunk_idx * grain_size;
            const std::size_t end_idx = std::min(start_idx + grain_size, n);
            const std::size_t chunk_size = end_idx - start_idx;

            double chunk_mean = detail::ZERO_DOUBLE;
            double chunk_m2 = detail::ZERO_DOUBLE;

            // Welford's algorithm on chunk - C++20 safe iteration
            auto chunk_range = values | std::views::drop(start_idx) | std::views::take(chunk_size);
            std::size_t local_count = 0;
            for (const double value : chunk_range) {
                ++local_count;
                const double delta = value - chunk_mean;
                const double count_inv = detail::ONE / static_cast<double>(local_count);
                chunk_mean += delta * count_inv;
                const double delta2 = value - chunk_mean;
                chunk_m2 += delta * delta2;
            }

            chunk_means[chunk_idx] = chunk_mean;
            chunk_m2s[chunk_idx] = chunk_m2;
            chunk_counts[chunk_idx] = chunk_size;
        });

        // Phase 2: Combine partial results using Chan's parallel algorithm
        running_mean = detail::ZERO_DOUBLE;
        double combined_m2 = detail::ZERO_DOUBLE;
        std::size_t combined_count = 0;

        for (std::size_t i = 0; i < num_chunks; ++i) {
            if (chunk_counts[i] > 0) {
                const double delta = chunk_means[i] - running_mean;
                const std::size_t new_count = combined_count + chunk_counts[i];

                running_mean +=
                    delta * static_cast<double>(chunk_counts[i]) / static_cast<double>(new_count);

                const double delta2 = chunk_means[i] - running_mean;
                combined_m2 += chunk_m2s[i] + delta * delta2 * static_cast<double>(combined_count) *
                                                  static_cast<double>(chunk_counts[i]) /
                                                  static_cast<double>(new_count);

                combined_count = new_count;
            }
        }

        sample_variance = combined_m2 / static_cast<double>(n - 1);

    } else {
        // Serial Welford's algorithm for smaller datasets - C++20 safe iteration
        running_mean = detail::ZERO_DOUBLE;
        double running_m2 = detail::ZERO_DOUBLE;

        std::size_t count = 0;
        for (const double value : values) {
            ++count;
            const double delta = value - running_mean;
            running_mean += delta / static_cast<double>(count);
            const double delta2 = value - running_mean;
            running_m2 += delta * delta2;
        }

        sample_variance = running_m2 / static_cast<double>(n - 1);
    }

    const double sample_std = std::sqrt(sample_variance);

    // Validate computed statistics
    if (sample_std <= detail::HIGH_PRECISION_TOLERANCE) {
        throw std::invalid_argument("Data has zero or near-zero variance - cannot fit Gaussian");
    }

    // Set parameters (this will validate and invalidate cache)
    setParameters(running_mean, sample_std);
}

void GaussianDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<GaussianDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void GaussianDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = detail::ZERO_DOUBLE;
    standardDeviation_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string GaussianDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "GaussianDistribution(mean=" << mean_ << ", stddev=" << standardDeviation_ << ")";
    return oss.str();
}

//==============================================================================
// 7. ADVANCED STATISTICAL METHODS
//==============================================================================

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
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

bool GaussianDistribution::isUsingStandardNormalOptimization() const {
    // No-op in release mode; only introspects when kDebugOptimizations is true.
    if constexpr (!stats::kDebugOptimizations) {
        return false;
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
    return isStandardNormal_;
}

// Utility methods moved from header for PIMPL optimization - no longer inline

double GaussianDistribution::getStandardizedValue(double x) const noexcept {
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
    return (x - mean_) * invStandardDeviation_;  // Use cached 1/σ for efficiency
}

double GaussianDistribution::getValueFromStandardized(double z) const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mean_ + standardDeviation_ * z;
}

bool GaussianDistribution::isStandardNormal() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (std::abs(mean_) <= detail::DEFAULT_TOLERANCE) &&
           (std::abs(standardDeviation_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
}

double GaussianDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // H(X) = 0.5 * (ln(2π) + 1 + 2*ln(σ))
    return detail::HALF_LN_2PI + detail::HALF + std::log(standardDeviation_);
}

double GaussianDistribution::getMedian() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mean_;  // For Gaussian, median = mean
}

double GaussianDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mean_;  // For Gaussian, mode = mean
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS (New Simplified API)
//==============================================================================

void GaussianDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const GaussianDistribution& dist, double value) { return dist.getProbability(value); },
        [](const GaussianDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_mean = dist.mean_;
            const double cached_norm_constant = dist.normalizationConstant_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_mean, cached_norm_constant,
                                               cached_neg_half_inv_var, cached_is_standard_normal);
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_mean = dist.mean_;
            const double cached_norm_constant = dist.normalizationConstant_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (cached_is_standard_normal) {
                        const double sq_diff = vals[i] * vals[i];
                        res[i] = detail::INV_SQRT_2PI * std::exp(detail::NEG_HALF * sq_diff);
                    } else {
                        const double diff = vals[i] - cached_mean;
                        const double sq_diff = diff * diff;
                        res[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (cached_is_standard_normal) {
                        const double sq_diff = vals[i] * vals[i];
                        res[i] = detail::INV_SQRT_2PI * std::exp(detail::NEG_HALF * sq_diff);
                    } else {
                        const double diff = vals[i] - cached_mean;
                        const double sq_diff = diff * diff;
                        res[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
                    }
                }
            }
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
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
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_mean = dist.mean_;
            const double cached_norm_constant = dist.normalizationConstant_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (cached_is_standard_normal) {
                    const double sq_diff = vals[i] * vals[i];
                    res[i] = detail::INV_SQRT_2PI * std::exp(detail::NEG_HALF * sq_diff);
                } else {
                    const double diff = vals[i] - cached_mean;
                    const double sq_diff = diff * diff;
                    res[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
                }
            });
        });
}

void GaussianDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const GaussianDistribution& dist, double value) {
            return dist.getLogProbability(value);
        },
        [](const GaussianDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_mean = dist.mean_;
            const double cached_log_std = dist.logStandardDeviation_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_mean, cached_log_std,
                                                  cached_neg_half_inv_var,
                                                  cached_is_standard_normal);
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_mean = dist.mean_;
            const double cached_log_std = dist.logStandardDeviation_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (cached_is_standard_normal) {
                        const double sq_diff = vals[i] * vals[i];
                        res[i] = detail::NEG_HALF_LN_2PI + detail::NEG_HALF * sq_diff;
                    } else {
                        const double diff = vals[i] - cached_mean;
                        const double sq_diff = diff * diff;
                        res[i] = detail::NEG_HALF_LN_2PI - cached_log_std +
                                 cached_neg_half_inv_var * sq_diff;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (cached_is_standard_normal) {
                        const double sq_diff = vals[i] * vals[i];
                        res[i] = detail::NEG_HALF_LN_2PI + detail::NEG_HALF * sq_diff;
                    } else {
                        const double diff = vals[i] - cached_mean;
                        const double sq_diff = diff * diff;
                        res[i] = detail::NEG_HALF_LN_2PI - cached_log_std +
                                 cached_neg_half_inv_var * sq_diff;
                    }
                }
            }
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
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
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_mean = dist.mean_;
            const double cached_log_std = dist.logStandardDeviation_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (cached_is_standard_normal) {
                    const double sq_diff = vals[i] * vals[i];
                    res[i] = detail::NEG_HALF_LN_2PI + detail::NEG_HALF * sq_diff;
                } else {
                    const double diff = vals[i] - cached_mean;
                    const double sq_diff = diff * diff;
                    res[i] = detail::NEG_HALF_LN_2PI - cached_log_std +
                             cached_neg_half_inv_var * sq_diff;
                }
            });
        });
}

void GaussianDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const GaussianDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const GaussianDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_mean = dist.mean_;
            const double cached_sigma_sqrt2 = dist.sigmaSqrt2_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(
                vals, res, count, cached_mean, cached_sigma_sqrt2, cached_is_standard_normal);
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_mean = dist.mean_;
            const double cached_sigma_sqrt2 = dist.sigmaSqrt2_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (cached_is_standard_normal) {
                        res[i] =
                            detail::HALF * (detail::ONE + std::erf(vals[i] * detail::INV_SQRT_2));
                    } else {
                        const double normalized = (vals[i] - cached_mean) / cached_sigma_sqrt2;
                        res[i] = detail::HALF * (detail::ONE + std::erf(normalized));
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (cached_is_standard_normal) {
                        res[i] =
                            detail::HALF * (detail::ONE + std::erf(vals[i] * detail::INV_SQRT_2));
                    } else {
                        const double normalized = (vals[i] - cached_mean) / cached_sigma_sqrt2;
                        res[i] = detail::HALF * (detail::ONE + std::erf(normalized));
                    }
                }
            }
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
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
                    const_cast<GaussianDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_mean = dist.mean_;
            const double cached_sigma_sqrt2 = dist.sigmaSqrt2_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (cached_is_standard_normal) {
                    res[i] = detail::HALF * (detail::ONE + std::erf(vals[i] * detail::INV_SQRT_2));
                } else {
                    const double normalized = (vals[i] - cached_mean) / cached_sigma_sqrt2;
                    res[i] = detail::HALF * (detail::ONE + std::erf(normalized));
                }
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS (Power User Interface)
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool GaussianDistribution::operator==(const GaussianDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    return std::abs(mean_ - other.mean_) <= detail::DEFAULT_TOLERANCE &&
           std::abs(standardDeviation_ - other.standardDeviation_) <= detail::DEFAULT_TOLERANCE;
}

//==============================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const GaussianDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, GaussianDistribution& distribution) {
    std::string line;
    double mean, stddev;

    // Expected format: "GaussianDistribution(mean=<value>, stddev=<value>)"
    // Read the entire line to handle spaces in the format

    // Skip leading whitespace and read the entire formatted string
    if (!std::getline(is, line)) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Trim leading whitespace
    size_t start = line.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    line = line.substr(start);

    if (!line.starts_with("GaussianDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract mean value
    if (line.find("mean=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t mean_pos = line.find("mean=") + 5;
    size_t comma_pos = line.find(",", mean_pos);
    if (comma_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string mean_str = line.substr(mean_pos, comma_pos - mean_pos);
        mean = std::stod(mean_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract stddev value
    if (line.find("stddev=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t stddev_pos = line.find("stddev=") + 7;
    size_t close_paren = line.find(")", stddev_pos);
    if (close_paren == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string stddev_str = line.substr(stddev_pos, close_paren - stddev_pos);
        stddev = std::stod(stddev_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Validate and set parameters using the safe API
    auto result = distribution.trySetParameters(mean, stddev);
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
// 18. PRIVATE BATCH IMPLEMENTATIONS
//
// These three methods are the computational core of Gaussian batch operations.
// They are called only after the public API validates inputs and extracts
// cached parameters under a read lock. They accept raw pointers to avoid
// the overhead of span bounds checking in the inner loop.
//
// Architecture:
//   SIMDPolicy::shouldUseSIMD(count) decides at runtime whether SIMD is
//   beneficial. Below the threshold (~8 elements on this machine), the scalar
//   path avoids SIMD dispatch overhead. Above it, VectorOps calls route
//   through simd_dispatch.cpp to the appropriate instruction set (AVX on this
//   Intel Ivy Bridge, NEON on Apple Silicon, SSE2 as fallback).
//
// In-place operations:
//   All three methods use `results` as a workspace to avoid heap allocations.
//   VectorOps operations are safe for in-place use (input read before write).
//
// Adding a new distribution — two patterns:
//
//   (A) Full-domain (Gaussian-style): no out-of-support inputs.
//       Chain scalar_add / scalar_multiply / vector_multiply / vector_exp or
//       vector_erf directly on the results buffer. No fixup needed.
//
//   (B) Bounded-support (Exponential-style): compute+fixup.
//       Run the SIMD chain unconditionally over all inputs, then a scalar
//       pass overwrites out-of-support elements (0 or -inf). Efficient when
//       most callers provide valid inputs, which is the common case.
//       See ExponentialDistribution::getProbabilityBatchUnsafeImpl.
//
//   Keep the scalar fallback path numerically consistent with the SIMD path.
//==============================================================================

void GaussianDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count, double mean,
                                                         double norm_constant,
                                                         double neg_half_inv_var,
                                                         bool is_standard_normal) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = detail::INV_SQRT_2PI * std::exp(detail::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - mean;
                const double sq_diff = diff * diff;
                results[i] = norm_constant * std::exp(neg_half_inv_var * sq_diff);
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // PERFORMANCE CRITICAL: Use results array as workspace to avoid allocations

    if (is_standard_normal) {
        // Standard normal: exp(-0.5 * x²) / sqrt(2π)
        // Step 1: results = x²
        arch::simd::VectorOps::vector_multiply(values, values, results, count);
        // Step 2: results = -0.5 * x²
        arch::simd::VectorOps::scalar_multiply(results, detail::NEG_HALF, results, count);
        // Step 3: results = exp(-0.5 * x²)
        arch::simd::VectorOps::vector_exp(results, results, count);
        // Step 4: results = exp(-0.5 * x²) / sqrt(2π)
        arch::simd::VectorOps::scalar_multiply(results, detail::INV_SQRT_2PI, results, count);
    } else {
        // General case: exp(-0.5 * ((x-μ)/σ)²) / (σ√(2π))
        // Step 1: results = x - μ (difference from mean)
        arch::simd::VectorOps::scalar_add(values, -mean, results, count);
        // Step 2: results = (x - μ)²
        arch::simd::VectorOps::vector_multiply(results, results, results, count);
        // Step 3: results = -0.5 * (x - μ)² / σ²
        arch::simd::VectorOps::scalar_multiply(results, neg_half_inv_var, results, count);
        // Step 4: results = exp(-0.5 * (x - μ)² / σ²)
        arch::simd::VectorOps::vector_exp(results, results, count);
        // Step 5: results = exp(-0.5 * (x - μ)² / σ²) / (σ√(2π))
        arch::simd::VectorOps::scalar_multiply(results, norm_constant, results, count);
    }
}

void GaussianDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double mean, double log_std,
    double neg_half_inv_var, bool is_standard_normal) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = detail::NEG_HALF_LN_2PI + detail::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - mean;
                const double sq_diff = diff * diff;
                results[i] = detail::NEG_HALF_LN_2PI - log_std + neg_half_inv_var * sq_diff;
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // PERFORMANCE CRITICAL: Use results array as workspace to avoid allocations

    if (is_standard_normal) {
        // Standard normal: -0.5 * ln(2π) - 0.5 * x²
        // Step 1: results = x²
        arch::simd::VectorOps::vector_multiply(values, values, results, count);
        // Step 2: results = -0.5 * x²
        arch::simd::VectorOps::scalar_multiply(results, detail::NEG_HALF, results, count);
        // Step 3: results = -0.5 * ln(2π) - 0.5 * x²
        arch::simd::VectorOps::scalar_add(results, detail::NEG_HALF_LN_2PI, results, count);
    } else {
        // General case: -0.5 * ln(2π) - ln(σ) - 0.5 * ((x-μ)/σ)²
        // Step 1: results = x - μ (difference from mean)
        arch::simd::VectorOps::scalar_add(values, -mean, results, count);
        // Step 2: results = (x - μ)²
        arch::simd::VectorOps::vector_multiply(results, results, results, count);
        // Step 3: results = -0.5 * (x - μ)² / σ²
        arch::simd::VectorOps::scalar_multiply(results, neg_half_inv_var, results, count);
        // Step 4: results = -0.5 * ln(2π) - ln(σ) - 0.5 * (x - μ)² / σ²
        const double log_norm_constant = detail::NEG_HALF_LN_2PI - log_std;
        arch::simd::VectorOps::scalar_add(results, log_norm_constant, results, count);
    }
}

void GaussianDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double mean, double sigma_sqrt2,
    bool is_standard_normal) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (is_standard_normal) {
                results[i] =
                    detail::HALF * (detail::ONE + std::erf(values[i] * detail::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - mean) / sigma_sqrt2;
                results[i] = detail::HALF * (detail::ONE + std::erf(normalized));
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // PERFORMANCE CRITICAL: Use results array as workspace to avoid allocations

    if (is_standard_normal) {
        // Standard normal case: normalized = values * INV_SQRT_2
        // Step 1: results = values * INV_SQRT_2 (normalized values)
        arch::simd::VectorOps::scalar_multiply(values, detail::INV_SQRT_2, results, count);
    } else {
        // General case: normalized = (values - mean) / sigma_sqrt2
        // Step 1: results = values - mean
        arch::simd::VectorOps::scalar_add(values, -mean, results, count);
        // Step 2: results = (values - mean) / sigma_sqrt2
        const double reciprocal_sigma_sqrt2 = detail::ONE / sigma_sqrt2;
        arch::simd::VectorOps::scalar_multiply(results, reciprocal_sigma_sqrt2, results, count);
    }

    // vector_erf is in-place safe: AVX loads each chunk before storing it,
    // and the SSE2/scalar fallback uses std::erf element-by-element.
    // No temporary allocation needed.
    arch::simd::VectorOps::vector_erf(results, results, count);

    // Final: results = 0.5 * (1 + erf(normalized))
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    arch::simd::VectorOps::scalar_multiply(results, detail::HALF, results, count);
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void GaussianDistribution::updateCacheUnsafe() const noexcept {
    // Core mathematical functions - primary cache
    normalizationConstant_ = detail::ONE / (standardDeviation_ * detail::SQRT_2PI);
    negHalfSigmaSquaredInv_ = detail::NEG_HALF / (standardDeviation_ * standardDeviation_);
    logStandardDeviation_ = std::log(standardDeviation_);
    sigmaSqrt2_ = standardDeviation_ * detail::SQRT_2;
    invStandardDeviation_ = detail::ONE / standardDeviation_;

    // Secondary cache values - performance optimizations
    cachedSigmaSquared_ = standardDeviation_ * standardDeviation_;
    cachedTwoSigmaSquared_ = detail::TWO * cachedSigmaSquared_;
    cachedLogTwoSigmaSquared_ = std::log(cachedTwoSigmaSquared_);
    cachedInvSigmaSquared_ = detail::ONE / cachedSigmaSquared_;
    cachedSqrtTwoPi_ = detail::SQRT_2PI;

    // Optimization flags - fast path detection
    isStandardNormal_ = (std::abs(mean_) <= detail::DEFAULT_TOLERANCE) &&
                        (std::abs(standardDeviation_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
    isUnitVariance_ = std::abs(cachedSigmaSquared_ - detail::ONE) <= detail::DEFAULT_TOLERANCE;
    isZeroMean_ = std::abs(mean_) <= detail::DEFAULT_TOLERANCE;
    isHighPrecision_ = standardDeviation_ < detail::HIGH_PRECISION_TOLERANCE ||
                       standardDeviation_ > detail::HIGH_PRECISION_UPPER_BOUND;
    isLowVariance_ = cachedSigmaSquared_ < 0.0625;  // σ² < 1/16

    // Update atomic parameters for lock-free access
    atomicMean_.store(mean_, std::memory_order_release);
    atomicStandardDeviation_.store(standardDeviation_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);

    // Cache is now valid
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
}

void GaussianDistribution::validateParameters(double mean, double stdDev) {
    if (!std::isfinite(mean)) {
        throw std::invalid_argument("Mean must be finite");
    }
    if (!std::isfinite(stdDev) || stdDev <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Standard deviation must be positive and finite");
    }
    if (stdDev > detail::MAX_STANDARD_DEVIATION) {
        throw std::invalid_argument("Standard deviation is too large for numerical stability");
    }
}

//==========================================================================
// 20. PRIVATE UTILITY METHODS
//==========================================================================

// Note: Currently no private utility methods needed for Gaussian distribution
// This section maintained for template compliance

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

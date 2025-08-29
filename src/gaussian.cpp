#include "../include/distributions/gaussian.h"

#include "../include/core/constants.h"
#include "../include/core/dispatch_utils.h"
#include "../include/core/safety.h"
#include "../include/core/validation.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_thresholds.h"
#include "../include/platform/simd.h"
#include "../include/platform/simd_policy.h"

#include <algorithm>
#include <cmath>
#include <concepts>  // C++20 concepts
#include <numeric>
#include <ranges>  // C++20 ranges
#include <span>    // C++20 span
#include <vector>
#include <version>  // C++20 feature detection

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
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
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

GaussianDistribution::GaussianDistribution(GaussianDistribution&& other)
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
    other.mean_ = detail::ZERO_DOUBLE;
    other.standardDeviation_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
}

GaussianDistribution& GaussianDistribution::operator=(GaussianDistribution&& other) noexcept {
    if (this != &other) {
        // C++11 Core Guidelines C.66 compliant: noexcept move assignment using atomic operations

        // Step 1: Invalidate both caches atomically (lock-free)
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);

        // Step 2: Try to acquire locks with timeout to avoid blocking indefinitely
        bool success = false;
        try {
            std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);

            // Use try_lock to avoid blocking - this is noexcept
            if (std::try_lock(lock1, lock2) == -1) {
                // Step 3: Move parameters
                mean_ = other.mean_;
                standardDeviation_ = other.standardDeviation_;
                other.mean_ = detail::ZERO_DOUBLE;
                other.standardDeviation_ = detail::ONE;
                cache_valid_ = false;
                other.cache_valid_ = false;
                success = true;
            }
        } catch (...) {
            // If locking fails, we still need to maintain noexcept guarantee
            // Fall back to atomic parameter exchange (lock-free)
        }

        // Step 4: Fallback for failed lock acquisition (still noexcept)
        if (!success) {
            // Use atomic exchange operations for thread-safe parameter swapping
            // This maintains basic correctness even if we can't acquire locks
            [[maybe_unused]] double temp_mean = mean_;
            [[maybe_unused]] double temp_stddev = standardDeviation_;

            // Atomic-like exchange (single assignment is atomic for built-in types)
            mean_ = other.mean_;
            standardDeviation_ = other.standardDeviation_;
            other.mean_ = detail::ZERO_DOUBLE;
            other.standardDeviation_ = detail::ONE;

            // Cache invalidation was already done atomically above
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
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
    // Copy current standard deviation for validation (thread-safe)
    double currentStdDev;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentStdDev = standardDeviation_;
    }

    validateParameters(mean, currentStdDev);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = mean;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GaussianDistribution::setStandardDeviation(double stdDev) {
    // Copy current mean for validation (thread-safe)
    double currentMean;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentMean = mean_;
    }

    validateParameters(currentMean, stdDev);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    standardDeviation_ = stdDev;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
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
    // Copy current standard deviation for validation (thread-safe)
    double currentStdDev;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentStdDev = standardDeviation_;
    }

    auto validation = validateGaussianParameters(mean, currentStdDev);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = mean;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok(true);
}

VoidResult GaussianDistribution::trySetStandardDeviation(double stdDev) noexcept {
    // Copy current mean for validation (thread-safe)
    double currentMean;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentMean = mean_;
    }

    auto validation = validateGaussianParameters(currentMean, stdDev);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    standardDeviation_ = stdDev;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok(true);
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

    return VoidResult::ok(true);
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

double GaussianDistribution::getLogProbability(double x) const noexcept {
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
    if (datasets.empty()) {
        // Handle empty datasets gracefully
        results.clear();
        return;
    }

    // Ensure results vector has correct size
    if (results.size() != datasets.size()) {
        results.resize(datasets.size());
    }

    const std::size_t num_datasets = datasets.size();

    // Use distribution-specific parallel thresholds for optimal work distribution
    if (arch::shouldUseDistributionParallel("gaussian", "batch_fit", num_datasets)) {
        // Thread-safe parallel execution with proper exception handling
        // Use a static mutex to synchronize access to the global thread pool from multiple threads
        static std::mutex pool_access_mutex;

        try {
            ThreadPool* pool_ptr = nullptr;
            {
                // Brief lock to get thread pool reference - minimize lock contention
                std::lock_guard<std::mutex> pool_lock(pool_access_mutex);
                pool_ptr = &ParallelUtils::getGlobalThreadPool();
            }

            const std::size_t optimal_grain_size = std::max(std::size_t{1}, num_datasets / 8);
            const std::size_t num_chunks =
                (num_datasets + optimal_grain_size - 1) / optimal_grain_size;

            // Pre-allocate futures with known size to avoid reallocation during concurrent access
            std::vector<std::future<void>> futures;
            futures.reserve(num_chunks);

            // Atomic counter for tracking completion and error handling
            std::atomic<std::size_t> completed_chunks{0};
            std::atomic<bool> has_error{false};
            std::mutex error_mutex;
            std::string error_message;

            // Submit all tasks with exception handling
            for (std::size_t i = 0; i < num_datasets; i += optimal_grain_size) {
                const std::size_t chunk_start = i;
                const std::size_t chunk_end = std::min(i + optimal_grain_size, num_datasets);

                auto future = pool_ptr->submit([&datasets, &results, chunk_start, chunk_end,
                                                &completed_chunks, &has_error, &error_mutex,
                                                &error_message]() {
                    try {
                        // Process chunk with local error handling
                        for (std::size_t j = chunk_start; j < chunk_end; ++j) {
                            results[j].fit(datasets[j]);
                        }
                        completed_chunks.fetch_add(1, std::memory_order_relaxed);
                    } catch (const std::exception& e) {
                        // Thread-safe error recording
                        {
                            std::lock_guard<std::mutex> error_lock(error_mutex);
                            if (!has_error.load()) {
                                error_message = "Parallel batch fit error in chunk [" +
                                                std::to_string(chunk_start) + ", " +
                                                std::to_string(chunk_end) + "): " + e.what();
                                has_error.store(true, std::memory_order_release);
                            }
                        }
                    } catch (...) {
                        // Handle non-standard exceptions
                        {
                            std::lock_guard<std::mutex> error_lock(error_mutex);
                            if (!has_error.load()) {
                                error_message = "Unknown error in parallel batch fit chunk [" +
                                                std::to_string(chunk_start) + ", " +
                                                std::to_string(chunk_end) + ")";
                                has_error.store(true, std::memory_order_release);
                            }
                        }
                    }
                });

                futures.push_back(std::move(future));
            }

            // Wait for all chunks to complete with timeout and error checking
            bool all_completed = true;
            for (auto& future : futures) {
                try {
                    // Use wait() instead of get() to avoid exception re-throwing from task
                    future.wait();
                } catch (const std::exception& e) {
                    // Handle future-related exceptions
                    std::lock_guard<std::mutex> error_lock(error_mutex);
                    if (!has_error.load()) {
                        error_message = "Future wait error: " + std::string(e.what());
                        has_error.store(true, std::memory_order_release);
                    }
                    all_completed = false;
                }
            }

            // Check for errors after all futures complete
            if (has_error.load()) {
                std::lock_guard<std::mutex> error_lock(error_mutex);
                throw std::runtime_error("Parallel batch fitting failed: " + error_message);
            }

            if (!all_completed) {
                throw std::runtime_error(
                    "Some parallel batch fitting tasks failed to complete properly");
            }

        } catch (const std::exception& e) {
            // If parallel execution fails, fall back to serial execution
            // This ensures robustness in case of thread pool issues
            for (std::size_t i = 0; i < num_datasets; ++i) {
                try {
                    results[i].fit(datasets[i]);
                } catch (const std::exception& fit_error) {
                    throw std::runtime_error("Serial fallback failed for dataset " +
                                             std::to_string(i) + ": " + fit_error.what() +
                                             " (original parallel error: " + e.what() + ")");
                }
            }
        }

    } else {
        // Serial processing for small numbers of datasets
        for (std::size_t i = 0; i < num_datasets; ++i) {
            try {
                results[i].fit(datasets[i]);
            } catch (const std::exception& e) {
                throw std::runtime_error("Serial batch fit failed for dataset " +
                                         std::to_string(i) + ": " + e.what());
            }
        }
    }
}

void GaussianDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = detail::ZERO_DOUBLE;
    standardDeviation_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
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

std::pair<double, double> GaussianDistribution::confidenceIntervalMean(
    const std::vector<double>& data, double confidence_level, bool population_variance_known) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= detail::ZERO_DOUBLE || confidence_level >= detail::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }

    const size_t n = data.size();
    const double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);

    double margin_of_error;

    if (population_variance_known || n >= 30) {
        // Use normal distribution (z-score)
        const double sample_var =
            std::inner_product(data.begin(), data.end(), data.begin(), detail::ZERO_DOUBLE) /
                static_cast<double>(n) -
            sample_mean * sample_mean;
        const double sample_std = std::sqrt(sample_var);
        const double alpha = detail::ONE - confidence_level;
        const double z_alpha_2 = detail::inverse_normal_cdf(detail::ONE - alpha * detail::HALF);
        margin_of_error = z_alpha_2 * sample_std / std::sqrt(static_cast<double>(n));
    } else {
        // Use t-distribution
        const double sample_var =
            std::inner_product(data.begin(), data.end(), data.begin(), detail::ZERO_DOUBLE,
                               std::plus<>(),
                               [sample_mean](double x, double y) {
                                   return (x - sample_mean) * (y - sample_mean);
                               }) /
            static_cast<double>(n - 1);
        const double sample_std = std::sqrt(sample_var);
        const double alpha = detail::ONE - confidence_level;
        const double t_alpha_2 =
            detail::inverse_t_cdf(detail::ONE - alpha * detail::HALF, static_cast<double>(n - 1));
        margin_of_error = t_alpha_2 * sample_std / std::sqrt(static_cast<double>(n));
    }

    return {sample_mean - margin_of_error, sample_mean + margin_of_error};
}

std::pair<double, double> GaussianDistribution::confidenceIntervalVariance(
    const std::vector<double>& data, double confidence_level) {
    if (data.size() < 2) {
        throw std::invalid_argument(
            "At least 2 data points required for variance confidence interval");
    }
    if (confidence_level <= detail::ZERO_DOUBLE || confidence_level >= detail::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }

    const size_t n = data.size();
    const double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);

    // Calculate sample variance
    const double sample_var =
        std::inner_product(
            data.begin(), data.end(), data.begin(), detail::ZERO_DOUBLE, std::plus<>(),
            [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }) /
        static_cast<double>(n - 1);

    const double alpha = detail::ONE - confidence_level;
    const double df = static_cast<double>(n - 1);

    // Chi-squared critical values
    const double alpha_half = alpha * detail::HALF;
    const double chi2_lower = detail::inverse_chi_squared_cdf(alpha_half, df);
    const double chi2_upper = detail::inverse_chi_squared_cdf(detail::ONE - alpha_half, df);

    const double lower_bound = (df * sample_var) / chi2_upper;
    const double upper_bound = (df * sample_var) / chi2_lower;

    return {lower_bound, upper_bound};
}

std::tuple<double, double, bool> GaussianDistribution::oneSampleTTest(
    const std::vector<double>& data, double hypothesized_mean, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    const size_t n = data.size();
    const double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);

    // Calculate sample standard deviation
    const double sample_var =
        std::inner_product(
            data.begin(), data.end(), data.begin(), detail::ZERO_DOUBLE, std::plus<>(),
            [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }) /
        static_cast<double>(n - 1);
    const double sample_std = std::sqrt(sample_var);

    // Calculate t-statistic
    const double t_statistic =
        (sample_mean - hypothesized_mean) / (sample_std / std::sqrt(static_cast<double>(n)));

    // Calculate p-value (two-tailed) using constants for 2.0 and 1.0
    const double p_value = detail::TWO * (detail::ONE - detail::t_cdf(std::abs(t_statistic),
                                                                      static_cast<double>(n - 1)));

    const bool reject_null = p_value < alpha;

    return {t_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::twoSampleTTest(
    const std::vector<double>& data1, const std::vector<double>& data2, bool equal_variances,
    double alpha) {
    if (data1.empty() || data2.empty()) {
        throw std::invalid_argument("Both data vectors must be non-empty");
    }
    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    const size_t n1 = data1.size();
    const size_t n2 = data2.size();

    // Sample means
    const double mean1 =
        std::accumulate(data1.begin(), data1.end(), detail::ZERO_DOUBLE) / static_cast<double>(n1);
    const double mean2 =
        std::accumulate(data2.begin(), data2.end(), detail::ZERO_DOUBLE) / static_cast<double>(n2);

    // Sample variances
    const double var1 =
        std::inner_product(data1.begin(), data1.end(), data1.begin(), detail::ZERO_DOUBLE,
                           std::plus<>(),
                           [mean1](double x, double y) { return (x - mean1) * (y - mean1); }) /
        static_cast<double>(n1 - 1);

    const double var2 =
        std::inner_product(data2.begin(), data2.end(), data2.begin(), detail::ZERO_DOUBLE,
                           std::plus<>(),
                           [mean2](double x, double y) { return (x - mean2) * (y - mean2); }) /
        static_cast<double>(n2 - 1);

    double t_statistic, degrees_of_freedom;

    if (equal_variances) {
        // Pooled t-test
        const double pooled_var =
            (static_cast<double>(n1 - 1) * var1 + static_cast<double>(n2 - 1) * var2) /
            static_cast<double>(n1 + n2 - 2);
        const double pooled_std = std::sqrt(pooled_var * (detail::ONE / static_cast<double>(n1) +
                                                          detail::ONE / static_cast<double>(n2)));
        t_statistic = (mean1 - mean2) / pooled_std;
        degrees_of_freedom = static_cast<double>(n1 + n2 - 2);
    } else {
        // Welch's t-test
        const double se =
            std::sqrt(var1 / static_cast<double>(n1) + var2 / static_cast<double>(n2));
        t_statistic = (mean1 - mean2) / se;

        // Welch-Satterthwaite equation for degrees of freedom
        const double numerator =
            std::pow(var1 / static_cast<double>(n1) + var2 / static_cast<double>(n2), detail::TWO);
        const double denominator =
            std::pow(var1 / static_cast<double>(n1), detail::TWO) / static_cast<double>(n1 - 1) +
            std::pow(var2 / static_cast<double>(n2), detail::TWO) / static_cast<double>(n2 - 1);
        degrees_of_freedom = numerator / denominator;
    }

    // Calculate p-value (two-tailed)
    const double p_value =
        detail::TWO * (detail::ONE - detail::t_cdf(std::abs(t_statistic), degrees_of_freedom));

    const bool reject_null = p_value < alpha;

    return {t_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::pairedTTest(const std::vector<double>& data1,
                                                                   const std::vector<double>& data2,
                                                                   double alpha) {
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("Data vectors must have the same size for paired t-test");
    }
    if (data1.empty()) {
        throw std::invalid_argument("Data vectors cannot be empty");
    }
    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    const size_t n = data1.size();

    // Calculate differences
    std::vector<double> differences(n);
    std::transform(data1.begin(), data1.end(), data2.begin(), differences.begin(),
                   [](double a, double b) { return a - b; });

    // Perform one-sample t-test on differences against mean = 0
    return oneSampleTTest(differences, detail::ZERO_DOUBLE, alpha);
}

std::tuple<double, double, double, double> GaussianDistribution::bayesianEstimation(
    const std::vector<double>& data, double prior_mean, double prior_precision, double prior_shape,
    double prior_rate) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    const size_t n = data.size();
    const double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);
    const double sample_sum_sq =
        std::inner_product(data.begin(), data.end(), data.begin(), detail::ZERO_DOUBLE);

    // Normal-Inverse-Gamma conjugate prior update
    const double posterior_precision = prior_precision + static_cast<double>(n);
    const double posterior_mean =
        (prior_precision * prior_mean + static_cast<double>(n) * sample_mean) / posterior_precision;
    const double posterior_shape = prior_shape + static_cast<double>(n) / detail::TWO;

    const double sum_sq_deviations =
        sample_sum_sq - static_cast<double>(n) * sample_mean * sample_mean;
    const double prior_mean_diff = sample_mean - prior_mean;
    const double posterior_rate =
        prior_rate + detail::HALF * sum_sq_deviations +
        detail::HALF *
            (prior_precision * static_cast<double>(n) * prior_mean_diff * prior_mean_diff) /
            posterior_precision;

    return {posterior_mean, posterior_precision, posterior_shape, posterior_rate};
}

std::pair<double, double> GaussianDistribution::bayesianCredibleInterval(
    const std::vector<double>& data, double credibility_level, double prior_mean,
    double prior_precision, double prior_shape, double prior_rate) {
    // Get posterior parameters
    auto [post_mean, post_precision, post_shape, post_rate] =
        bayesianEstimation(data, prior_mean, prior_precision, prior_shape, prior_rate);

    // Posterior marginal for mean follows t-distribution
    const double df = detail::TWO * post_shape;
    const double scale = std::sqrt(post_rate / (post_precision * post_shape));

    const double alpha = detail::ONE - credibility_level;
    const double t_critical = detail::inverse_t_cdf(detail::ONE - alpha * detail::HALF, df);

    const double margin_of_error = t_critical * scale;

    return {post_mean - margin_of_error, post_mean + margin_of_error};
}

std::pair<double, double> GaussianDistribution::robustEstimation(const std::vector<double>& data,
                                                                 const std::string& estimator_type,
                                                                 double tuning_constant) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Initial estimates using median and MAD
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    const double median = (sorted_data.size() % 2 == 0)
                              ? (sorted_data.at(sorted_data.size() / 2 - 1) +
                                 sorted_data.at(sorted_data.size() / 2)) /
                                    detail::TWO
                              : sorted_data.at(sorted_data.size() / 2);

    // Median Absolute Deviation (MAD)
    std::vector<double> abs_deviations(data.size());
    std::transform(data.begin(), data.end(), abs_deviations.begin(),
                   [median](double x) { return std::abs(x - median); });
    std::sort(abs_deviations.begin(), abs_deviations.end());

    const double mad = (abs_deviations.size() % 2 == 0)
                           ? (abs_deviations.at(abs_deviations.size() / 2 - 1) +
                              abs_deviations.at(abs_deviations.size() / 2)) /
                                 detail::TWO
                           : abs_deviations.at(abs_deviations.size() / 2);

    // Convert MAD to robust scale estimate
    double robust_location = median;
    double robust_scale = mad * detail::MAD_SCALING_FACTOR;  // Use named constant instead of 1.4826

    // Iterative M-estimation
    const int max_iterations = 50;
    const double convergence_tol = 1e-6;

    for (int iter = 0; iter < max_iterations; ++iter) {
        double sum_weights = detail::ZERO_DOUBLE;
        double weighted_sum = detail::ZERO_DOUBLE;

        for (double x : data) {
            const double standardized_residual = (x - robust_location) / robust_scale;
            double weight = detail::ONE;

            if (estimator_type == "huber") {
                weight = (std::abs(standardized_residual) <= tuning_constant)
                             ? detail::ONE
                             : tuning_constant / std::abs(standardized_residual);
            } else if (estimator_type == "tukey") {
                weight =
                    (std::abs(standardized_residual) <= tuning_constant)
                        ? std::pow(detail::ONE - std::pow(standardized_residual / tuning_constant,
                                                          detail::TWO),
                                   detail::TWO)
                        : detail::ZERO_DOUBLE;
            } else if (estimator_type == "hampel") {
                const double abs_res = std::abs(standardized_residual);
                if (abs_res <= tuning_constant) {
                    weight = detail::ONE;
                } else if (abs_res <= detail::TWO * tuning_constant) {
                    weight = tuning_constant / abs_res;
                } else if (abs_res <= detail::THREE * tuning_constant) {
                    weight = tuning_constant * (detail::THREE - abs_res / tuning_constant) /
                             (detail::TWO * abs_res);
                } else {
                    weight = detail::ZERO_DOUBLE;
                }
            } else {
                throw std::invalid_argument(
                    "Unknown estimator type. Use 'huber', 'tukey', or 'hampel'");
            }

            sum_weights += weight;
            weighted_sum += weight * x;
        }

        const double new_location = weighted_sum / sum_weights;

        // Update scale estimate
        double weighted_scale_sum = detail::ZERO_DOUBLE;
        for (double x : data) {
            const double residual = x - new_location;
            const double standardized_residual = residual / robust_scale;
            double weight = detail::ONE;

            if (estimator_type == "huber") {
                weight = (std::abs(standardized_residual) <= tuning_constant)
                             ? detail::ONE
                             : tuning_constant / std::abs(standardized_residual);
            } else if (estimator_type == "tukey") {
                weight =
                    (std::abs(standardized_residual) <= tuning_constant)
                        ? std::pow(detail::ONE - std::pow(standardized_residual / tuning_constant,
                                                          detail::TWO),
                                   detail::TWO)
                        : detail::ZERO_DOUBLE;
            }

            weighted_scale_sum += weight * residual * residual;
        }

        const double new_scale = std::sqrt(weighted_scale_sum / sum_weights);

        // Check convergence
        if (std::abs(new_location - robust_location) < convergence_tol &&
            std::abs(new_scale - robust_scale) < convergence_tol) {
            break;
        }

        robust_location = new_location;
        robust_scale = new_scale;
    }

    return {robust_location, robust_scale};
}

std::pair<double, double> GaussianDistribution::methodOfMomentsEstimation(
    const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    const size_t n = data.size();

    // First moment (mean)
    const double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);

    // Second central moment (variance)
    const double sample_variance =
        std::inner_product(
            data.begin(), data.end(), data.begin(), detail::ZERO_DOUBLE, std::plus<>(),
            [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }) /
        static_cast<double>(n);  // Population variance (divide by n, not n-1)

    const double sample_stddev = std::sqrt(sample_variance);

    return {sample_mean, sample_stddev};
}

std::pair<double, double> GaussianDistribution::lMomentsEstimation(
    const std::vector<double>& data) {
    if (data.size() < 2) {
        throw std::invalid_argument("At least 2 data points required for L-moments estimation");
    }

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    const size_t n = sorted_data.size();

    // Calculate L-moments
    double l1 = detail::ZERO_DOUBLE;  // L-mean
    double l2 = detail::ZERO_DOUBLE;  // L-scale

    // L1 (L-mean) = mean of order statistics
    l1 = std::accumulate(sorted_data.begin(), sorted_data.end(), detail::ZERO_DOUBLE) /
         static_cast<double>(n);

    // L2 (L-scale) = 0.5 * E[X_{2:2} - X_{1:2}]
    for (size_t i = 0; i < n; ++i) {
        const double weight =
            (detail::TWO * static_cast<double>(i) + detail::ONE - static_cast<double>(n)) /
            static_cast<double>(n);
        l2 += weight * sorted_data[i];
    }
    l2 = detail::HALF * l2;

    // For Gaussian distribution:
    // L1 = μ (location parameter)
    // L2 = σ/√π (scale parameter relationship)
    const double location_param = l1;
    const double scale_param = l2 * std::sqrt(detail::PI);

    return {location_param, scale_param};
}

std::vector<double> GaussianDistribution::calculateHigherMoments(const std::vector<double>& data,
                                                                 bool center_on_mean) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    const size_t n = data.size();
    const double sample_mean =
        center_on_mean ? std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n)
                       : 0.0;

    std::vector<double> moments(6, 0.0);

    // Calculate raw or central moments up to 6th order
    for (double x : data) {
        const double deviation = x - sample_mean;

        for (int k = 0; k < 6; ++k) {
            if (center_on_mean) {
                moments[static_cast<std::size_t>(k)] += std::pow(deviation, k + 1);
            } else {
                moments[static_cast<std::size_t>(k)] += std::pow(x, k + 1);
            }
        }
    }

    // Normalize by sample size
    for (double& moment : moments) {
        moment /= static_cast<double>(n);
    }

    return moments;
}

std::tuple<double, double, bool> GaussianDistribution::jarqueBeraTest(
    const std::vector<double>& data, double alpha) {
    if (data.size() < 8) {
        throw std::invalid_argument("At least 8 data points required for Jarque-Bera test");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    const size_t n = data.size();
    const double sample_mean =
        std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);

    // Calculate sample variance, skewness, and kurtosis
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;

    for (double x : data) {
        const double deviation = x - sample_mean;
        const double dev2 = deviation * deviation;
        const double dev3 = dev2 * deviation;
        const double dev4 = dev3 * deviation;

        m2 += dev2;
        m3 += dev3;
        m4 += dev4;
    }

    m2 /= static_cast<double>(n);
    m3 /= static_cast<double>(n);
    m4 /= static_cast<double>(n);

    const double skewness = m3 / std::pow(m2, 1.5);
    const double kurtosis = m4 / (m2 * m2) - detail::EXCESS_KURTOSIS_OFFSET;  // Excess kurtosis

    // Jarque-Bera statistic
    const double jb_statistic =
        static_cast<double>(n) *
        (skewness * skewness / detail::SIX + kurtosis * kurtosis / detail::TWO_TWENTY_FIVE);

    // P-value from chi-squared distribution with 2 degrees of freedom
    const double p_value = detail::ONE - detail::chi_squared_cdf(jb_statistic, detail::TWO);

    const bool reject_null = p_value < alpha;

    return {jb_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::shapiroWilkTest(
    const std::vector<double>& data, double alpha) {
    if (data.size() < 3 || data.size() > 5000) {
        throw std::invalid_argument("Shapiro-Wilk test requires 3 to 5000 data points");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    const size_t n = sorted_data.size();

    // Simplified Shapiro-Wilk implementation
    // For full implementation, would need lookup tables for coefficients

    // Calculate sample variance
    const double sample_mean =
        std::accumulate(sorted_data.begin(), sorted_data.end(), 0.0) / static_cast<double>(n);
    double ss = 0.0;
    for (double x : sorted_data) {
        ss += (x - sample_mean) * (x - sample_mean);
    }

    // Simplified W statistic calculation
    // This is a basic approximation - full implementation would use proper coefficients
    double numerator = 0.0;
    for (size_t i = 0; i < n / 2; ++i) {
        const double coeff =
            detail::inverse_normal_cdf((static_cast<double>(i) + detail::THREE_QUARTERS) /
                                       (static_cast<double>(n) + detail::HALF));
        numerator += coeff * (sorted_data[n - 1 - i] - sorted_data[i]);
    }

    const double w_statistic = (numerator * numerator) / ss;

    // Approximate p-value (simplified)
    // Full implementation would use proper lookup tables or approximations
    const double log_p = detail::NEG_HALF * std::log(w_statistic) -
                         detail::ONE_POINT_FIVE * std::log(n) + detail::TWO;
    const double p_value = std::exp(log_p);

    const bool reject_null = p_value < alpha;

    return {w_statistic, std::min(p_value, detail::ONE), reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::likelihoodRatioTest(
    const std::vector<double>& data, const GaussianDistribution& restricted_model,
    const GaussianDistribution& unrestricted_model, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Calculate log-likelihoods
    double log_likelihood_restricted = 0.0;
    double log_likelihood_unrestricted = 0.0;

    for (double x : data) {
        log_likelihood_restricted += restricted_model.getLogProbability(x);
        log_likelihood_unrestricted += unrestricted_model.getLogProbability(x);
    }

    // Likelihood ratio statistic
    const double lr_statistic =
        detail::TWO * (log_likelihood_unrestricted - log_likelihood_restricted);

    // Degrees of freedom = difference in number of parameters
    const int df = unrestricted_model.getNumParameters() - restricted_model.getNumParameters();

    if (df <= 0) {
        throw std::invalid_argument(
            "Unrestricted model must have more parameters than restricted model");
    }

    // P-value from chi-squared distribution
    const double p_value = detail::ONE - detail::chi_squared_cdf(lr_statistic, df);

    const bool reject_null = p_value < alpha;

    return {lr_statistic, p_value, reject_null};
}

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
//==========================================================================

std::tuple<double, double, bool> GaussianDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data, const GaussianDistribution& distribution, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Use the KS statistic calculation from math_utils
    double ks_statistic = detail::calculate_ks_statistic(data, distribution);

    // Asymptotic p-value approximation for KS test
    // P-value ≈ 2 * exp(-2 * n * D²) for large n
    const double n = static_cast<double>(data.size());
    const double p_value_approx =
        detail::TWO * std::exp(-detail::TWO * n * ks_statistic * ks_statistic);

    // Clamp p-value to [0, 1]
    const double p_value = std::min(detail::ONE, std::max(detail::ZERO, p_value_approx));

    const bool reject_null = p_value < alpha;

    return {ks_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::andersonDarlingTest(
    const std::vector<double>& data, const GaussianDistribution& distribution, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Use the AD statistic calculation from math_utils
    double ad_statistic = detail::calculate_ad_statistic(data, distribution);

    // Asymptotic p-value approximation for AD test with Gaussian distribution
    // This is a simplified approximation - full implementation would use
    // more sophisticated lookup tables or better approximation formulas
    const double n = static_cast<double>(data.size());
    const double modified_stat = ad_statistic * (detail::ONE + detail::THREE_QUARTERS / n +
                                                 detail::TWO_TWENTY_FIVE / (n * n));

    // Approximate p-value using exponential approximation
    double p_value;
    if (modified_stat >= detail::THIRTEEN) {
        p_value = detail::ZERO;
    } else if (modified_stat >= detail::SIX) {
        p_value = std::exp(-detail::ONE_POINT_TWO_EIGHT * modified_stat);
    } else {
        p_value = std::exp(-detail::ONE_POINT_EIGHT * modified_stat + detail::ONE_POINT_FIVE);
    }

    // Clamp p-value to [0, 1]
    p_value = std::min(detail::ONE, std::max(detail::ZERO, p_value));

    const bool reject_null = p_value < alpha;

    return {ad_statistic, p_value, reject_null};
}

//==============================================================================
// 9. CROSS-VALIDATION METHODS
//==============================================================================

std::vector<std::tuple<double, double, double>> GaussianDistribution::kFoldCrossValidation(
    const std::vector<double>& data, int k, unsigned int random_seed) {
    if (data.size() < static_cast<size_t>(k)) {
        throw std::invalid_argument("Data size must be at least k for k-fold cross-validation");
    }
    if (k <= 1) {
        throw std::invalid_argument("Number of folds k must be greater than 1");
    }

    const size_t n = data.size();
    const size_t fold_size = n / static_cast<std::size_t>(k);

    // Create shuffled indices for random fold assignment
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<std::tuple<double, double, double>> results;
    results.reserve(static_cast<std::size_t>(k));

    for (int fold = 0; fold < k; ++fold) {
        // Define validation set indices for this fold
        const size_t start_idx = static_cast<std::size_t>(fold) * fold_size;
        const size_t end_idx =
            (fold == k - 1) ? n : (static_cast<std::size_t>(fold) + 1) * fold_size;

        // Create training and validation sets
        std::vector<double> training_data;
        std::vector<double> validation_data;
        training_data.reserve(n - (end_idx - start_idx));
        validation_data.reserve(end_idx - start_idx);

        for (size_t i = 0; i < n; ++i) {
            if (i >= start_idx && i < end_idx) {
                validation_data.push_back(data[indices[i]]);
            } else {
                training_data.push_back(data[indices[i]]);
            }
        }

        // Fit model on training data
        GaussianDistribution fitted_model;
        fitted_model.fit(training_data);

        // Evaluate on validation data
        // double mean_error = 0.0;  // No longer used - MAE calculated directly from errors vector
        // double std_error = 0.0;   // No longer used
        double log_likelihood = 0.0;

        // Calculate prediction errors and log-likelihood
        std::vector<double> errors;
        errors.reserve(validation_data.size());

        for (double val : validation_data) {
            double predicted_mean = fitted_model.getMean();
            double error = std::abs(val - predicted_mean);
            errors.push_back(error);

            log_likelihood += fitted_model.getLogProbability(val);
        }

        // Calculate MAE and RMSE
        double mae =
            std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());

        // Calculate RMSE = sqrt(mean(squared_errors))
        double mse = 0.0;
        for (double error : errors) {
            mse += error * error;
        }
        mse /= static_cast<double>(errors.size());
        double rmse = std::sqrt(mse);

        results.emplace_back(mae, rmse, log_likelihood);
    }

    return results;
}

std::tuple<double, double, double> GaussianDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    if (data.size() < 3) {
        throw std::invalid_argument("At least 3 data points required for LOOCV");
    }

    const size_t n = data.size();
    std::vector<double> absolute_errors;
    std::vector<double> squared_errors;
    double total_log_likelihood = 0.0;

    absolute_errors.reserve(n);
    squared_errors.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        // Create training set excluding point i
        std::vector<double> training_data;
        training_data.reserve(n - 1);

        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                training_data.push_back(data[j]);
            }
        }

        // Fit model on training data
        GaussianDistribution fitted_model;
        fitted_model.fit(training_data);

        // Evaluate on left-out point
        double predicted_mean = fitted_model.getMean();
        double actual_value = data[i];

        double absolute_error = std::abs(actual_value - predicted_mean);
        double squared_error = (actual_value - predicted_mean) * (actual_value - predicted_mean);

        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);

        total_log_likelihood += fitted_model.getLogProbability(actual_value);
    }

    // Calculate summary statistics
    double mean_absolute_error =
        std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) /
        static_cast<double>(n);
    double mean_squared_error =
        std::accumulate(squared_errors.begin(), squared_errors.end(), 0.0) / static_cast<double>(n);
    double root_mean_squared_error = std::sqrt(mean_squared_error);

    return {mean_absolute_error, root_mean_squared_error, total_log_likelihood};
}

//==============================================================================
// 10. INFORMATION CRITERIA
//==============================================================================

std::tuple<double, double, double, double> GaussianDistribution::computeInformationCriteria(
    const std::vector<double>& data, const GaussianDistribution& fitted_distribution) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    const double n = static_cast<double>(data.size());
    const int k = fitted_distribution.getNumParameters();  // 2 for Gaussian

    // Calculate log-likelihood
    double log_likelihood = 0.0;
    for (double val : data) {
        log_likelihood += fitted_distribution.getLogProbability(val);
    }

    // Compute information criteria
    const double aic = detail::TWO * k - detail::TWO * log_likelihood;
    const double bic = std::log(n) * k - detail::TWO * log_likelihood;

    // AICc (corrected AIC for small sample sizes)
    double aicc;
    if (n - k - 1 > 0) {
        aicc = aic + (detail::TWO * k * (k + 1)) / (n - k - 1);
    } else {
        aicc = std::numeric_limits<double>::infinity();  // Undefined for small samples
    }

    return {aic, bic, aicc, log_likelihood};
}

//==============================================================================
// 11. BOOTSTRAP METHODS
//==============================================================================

std::tuple<std::pair<double, double>, std::pair<double, double>>
GaussianDistribution::bootstrapParameterConfidenceIntervals(const std::vector<double>& data,
                                                            double confidence_level,
                                                            int n_bootstrap,
                                                            unsigned int random_seed) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    if (n_bootstrap <= 0) {
        throw std::invalid_argument("Number of bootstrap samples must be positive");
    }

    const size_t n = data.size();
    std::vector<double> bootstrap_means;
    std::vector<double> bootstrap_stds;
    bootstrap_means.reserve(static_cast<std::size_t>(n_bootstrap));
    bootstrap_stds.reserve(static_cast<std::size_t>(n_bootstrap));

    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<size_t> dist(0, n - 1);

    // Generate bootstrap samples
    for (int b = 0; b < n_bootstrap; ++b) {
        std::vector<double> bootstrap_sample;
        bootstrap_sample.reserve(n);

        // Sample with replacement
        for (size_t i = 0; i < n; ++i) {
            bootstrap_sample.push_back(data[dist(rng)]);
        }

        // Fit model to bootstrap sample
        GaussianDistribution bootstrap_model;
        bootstrap_model.fit(bootstrap_sample);

        bootstrap_means.push_back(bootstrap_model.getMean());
        bootstrap_stds.push_back(bootstrap_model.getStandardDeviation());
    }

    // Sort for quantile calculation
    std::sort(bootstrap_means.begin(), bootstrap_means.end());
    std::sort(bootstrap_stds.begin(), bootstrap_stds.end());

    // Calculate confidence intervals using percentile method
    const double alpha = detail::ONE - confidence_level;
    const double lower_percentile = alpha * detail::HALF;
    const double upper_percentile = detail::ONE - alpha * detail::HALF;

    const size_t lower_idx = static_cast<size_t>(lower_percentile * (n_bootstrap - 1));
    const size_t upper_idx = static_cast<size_t>(upper_percentile * (n_bootstrap - 1));

    std::pair<double, double> mean_ci = {bootstrap_means[lower_idx], bootstrap_means[upper_idx]};
    std::pair<double, double> std_ci = {bootstrap_stds[lower_idx], bootstrap_stds[upper_idx]};

    return {mean_ci, std_ci};
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

#ifdef DEBUG

bool GaussianDistribution::isUsingStandardNormalOptimization() const {
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

    // Return the current state of the standard normal optimization flag
    return isStandardNormal_;
}

#endif  // DEBUG

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS (New Simplified API)
//==============================================================================

void GaussianDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<GaussianDistribution>::distType(),
        detail::DistributionTraits<GaussianDistribution>::complexity(),
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
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
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

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_mean = dist.mean_;
            const double cached_norm_constant = dist.normalizationConstant_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
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
        *this, values, results, hint, detail::DistributionTraits<GaussianDistribution>::distType(),
        detail::DistributionTraits<GaussianDistribution>::complexity(),
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
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
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

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_mean = dist.mean_;
            const double cached_log_std = dist.logStandardDeviation_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
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
        *this, values, results, hint, detail::DistributionTraits<GaussianDistribution>::distType(),
        detail::DistributionTraits<GaussianDistribution>::complexity(),
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
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
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

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_mean = dist.mean_;
            const double cached_sigma_sqrt2 = dist.sigmaSqrt2_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
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

void GaussianDistribution::getProbabilityWithStrategy(std::span<const double> values,
                                                      std::span<double> results,
                                                      detail::Strategy strategy) const {
    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
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

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
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
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
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

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_mean = dist.mean_;
            const double cached_log_std = dist.logStandardDeviation_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
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

void GaussianDistribution::getLogProbabilityWithStrategy(std::span<const double> values,
                                                         std::span<double> results,
                                                         detail::Strategy strategy) const {
    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
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

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
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
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
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

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_mean = dist.mean_;
            const double cached_log_std = dist.logStandardDeviation_;
            const double cached_neg_half_inv_var = dist.negHalfSigmaSquaredInv_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
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

void GaussianDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                                                std::span<double> results,
                                                                detail::Strategy strategy) const {
    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
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

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (cached_is_standard_normal) {
                    res[i] = detail::HALF * (detail::ONE + std::erf(vals[i] * detail::INV_SQRT_2));
                } else {
                    const double normalized = (vals[i] - cached_mean) / cached_sigma_sqrt2;
                    res[i] = detail::HALF * (detail::ONE + std::erf(normalized));
                }
            });
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
        },
        [](const GaussianDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
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

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_mean = dist.mean_;
            const double cached_sigma_sqrt2 = dist.sigmaSqrt2_;
            const bool cached_is_standard_normal = dist.isStandardNormal_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
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

    if (line.find("GaussianDistribution(") != 0) {
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
// 18. PRIVATE BATCH IMPLEMENTATION USING VECTOROPS
//
// CRITICAL SAFETY DOCUMENTATION FOR LOW-LEVEL SIMD OPERATIONS:
//
// These private implementation methods contain the actual "unsafe" raw pointer
// operations that enable high-performance SIMD vectorization. They are called only
// by the safe public wrapper methods above after proper validation.
//
// ⚠️  DANGER ZONE - RAW POINTER OPERATIONS ⚠️
//
// WHY THESE METHODS USE RAW POINTERS:
// 1. SIMD vectorization requires direct memory access with specific alignment
// 2. std::vector::data() returns raw pointers for optimal SIMD performance
// 3. SIMD intrinsics (AVX, SSE) operate on contiguous memory blocks
// 4. Zero abstraction penalty: direct hardware instruction mapping
//
// SAFETY PRECAUTIONS ENFORCED:
// 1. ✅ These methods are private - only callable from validated public interfaces
// 2. ✅ All callers perform bounds checking before invoking these methods
// 3. ✅ Memory alignment is handled by arch::simd::aligned_allocator
// 4. ✅ CPU feature detection prevents crashes on unsupported hardware
// 5. ✅ Scalar fallback path for small arrays or SIMD-unsupported systems
//
// SIMD OPERATION SAFETY GUARANTEES:
// - arch::simd::VectorOps methods internally validate pointer alignment
// - Vector operations are bounds-checked at the SIMD library level
// - Aligned memory allocation ensures optimal cache performance
// - Runtime CPU detection prevents using unsupported instructions
//
// ⚠️  DO NOT CALL THESE METHODS DIRECTLY ⚠️
// Always use the safe public interfaces like:
// - getProbabilityBatch() for thread-safe validation
// - getProbabilityBatchParallel() for C++20 std::span safety
// - getProbabilityBatchCacheAware() for additional safety checks
//
// FOR MAINTENANCE DEVELOPERS:
// When modifying these methods:
// 1. Ensure all array accesses use validated indices
// 2. Test both SIMD and scalar code paths
// 3. Verify alignment requirements are met
// 4. Update unit tests for both performance and correctness
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

    // Note: We need to use a temporary for erf since arch::simd::VectorOps::vector_erf
    // may not support in-place operation. This is unavoidable for the erf function.
    // However, we minimize allocations by reusing the results array for intermediate steps.

    // For systems where vector_erf supports in-place operations, this could be:
    // arch::simd::VectorOps::vector_erf(results, results, count);
    // But for safety, we allocate only one temporary array:
    std::vector<double, arch::simd::aligned_allocator<double>> erf_values(count);
    arch::simd::VectorOps::vector_erf(results, erf_values.data(), count);

    // Final computation: results = 0.5 * (1 + erf_values)
    // Step 1: results = 1 + erf_values
    arch::simd::VectorOps::scalar_add(erf_values.data(), detail::ONE, results, count);
    // Step 2: results = 0.5 * (1 + erf_values)
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

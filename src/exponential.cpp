#include "../include/distributions/exponential.h"

#include "../include/core/constants.h"
#include "../include/core/log_space_ops.h"
#include "../include/core/math_utils.h"
#include "../include/core/validation.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_execution.h"  // For parallel execution policies
#include "../include/platform/work_stealing_pool.h"  // For WorkStealingPool
// ParallelUtils functionality is provided by parallel_execution.h
#include "../include/core/dispatch_utils.h"  // For DispatchUtils::autoDispatch
#include "../include/core/threshold_constants.h"
#include "../include/platform/thread_pool.h"  // For ThreadPool

#include <algorithm>
#include <cmath>
#include <execution>   // For parallel algorithms
#include <functional>  // For std::plus and std::divides
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>  // C++20 ranges
#include <sstream>
#include <vector>

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

ExponentialDistribution::ExponentialDistribution(ExponentialDistribution&& other)
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    other.lambda_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

ExponentialDistribution& ExponentialDistribution::operator=(
    ExponentialDistribution&& other) noexcept {
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
                lambda_ = other.lambda_;
                other.lambda_ = detail::ONE;
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
            [[maybe_unused]] double temp_lambda = lambda_;

            // Atomic-like exchange (single assignment is atomic for built-in types)
            lambda_ = other.lambda_;
            other.lambda_ = detail::ONE;

            // Cache invalidation was already done atomically above
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
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

    return VoidResult::ok(true);
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

    return VoidResult::ok(true);
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
    if (!std::ranges::all_of(values, [](double value) { return value > detail::ZERO_DOUBLE; })) {
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
    if (arch::shouldUseDistributionParallel("exponential", "batch_fit", num_datasets)) {
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

void ExponentialDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
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

std::pair<double, double> ExponentialDistribution::confidenceIntervalRate(
    const std::vector<double>& data, double confidence_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    if (confidence_level <= detail::ZERO_DOUBLE || confidence_level >= detail::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);

    // For exponential distribution, confidence interval for λ using chi-squared distribution
    // The exact statistic is: 2n*X̄*λ ~ χ²(2n), where X̄ is the sample mean
    // Rearranging: λ ~ χ²(2n) / (2n*X̄) = χ²(2n) / (2*ΣXᵢ)
    // For confidence interval: P(χ²_{α/2,2n} < 2n*X̄*λ < χ²_{1-α/2,2n}) = confidence_level
    const double alpha = detail::ONE - confidence_level;
    const double dof = detail::TWO * static_cast<double>(n);

    // Get chi-squared quantiles - note the order for proper bounds
    const double chi_lower = detail::inverse_chi_squared_cdf(alpha * detail::HALF, dof);
    const double chi_upper =
        detail::inverse_chi_squared_cdf(detail::ONE - alpha * detail::HALF, dof);

    // Transform to rate parameter confidence interval
    // λ_lower = χ²{α/2,2n} / (2*ΣXᵢ), λ_upper = χ²{1-α/2,2n} / (2*ΣXᵢ)
    const double lambda_lower = chi_lower / (detail::TWO * sample_sum);
    const double lambda_upper = chi_upper / (detail::TWO * sample_sum);

    return {lambda_lower, lambda_upper};
}

std::pair<double, double> ExponentialDistribution::confidenceIntervalScale(
    const std::vector<double>& data, double confidence_level) {
    // Get rate parameter confidence interval
    const auto [lambda_lower, lambda_upper] = confidenceIntervalRate(data, confidence_level);

    // Transform to scale parameter (reciprocal relationship)
    const double scale_lower = detail::ONE / lambda_upper;
    const double scale_upper = detail::ONE / lambda_lower;

    return {scale_lower, scale_upper};
}

std::tuple<double, double, bool> ExponentialDistribution::likelihoodRatioTest(
    const std::vector<double>& data, double null_lambda, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    if (null_lambda <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Null hypothesis lambda must be positive");
    }

    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);
    const double sample_mean = sample_sum / static_cast<double>(n);
    const double mle_lambda = detail::ONE / sample_mean;

    // Log-likelihood under null hypothesis: n*ln(λ₀) - λ₀*Σxᵢ
    const double log_likelihood_null =
        static_cast<double>(n) * std::log(null_lambda) - null_lambda * sample_sum;

    // Log-likelihood under alternative (MLE): n*ln(λ̂) - λ̂*Σxᵢ = n*ln(λ̂) - n
    const double log_likelihood_alt =
        static_cast<double>(n) * std::log(mle_lambda) - static_cast<double>(n);

    // Likelihood ratio statistic: -2ln(Λ) = 2(ℓ(λ̂) - ℓ(λ₀))
    const double lr_statistic = detail::TWO * (log_likelihood_alt - log_likelihood_null);

    // Under H₀: LR ~ χ²(1)
    const double p_value = detail::ONE - detail::chi_squared_cdf(lr_statistic, detail::ONE);
    const bool reject_null = p_value < alpha;

    return {lr_statistic, p_value, reject_null};
}

std::pair<double, double> ExponentialDistribution::bayesianEstimation(
    const std::vector<double>& data, double prior_shape, double prior_rate) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    if (prior_shape <= detail::ZERO_DOUBLE || prior_rate <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Prior parameters must be positive");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);

    // Posterior parameters for Gamma(α, β) conjugate prior
    // Prior: λ ~ Gamma(α, β)
    // Likelihood: xᵢ ~ Exponential(λ)
    // Posterior: λ|x ~ Gamma(α + n, β + Σxᵢ)
    const double posterior_shape = prior_shape + static_cast<double>(n);
    const double posterior_rate = prior_rate + sample_sum;

    return {posterior_shape, posterior_rate};
}

std::pair<double, double> ExponentialDistribution::bayesianCredibleInterval(
    const std::vector<double>& data, double credibility_level, double prior_shape,
    double prior_rate) {
    if (credibility_level <= detail::ZERO_DOUBLE || credibility_level >= detail::ONE) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }

    // Get posterior parameters
    const auto [post_shape, post_rate] = bayesianEstimation(data, prior_shape, prior_rate);

    // Calculate credible interval from posterior Gamma distribution
    // For now, use a simple approximation - implement proper gamma quantile later
    [[maybe_unused]] const double alpha = detail::ONE - credibility_level;
    const double mean = post_shape / post_rate;
    const double std_dev = std::sqrt(post_shape) / post_rate;
    const double z_alpha_2 = detail::ONE + detail::HALF;  // Approximate normal quantile
    const double lower_quantile = mean - z_alpha_2 * std_dev;
    const double upper_quantile = mean + z_alpha_2 * std_dev;

    return {lower_quantile, upper_quantile};
}

double ExponentialDistribution::robustEstimation(const std::vector<double>& data,
                                                 const std::string& estimator_type,
                                                 double trim_proportion) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    if (trim_proportion < detail::ZERO_DOUBLE || trim_proportion >= detail::HALF) {
        throw std::invalid_argument("Trim proportion must be between 0 and detail::HALF");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);

    const std::size_t n = sorted_data.size();
    const std::size_t trim_count =
        static_cast<std::size_t>(std::floor(trim_proportion * static_cast<double>(n)));

    if (estimator_type == "winsorized") {
        // Winsorized estimation: replace extreme values with boundary values
        if (trim_count > 0) {
            const double lower_bound = sorted_data[trim_count];
            const double upper_bound = sorted_data[n - detail::ONE_INT - trim_count];

            for (std::size_t i = 0; i < trim_count; ++i) {
                sorted_data[i] = lower_bound;
                sorted_data[n - detail::ONE_INT - i] = upper_bound;
            }
        }
    } else if (estimator_type == "trimmed") {
        // Trimmed estimation: remove extreme values
        if (trim_count > 0) {
            sorted_data.erase(sorted_data.begin(),
                              sorted_data.begin() + static_cast<std::ptrdiff_t>(trim_count));
            sorted_data.erase(sorted_data.end() - static_cast<std::ptrdiff_t>(trim_count),
                              sorted_data.end());
        }
    } else {
        throw std::invalid_argument("Estimator type must be 'winsorized' or 'trimmed'");
    }

    if (sorted_data.empty()) {
        throw std::runtime_error("No data remaining after trimming");
    }

    // Calculate robust mean
    const double robust_sum =
        std::accumulate(sorted_data.begin(), sorted_data.end(), detail::ZERO_DOUBLE);
    const double robust_mean = robust_sum / static_cast<double>(sorted_data.size());

    return detail::ONE / robust_mean;
}

double ExponentialDistribution::methodOfMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    // For exponential distribution: E[X] = 1/λ, so λ = 1/sample_mean
    const double sample_sum = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);
    const double sample_mean = sample_sum / static_cast<double>(data.size());

    return detail::ONE / sample_mean;
}

double ExponentialDistribution::lMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    // For exponential distribution, L₁ = mean = 1/λ
    // So this is equivalent to method of moments for exponential
    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);

    const std::size_t n = sorted_data.size();
    double l1 = detail::ZERO_DOUBLE;  // First L-moment (mean)

    // Calculate L₁ using order statistics
    for (std::size_t i = 0; i < n; ++i) {
        l1 += sorted_data[i];
    }
    l1 /= static_cast<double>(n);

    return detail::ONE / l1;
}

std::tuple<double, double, bool> ExponentialDistribution::coefficientOfVariationTest(
    const std::vector<double>& data, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const size_t n = data.size();
    if (n < 2) {
        throw std::invalid_argument(
            "At least 2 data points required for coefficient of variation test");
    }

    // Calculate sample mean and sample standard deviation
    const double sum = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);
    const double sample_mean = sum / static_cast<double>(n);

    // Calculate sample variance (unbiased estimator)
    double sum_squared_deviations = detail::ZERO_DOUBLE;
    for (double value : data) {
        const double deviation = value - sample_mean;
        sum_squared_deviations += deviation * deviation;
    }
    const double sample_variance = sum_squared_deviations / static_cast<double>(n - 1);
    const double sample_std_dev = std::sqrt(sample_variance);

    // Calculate coefficient of variation
    const double cv = sample_std_dev / sample_mean;

    // For exponential distribution, the theoretical CV = 1
    // Test statistic: how far the observed CV is from 1
    const double cv_statistic = std::abs(cv - detail::ONE);

    // For large n, the CV of exponential follows approximately normal distribution
    // with mean = 1 and variance ≈ 1/n (asymptotic result)
    const double cv_std_error = detail::ONE / std::sqrt(static_cast<double>(n));
    const double z_statistic = cv_statistic / cv_std_error;

    // Two-tailed test p-value using normal approximation
    const double p_value = detail::TWO * (detail::ONE - detail::normal_cdf(z_statistic));

    const bool reject_null = p_value < alpha;

    return {cv_statistic, p_value, reject_null};
}

//==============================================================================
// 8. GOODNESS-OF-FIT TESTS
//==============================================================================

std::tuple<double, double, bool> ExponentialDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data, const ExponentialDistribution& distribution, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    // Use the overflow-safe KS statistic calculation from math_utils
    double ks_statistic = detail::calculate_ks_statistic(data, distribution);

    // Asymptotic p-value approximation for KS test
    // P-value ≈ 2 * exp(-2 * n * D²) for large n
    const double n_double = static_cast<double>(data.size());
    const double p_value_approx =
        detail::TWO * std::exp(-detail::TWO * n_double * ks_statistic * ks_statistic);

    // Clamp p-value to [0, 1]
    const double p_value = std::min(detail::ONE, std::max(detail::ZERO_DOUBLE, p_value_approx));

    const bool reject_null = p_value < alpha;

    return {ks_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> ExponentialDistribution::andersonDarlingTest(
    const std::vector<double>& data, const ExponentialDistribution& distribution, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    // Use the overflow-safe AD statistic calculation from math_utils
    double ad_statistic = detail::calculate_ad_statistic(data, distribution);

    // Adjust for exponential distribution (modification for known parameters)
    // For exponential distribution with estimated parameter, adjust the statistic
    const size_t n = data.size();
    const double n_double = static_cast<double>(n);
    const double ad_adjusted = ad_statistic * (detail::ONE + 0.6 / n_double);

    // Improved p-value approximation for exponential distribution Anderson-Darling test
    // Based on D'Agostino and Stephens (1986) formulas for exponential distribution
    // with enhanced handling for large statistics
    double p_value;
    if (ad_adjusted < detail::SMALL_EFFECT) {
        p_value = detail::ONE -
                  std::exp(-13.436 + 101.14 * ad_adjusted - 223.73 * ad_adjusted * ad_adjusted);
    } else if (ad_adjusted < 0.34) {
        p_value = detail::ONE -
                  std::exp(-8.318 + 42.796 * ad_adjusted - 59.938 * ad_adjusted * ad_adjusted);
    } else if (ad_adjusted < 0.6) {
        p_value = std::exp(0.9177 - 4.279 * ad_adjusted - 1.38 * ad_adjusted * ad_adjusted);
    } else if (ad_adjusted < detail::TWO) {
        p_value = std::exp(1.2937 - 5.709 * ad_adjusted + 0.0186 * ad_adjusted * ad_adjusted);
    } else {
        // For very large AD statistics, p-value should be very small (close to 0)
        // Use asymptotic approximation for extreme values
        p_value = std::exp(-ad_adjusted * detail::TWO);
    }

    // Clamp p-value to [0, 1]
    p_value = std::min(detail::ONE, std::max(detail::ZERO_DOUBLE, p_value));

    const bool reject_null = p_value < alpha;

    return {ad_adjusted, p_value, reject_null};
}

//==============================================================================
// 9. CROSS-VALIDATION METHODS
//==============================================================================

std::vector<std::tuple<double, double, double>> ExponentialDistribution::kFoldCrossValidation(
    const std::vector<double>& data, int k, unsigned int random_seed) {
    if (data.size() < static_cast<size_t>(k)) {
        throw std::invalid_argument("Data size must be at least k for k-fold cross-validation");
    }
    if (k <= 1) {
        throw std::invalid_argument("Number of folds k must be greater than 1");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
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

        // Fit model on training data (MLE estimation)
        const double training_sum =
            std::accumulate(training_data.begin(), training_data.end(), detail::ZERO_DOUBLE);
        const double training_mean = training_sum / static_cast<double>(training_data.size());
        const double fitted_rate = detail::ONE / training_mean;

        ExponentialDistribution fitted_model(fitted_rate);

        // Evaluate on validation data
        std::vector<double> absolute_errors;
        std::vector<double> squared_errors;
        double log_likelihood = detail::ZERO_DOUBLE;

        absolute_errors.reserve(validation_data.size());
        squared_errors.reserve(validation_data.size());

        // Calculate prediction errors and log-likelihood
        for (double val : validation_data) {
            // For exponential distribution, the "prediction" is the mean (1/λ)
            const double predicted_mean = detail::ONE / fitted_rate;

            const double absolute_error = std::abs(val - predicted_mean);
            const double squared_error = (val - predicted_mean) * (val - predicted_mean);

            absolute_errors.push_back(absolute_error);
            squared_errors.push_back(squared_error);

            log_likelihood += fitted_model.getLogProbability(val);
        }

        // Calculate MAE and RMSE
        const double mae =
            std::accumulate(absolute_errors.begin(), absolute_errors.end(), detail::ZERO_DOUBLE) /
            static_cast<double>(absolute_errors.size());
        const double mse =
            std::accumulate(squared_errors.begin(), squared_errors.end(), detail::ZERO_DOUBLE) /
            static_cast<double>(squared_errors.size());
        const double rmse = std::sqrt(mse);

        results.emplace_back(mae, rmse, log_likelihood);
    }

    return results;
}

std::tuple<double, double, double> ExponentialDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    if (data.size() < 3) {
        throw std::invalid_argument("At least 3 data points required for LOOCV");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const size_t n = data.size();
    std::vector<double> absolute_errors;
    std::vector<double> squared_errors;
    double total_log_likelihood = detail::ZERO_DOUBLE;

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

        // Fit model on training data (MLE estimation)
        const double training_sum =
            std::accumulate(training_data.begin(), training_data.end(), detail::ZERO_DOUBLE);
        const double training_mean = training_sum / static_cast<double>(training_data.size());
        const double fitted_rate = detail::ONE / training_mean;

        ExponentialDistribution fitted_model(fitted_rate);

        // Evaluate on left-out point
        // For exponential distribution, the "prediction" is the mean (1/λ)
        const double predicted_mean = detail::ONE / fitted_rate;
        const double actual_value = data[i];

        const double absolute_error = std::abs(actual_value - predicted_mean);
        const double squared_error =
            (actual_value - predicted_mean) * (actual_value - predicted_mean);

        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);

        total_log_likelihood += fitted_model.getLogProbability(actual_value);
    }

    // Calculate summary statistics
    const double mean_absolute_error =
        std::accumulate(absolute_errors.begin(), absolute_errors.end(), detail::ZERO_DOUBLE) /
        static_cast<double>(n);
    const double mean_squared_error =
        std::accumulate(squared_errors.begin(), squared_errors.end(), detail::ZERO_DOUBLE) /
        static_cast<double>(n);
    const double root_mean_squared_error = std::sqrt(mean_squared_error);

    return {mean_absolute_error, root_mean_squared_error, total_log_likelihood};
}

//==============================================================================
// 10. INFORMATION CRITERIA
//==============================================================================

std::tuple<double, double, double, double> ExponentialDistribution::computeInformationCriteria(
    const std::vector<double>& data, const ExponentialDistribution& fitted_distribution) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const double n = static_cast<double>(data.size());
    const int k = 1;  // Exponential distribution has 1 parameter (λ)

    // Calculate log-likelihood
    double log_likelihood = detail::ZERO_DOUBLE;
    for (double val : data) {
        log_likelihood += fitted_distribution.getLogProbability(val);
    }

    // Compute information criteria
    const double aic = detail::TWO * static_cast<double>(k) - detail::TWO * log_likelihood;
    const double bic = std::log(n) * static_cast<double>(k) - detail::TWO * log_likelihood;

    // AICc (corrected AIC for small sample sizes)
    double aicc;
    if (n - static_cast<double>(k) - detail::ONE > detail::ZERO_DOUBLE) {
        aicc =
            aic + (detail::TWO * static_cast<double>(k) * (static_cast<double>(k) + detail::ONE)) /
                      (n - static_cast<double>(k) - detail::ONE);
    } else {
        aicc = std::numeric_limits<double>::infinity();  // Undefined for small samples
    }

    return {aic, bic, aicc, log_likelihood};
}

//==============================================================================
// 11. BOOTSTRAP METHODS
//==============================================================================

std::pair<double, double> ExponentialDistribution::bootstrapParameterConfidenceIntervals(
    const std::vector<double>& data, double confidence_level, int n_bootstrap,
    unsigned int random_seed) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= detail::ZERO_DOUBLE || confidence_level >= detail::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    if (n_bootstrap <= 0) {
        throw std::invalid_argument("Number of bootstrap samples must be positive");
    }

    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > detail::ZERO_DOUBLE; })) {
        throw std::invalid_argument(
            "All data values must be positive for exponential distribution");
    }

    const size_t n = data.size();
    std::vector<double> bootstrap_rates;
    bootstrap_rates.reserve(static_cast<std::size_t>(n_bootstrap));

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

        // Estimate rate parameter for bootstrap sample (MLE)
        const double bootstrap_sum =
            std::accumulate(bootstrap_sample.begin(), bootstrap_sample.end(), detail::ZERO_DOUBLE);
        const double bootstrap_mean = bootstrap_sum / static_cast<double>(bootstrap_sample.size());
        const double bootstrap_rate = detail::ONE / bootstrap_mean;

        bootstrap_rates.push_back(bootstrap_rate);
    }

    // Sort for quantile calculation
    std::sort(bootstrap_rates.begin(), bootstrap_rates.end());

    // Calculate confidence intervals using percentile method
    const double alpha = detail::ONE - confidence_level;
    const double lower_percentile = alpha * detail::HALF;
    const double upper_percentile = detail::ONE - alpha * detail::HALF;

    const size_t lower_idx = static_cast<size_t>(lower_percentile * (n_bootstrap - 1));
    const size_t upper_idx = static_cast<size_t>(upper_percentile * (n_bootstrap - 1));

    return {bootstrap_rates[lower_idx], bootstrap_rates[upper_idx]};
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

// Note: Most distribution-specific utility methods are implemented inline in the header for
// performance Only complex methods requiring implementation are included here

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void ExponentialDistribution::getProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint,
        detail::DistributionTraits<ExponentialDistribution>::distType(),
        detail::DistributionTraits<ExponentialDistribution>::complexity(),
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
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: forwards to work-stealing until GPU implementation available
            // NOTE: GPU acceleration not yet implemented - using work-stealing for optimal CPU
            // performance
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

            // Cache parameters for thread-safe GPU-accelerated (work-stealing fallback) access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing (GPU fallback)
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
        *this, values, results, hint,
        detail::DistributionTraits<ExponentialDistribution>::distType(),
        detail::DistributionTraits<ExponentialDistribution>::complexity(),
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
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: forwards to work-stealing until GPU implementation available
            // NOTE: GPU acceleration not yet implemented - using work-stealing for optimal CPU
            // performance
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

            // Cache parameters for thread-safe GPU-accelerated (work-stealing fallback) access
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing (GPU fallback)
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
        *this, values, results, hint,
        detail::DistributionTraits<ExponentialDistribution>::distType(),
        detail::DistributionTraits<ExponentialDistribution>::complexity(),
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
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: forwards to work-stealing until GPU implementation available
            // NOTE: GPU acceleration not yet implemented - using work-stealing for optimal CPU
            // performance
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

            // Cache parameters for thread-safe GPU-accelerated (work-stealing fallback) access
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing (GPU fallback)
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

void ExponentialDistribution::getProbabilityWithStrategy(std::span<const double> values,
                                                         std::span<double> results,
                                                         detail::Strategy strategy) const {
    // GPU acceleration fallback - GPU implementation not yet available, use optimal CPU strategy
    if (strategy == detail::Strategy::GPU_ACCELERATED) {
        strategy = detail::Strategy::WORK_STEALING;
    }

    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
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
            // Parallel-SIMD lambda: WithStrategy power user method - execute parallel directly
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

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
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
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: forwards to work-stealing until GPU implementation available
            // NOTE: GPU acceleration not yet implemented - using work-stealing for optimal CPU
            // performance
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

            // Cache parameters for thread-safe GPU-accelerated (work-stealing fallback) access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing (GPU fallback)
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

void ExponentialDistribution::getLogProbabilityWithStrategy(std::span<const double> values,
                                                            std::span<double> results,
                                                            detail::Strategy strategy) const {
    // GPU acceleration fallback - GPU implementation not yet available, use optimal CPU strategy
    if (strategy == detail::Strategy::GPU_ACCELERATED) {
        strategy = detail::Strategy::WORK_STEALING;
    }

    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
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

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
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

            // Cache parameters for thread-safe work-stealing processing
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
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: forwards to work-stealing until GPU implementation available
            // NOTE: GPU acceleration not yet implemented - using work-stealing for optimal CPU
            // performance
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

            // Cache parameters for thread-safe GPU-accelerated (work-stealing fallback) processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing (GPU fallback)
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

void ExponentialDistribution::getCumulativeProbabilityWithStrategy(
    std::span<const double> values, std::span<double> results, detail::Strategy strategy) const {
    // GPU acceleration fallback - GPU implementation not yet available, use optimal CPU strategy
    if (strategy == detail::Strategy::GPU_ACCELERATED) {
        strategy = detail::Strategy::WORK_STEALING;
    }

    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
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

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
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

            // Cache parameters for thread-safe work-stealing processing
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
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: forwards to work-stealing until GPU implementation available
            // NOTE: GPU acceleration not yet implemented - using work-stealing for optimal CPU
            // performance
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

            // Cache parameters for thread-safe GPU-accelerated (work-stealing fallback) processing
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing (GPU fallback)
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
    if (token.find("ExponentialDistribution(") != 0) {
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
//==============================================================================

void ExponentialDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, size_t count, double cached_lambda,
    double cached_neg_lambda) const noexcept {
    for (size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else {
            // Fast path check for unit rate (λ = 1)
            if (std::abs(cached_lambda - detail::ONE) <= detail::DEFAULT_TOLERANCE) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, size_t count, double cached_log_lambda,
    double cached_neg_lambda) const noexcept {
    for (size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x < detail::ZERO_DOUBLE) {
            results[i] = detail::NEGATIVE_INFINITY;
        } else {
            // Fast path check for unit rate (λ = 1)
            if (std::abs(cached_log_lambda - detail::ZERO_DOUBLE) <= detail::DEFAULT_TOLERANCE) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, size_t count, double cached_neg_lambda) const noexcept {
    for (size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else {
            // Fast path check for unit rate (λ = 1)
            if (std::abs(cached_neg_lambda + detail::ONE) <= detail::DEFAULT_TOLERANCE) {
                results[i] = detail::ONE - std::exp(-x);
            } else {
                results[i] = detail::ONE - std::exp(cached_neg_lambda * x);
            }
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

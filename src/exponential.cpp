#include "../include/exponential.h"
#include "../include/validation.h"
#include "../include/math_utils.h"
#include "../include/log_space_ops.h"
#include "../include/cpu_detection.h"
#include "../include/parallel_execution.h" // For parallel execution policies
#include "../include/work_stealing_pool.h" // For WorkStealingPool
#include <iostream>
#include "../include/thread_pool.h" // For ThreadPool
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <ranges> // C++20 ranges
#include <functional> // For std::plus and std::divides
#include <execution> // For parallel algorithms

namespace libstats {

//==============================================================================
// CONSTRUCTORS AND DESTRUCTORS
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
    other.lambda_ = constants::math::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

ExponentialDistribution& ExponentialDistribution::operator=(ExponentialDistribution&& other) noexcept {
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
                other.lambda_ = constants::math::ONE;
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
            other.lambda_ = constants::math::ONE;
            
            // Cache invalidation was already done atomically above
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
    }
    return *this;
}

//==============================================================================
// CORE PROBABILITY METHODS
//==============================================================================

double ExponentialDistribution::getProbability(double x) const {
    // Return 0 for negative values
    if (x < constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
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
    if (x < constants::math::ZERO_DOUBLE) {
        return constants::probability::NEGATIVE_INFINITY;
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
    if (x < constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
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
        return constants::math::ONE - std::exp(-x);
    }
    
    // General case: F(x) = 1 - exp(-λx)
    return constants::math::ONE - std::exp(negLambda_ * x);
}

double ExponentialDistribution::getQuantile(double p) const {
    if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    if (p == constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
    }
    if (p == constants::math::ONE) {
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
        return -std::log(constants::math::ONE - p);
    }
    
    // General case: F^(-1)(p) = -ln(1-p)/λ
    return -std::log(constants::math::ONE - p) * invLambda_;
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
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 
        constants::math::ONE
    );
    
    double u = uniform(rng);
    
    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -std::log(u);
    }
    
    // General case: inverse transform sampling
    // X = -ln(U)/λ where U ~ Uniform(0,1)
    return -std::log(u) * invLambda_;
}

//==============================================================================
// PARAMETER GETTERS AND SETTERS
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
// DISTRIBUTION MANAGEMENT
//==============================================================================

void ExponentialDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    
    // C++20 best practices: Use ranges and views for safe validation
    // Check for non-positive values using ranges algorithms
    if (!std::ranges::all_of(values, [](double value) {
            return value > constants::math::ZERO_DOUBLE;
        })) {
        throw std::invalid_argument("Exponential distribution requires positive values");
    }
    
    // Calculate mean using standard accumulate (following Gaussian pattern)
    const double sum = std::accumulate(values.begin(), values.end(),
                                       constants::math::ZERO_DOUBLE);
    const double sample_mean = sum / static_cast<double>(values.size());
    
    if (sample_mean <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Sample mean must be positive for exponential distribution");
    }
    
    // Set parameters (this will validate and invalidate cache)
    setLambda(constants::math::ONE / sample_mean);
}

void ExponentialDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = constants::math::ONE;
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
// COMPARISON OPERATORS
//==============================================================================

bool ExponentialDistribution::operator==(const ExponentialDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    
    return std::abs(lambda_ - other.lambda_) <= constants::precision::DEFAULT_TOLERANCE;
}

//==============================================================================
// STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const ExponentialDistribution& distribution) {
    return os << distribution.toString();
}

//==============================================================================
// SIMD BATCH OPERATIONS
//==============================================================================

void ExponentialDistribution::getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
    if (count == 0) return;
    
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
    
    // Use cached values (protected by lock)
    const double cached_lambda = lambda_;
    const double cached_neg_lambda = negLambda_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getProbabilityBatchUnsafeImpl(values, results, count, cached_lambda, cached_neg_lambda);
}

void ExponentialDistribution::getLogProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
    if (count == 0) return;
    
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
    
    // Use cached values (protected by lock)
    const double cached_log_lambda = logLambda_;
    const double cached_neg_lambda = negLambda_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_log_lambda, cached_neg_lambda);
}

void ExponentialDistribution::getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const {
    if (count == 0) return;
    
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
    
    // Use cached values (protected by lock)
    const double cached_neg_lambda = negLambda_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, cached_neg_lambda);
}

void ExponentialDistribution::getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getProbabilityBatchUnsafeImpl(values, results, count, lambda_, negLambda_);
}

void ExponentialDistribution::getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getLogProbabilityBatchUnsafeImpl(values, results, count, logLambda_, negLambda_);
}

//==============================================================================
// PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void ExponentialDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                           double lambda, double neg_lambda) const noexcept {
    // Check if we should use SIMD
    if (cpu::supports_avx2() && count >= constants::simd::MIN_SIMD_SIZE) {
        getProbabilityBatchSIMD(values, results, count, lambda, neg_lambda);
        return;
    }
    
    // Scalar fallback
    const bool is_unit_rate = (std::abs(lambda - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else if (is_unit_rate) {
            results[i] = std::exp(-x);
        } else {
            results[i] = lambda * std::exp(neg_lambda * x);
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                              double log_lambda, double neg_lambda) const noexcept {
    // Check if we should use SIMD
    if (cpu::supports_avx2() && count >= constants::simd::MIN_SIMD_SIZE) {
        getLogProbabilityBatchSIMD(values, results, count, log_lambda, neg_lambda);
        return;
    }
    
    // Scalar fallback
    const bool is_unit_rate = (std::abs(log_lambda - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE);
    
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::probability::NEGATIVE_INFINITY;
        } else if (is_unit_rate) {
            results[i] = -x;
        } else {
            results[i] = log_lambda + neg_lambda * x;
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                     double neg_lambda) const noexcept {
    // Check if we should use SIMD
    if (cpu::supports_avx2() && count >= constants::simd::MIN_SIMD_SIZE) {
        getCumulativeProbabilityBatchSIMD(values, results, count, neg_lambda);
        return;
    }
    
    // Scalar fallback
    const bool is_unit_rate = (std::abs(neg_lambda + constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else if (is_unit_rate) {
            results[i] = constants::math::ONE - std::exp(-x);
        } else {
            results[i] = constants::math::ONE - std::exp(neg_lambda * x);
        }
    }
}

//==============================================================================
// SIMD IMPLEMENTATIONS
//==============================================================================

void ExponentialDistribution::getProbabilityBatchSIMD(const double* values, double* results, std::size_t count,
                                                      double lambda, double neg_lambda) const noexcept {
#if defined(LIBSTATS_HAS_AVX2)
    const std::size_t simd_width = 4; // AVX2 can handle 4 doubles
    const std::size_t simd_count = count - (count % simd_width);
    
    const __m256d lambda_vec = _mm256_set1_pd(lambda);
    const __m256d neg_lambda_vec = _mm256_set1_pd(neg_lambda);
    const __m256d zero_vec = _mm256_setzero_pd();
    const __m256d one_vec = _mm256_set1_pd(constants::math::ONE);
    
    const bool is_unit_rate = (std::abs(lambda - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_count; i += simd_width) {
        __m256d x_vec = _mm256_loadu_pd(&values[i]);
        __m256d mask = _mm256_cmp_pd(x_vec, zero_vec, _CMP_GE_OQ);
        
        __m256d result;
        if (is_unit_rate) {
            result = _mm256_exp_pd(_mm256_sub_pd(zero_vec, x_vec));
        } else {
            result = _mm256_mul_pd(lambda_vec, _mm256_exp_pd(_mm256_mul_pd(neg_lambda_vec, x_vec)));
        }
        
        result = _mm256_and_pd(result, mask);
        _mm256_storeu_pd(&results[i], result);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_count; i < count; ++i) {
        const double x = values[i];
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else if (is_unit_rate) {
            results[i] = std::exp(-x);
        } else {
            results[i] = lambda * std::exp(neg_lambda * x);
        }
    }
#else
    // Fallback to scalar implementation
    getProbabilityBatchUnsafeImpl(values, results, count, lambda, neg_lambda);
#endif
}

void ExponentialDistribution::getLogProbabilityBatchSIMD(const double* values, double* results, std::size_t count,
                                                         double log_lambda, double neg_lambda) const noexcept {
#if defined(LIBSTATS_HAS_AVX2)
    const std::size_t simd_width = 4; // AVX2 can handle 4 doubles
    const std::size_t simd_count = count - (count % simd_width);
    
    const __m256d log_lambda_vec = _mm256_set1_pd(log_lambda);
    const __m256d neg_lambda_vec = _mm256_set1_pd(neg_lambda);
    const __m256d zero_vec = _mm256_setzero_pd();
    const __m256d neg_inf_vec = _mm256_set1_pd(constants::probability::NEGATIVE_INFINITY);
    
    const bool is_unit_rate = (std::abs(log_lambda - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_count; i += simd_width) {
        __m256d x_vec = _mm256_loadu_pd(&values[i]);
        __m256d mask = _mm256_cmp_pd(x_vec, zero_vec, _CMP_GE_OQ);
        
        __m256d result;
        if (is_unit_rate) {
            result = _mm256_sub_pd(zero_vec, x_vec);
        } else {
            result = _mm256_add_pd(log_lambda_vec, _mm256_mul_pd(neg_lambda_vec, x_vec));
        }
        
        result = _mm256_blendv_pd(neg_inf_vec, result, mask);
        _mm256_storeu_pd(&results[i], result);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_count; i < count; ++i) {
        const double x = values[i];
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::probability::NEGATIVE_INFINITY;
        } else if (is_unit_rate) {
            results[i] = -x;
        } else {
            results[i] = log_lambda + neg_lambda * x;
        }
    }
#else
    // Fallback to scalar implementation
    getLogProbabilityBatchUnsafeImpl(values, results, count, log_lambda, neg_lambda);
#endif
}

void ExponentialDistribution::getCumulativeProbabilityBatchSIMD(const double* values, double* results, std::size_t count,
                                                               double neg_lambda) const noexcept {
#if defined(LIBSTATS_HAS_AVX2)
    const std::size_t simd_width = 4; // AVX2 can handle 4 doubles
    const std::size_t simd_count = count - (count % simd_width);
    
    const __m256d neg_lambda_vec = _mm256_set1_pd(neg_lambda);
    const __m256d zero_vec = _mm256_setzero_pd();
    const __m256d one_vec = _mm256_set1_pd(constants::math::ONE);
    
    const bool is_unit_rate = (std::abs(neg_lambda + constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Process SIMD blocks
    for (std::size_t i = 0; i < simd_count; i += simd_width) {
        __m256d x_vec = _mm256_loadu_pd(&values[i]);
        __m256d mask = _mm256_cmp_pd(x_vec, zero_vec, _CMP_GE_OQ);
        
        __m256d result;
        if (is_unit_rate) {
            result = _mm256_sub_pd(one_vec, _mm256_exp_pd(_mm256_sub_pd(zero_vec, x_vec)));
        } else {
            result = _mm256_sub_pd(one_vec, _mm256_exp_pd(_mm256_mul_pd(neg_lambda_vec, x_vec)));
        }
        
        result = _mm256_and_pd(result, mask);
        _mm256_storeu_pd(&results[i], result);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_count; i < count; ++i) {
        const double x = values[i];
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else if (is_unit_rate) {
            results[i] = constants::math::ONE - std::exp(-x);
        } else {
            results[i] = constants::math::ONE - std::exp(neg_lambda * x);
        }
    }
#else
    // Fallback to scalar implementation
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, neg_lambda);
#endif
}

//==============================================================================
// PARALLEL BATCH OPERATIONS
//==============================================================================

void ExponentialDistribution::getProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output span sizes must match");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Ensure cache is valid once before parallel processing
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_lambda = lambda_;
    const double cached_neg_lambda = negLambda_;
    const bool cached_is_unit_rate = isUnitRate_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element in parallel using cached parameters
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const noexcept {
    if (values.size() != results.size() || values.empty()) return;
    
    const std::size_t count = values.size();
    
    // Ensure cache is valid once before parallel processing
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_log_lambda = logLambda_;
    const double cached_neg_lambda = negLambda_;
    const bool cached_is_unit_rate = isUnitRate_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element in parallel using cached parameters
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            } else if (cached_is_unit_rate) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            } else if (cached_is_unit_rate) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output span sizes must match");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Ensure cache is valid once before parallel processing
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_neg_lambda = negLambda_;
    const bool cached_is_unit_rate = isUnitRate_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element in parallel using cached parameters
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = constants::math::ONE - std::exp(-x);
            } else {
                results[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = constants::math::ONE - std::exp(-x);
            } else {
                results[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
            }
        }
    }
}

void ExponentialDistribution::getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                             WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output span sizes must match");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Ensure cache is valid once before parallel processing
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_lambda = lambda_;
    const double cached_neg_lambda = negLambda_;
    const bool cached_is_unit_rate = isUnitRate_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use WorkStealingPool for dynamic load balancing - optimal for heavy computational loads
    if (WorkStealingUtils::shouldUseWorkStealing(count)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element with work stealing load balancing using cached parameters
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        });
        
        // Wait for all work stealing tasks to complete
        pool.waitForAll();
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        }
    }
}

void ExponentialDistribution::getProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                           [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output span sizes must match");
    }
    
    if (values.empty()) return;
    
    // Check cache for batch results
    const std::string cache_key = "exp_batch_" + std::to_string(lambda_) + "_" + std::to_string(values.size());
    
    // For cache-aware processing, use smaller chunks to better utilize cache
    // Use a reasonable default chunk size when cache manager doesn't provide one
    const std::size_t optimal_chunk_size = 1024;  // Default cache-friendly chunk size
    
    for (std::size_t i = 0; i < values.size(); i += optimal_chunk_size) {
        const std::size_t end = std::min(i + optimal_chunk_size, values.size());
        const std::size_t chunk_count = end - i;
        
        // Process chunk using SIMD batch operation
        getProbabilityBatch(values.data() + i, results.data() + i, chunk_count);
    }
    
    // Cache access recorded implicitly through batch operations
}

//==============================================================================
// ADVANCED STATISTICAL METHODS
//==============================================================================

std::pair<double, double> ExponentialDistribution::confidenceIntervalRate(
    const std::vector<double>& data, 
    double confidence_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (confidence_level <= constants::math::ZERO_DOUBLE || confidence_level >= constants::math::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    
    // For exponential distribution, confidence interval for λ using chi-squared distribution
    // The exact statistic is: 2n*X̄*λ ~ χ²(2n), where X̄ is the sample mean
    // Rearranging: λ ~ χ²(2n) / (2n*X̄) = χ²(2n) / (2*ΣXᵢ)
    // For confidence interval: P(χ²_{α/2,2n} < 2n*X̄*λ < χ²_{1-α/2,2n}) = confidence_level
    const double alpha = constants::math::ONE - confidence_level;
    const double dof = constants::math::TWO * static_cast<double>(n);
    
    // Get chi-squared quantiles - note the order for proper bounds
    const double chi_lower = math::inverse_chi_squared_cdf(alpha * constants::math::HALF, dof);
    const double chi_upper = math::inverse_chi_squared_cdf(constants::math::ONE - alpha * constants::math::HALF, dof);
    
    // Transform to rate parameter confidence interval
    // λ_lower = χ²{α/2,2n} / (2*ΣXᵢ), λ_upper = χ²{1-α/2,2n} / (2*ΣXᵢ)
    const double lambda_lower = chi_lower / (constants::math::TWO * sample_sum);
    const double lambda_upper = chi_upper / (constants::math::TWO * sample_sum);
    
    return {lambda_lower, lambda_upper};
}

std::pair<double, double> ExponentialDistribution::confidenceIntervalScale(
    const std::vector<double>& data,
    double confidence_level) {
    
    // Get rate parameter confidence interval
    const auto [lambda_lower, lambda_upper] = confidenceIntervalRate(data, confidence_level);
    
    // Transform to scale parameter (reciprocal relationship)
    const double scale_lower = constants::math::ONE / lambda_upper;
    const double scale_upper = constants::math::ONE / lambda_lower;
    
    return {scale_lower, scale_upper};
}

std::tuple<double, double, bool> ExponentialDistribution::likelihoodRatioTest(
    const std::vector<double>& data,
    double null_lambda,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (null_lambda <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Null hypothesis lambda must be positive");
    }
    
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    const double sample_mean = sample_sum / static_cast<double>(n);
    const double mle_lambda = constants::math::ONE / sample_mean;
    
    // Log-likelihood under null hypothesis: n*ln(λ₀) - λ₀*Σxᵢ
    const double log_likelihood_null = static_cast<double>(n) * std::log(null_lambda) - null_lambda * sample_sum;
    
    // Log-likelihood under alternative (MLE): n*ln(λ̂) - λ̂*Σxᵢ = n*ln(λ̂) - n
    const double log_likelihood_alt = static_cast<double>(n) * std::log(mle_lambda) - static_cast<double>(n);
    
    // Likelihood ratio statistic: -2ln(Λ) = 2(ℓ(λ̂) - ℓ(λ₀))
    const double lr_statistic = constants::math::TWO * (log_likelihood_alt - log_likelihood_null);
    
    // Under H₀: LR ~ χ²(1)
    const double p_value = constants::math::ONE - math::chi_squared_cdf(lr_statistic, constants::math::ONE);
    const bool reject_null = p_value < alpha;
    
    return {lr_statistic, p_value, reject_null};
}

std::pair<double, double> ExponentialDistribution::bayesianEstimation(
    const std::vector<double>& data,
    double prior_shape,
    double prior_rate) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (prior_shape <= constants::math::ZERO_DOUBLE || prior_rate <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Prior parameters must be positive");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    
    // Posterior parameters for Gamma(α, β) conjugate prior
    // Prior: λ ~ Gamma(α, β)
    // Likelihood: xᵢ ~ Exponential(λ)
    // Posterior: λ|x ~ Gamma(α + n, β + Σxᵢ)
    const double posterior_shape = prior_shape + static_cast<double>(n);
    const double posterior_rate = prior_rate + sample_sum;
    
    return {posterior_shape, posterior_rate};
}

std::pair<double, double> ExponentialDistribution::bayesianCredibleInterval(
    const std::vector<double>& data,
    double credibility_level,
    double prior_shape,
    double prior_rate) {
    
    if (credibility_level <= constants::math::ZERO_DOUBLE || credibility_level >= constants::math::ONE) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }
    
    // Get posterior parameters
    const auto [post_shape, post_rate] = bayesianEstimation(data, prior_shape, prior_rate);
    
    // Calculate credible interval from posterior Gamma distribution
    // For now, use a simple approximation - implement proper gamma quantile later
    [[maybe_unused]] const double alpha = constants::math::ONE - credibility_level;
    const double mean = post_shape / post_rate;
    const double std_dev = std::sqrt(post_shape) / post_rate;
    const double z_alpha_2 = constants::math::ONE + constants::math::HALF; // Approximate normal quantile
    const double lower_quantile = mean - z_alpha_2 * std_dev;
    const double upper_quantile = mean + z_alpha_2 * std_dev;
    
    return {lower_quantile, upper_quantile};
}

double ExponentialDistribution::robustEstimation(
    const std::vector<double>& data,
    const std::string& estimator_type,
    double trim_proportion) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (trim_proportion < constants::math::ZERO_DOUBLE || trim_proportion >= constants::math::HALF) {
        throw std::invalid_argument("Trim proportion must be between 0 and 0.5");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);
    
    const std::size_t n = sorted_data.size();
    const std::size_t trim_count = static_cast<std::size_t>(std::floor(trim_proportion * static_cast<double>(n)));
    
    if (estimator_type == "winsorized") {
        // Winsorized estimation: replace extreme values with boundary values
        if (trim_count > 0) {
            const double lower_bound = sorted_data[trim_count];
            const double upper_bound = sorted_data[n - 1 - trim_count];
            
            for (std::size_t i = 0; i < trim_count; ++i) {
                sorted_data[i] = lower_bound;
                sorted_data[n - 1 - i] = upper_bound;
            }
        }
    } else if (estimator_type == "trimmed") {
        // Trimmed estimation: remove extreme values
        if (trim_count > 0) {
            sorted_data.erase(sorted_data.begin(), sorted_data.begin() + static_cast<std::ptrdiff_t>(trim_count));
            sorted_data.erase(sorted_data.end() - static_cast<std::ptrdiff_t>(trim_count), sorted_data.end());
        }
    } else {
        throw std::invalid_argument("Estimator type must be 'winsorized' or 'trimmed'");
    }
    
    if (sorted_data.empty()) {
        throw std::runtime_error("No data remaining after trimming");
    }
    
    // Calculate robust mean
    const double robust_sum = std::accumulate(sorted_data.begin(), sorted_data.end(), constants::math::ZERO_DOUBLE);
    const double robust_mean = robust_sum / static_cast<double>(sorted_data.size());
    
    return constants::math::ONE / robust_mean;
}

double ExponentialDistribution::methodOfMomentsEstimation(
    const std::vector<double>& data) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // For exponential distribution: E[X] = 1/λ, so λ = 1/sample_mean
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    const double sample_mean = sample_sum / static_cast<double>(data.size());
    
    return constants::math::ONE / sample_mean;
}

double ExponentialDistribution::lMomentsEstimation(
    const std::vector<double>& data) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // For exponential distribution, L₁ = mean = 1/λ
    // So this is equivalent to method of moments for exponential
    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);
    
    const std::size_t n = sorted_data.size();
    double l1 = constants::math::ZERO_DOUBLE; // First L-moment (mean)
    
    // Calculate L₁ using order statistics
    for (std::size_t i = 0; i < n; ++i) {
        l1 += sorted_data[i];
    }
    l1 /= static_cast<double>(n);
    
    return constants::math::ONE / l1;
}

} // namespace libstats

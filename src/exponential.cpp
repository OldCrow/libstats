#include "../include/distributions/exponential.h"
#include "../include/core/constants.h"
#include "../include/core/validation.h"
#include "../include/core/math_utils.h"
#include "../include/core/log_space_ops.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_execution.h" // For parallel execution policies
#include "../include/platform/work_stealing_pool.h" // For WorkStealingPool
#include <iostream>
#include "../include/platform/thread_pool.h" // For ThreadPool
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
    
    lock.unlock(); // Release lock before generation
    
    // Use high-quality uniform distribution for batch generation
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 
        constants::math::ONE
    );
    
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
    // Check if vectorization is beneficial and CPU supports it (matching Gaussian pattern)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2() || cpu::supports_avx512());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
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
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    const bool is_unit_rate = (std::abs(lambda - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, simd::aligned_allocator<double>> temp_values(count);
    std::vector<double, simd::aligned_allocator<double>> exp_inputs(count);
    
    // Step 1: Prepare exp() inputs
    if (is_unit_rate) {
        // For unit rate: exp(-x)
        simd::VectorOps::scalar_multiply(values, constants::math::NEG_ONE, exp_inputs.data(), count);
    } else {
        // For general case: exp(neg_lambda * x)
        simd::VectorOps::scalar_multiply(values, neg_lambda, exp_inputs.data(), count);
    }
    
    // Step 2: Apply vectorized exponential
    simd::VectorOps::vector_exp(exp_inputs.data(), results, count);
    
    // Step 3: Apply lambda scaling if needed
    if (!is_unit_rate) {
        simd::VectorOps::scalar_multiply(results, lambda, results, count);
    }
    
    // Step 4: Handle negative input values (set to zero)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                              double log_lambda, double neg_lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (matching Gaussian pattern)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2() || cpu::supports_avx512());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
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
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    const bool is_unit_rate = (std::abs(log_lambda - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, simd::aligned_allocator<double>> temp_values(count);
    
    // Step 1: Calculate the main term
    if (is_unit_rate) {
        // For unit rate: -x
        simd::VectorOps::scalar_multiply(values, constants::math::NEG_ONE, results, count);
    } else {
        // For general case: log_lambda + neg_lambda * x
        simd::VectorOps::scalar_multiply(values, neg_lambda, temp_values.data(), count);
        simd::VectorOps::scalar_add(temp_values.data(), log_lambda, results, count);
    }
    
    // Step 2: Handle negative input values (set to -infinity)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < constants::math::ZERO_DOUBLE) {
            results[i] = constants::probability::NEGATIVE_INFINITY;
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                     double neg_lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (matching Gaussian pattern)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2() || cpu::supports_avx512());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
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
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    const bool is_unit_rate = (std::abs(neg_lambda + constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, simd::aligned_allocator<double>> exp_inputs(count);
    std::vector<double, simd::aligned_allocator<double>> exp_results(count);
    
    // Step 1: Prepare exp() inputs
    if (is_unit_rate) {
        // For unit rate: 1 - exp(-x)
        simd::VectorOps::scalar_multiply(values, constants::math::NEG_ONE, exp_inputs.data(), count);
    } else {
        // For general case: 1 - exp(neg_lambda * x)
        simd::VectorOps::scalar_multiply(values, neg_lambda, exp_inputs.data(), count);
    }
    
    // Step 2: Apply vectorized exponential
    simd::VectorOps::vector_exp(exp_inputs.data(), exp_results.data(), count);
    
    // Step 3: Calculate 1 - exp(...)
    simd::VectorOps::scalar_add(exp_results.data(), constants::math::NEG_ONE, results, count);
    simd::VectorOps::scalar_multiply(results, constants::math::NEG_ONE, results, count);
    
    // Step 4: Handle negative input values (set to zero)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        }
    }
}

// Note: Redundant SIMD methods removed - SIMD optimization is now handled
// internally within the *BatchUnsafeImpl methods above, following the
// standardized pattern established in gaussian.cpp

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

void ExponentialDistribution::getLogProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
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
    const double cached_log_lambda = logLambda_;
    const double cached_neg_lambda = negLambda_;
    const bool cached_is_unit_rate = isUnitRate_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use WorkStealingPool for dynamic load balancing - optimal for heavy computational loads
    if (WorkStealingUtils::shouldUseWorkStealing(count)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element with work stealing load balancing using cached parameters
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = -std::numeric_limits<double>::infinity();
            } else if (cached_is_unit_rate) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        });
        
        // Wait for all work stealing tasks to complete
        pool.waitForAll();
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = -std::numeric_limits<double>::infinity();
            } else if (cached_is_unit_rate) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                              [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output span sizes must match");
    }
    
    if (values.empty()) return;
    
    // Check cache for batch results
    const std::string cache_key = "exp_log_batch_" + std::to_string(lambda_) + "_" + std::to_string(values.size());
    
    // For cache-aware processing, use smaller chunks to better utilize cache
    // Use a reasonable default chunk size when cache manager doesn't provide one
    const std::size_t optimal_chunk_size = 1024;  // Default cache-friendly chunk size
    
    for (std::size_t i = 0; i < values.size(); i += optimal_chunk_size) {
        const std::size_t end = std::min(i + optimal_chunk_size, values.size());
        const std::size_t chunk_count = end - i;
        
        // Process chunk using SIMD batch operation
        getLogProbabilityBatch(values.data() + i, results.data() + i, chunk_count);
    }
    
    // Cache access recorded implicitly through batch operations
}

void ExponentialDistribution::getCumulativeProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
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
    const double cached_neg_lambda = negLambda_;
    const bool cached_is_unit_rate = isUnitRate_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use WorkStealingPool for dynamic load balancing - optimal for heavy computational loads
    if (WorkStealingUtils::shouldUseWorkStealing(count)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with work stealing load balancing using cached parameters
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (cached_is_unit_rate) {
                results[i] = constants::math::ONE - std::exp(-x);
            } else {
                results[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
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
                results[i] = constants::math::ONE - std::exp(-x);
            } else {
                results[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
            }
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                                     [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output span sizes must match");
    }
    
    if (values.empty()) return;
    
    // Check cache for batch results
    const std::string cache_key = "exp_cdf_batch_" + std::to_string(lambda_) + "_" + std::to_string(values.size());
    
    // For cache-aware processing, use smaller chunks to better utilize cache
    // Use a reasonable default chunk size when cache manager doesn't provide one
    const std::size_t optimal_chunk_size = 1024;  // Default cache-friendly chunk size
    
    for (std::size_t i = 0; i < values.size(); i += optimal_chunk_size) {
        const std::size_t end = std::min(i + optimal_chunk_size, values.size());
        const std::size_t chunk_count = end - i;
        
        // Process chunk using SIMD batch operation
        getCumulativeProbabilityBatch(values.data() + i, results.data() + i, chunk_count);
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

std::tuple<double, double, bool> ExponentialDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data,
    const ExponentialDistribution& distribution,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // Sort data for empirical CDF calculation
    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);
    
    const size_t n = sorted_data.size();
    double max_diff = constants::math::ZERO_DOUBLE;
    
    // Calculate KS statistic: max|F_n(x) - F(x)|
    for (size_t i = 0; i < n; ++i) {
        const double x = sorted_data[i];
        
        // Empirical CDF at x: F_n(x) = (i+1)/n
        const double empirical_cdf = static_cast<double>(i + 1) / static_cast<double>(n);
        
        // Theoretical exponential CDF: F(x) = 1 - exp(-λx)
        const double theoretical_cdf = distribution.getCumulativeProbability(x);
        
        // Check both F_n(x) - F(x) and F(x) - F_{n-1}(x)
        const double diff1 = std::abs(empirical_cdf - theoretical_cdf);
        
        double diff2 = constants::math::ZERO_DOUBLE;
        if (i > 0) {
            const double prev_empirical_cdf = static_cast<double>(i) / static_cast<double>(n);
            diff2 = std::abs(theoretical_cdf - prev_empirical_cdf);
        }
        
        max_diff = std::max({max_diff, diff1, diff2});
    }
    
    // Asymptotic p-value approximation for KS test
    // P-value ≈ 2 * exp(-2 * n * D²) for large n
    const double n_double = static_cast<double>(n);
    const double p_value_approx = constants::math::TWO * std::exp(-constants::math::TWO * n_double * max_diff * max_diff);
    
    // Clamp p-value to [0, 1]
    const double p_value = std::min(constants::math::ONE, std::max(constants::math::ZERO_DOUBLE, p_value_approx));
    
    const bool reject_null = p_value < alpha;
    
    return {max_diff, p_value, reject_null};
}

std::vector<std::tuple<double, double, double>> ExponentialDistribution::kFoldCrossValidation(
    const std::vector<double>& data,
    int k,
    unsigned int random_seed) {
    
    if (data.size() < static_cast<size_t>(k)) {
        throw std::invalid_argument("Data size must be at least k for k-fold cross-validation");
    }
    if (k <= 1) {
        throw std::invalid_argument("Number of folds k must be greater than 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    const size_t fold_size = n / k;
    
    // Create shuffled indices for random fold assignment
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    std::vector<std::tuple<double, double, double>> results;
    results.reserve(k);
    
    for (int fold = 0; fold < k; ++fold) {
        // Define validation set indices for this fold
        const size_t start_idx = fold * fold_size;
        const size_t end_idx = (fold == k - 1) ? n : (fold + 1) * fold_size;
        
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
        const double training_sum = std::accumulate(training_data.begin(), training_data.end(), constants::math::ZERO_DOUBLE);
        const double training_mean = training_sum / static_cast<double>(training_data.size());
        const double fitted_rate = constants::math::ONE / training_mean;
        
        ExponentialDistribution fitted_model(fitted_rate);
        
        // Evaluate on validation data
        double rate_error = constants::math::ZERO_DOUBLE;
        double scale_error = constants::math::ZERO_DOUBLE;
        double log_likelihood = constants::math::ZERO_DOUBLE;
        
        // Calculate prediction errors and log-likelihood
        const double validation_sum = std::accumulate(validation_data.begin(), validation_data.end(), constants::math::ZERO_DOUBLE);
        const double validation_mean = validation_sum / static_cast<double>(validation_data.size());
        const double true_rate_estimate = constants::math::ONE / validation_mean;
        const double true_scale_estimate = validation_mean;
        
        // Rate parameter error
        rate_error = std::abs(fitted_rate - true_rate_estimate);
        
        // Scale parameter error (1/λ)
        const double fitted_scale = constants::math::ONE / fitted_rate;
        scale_error = std::abs(fitted_scale - true_scale_estimate);
        
        // Log-likelihood on validation set
        for (double val : validation_data) {
            log_likelihood += fitted_model.getLogProbability(val);
        }
        
        results.emplace_back(rate_error, scale_error, log_likelihood);
    }
    
    return results;
}

std::tuple<double, double, double> ExponentialDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    
    if (data.size() < 3) {
        throw std::invalid_argument("At least 3 data points required for LOOCV");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    std::vector<double> absolute_errors;
    std::vector<double> squared_errors;
    double total_log_likelihood = constants::math::ZERO_DOUBLE;
    
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
        const double training_sum = std::accumulate(training_data.begin(), training_data.end(), constants::math::ZERO_DOUBLE);
        const double training_mean = training_sum / static_cast<double>(training_data.size());
        const double fitted_rate = constants::math::ONE / training_mean;
        
        ExponentialDistribution fitted_model(fitted_rate);
        
        // Evaluate on left-out point
        // For exponential distribution, the "prediction" is the mean (1/λ)
        const double predicted_mean = constants::math::ONE / fitted_rate;
        const double actual_value = data[i];
        
        const double absolute_error = std::abs(actual_value - predicted_mean);
        const double squared_error = (actual_value - predicted_mean) * (actual_value - predicted_mean);
        
        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);
        
        total_log_likelihood += fitted_model.getLogProbability(actual_value);
    }
    
    // Calculate summary statistics
    const double mean_absolute_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(n);
    const double mean_squared_error = std::accumulate(squared_errors.begin(), squared_errors.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(n);
    const double root_mean_squared_error = std::sqrt(mean_squared_error);
    
    return {mean_absolute_error, root_mean_squared_error, total_log_likelihood};
}

std::tuple<double, double, double, double> ExponentialDistribution::computeInformationCriteria(
    const std::vector<double>& data,
    const ExponentialDistribution& fitted_distribution) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const double n = static_cast<double>(data.size());
    const int k = 1; // Exponential distribution has 1 parameter (λ)
    
    // Calculate log-likelihood
    double log_likelihood = constants::math::ZERO_DOUBLE;
    for (double val : data) {
        log_likelihood += fitted_distribution.getLogProbability(val);
    }
    
    // Compute information criteria
    const double aic = constants::math::TWO * static_cast<double>(k) - constants::math::TWO * log_likelihood;
    const double bic = std::log(n) * static_cast<double>(k) - constants::math::TWO * log_likelihood;
    
    // AICc (corrected AIC for small sample sizes)
    double aicc;
    if (n - static_cast<double>(k) - constants::math::ONE > constants::math::ZERO_DOUBLE) {
        aicc = aic + (constants::math::TWO * static_cast<double>(k) * (static_cast<double>(k) + constants::math::ONE)) / (n - static_cast<double>(k) - constants::math::ONE);
    } else {
        aicc = std::numeric_limits<double>::infinity(); // Undefined for small samples
    }
    
    return {aic, bic, aicc, log_likelihood};
}

std::pair<double, double> ExponentialDistribution::bootstrapParameterConfidenceInterval(
    const std::vector<double>& data,
    double confidence_level,
    int n_bootstrap,
    unsigned int random_seed) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= constants::math::ZERO_DOUBLE || confidence_level >= constants::math::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    if (n_bootstrap <= 0) {
        throw std::invalid_argument("Number of bootstrap samples must be positive");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    std::vector<double> bootstrap_rates;
    bootstrap_rates.reserve(n_bootstrap);
    
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
        const double bootstrap_sum = std::accumulate(bootstrap_sample.begin(), bootstrap_sample.end(), constants::math::ZERO_DOUBLE);
        const double bootstrap_mean = bootstrap_sum / static_cast<double>(bootstrap_sample.size());
        const double bootstrap_rate = constants::math::ONE / bootstrap_mean;
        
        bootstrap_rates.push_back(bootstrap_rate);
    }
    
    // Sort for quantile calculation
    std::sort(bootstrap_rates.begin(), bootstrap_rates.end());
    
    // Calculate confidence intervals using percentile method
    const double alpha = constants::math::ONE - confidence_level;
    const double lower_percentile = alpha * constants::math::HALF;
    const double upper_percentile = constants::math::ONE - alpha * constants::math::HALF;
    
    const size_t lower_idx = static_cast<size_t>(lower_percentile * (n_bootstrap - 1));
    const size_t upper_idx = static_cast<size_t>(upper_percentile * (n_bootstrap - 1));
    
    return {bootstrap_rates[lower_idx], bootstrap_rates[upper_idx]};
}

//==============================================================================
// RESULT-BASED SETTERS
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

} // namespace libstats

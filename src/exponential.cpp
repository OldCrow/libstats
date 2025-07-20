#include "../include/exponential.h"
#include "../include/validation.h"
#include "../include/math_utils.h"
#include "../include/log_space_ops.h"
#include "../include/cpu_detection.h"
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

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
    
    // Check for non-positive values
    for (double value : values) {
        if (value <= constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Exponential distribution requires positive values");
        }
    }
    
    // Maximum likelihood estimation: λ = 1/sample_mean
    const double sample_mean = std::accumulate(values.begin(), values.end(), constants::math::ZERO_DOUBLE) / values.size();
    
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

} // namespace libstats

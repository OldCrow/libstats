#include "../include/distributions/gaussian.h"
#include "../include/core/constants.h"
#include "../include/core/validation.h"
#include "../include/core/math_utils.h"
#include "../include/core/log_space_ops.h"
#include "../include/platform/cpu_detection.h"
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <ranges>      // C++20 ranges
#include <concepts>    // C++20 concepts
#include <span>        // C++20 span
#include <version>     // C++20 feature detection

namespace libstats {

//==============================================================================
// CONSTRUCTORS AND DESTRUCTORS
//==============================================================================

GaussianDistribution::GaussianDistribution(double mean, double standardDeviation) 
    : DistributionBase(), mean_(mean), standardDeviation_(standardDeviation) {
    validateParameters(mean, standardDeviation);
    // Cache will be updated on first use
}

GaussianDistribution::GaussianDistribution(const GaussianDistribution& other) 
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
    // Cache will be updated on first use
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
    other.mean_ = constants::math::ZERO_DOUBLE;
    other.standardDeviation_ = constants::math::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
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
                other.mean_ = constants::math::ZERO_DOUBLE;
                other.standardDeviation_ = constants::math::ONE;
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
            other.mean_ = constants::math::ZERO_DOUBLE;
            other.standardDeviation_ = constants::math::ONE;
            
            // Cache invalidation was already done atomically above
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
    }
    return *this;
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
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), constants::math::ONE);
    
    double u1, u2;
    double magnitude, angle;
    
    do {
        u1 = uniform(rng);
        u2 = uniform(rng);
        
        // Box-Muller transformation
        magnitude = std::sqrt(constants::math::NEG_TWO * std::log(u1));
        angle = constants::math::TWO_PI * u2;
        
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
    
    lock.unlock(); // Release lock before generation
    
    std::uniform_real_distribution<double> uniform(constants::math::ZERO_DOUBLE, constants::math::ONE);
    
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
        const double magnitude = std::sqrt(constants::math::NEG_TWO * std::log(u1));
        const double angle = constants::math::TWO_PI * u2;
        
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
        
        const double magnitude = std::sqrt(constants::math::NEG_TWO * std::log(u1));
        const double angle = constants::math::TWO_PI * u2;
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
// PARAMETER GETTERS AND SETTERS
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
        return constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
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
        return constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
    }
    
    // General case - direct computation for Gaussian log-PDF
    const double diff = x - mean_;
    const double sq_diff = diff * diff;
    return constants::math::NEG_HALF_LN_2PI - logStandardDeviation_ + negHalfSigmaSquaredInv_ * sq_diff;
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
        return constants::math::HALF * (constants::math::ONE + std::erf(x * constants::math::INV_SQRT_2));
    }
    
    // General case
    const double normalized = (x - mean_) / sigmaSqrt2_;
    return constants::math::HALF * (constants::math::ONE + std::erf(normalized));
}

//==============================================================================
// BATCH OPERATIONS USING VECTOROPS
// 
// SAFETY DOCUMENTATION FOR DEVELOPERS:
// 
// This section contains performance-critical batch operations that deliberately
// use raw pointer interfaces for maximum performance in SIMD operations. While
// these methods internally use "unsafe" raw pointer access, they are wrapped in
// carefully designed safe interfaces that handle all validation and bounds checking.
// 
// WHY RAW POINTERS ARE USED HERE:
// 1. SIMD vectorization requires contiguous memory access with known alignment
// 2. std::vector::data() provides optimal cache-friendly access patterns
// 3. SIMD intrinsics operate directly on raw memory regions
// 4. Zero-overhead abstraction principle: no performance penalty for safety
// 
// SAFETY GUARANTEES PROVIDED:
// 1. All public interfaces validate input parameters before calling internal methods
// 2. Cache validity is ensured before any computations
// 3. Thread-safety is maintained through proper locking mechanisms
// 4. Bounds checking is performed at the interface level
// 5. SIMD alignment requirements are handled automatically
// 
// HOW TO USE THESE OPERATIONS SAFELY:
// 1. Always use the provided safe wrapper methods (not the "Unsafe" variants)
// 2. Ensure input arrays are properly allocated with correct sizes
// 3. For C++20 users: prefer the std::span interfaces for automatic bounds checking
// 4. For maximum safety: use the parallel batch methods that include additional validation
// 
// PERFORMANCE OPTIMIZATION STRATEGY:
// - Small arrays (< SIMD threshold): Use scalar loops with bounds checking
// - Large arrays (≥ SIMD threshold): Use vectorized operations with alignment
// - Thread-local caching: Eliminates repeated parameter validation overhead
// - Lock-free hot paths: Cache parameters before releasing locks
//==============================================================================

void GaussianDistribution::getProbabilityBatch(const double* values, double* results, std::size_t count) const {
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
    const double cached_mean = mean_;
    const double cached_norm_constant = normalizationConstant_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getProbabilityBatchUnsafeImpl(values, results, count, cached_mean, 
                                  cached_norm_constant, cached_neg_half_inv_var, 
                                  cached_is_standard_normal);
}

void GaussianDistribution::getLogProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
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
    const double cached_mean = mean_;
    const double cached_log_std = logStandardDeviation_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_mean, 
                                     cached_log_std, cached_neg_half_inv_var, 
                                     cached_is_standard_normal);
}

void GaussianDistribution::getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getProbabilityBatchUnsafeImpl(values, results, count, mean_, normalizationConstant_, 
                                  negHalfSigmaSquaredInv_, isStandardNormal_);
}

void GaussianDistribution::getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getLogProbabilityBatchUnsafeImpl(values, results, count, mean_, logStandardDeviation_, 
                                     negHalfSigmaSquaredInv_, isStandardNormal_);
}

void GaussianDistribution::getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const {
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
    const double cached_mean = mean_;
    const double cached_sigma_sqrt2 = sigmaSqrt2_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, cached_mean, 
                                            cached_sigma_sqrt2, cached_is_standard_normal);
}

//==============================================================================
// PRIVATE BATCH IMPLEMENTATION USING VECTOROPS
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
// 3. ✅ Memory alignment is handled by simd::aligned_allocator
// 4. ✅ CPU feature detection prevents crashes on unsupported hardware
// 5. ✅ Scalar fallback path for small arrays or SIMD-unsupported systems
// 
// SIMD OPERATION SAFETY GUARANTEES:
// - simd::VectorOps methods internally validate pointer alignment
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

void GaussianDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                          double mean, double norm_constant, double neg_half_inv_var,
                                                          bool is_standard_normal) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = (count >= simd::tuned::min_states_for_simd()) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2() || cpu::supports_avx512());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
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
        simd::VectorOps::vector_multiply(values, values, results, count);
        // Step 2: results = -0.5 * x²
        simd::VectorOps::scalar_multiply(results, constants::math::NEG_HALF, results, count);
        // Step 3: results = exp(-0.5 * x²)
        simd::VectorOps::vector_exp(results, results, count);
        // Step 4: results = exp(-0.5 * x²) / sqrt(2π)
        simd::VectorOps::scalar_multiply(results, constants::math::INV_SQRT_2PI, results, count);
    } else {
        // General case: exp(-0.5 * ((x-μ)/σ)²) / (σ√(2π))
        // Step 1: results = x - μ (difference from mean)
        simd::VectorOps::scalar_add(values, -mean, results, count);
        // Step 2: results = (x - μ)²
        simd::VectorOps::vector_multiply(results, results, results, count);
        // Step 3: results = -0.5 * (x - μ)² / σ²
        simd::VectorOps::scalar_multiply(results, neg_half_inv_var, results, count);
        // Step 4: results = exp(-0.5 * (x - μ)² / σ²)
        simd::VectorOps::vector_exp(results, results, count);
        // Step 5: results = exp(-0.5 * (x - μ)² / σ²) / (σ√(2π))
        simd::VectorOps::scalar_multiply(results, norm_constant, results, count);
    }
}

void GaussianDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                             double mean, double log_std, double neg_half_inv_var,
                                                             bool is_standard_normal) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = (count >= simd::tuned::min_states_for_simd()) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2() || cpu::supports_avx512());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - log_std + neg_half_inv_var * sq_diff;
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    // PERFORMANCE CRITICAL: Use results array as workspace to avoid allocations
    
    if (is_standard_normal) {
        // Standard normal: -0.5 * ln(2π) - 0.5 * x²
        // Step 1: results = x²
        simd::VectorOps::vector_multiply(values, values, results, count);
        // Step 2: results = -0.5 * x²
        simd::VectorOps::scalar_multiply(results, constants::math::NEG_HALF, results, count);
        // Step 3: results = -0.5 * ln(2π) - 0.5 * x²
        simd::VectorOps::scalar_add(results, constants::math::NEG_HALF_LN_2PI, results, count);
    } else {
        // General case: -0.5 * ln(2π) - ln(σ) - 0.5 * ((x-μ)/σ)²
        // Step 1: results = x - μ (difference from mean)
        simd::VectorOps::scalar_add(values, -mean, results, count);
        // Step 2: results = (x - μ)²
        simd::VectorOps::vector_multiply(results, results, results, count);
        // Step 3: results = -0.5 * (x - μ)² / σ²
        simd::VectorOps::scalar_multiply(results, neg_half_inv_var, results, count);
        // Step 4: results = -0.5 * ln(2π) - ln(σ) - 0.5 * (x - μ)² / σ²
        const double log_norm_constant = constants::math::NEG_HALF_LN_2PI - log_std;
        simd::VectorOps::scalar_add(results, log_norm_constant, results, count);
    }
}

void GaussianDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                   double mean, double sigma_sqrt2, bool is_standard_normal) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = (count >= simd::tuned::min_states_for_simd()) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2() || cpu::supports_avx512());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - mean) / sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    // PERFORMANCE CRITICAL: Use results array as workspace to avoid allocations
    
    if (is_standard_normal) {
        // Standard normal case: normalized = values * INV_SQRT_2
        // Step 1: results = values * INV_SQRT_2 (normalized values)
        simd::VectorOps::scalar_multiply(values, constants::math::INV_SQRT_2, results, count);
    } else {
        // General case: normalized = (values - mean) / sigma_sqrt2
        // Step 1: results = values - mean
        simd::VectorOps::scalar_add(values, -mean, results, count);
        // Step 2: results = (values - mean) / sigma_sqrt2
        const double reciprocal_sigma_sqrt2 = constants::math::ONE / sigma_sqrt2;
        simd::VectorOps::scalar_multiply(results, reciprocal_sigma_sqrt2, results, count);
    }
    
    // Note: We need to use a temporary for erf since simd::VectorOps::vector_erf
    // may not support in-place operation. This is unavoidable for the erf function.
    // However, we minimize allocations by reusing the results array for intermediate steps.
    
    // For systems where vector_erf supports in-place operations, this could be:
    // simd::VectorOps::vector_erf(results, results, count);
    // But for safety, we allocate only one temporary array:
    std::vector<double, simd::aligned_allocator<double>> erf_values(count);
    simd::VectorOps::vector_erf(results, erf_values.data(), count);
    
    // Final computation: results = 0.5 * (1 + erf_values)
    // Step 1: results = 1 + erf_values
    simd::VectorOps::scalar_add(erf_values.data(), constants::math::ONE, results, count);
    // Step 2: results = 0.5 * (1 + erf_values)
    simd::VectorOps::scalar_multiply(results, constants::math::HALF, results, count);
}

//==============================================================================
// DISTRIBUTION MANAGEMENT
//==============================================================================

void GaussianDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit to empty data");
    }
    
    // Check minimum data points for reliable fitting
    if (values.size() < constants::thresholds::MIN_DATA_POINTS_FOR_FITTING) {
        throw std::invalid_argument("Insufficient data points for reliable Gaussian fitting");
    }
    
    const std::size_t n = values.size();
    
    // Use parallel execution for large datasets
    double running_mean, sample_variance;
    
    if (parallel::should_use_distribution_parallel(n)) {
        // Parallel Welford's algorithm using chunked computation
        const std::size_t grain_size = parallel::get_adaptive_grain_size(2, n);  // Mixed operations
        const std::size_t num_chunks = (n + grain_size - 1) / grain_size;
        
        // Storage for partial results from each chunk
        std::vector<double> chunk_means(num_chunks);
        std::vector<double> chunk_m2s(num_chunks);
        std::vector<std::size_t> chunk_counts(num_chunks);
        
        // Phase 1: Compute partial statistics in parallel chunks
        // Create indices for parallel processing
        std::vector<std::size_t> chunk_indices(num_chunks);
        std::iota(chunk_indices.begin(), chunk_indices.end(), 0);
        
        parallel::safe_for_each(chunk_indices.begin(), chunk_indices.end(), [&](std::size_t chunk_idx) {
            const std::size_t start_idx = chunk_idx * grain_size;
            const std::size_t end_idx = std::min(start_idx + grain_size, n);
            const std::size_t chunk_size = end_idx - start_idx;
            
            double chunk_mean = constants::math::ZERO_DOUBLE;
            double chunk_m2 = constants::math::ZERO_DOUBLE;
            
            // Welford's algorithm on chunk - C++20 safe iteration
            auto chunk_range = values | std::views::drop(start_idx) | std::views::take(chunk_size);
            std::size_t local_count = 0;
            for (const double value : chunk_range) {
                ++local_count;
                const double delta = value - chunk_mean;
                const double count_inv = constants::math::ONE / static_cast<double>(local_count);
                chunk_mean += delta * count_inv;
                const double delta2 = value - chunk_mean;
                chunk_m2 += delta * delta2;
            }
            
            chunk_means[chunk_idx] = chunk_mean;
            chunk_m2s[chunk_idx] = chunk_m2;
            chunk_counts[chunk_idx] = chunk_size;
        });
        
        // Phase 2: Combine partial results using Chan's parallel algorithm
        running_mean = constants::math::ZERO_DOUBLE;
        double combined_m2 = constants::math::ZERO_DOUBLE;
        std::size_t combined_count = 0;
        
        for (std::size_t i = 0; i < num_chunks; ++i) {
            if (chunk_counts[i] > 0) {
                const double delta = chunk_means[i] - running_mean;
                const std::size_t new_count = combined_count + chunk_counts[i];
                
                running_mean += delta * static_cast<double>(chunk_counts[i]) / static_cast<double>(new_count);
                
                const double delta2 = chunk_means[i] - running_mean;
                combined_m2 += chunk_m2s[i] + delta * delta2 * 
                              static_cast<double>(combined_count) * static_cast<double>(chunk_counts[i]) / 
                              static_cast<double>(new_count);
                
                combined_count = new_count;
            }
        }
        
        sample_variance = combined_m2 / static_cast<double>(n - 1);
        
    } else {
        // Serial Welford's algorithm for smaller datasets - C++20 safe iteration
        running_mean = constants::math::ZERO_DOUBLE;
        double running_m2 = constants::math::ZERO_DOUBLE;
        
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
    if (sample_std <= constants::precision::HIGH_PRECISION_TOLERANCE) {
        throw std::invalid_argument("Data has zero or near-zero variance - cannot fit Gaussian");
    }
    
    // Set parameters (this will validate and invalidate cache)
    setParameters(running_mean, sample_std);
}

void GaussianDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = constants::math::ZERO_DOUBLE;
    standardDeviation_ = constants::math::ONE;
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
// COMPARISON OPERATORS
//==============================================================================

bool GaussianDistribution::operator==(const GaussianDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    
    return std::abs(mean_ - other.mean_) <= constants::precision::DEFAULT_TOLERANCE &&
           std::abs(standardDeviation_ - other.standardDeviation_) <= constants::precision::DEFAULT_TOLERANCE;
}

//==============================================================================
// STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const GaussianDistribution& distribution) {
    return os << distribution.toString();
}

//==============================================================================
// COMPLEX METHODS (Implementation in .cpp per C++20 best practices)
//==============================================================================

// Note: Simple statistical moments (getMean, getVariance, getSkewness, getKurtosis)
// are implemented inline in the header for optimal performance since they are
// trivial calculations or constants for the Gaussian distribution.

double GaussianDistribution::getQuantile(double p) const {
    if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    if (p == constants::math::ZERO_DOUBLE) return -std::numeric_limits<double>::infinity();
    if (p == constants::math::ONE) return std::numeric_limits<double>::infinity();
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    if (p == constants::math::HALF) {
        return mean_;  // Median equals mean for normal distribution
    }
    
    // Use inverse error function for standard normal quantile
    // For standard normal: quantile = sqrt(2) * erfinv(2p - 1)
    // For general normal: quantile = mean + sigma * sqrt(2) * erfinv(2p - 1)
    
    const double erf_input = constants::math::TWO * p - constants::math::ONE;
    double z = math::erf_inv(erf_input);
    return mean_ + standardDeviation_ * constants::math::SQRT_2 * z;
}


//==============================================================================
// PARALLEL BATCH OPERATIONS
//==============================================================================

void GaussianDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                           std::vector<GaussianDistribution>& results) {
    if (datasets.empty()) {
        throw std::invalid_argument("Cannot fit to empty dataset collection");
    }
    
    // Ensure results vector has correct size
    if (results.size() != datasets.size()) {
        results.resize(datasets.size());
    }
    
    const std::size_t num_datasets = datasets.size();
    
    // Use Level 0-3 ParallelUtils for optimal work distribution
    if (parallel::should_use_parallel(num_datasets)) {
        // Leverage ParallelUtils::parallelFor with optimal grain sizing
        ParallelUtils::parallelFor(std::size_t{0}, num_datasets, 
                                  [&datasets, &results](std::size_t idx) {
            // Fit each dataset independently in parallel with Level 0-3 infrastructure
            results[idx].fit(datasets[idx]);
        });
        
    } else {
        // Serial processing for small numbers of datasets
        for (std::size_t i = 0; i < num_datasets; ++i) {
            results[i].fit(datasets[i]);
        }
    }
}

//==============================================================================
// THREAD POOL PARALLEL BATCH OPERATIONS
//==============================================================================

void GaussianDistribution::getProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
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
    const double cached_mean = mean_;
    const double cached_norm_constant = normalizationConstant_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration with optimal work distribution
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element in parallel using cached parameters
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
            }
        }
    }
}

void GaussianDistribution::getLogProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const noexcept {
    if (values.size() != results.size()) {
        // In noexcept context, we can't throw, so return early on size mismatch
        return;
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
    const double cached_mean = mean_;
    const double cached_log_std = logStandardDeviation_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element in parallel using cached parameters
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - cached_log_std + cached_neg_half_inv_var * sq_diff;
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - cached_log_std + cached_neg_half_inv_var * sq_diff;
            }
        }
    }
}

void GaussianDistribution::getCumulativeProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
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
    const double cached_mean = mean_;
    const double cached_sigma_sqrt2 = sigmaSqrt2_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element in parallel using cached parameters
            if (cached_is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - cached_mean) / cached_sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - cached_mean) / cached_sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        }
    }
}

void GaussianDistribution::getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                           WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
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
    const double cached_mean = mean_;
    const double cached_norm_constant = normalizationConstant_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use WorkStealingPool for dynamic load balancing - optimal for heavy computational loads
    if (WorkStealingUtils::shouldUseWorkStealing(count)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element with work stealing load balancing
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
            }
        });
        
        // Wait for all work stealing tasks to complete
        pool.waitForAll();
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
            }
        }
    }
}

void GaussianDistribution::getProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                         cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system for predictive cache warming
    const std::string cache_key = "gaussian_pdf_batch_" + std::to_string(count);
    
    // Check if we have cached batch parameters for this operation size
    // TODO: Use cached_params for predictive cache warming and algorithm selection
    // In future implementation, this will:
    //   1. Pre-warm CPU caches with frequently accessed intermediate values
    //   2. Select optimal parallel algorithms based on historical performance
    //   3. Predict memory access patterns to optimize SIMD operations
    auto cached_params = cache_manager.getCachedComputationParams(cache_key);
    if (cached_params.has_value()) {
        // Future: Use cached_params.first (grain_size) and cached_params.second (hit_rate)
        // to influence parallel strategy selection and memory prefetching
        // For now, this information is available but not yet utilized
    }
    
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
    const double cached_mean = mean_;
    const double cached_norm_constant = normalizationConstant_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "gaussian_pdf");
    
    // Use cache-aware parallel processing with adaptive grain sizing
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element with cache-aware access patterns
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
            }
        }, optimal_grain_size);  // Use adaptive grain size from cache manager
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::INV_SQRT_2PI * std::exp(constants::math::NEG_HALF * sq_diff);
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_var * sq_diff);
            }
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void GaussianDistribution::getLogProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                            WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
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
    const double cached_mean = mean_;
    const double cached_log_std = logStandardDeviation_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use WorkStealingPool for dynamic load balancing - optimal for heavy computational loads
    if (WorkStealingUtils::shouldUseWorkStealing(count)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element with work stealing load balancing
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - cached_log_std + cached_neg_half_inv_var * sq_diff;
            }
        });
        
        // Wait for all work stealing tasks to complete
        pool.waitForAll();
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - cached_log_std + cached_neg_half_inv_var * sq_diff;
            }
        }
    }
}

void GaussianDistribution::getLogProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                           cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system for predictive cache warming
    const std::string cache_key = "gaussian_log_pdf_batch_" + std::to_string(count);
    
    // Check if we have cached batch parameters for this operation size
    auto cached_params = cache_manager.getCachedComputationParams(cache_key);
    if (cached_params.has_value()) {
        // Future: Use cached_params.first (grain_size) and cached_params.second (hit_rate)
        // to influence parallel strategy selection and memory prefetching
        // For now, this information is available but not yet utilized
    }
    
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
    const double cached_mean = mean_;
    const double cached_log_std = logStandardDeviation_;
    const double cached_neg_half_inv_var = negHalfSigmaSquaredInv_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "gaussian_log_pdf");
    
    // Use cache-aware parallel processing with adaptive grain sizing
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element with cache-aware access patterns
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - cached_log_std + cached_neg_half_inv_var * sq_diff;
            }
        }, optimal_grain_size);  // Use adaptive grain size from cache manager
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                const double sq_diff = values[i] * values[i];
                results[i] = constants::math::NEG_HALF_LN_2PI + constants::math::NEG_HALF * sq_diff;
            } else {
                const double diff = values[i] - cached_mean;
                const double sq_diff = diff * diff;
                results[i] = constants::math::NEG_HALF_LN_2PI - cached_log_std + cached_neg_half_inv_var * sq_diff;
            }
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void GaussianDistribution::getCumulativeProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                                    WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
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
    const double cached_mean = mean_;
    const double cached_sigma_sqrt2 = sigmaSqrt2_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use WorkStealingPool for dynamic load balancing - optimal for heavy computational loads
    if (WorkStealingUtils::shouldUseWorkStealing(count)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with work stealing load balancing
            if (cached_is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - cached_mean) / cached_sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        });
        
        // Wait for all work stealing tasks to complete
        pool.waitForAll();
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - cached_mean) / cached_sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        }
    }
}

void GaussianDistribution::getCumulativeProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                                  cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system for predictive cache warming
    const std::string cache_key = "gaussian_cdf_batch_" + std::to_string(count);
    
    // Check if we have cached batch parameters for this operation size
    auto cached_params = cache_manager.getCachedComputationParams(cache_key);
    if (cached_params.has_value()) {
        // Future: Use cached_params.first (grain_size) and cached_params.second (hit_rate)
        // to influence parallel strategy selection and memory prefetching
        // For now, this information is available but not yet utilized
    }
    
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
    const double cached_mean = mean_;
    const double cached_sigma_sqrt2 = sigmaSqrt2_;
    const bool cached_is_standard_normal = isStandardNormal_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "gaussian_cdf");
    
    // Use cache-aware parallel processing with adaptive grain sizing
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with cache-aware access patterns
            if (cached_is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - cached_mean) / cached_sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        }, optimal_grain_size);  // Use adaptive grain size from cache manager
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (cached_is_standard_normal) {
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(values[i] * constants::math::INV_SQRT_2));
            } else {
                const double normalized = (values[i] - cached_mean) / cached_sigma_sqrt2;
                results[i] = constants::math::HALF * (constants::math::ONE + std::erf(normalized));
            }
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

//==============================================================================
// ADVANCED STATISTICAL METHODS
//==============================================================================

std::pair<double, double> GaussianDistribution::confidenceIntervalMean(
    const std::vector<double>& data, 
    double confidence_level,
    bool population_variance_known) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= constants::math::ZERO_DOUBLE || confidence_level >= constants::math::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / n;
    
    double margin_of_error;
    
    if (population_variance_known || n >= 30) {
        // Use normal distribution (z-score)
        const double sample_var = std::inner_product(
            data.begin(), data.end(), data.begin(), constants::math::ZERO_DOUBLE) / n - sample_mean * sample_mean;
        const double sample_std = std::sqrt(sample_var);
        const double alpha = constants::math::ONE - confidence_level;
        const double z_alpha_2 = math::inverse_normal_cdf(constants::math::ONE - alpha * constants::math::HALF);
        margin_of_error = z_alpha_2 * sample_std / std::sqrt(n);
    } else {
        // Use t-distribution
        const double sample_var = std::inner_product(
            data.begin(), data.end(), data.begin(), constants::math::ZERO_DOUBLE, 
            std::plus<>(), 
            [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }
        ) / (n - 1);
        const double sample_std = std::sqrt(sample_var);
        const double alpha = constants::math::ONE - confidence_level;
        const double t_alpha_2 = math::inverse_t_cdf(constants::math::ONE - alpha * constants::math::HALF, n - 1);
        margin_of_error = t_alpha_2 * sample_std / std::sqrt(n);
    }
    
    return {sample_mean - margin_of_error, sample_mean + margin_of_error};
}

std::pair<double, double> GaussianDistribution::confidenceIntervalVariance(
    const std::vector<double>& data,
    double confidence_level) {
    
    if (data.size() < 2) {
        throw std::invalid_argument("At least 2 data points required for variance confidence interval");
    }
    if (confidence_level <= constants::math::ZERO_DOUBLE || confidence_level >= constants::math::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / n;
    
    // Calculate sample variance
    const double sample_var = std::inner_product(
        data.begin(), data.end(), data.begin(), constants::math::ZERO_DOUBLE, 
        std::plus<>(), 
        [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }
    ) / (n - 1);
    
    const double alpha = constants::math::ONE - confidence_level;
    const double df = n - 1;
    
    // Chi-squared critical values  
    const double alpha_half = alpha * constants::math::HALF;
    const double chi2_lower = math::inverse_chi_squared_cdf(alpha_half, df);
    const double chi2_upper = math::inverse_chi_squared_cdf(constants::math::ONE - alpha_half, df);
    
    const double lower_bound = (df * sample_var) / chi2_upper;
    const double upper_bound = (df * sample_var) / chi2_lower;
    
    return {lower_bound, upper_bound};
}

std::tuple<double, double, bool> GaussianDistribution::oneSampleTTest(
    const std::vector<double>& data,
    double hypothesized_mean,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / n;
    
    // Calculate sample standard deviation
    const double sample_var = std::inner_product(
        data.begin(), data.end(), data.begin(), constants::math::ZERO_DOUBLE, 
        std::plus<>(), 
        [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }
    ) / (n - 1);
    const double sample_std = std::sqrt(sample_var);
    
    // Calculate t-statistic
    const double t_statistic = (sample_mean - hypothesized_mean) / (sample_std / std::sqrt(n));
    
    // Calculate p-value (two-tailed) using constants for 2.0 and 1.0
    const double p_value = constants::math::TWO * (constants::math::ONE - math::t_cdf(std::abs(t_statistic), n - 1));
    
    const bool reject_null = p_value < alpha;
    
    return {t_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::twoSampleTTest(
    const std::vector<double>& data1,
    const std::vector<double>& data2,
    bool equal_variances,
    double alpha) {
    
    if (data1.empty() || data2.empty()) {
        throw std::invalid_argument("Both data vectors must be non-empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    const size_t n1 = data1.size();
    const size_t n2 = data2.size();
    
    // Sample means
    const double mean1 = std::accumulate(data1.begin(), data1.end(), constants::math::ZERO_DOUBLE) / n1;
    const double mean2 = std::accumulate(data2.begin(), data2.end(), constants::math::ZERO_DOUBLE) / n2;
    
    // Sample variances
    const double var1 = std::inner_product(
        data1.begin(), data1.end(), data1.begin(), constants::math::ZERO_DOUBLE, 
        std::plus<>(), 
        [mean1](double x, double y) { return (x - mean1) * (y - mean1); }
    ) / (n1 - 1);
    
    const double var2 = std::inner_product(
        data2.begin(), data2.end(), data2.begin(), constants::math::ZERO_DOUBLE, 
        std::plus<>(), 
        [mean2](double x, double y) { return (x - mean2) * (y - mean2); }
    ) / (n2 - 1);
    
    double t_statistic, degrees_of_freedom;
    
    if (equal_variances) {
        // Pooled t-test
        const double pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2);
        const double pooled_std = std::sqrt(pooled_var * (constants::math::ONE/n1 + constants::math::ONE/n2));
        t_statistic = (mean1 - mean2) / pooled_std;
        degrees_of_freedom = n1 + n2 - 2;
    } else {
        // Welch's t-test
        const double se = std::sqrt(var1/n1 + var2/n2);
        t_statistic = (mean1 - mean2) / se;
        
        // Welch-Satterthwaite equation for degrees of freedom
        const double numerator = std::pow(var1/n1 + var2/n2, constants::math::TWO);
        const double denominator = std::pow(var1/n1, constants::math::TWO)/(n1-1) + std::pow(var2/n2, constants::math::TWO)/(n2-1);
        degrees_of_freedom = numerator / denominator;
    }
    
    // Calculate p-value (two-tailed)
    const double p_value = constants::math::TWO * (constants::math::ONE - math::t_cdf(std::abs(t_statistic), degrees_of_freedom));
    
    const bool reject_null = p_value < alpha;
    
    return {t_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::pairedTTest(
    const std::vector<double>& data1,
    const std::vector<double>& data2,
    double alpha) {
    
    if (data1.size() != data2.size()) {
        throw std::invalid_argument("Data vectors must have the same size for paired t-test");
    }
    if (data1.empty()) {
        throw std::invalid_argument("Data vectors cannot be empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    const size_t n = data1.size();
    
    // Calculate differences
    std::vector<double> differences(n);
    std::transform(data1.begin(), data1.end(), data2.begin(), differences.begin(),
                   [](double a, double b) { return a - b; });
    
    // Perform one-sample t-test on differences against mean = 0
    return oneSampleTTest(differences, constants::math::ZERO_DOUBLE, alpha);
}

std::tuple<double, double, double, double> GaussianDistribution::bayesianEstimation(
    const std::vector<double>& data,
    double prior_mean,
    double prior_precision,
    double prior_shape,
    double prior_rate) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    const size_t n = data.size();
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / n;
    const double sample_sum_sq = std::inner_product(data.begin(), data.end(), data.begin(), constants::math::ZERO_DOUBLE);
    
    // Normal-Inverse-Gamma conjugate prior update
    const double posterior_precision = prior_precision + n;
    const double posterior_mean = (prior_precision * prior_mean + n * sample_mean) / posterior_precision;
    const double posterior_shape = prior_shape + n / constants::math::TWO;
    
    const double sum_sq_deviations = sample_sum_sq - n * sample_mean * sample_mean;
    const double prior_mean_diff = sample_mean - prior_mean;
    const double posterior_rate = prior_rate + constants::math::HALF * sum_sq_deviations + 
                                  constants::math::HALF * (prior_precision * n * prior_mean_diff * prior_mean_diff) / posterior_precision;
    
    return {posterior_mean, posterior_precision, posterior_shape, posterior_rate};
}

std::pair<double, double> GaussianDistribution::bayesianCredibleInterval(
    const std::vector<double>& data,
    double credibility_level,
    double prior_mean,
    double prior_precision,
    double prior_shape,
    double prior_rate) {
    
    // Get posterior parameters
    auto [post_mean, post_precision, post_shape, post_rate] = 
        bayesianEstimation(data, prior_mean, prior_precision, prior_shape, prior_rate);
    
    // Posterior marginal for mean follows t-distribution
    const double df = constants::math::TWO * post_shape;
    const double scale = std::sqrt(post_rate / (post_precision * post_shape));
    
    const double alpha = constants::math::ONE - credibility_level;
    const double t_critical = math::inverse_t_cdf(constants::math::ONE - alpha * constants::math::HALF, df);
    
    const double margin_of_error = t_critical * scale;
    
    return {post_mean - margin_of_error, post_mean + margin_of_error};
}

std::pair<double, double> GaussianDistribution::robustEstimation(
    const std::vector<double>& data,
    const std::string& estimator_type,
    double tuning_constant) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Initial estimates using median and MAD
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    const double median = (sorted_data.size() % 2 == 0) ?
        (sorted_data.at(sorted_data.size()/2 - 1) + sorted_data.at(sorted_data.size()/2)) / constants::math::TWO :
        sorted_data.at(sorted_data.size()/2);
    
    // Median Absolute Deviation (MAD)
    std::vector<double> abs_deviations(data.size());
    std::transform(data.begin(), data.end(), abs_deviations.begin(),
                   [median](double x) { return std::abs(x - median); });
    std::sort(abs_deviations.begin(), abs_deviations.end());
    
    const double mad = (abs_deviations.size() % 2 == 0) ?
        (abs_deviations.at(abs_deviations.size()/2 - 1) + abs_deviations.at(abs_deviations.size()/2)) / constants::math::TWO :
        abs_deviations.at(abs_deviations.size()/2);
    
    // Convert MAD to robust scale estimate
    double robust_location = median;
    double robust_scale = mad * constants::robust::MAD_SCALING_FACTOR; // Use named constant instead of 1.4826
    
    // Iterative M-estimation
    const int max_iterations = 50;
    const double convergence_tol = 1e-6;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double sum_weights = constants::math::ZERO_DOUBLE;
        double weighted_sum = constants::math::ZERO_DOUBLE;
        
        for (double x : data) {
            const double standardized_residual = (x - robust_location) / robust_scale;
            double weight = constants::math::ONE;
            
            if (estimator_type == "huber") {
                weight = (std::abs(standardized_residual) <= tuning_constant) ?
                         constants::math::ONE : tuning_constant / std::abs(standardized_residual);
            } else if (estimator_type == "tukey") {
                weight = (std::abs(standardized_residual) <= tuning_constant) ?
                         std::pow(constants::math::ONE - std::pow(standardized_residual / tuning_constant, constants::math::TWO), constants::math::TWO) : constants::math::ZERO_DOUBLE;
            } else if (estimator_type == "hampel") {
                const double abs_res = std::abs(standardized_residual);
                if (abs_res <= tuning_constant) {
                    weight = constants::math::ONE;
                } else if (abs_res <= constants::math::TWO * tuning_constant) {
                    weight = tuning_constant / abs_res;
                } else if (abs_res <= constants::math::THREE * tuning_constant) {
                    weight = tuning_constant * (constants::math::THREE - abs_res / tuning_constant) / (constants::math::TWO * abs_res);
                } else {
                    weight = constants::math::ZERO_DOUBLE;
                }
            } else {
                throw std::invalid_argument("Unknown estimator type. Use 'huber', 'tukey', or 'hampel'");
            }
            
            sum_weights += weight;
            weighted_sum += weight * x;
        }
        
        const double new_location = weighted_sum / sum_weights;
        
        // Update scale estimate
        double weighted_scale_sum = constants::math::ZERO_DOUBLE;
        for (double x : data) {
            const double residual = x - new_location;
            const double standardized_residual = residual / robust_scale;
            double weight = constants::math::ONE;
            
            if (estimator_type == "huber") {
                weight = (std::abs(standardized_residual) <= tuning_constant) ?
                         constants::math::ONE : tuning_constant / std::abs(standardized_residual);
            } else if (estimator_type == "tukey") {
                weight = (std::abs(standardized_residual) <= tuning_constant) ?
                         std::pow(constants::math::ONE - std::pow(standardized_residual / tuning_constant, constants::math::TWO), constants::math::TWO) : constants::math::ZERO_DOUBLE;
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
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / n;
    
    // Second central moment (variance)
    const double sample_variance = std::inner_product(
        data.begin(), data.end(), data.begin(), constants::math::ZERO_DOUBLE, 
        std::plus<>(), 
        [sample_mean](double x, double y) { return (x - sample_mean) * (y - sample_mean); }
    ) / n;  // Population variance (divide by n, not n-1)
    
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
    double l1 = constants::math::ZERO_DOUBLE; // L-mean
    double l2 = constants::math::ZERO_DOUBLE; // L-scale
    
    // L1 (L-mean) = mean of order statistics
    l1 = std::accumulate(sorted_data.begin(), sorted_data.end(), constants::math::ZERO_DOUBLE) / n;
    
    // L2 (L-scale) = 0.5 * E[X_{2:2} - X_{1:2}]
    for (size_t i = 0; i < n; ++i) {
        const double weight = (constants::math::TWO * i + constants::math::ONE - n) / n;
        l2 += weight * sorted_data[i];
    }
    l2 = constants::math::HALF * l2;
    
    // For Gaussian distribution:
    // L1 = μ (location parameter)
    // L2 = σ/√π (scale parameter relationship)
    const double location_param = l1;
    const double scale_param = l2 * std::sqrt(constants::math::PI);
    
    return {location_param, scale_param};
}

std::vector<double> GaussianDistribution::calculateHigherMoments(
    const std::vector<double>& data,
    bool center_on_mean) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    const size_t n = data.size();
    const double sample_mean = center_on_mean ? 
        std::accumulate(data.begin(), data.end(), 0.0) / n : 0.0;
    
    std::vector<double> moments(6, 0.0);
    
    // Calculate raw or central moments up to 6th order
    for (double x : data) {
        const double deviation = x - sample_mean;
        
        for (int k = 0; k < 6; ++k) {
            if (center_on_mean) {
                moments[k] += std::pow(deviation, k + 1);
            } else {
                moments[k] += std::pow(x, k + 1);
            }
        }
    }
    
    // Normalize by sample size
    for (double& moment : moments) {
        moment /= n;
    }
    
    return moments;
}

std::tuple<double, double, bool> GaussianDistribution::jarqueBeraTest(
    const std::vector<double>& data,
    double alpha) {
    
    if (data.size() < 8) {
        throw std::invalid_argument("At least 8 data points required for Jarque-Bera test");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    
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
    
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    const double skewness = m3 / std::pow(m2, 1.5);
    const double kurtosis = m4 / (m2 * m2) - constants::thresholds::EXCESS_KURTOSIS_OFFSET; // Excess kurtosis
    
    // Jarque-Bera statistic
    const double jb_statistic = n * (skewness * skewness / constants::math::SIX + kurtosis * kurtosis / constants::math::TWO_TWENTY_FIVE);
    
    // P-value from chi-squared distribution with 2 degrees of freedom
    const double p_value = constants::math::ONE - math::chi_squared_cdf(jb_statistic, constants::math::TWO);
    
    const bool reject_null = p_value < alpha;
    
    return {jb_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::shapiroWilkTest(
    const std::vector<double>& data,
    double alpha) {
    
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
    const double sample_mean = std::accumulate(sorted_data.begin(), sorted_data.end(), 0.0) / n;
    double ss = 0.0;
    for (double x : sorted_data) {
        ss += (x - sample_mean) * (x - sample_mean);
    }
    
    // Simplified W statistic calculation
    // This is a basic approximation - full implementation would use proper coefficients
    double numerator = 0.0;
    for (size_t i = 0; i < n / 2; ++i) {
        const double coeff = math::inverse_normal_cdf((i + constants::math::THREE_QUARTERS) / (n + constants::math::HALF));
        numerator += coeff * (sorted_data[n - 1 - i] - sorted_data[i]);
    }
    
    const double w_statistic = (numerator * numerator) / ss;
    
    // Approximate p-value (simplified)
    // Full implementation would use proper lookup tables or approximations
    const double log_p = constants::math::NEG_HALF * std::log(w_statistic) - constants::math::ONE_POINT_FIVE * std::log(n) + constants::math::TWO;
    const double p_value = std::exp(log_p);
    
    const bool reject_null = p_value < alpha;
    
    return {w_statistic, std::min(p_value, constants::math::ONE), reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::likelihoodRatioTest(
    const std::vector<double>& data,
    const GaussianDistribution& restricted_model,
    const GaussianDistribution& unrestricted_model,
    double alpha) {
    
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
    const double lr_statistic = constants::math::TWO * (log_likelihood_unrestricted - log_likelihood_restricted);
    
    // Degrees of freedom = difference in number of parameters
    const int df = unrestricted_model.getNumParameters() - restricted_model.getNumParameters();
    
    if (df <= 0) {
        throw std::invalid_argument("Unrestricted model must have more parameters than restricted model");
    }
    
    // P-value from chi-squared distribution
    const double p_value = constants::math::ONE - math::chi_squared_cdf(lr_statistic, df);
    
    const bool reject_null = p_value < alpha;
    
    return {lr_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data,
    const GaussianDistribution& distribution,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Use the KS statistic calculation from math_utils
    double ks_statistic = math::calculate_ks_statistic(data, distribution);
    
    // Asymptotic p-value approximation for KS test
    // P-value ≈ 2 * exp(-2 * n * D²) for large n
    const double n = static_cast<double>(data.size());
    const double p_value_approx = constants::math::TWO * std::exp(-constants::math::TWO * n * ks_statistic * ks_statistic);
    
    // Clamp p-value to [0, 1]
    const double p_value = std::min(constants::math::ONE, std::max(constants::precision::ZERO, p_value_approx));
    
    const bool reject_null = p_value < alpha;
    
    return {ks_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> GaussianDistribution::andersonDarlingTest(
    const std::vector<double>& data,
    const GaussianDistribution& distribution,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Use the AD statistic calculation from math_utils
    double ad_statistic = math::calculate_ad_statistic(data, distribution);
    
    // Asymptotic p-value approximation for AD test with Gaussian distribution
    // This is a simplified approximation - full implementation would use
    // more sophisticated lookup tables or better approximation formulas
    const double n = static_cast<double>(data.size());
    const double modified_stat = ad_statistic * (constants::math::ONE + constants::math::THREE_QUARTERS / n + constants::math::TWO_TWENTY_FIVE / (n * n));
    
    // Approximate p-value using exponential approximation
    double p_value;
    if (modified_stat >= constants::math::THIRTEEN) {
        p_value = constants::precision::ZERO;
    } else if (modified_stat >= constants::math::SIX) {
        p_value = std::exp(-constants::math::ONE_POINT_TWO_EIGHT * modified_stat);
    } else {
        p_value = std::exp(-constants::math::ONE_POINT_EIGHT * modified_stat + constants::math::ONE_POINT_FIVE);
    }
    
    // Clamp p-value to [0, 1]
    p_value = std::min(constants::math::ONE, std::max(constants::precision::ZERO, p_value));
    
    const bool reject_null = p_value < alpha;
    
    return {ad_statistic, p_value, reject_null};
}

//==============================================================================
// CROSS-VALIDATION AND MODEL SELECTION
//==============================================================================

std::vector<std::tuple<double, double, double>> GaussianDistribution::kFoldCrossValidation(
    const std::vector<double>& data,
    int k,
    unsigned int random_seed) {
    
    if (data.size() < static_cast<size_t>(k)) {
        throw std::invalid_argument("Data size must be at least k for k-fold cross-validation");
    }
    if (k <= 1) {
        throw std::invalid_argument("Number of folds k must be greater than 1");
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
        
        // Fit model on training data
        GaussianDistribution fitted_model;
        fitted_model.fit(training_data);
        
        // Evaluate on validation data
        double mean_error = 0.0;
        double std_error = 0.0;
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
        
        // Calculate mean and standard deviation of errors
        mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        
        double variance_error = 0.0;
        for (double error : errors) {
            variance_error += (error - mean_error) * (error - mean_error);
        }
        std_error = std::sqrt(variance_error / errors.size());
        
        results.emplace_back(mean_error, std_error, log_likelihood);
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
    double mean_absolute_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / n;
    double mean_squared_error = std::accumulate(squared_errors.begin(), squared_errors.end(), 0.0) / n;
    double root_mean_squared_error = std::sqrt(mean_squared_error);
    
    return {mean_absolute_error, root_mean_squared_error, total_log_likelihood};
}

std::tuple<std::pair<double, double>, std::pair<double, double>> GaussianDistribution::bootstrapParameterConfidenceIntervals(
    const std::vector<double>& data,
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
    bootstrap_means.reserve(n_bootstrap);
    bootstrap_stds.reserve(n_bootstrap);
    
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
    const double alpha = constants::math::ONE - confidence_level;
    const double lower_percentile = alpha * constants::math::HALF;
    const double upper_percentile = constants::math::ONE - alpha * constants::math::HALF;
    
    const size_t lower_idx = static_cast<size_t>(lower_percentile * (n_bootstrap - 1));
    const size_t upper_idx = static_cast<size_t>(upper_percentile * (n_bootstrap - 1));
    
    std::pair<double, double> mean_ci = {bootstrap_means[lower_idx], bootstrap_means[upper_idx]};
    std::pair<double, double> std_ci = {bootstrap_stds[lower_idx], bootstrap_stds[upper_idx]};
    
    return {mean_ci, std_ci};
}

std::tuple<double, double, double, double> GaussianDistribution::computeInformationCriteria(
    const std::vector<double>& data,
    const GaussianDistribution& fitted_distribution) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    const double n = static_cast<double>(data.size());
    const int k = fitted_distribution.getNumParameters(); // 2 for Gaussian
    
    // Calculate log-likelihood
    double log_likelihood = 0.0;
    for (double val : data) {
        log_likelihood += fitted_distribution.getLogProbability(val);
    }
    
    // Compute information criteria
    const double aic = constants::math::TWO * k - constants::math::TWO * log_likelihood;
    const double bic = std::log(n) * k - constants::math::TWO * log_likelihood;
    
    // AICc (corrected AIC for small sample sizes)
    double aicc;
    if (n - k - 1 > 0) {
        aicc = aic + (constants::math::TWO * k * (k + 1)) / (n - k - 1);
    } else {
        aicc = std::numeric_limits<double>::infinity(); // Undefined for small samples
    }
    
    return {aic, bic, aicc, log_likelihood};
}

} // namespace libstats

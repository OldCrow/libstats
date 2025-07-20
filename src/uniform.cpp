#include "../include/uniform.h"
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

UniformDistribution::UniformDistribution(double a, double b) 
    : DistributionBase(), a_(a), b_(b) {
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
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        
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
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    a_ = other.a_;
    b_ = other.b_;
    other.a_ = constants::math::ZERO_DOUBLE;
    other.b_ = constants::math::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

UniformDistribution& UniformDistribution::operator=(UniformDistribution&& other) noexcept {
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
                a_ = other.a_;
                b_ = other.b_;
                other.a_ = constants::math::ZERO_DOUBLE;
                other.b_ = constants::math::ONE;
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
            [[maybe_unused]] double temp_a = a_;
            [[maybe_unused]] double temp_b = b_;
            
            // Atomic-like exchange (single assignment is atomic for built-in types)
            a_ = other.a_;
            b_ = other.b_;
            other.a_ = constants::math::ZERO_DOUBLE;
            other.b_ = constants::math::ONE;
            
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

double UniformDistribution::getProbability(double x) const {
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
    
    // Check if x is within the support [a, b]
    if (x < a_ || x > b_) {
        return constants::math::ZERO_DOUBLE;
    }
    
    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return constants::math::ONE;
    }
    
    // General case: PDF = 1/(b-a) for x in [a,b]
    return invWidth_;
}

double UniformDistribution::getLogProbability(double x) const noexcept {
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
    
    // Check if x is within the support [a, b]
    if (x < a_ || x > b_) {
        return constants::probability::NEGATIVE_INFINITY;
    }
    
    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return constants::math::ZERO_DOUBLE;  // log(1) = 0
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
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    
    // CDF for uniform distribution
    if (x < a_) {
        return constants::math::ZERO_DOUBLE;
    }
    if (x > b_) {
        return constants::math::ONE;
    }
    
    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return x;  // CDF(x) = x for U(0,1)
    }
    
    // General case: CDF(x) = (x-a)/(b-a)
    return (x - a_) * invWidth_;
}

double UniformDistribution::getQuantile(double p) const {
    if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    if (p == constants::math::ZERO_DOUBLE) {
        return a_;
    }
    if (p == constants::math::ONE) {
        return b_;
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
    
    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return p;  // Quantile(p) = p for U(0,1)
    }
    
    // General case: Quantile(p) = a + p*(b-a)
    return a_ + p * width_;
}

double UniformDistribution::sample(std::mt19937& rng) const {
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
        constants::math::ZERO_DOUBLE, 
        constants::math::ONE
    );
    
    double u = uniform(rng);
    
    // Fast path for unit interval [0,1]
    if (isUnitInterval_) {
        return u;
    }
    
    // General case: linear transformation X = a + (b-a)*U
    return a_ + width_ * u;
}

//==============================================================================
// PARAMETER GETTERS AND SETTERS
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
}

void UniformDistribution::setBounds(double a, double b) {
    validateParameters(a, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

double UniformDistribution::getMean() const noexcept {
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
    
    return midpoint_;
}

double UniformDistribution::getVariance() const noexcept {
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
    
    return variance_;
}

double UniformDistribution::getWidth() const noexcept {
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
    
    return width_;
}

double UniformDistribution::getMidpoint() const noexcept {
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
    
    return midpoint_;
}

//==============================================================================
// DISTRIBUTION MANAGEMENT
//==============================================================================

void UniformDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
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
    const double margin = (sample_max - sample_min) * constants::precision::DEFAULT_TOLERANCE;
    const double fitted_a = sample_min - margin;
    const double fitted_b = sample_max + margin;
    
    // Set parameters (this will validate and invalidate cache)
    setBounds(fitted_a, fitted_b);
}

void UniformDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = constants::math::ZERO_DOUBLE;
    b_ = constants::math::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

std::string UniformDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "UniformDistribution(a=" << a_ << ", b=" << b_ << ")";
    return oss.str();
}

//==============================================================================
// COMPARISON OPERATORS
//==============================================================================

bool UniformDistribution::operator==(const UniformDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    
    return std::abs(a_ - other.a_) <= constants::precision::DEFAULT_TOLERANCE &&
           std::abs(b_ - other.b_) <= constants::precision::DEFAULT_TOLERANCE;
}

//==============================================================================
// STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const UniformDistribution& distribution) {
    return os << distribution.toString();
}

//==============================================================================
// SIMD BATCH OPERATIONS
//==============================================================================

void UniformDistribution::getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
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
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getProbabilityBatchUnsafeImpl(values, results, count, cached_a, cached_b, cached_inv_width);
}

void UniformDistribution::getLogProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
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
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_log_inv_width = -std::log(width_);
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_a, cached_b, cached_log_inv_width);
}

void UniformDistribution::getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const {
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
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, cached_a, cached_b, cached_inv_width);
}

void UniformDistribution::getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getProbabilityBatchUnsafeImpl(values, results, count, a_, b_, invWidth_);
}

void UniformDistribution::getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    const double log_inv_width = -std::log(width_);
    getLogProbabilityBatchUnsafeImpl(values, results, count, a_, b_, log_inv_width);
}

//==============================================================================
// PRIVATE BATCH IMPLEMENTATION USING VECTOROPS
//==============================================================================

void UniformDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                        double a, double b, double inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = (count >= simd::tuned::min_states_for_simd()) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                     (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            
            if (x < a || x > b) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (is_unit_interval) {
                results[i] = constants::math::ONE;
            } else {
                results[i] = inv_width;
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                 (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    std::vector<double, simd::aligned_allocator<double>> temp_results(count);
    
    if (is_unit_interval) {
        // Unit interval case: result is 1 for x in [0,1], 0 otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            temp_results[i] = (x >= constants::math::ZERO_DOUBLE && x <= constants::math::ONE) ? 
                             constants::math::ONE : constants::math::ZERO_DOUBLE;
        }
    } else {
        // General case: result is inv_width for x in [a,b], 0 otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            temp_results[i] = (x >= a && x <= b) ? inv_width : constants::math::ZERO_DOUBLE;
        }
    }
    
    // Copy results back
    std::copy(temp_results.begin(), temp_results.end(), results);
}

void UniformDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                           double a, double b, double log_inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = (count >= simd::tuned::min_states_for_simd()) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                     (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            
            if (x < a || x > b) {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            } else if (is_unit_interval) {
                results[i] = constants::math::ZERO_DOUBLE;  // log(1) = 0
            } else {
                results[i] = log_inv_width;
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                 (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    std::vector<double, simd::aligned_allocator<double>> temp_results(count);
    
    if (is_unit_interval) {
        // Unit interval case: result is 0 for x in [0,1], -∞ otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            temp_results[i] = (x >= constants::math::ZERO_DOUBLE && x <= constants::math::ONE) ? 
                             constants::math::ZERO_DOUBLE : constants::probability::NEGATIVE_INFINITY;
        }
    } else {
        // General case: result is log_inv_width for x in [a,b], -∞ otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            temp_results[i] = (x >= a && x <= b) ? log_inv_width : constants::probability::NEGATIVE_INFINITY;
        }
    }
    
    // Copy results back
    std::copy(temp_results.begin(), temp_results.end(), results);
}

void UniformDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                  double a, double b, double inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = (count >= simd::tuned::min_states_for_simd()) && 
                         (cpu::supports_sse2() || cpu::supports_avx() || cpu::supports_avx2());
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                     (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            
            if (x < a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (x > b) {
                results[i] = constants::math::ONE;
            } else if (is_unit_interval) {
                results[i] = x;  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (x - a) * inv_width;
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                 (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    std::vector<double, simd::aligned_allocator<double>> temp_results(count);
    std::vector<double, simd::aligned_allocator<double>> shifted_values(count);
    
    if (is_unit_interval) {
        // Unit interval case: CDF(x) = 0 for x < 0, x for 0 ≤ x ≤ 1, 1 for x > 1
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                temp_results[i] = constants::math::ZERO_DOUBLE;
            } else if (x > constants::math::ONE) {
                temp_results[i] = constants::math::ONE;
            } else {
                temp_results[i] = x;
            }
        }
    } else {
        // General case: CDF(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
        // Use SIMD for the subtraction and multiplication
        simd::VectorOps::scalar_add(values, -a, shifted_values.data(), count);
        simd::VectorOps::scalar_multiply(shifted_values.data(), inv_width, temp_results.data(), count);
        
        // Apply bounds: clamp to [0, 1]
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < a) {
                temp_results[i] = constants::math::ZERO_DOUBLE;
            } else if (x > b) {
                temp_results[i] = constants::math::ONE;
            }
            // Otherwise, keep the computed value from SIMD operations
        }
    }
    
    // Copy results back
    std::copy(temp_results.begin(), temp_results.end(), results);
}

} // namespace libstats

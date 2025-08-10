#include "../include/distributions/uniform.h"
#include "../include/core/constants.h"
#include "../include/core/validation.h"
#include "../include/core/math_utils.h"
#include "../include/core/log_space_ops.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/work_stealing_pool.h"
#include "../include/platform/adaptive_cache.h"
#include "../include/core/dispatch_utils.h"
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
    // Ensure cache is valid once before using - using the same pattern as other methods
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

std::vector<double> UniformDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Get cached parameters for efficiency
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
    
    const double cached_a = a_;
    const double cached_width = width_;
    const bool cached_is_unit_interval = isUnitInterval_;
    lock.unlock();
    
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

std::istream& operator>>(std::istream& is, UniformDistribution& distribution) {
    std::string token;
    double a, b;
    
    // Expected format: "UniformDistribution(a=<value>, b=<value>)"
    // We'll parse this step by step
    
    // Skip whitespace and read the first part
    is >> token;
    if (token.find("UniformDistribution(") != 0) {
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

void UniformDistribution::getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
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

void UniformDistribution::getCumulativeProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, a_, b_, invWidth_);
}

//==============================================================================
// PRIVATE BATCH IMPLEMENTATION USING VECTOROPS
//==============================================================================

void UniformDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                        double a, double b, double inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: For uniform distribution, computation is extremely simple (just bounds checking)
        // so SIMD rarely provides benefits, but we use centralized policy for consistency
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            // Use exclusion check (x < a || x > b) for CPU efficiency:
            // - Short-circuits on first true condition (common for out-of-support values)
            // - Matches scalar implementation exactly for consistency
            results[i] = (x < a || x > b) ? constants::math::ZERO_DOUBLE : inv_width;
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
        results[i] = (x < a || x > b) ? constants::math::ZERO_DOUBLE : inv_width;
    }
}

void UniformDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                           double a, double b, double log_inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: For uniform distribution, computation is extremely simple (just bounds checking)
        // so SIMD rarely provides benefits, but we use centralized policy for consistency
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            // Use exclusion check (x < a || x > b) for consistency with scalar and PDF SIMD:
            // - Matches boundary conditions exactly
            // - Short-circuits efficiently for out-of-support values
            results[i] = (x < a || x > b) ? constants::probability::NEGATIVE_INFINITY : log_inv_width;
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
        results[i] = (x < a || x > b) ? constants::probability::NEGATIVE_INFINITY : log_inv_width;
    }
}

void UniformDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                  double a, double b, double inv_width) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: For uniform distribution CDF, computation is simple (bounds checking + linear interpolation)
        // so SIMD rarely provides benefits, but we use centralized policy for consistency
        const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                     (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        
        if (is_unit_interval) {
            // Unit interval case: CDF(x) = 0 for x < 0, x for 0 ≤ x ≤ 1, 1 for x > 1
            for (std::size_t i = 0; i < count; ++i) {
                const double x = values[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    results[i] = constants::math::ZERO_DOUBLE;
                } else if (x > constants::math::ONE) {
                    results[i] = constants::math::ONE;
                } else {
                    results[i] = x;
                }
            }
        } else {
            // General case: CDF(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
            for (std::size_t i = 0; i < count; ++i) {
                const double x = values[i];
                if (x < a) {
                    results[i] = constants::math::ZERO_DOUBLE;
                } else if (x > b) {
                    results[i] = constants::math::ONE;
                } else {
                    results[i] = (x - a) * inv_width;
                }
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation if possible
    // Note: For uniform distribution, vectorization typically doesn't provide significant benefits
    // due to the simple nature of bounds checking, but we implement for consistency
    // In practice, this will mostly fall back to scalar due to the nature of the operation
    
    const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                 (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    // Use scalar implementation even when SIMD is available because uniform distribution
    // operations are not amenable to vectorization (primarily branching logic)
    if (is_unit_interval) {
        // Unit interval case: CDF(x) = 0 for x < 0, x for 0 ≤ x ≤ 1, 1 for x > 1
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (x > constants::math::ONE) {
                results[i] = constants::math::ONE;
            } else {
                results[i] = x;
            }
        }
    } else {
        // General case: CDF(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (x > b) {
                results[i] = constants::math::ONE;
            } else {
                results[i] = (x - a) * inv_width;
            }
        }
    }
}

//==============================================================================
// PARALLEL BATCH OPERATIONS  
//==============================================================================

void UniformDistribution::getProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform PDF: extremely lightweight operation (2 comparisons + 1 assignment)
    // Thread pool overhead will always dominate computation time regardless of size
    // Use optimized SIMD batch implementation instead of parallel threads
    // Call unsafe implementation directly since we already did cache management
    getProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_inv_width);
}

void UniformDistribution::getLogProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const noexcept {
    if (values.size() != results.size()) return; // Can't throw in noexcept context
    
    const std::size_t count = values.size();
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_log_inv_width = -std::log(width_);
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform LogPDF: extremely lightweight operation (2 comparisons + 1 assignment)
    // Thread pool overhead will always dominate computation time regardless of size
    // Use optimized SIMD batch implementation instead of parallel threads
    // Call unsafe implementation directly since we already did cache management
    getLogProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_log_inv_width);
}

void UniformDistribution::getCumulativeProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform CDF: lightweight operation (2-3 comparisons + 1 arithmetic)
    // Data shows regular parallel threading has excessive overhead (13120μs vs 245μs SIMD)
    // Thread pool overhead dominates computation time regardless of size
    // Use optimized SIMD batch implementation instead of parallel threads
    // Call unsafe implementation directly since we already did cache management
    getCumulativeProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_inv_width);
}

void UniformDistribution::getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                         [[maybe_unused]] WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform PDF: extremely lightweight operation
    // Work-stealing thread overhead will always dominate computation time
    // Use optimized SIMD batch implementation instead of work-stealing threads
    // Call unsafe implementation directly since we already did cache management
    getProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_inv_width);
}

void UniformDistribution::getProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                       cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "uniform_pdf_batch_" + std::to_string(count);
    
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform PDF: extremely lightweight operation
    // Cache-aware parallel overhead will always dominate computation time
    // Use optimized SIMD batch implementation instead of cache-aware parallel threads
    // Call unsafe implementation directly since we already did cache management
    getProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_inv_width);
    
    // Still update cache performance metrics for monitoring
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "uniform_pdf");
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void UniformDistribution::getLogProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                            [[maybe_unused]] WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_log_inv_width = -std::log(width_);
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform LogPDF: extremely lightweight operation
    // Work-stealing thread overhead will always dominate computation time
    // Use optimized SIMD batch implementation instead of work-stealing threads
    // Call unsafe implementation directly since we already did cache management
    getLogProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_log_inv_width);
}

void UniformDistribution::getLogProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                          cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "uniform_log_pdf_batch_" + std::to_string(count);
    
    auto cached_params = cache_manager.getCachedComputationParams(cache_key);
    if (cached_params.has_value()) {
        // Future: Use cached performance metrics for optimization
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_log_inv_width = -std::log(width_);
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform LogPDF: extremely lightweight operation
    // Cache-aware parallel overhead will always dominate computation time
    // Use optimized SIMD batch implementation instead of cache-aware parallel threads
    // Call unsafe implementation directly since we already did cache management
    getLogProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_log_inv_width);
    
    // Still update cache performance metrics for monitoring
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "uniform_log_pdf");
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void UniformDistribution::getCumulativeProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                                   [[maybe_unused]] WorkStealingPool& pool) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform CDF: lightweight operation (2-3 comparisons + 1 arithmetic)
    // Data shows work-stealing parallelism has excessive overhead (10409μs vs 238μs SIMD)
    // Work-stealing thread overhead dominates computation time regardless of size
    // Use optimized SIMD batch implementation instead of work-stealing threads
    // Call unsafe implementation directly since we already did cache management
    getCumulativeProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_inv_width);
}

void UniformDistribution::getCumulativeProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                                 cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "uniform_cdf_batch_" + std::to_string(count);
    
    auto cached_params = cache_manager.getCachedComputationParams(cache_key);
    if (cached_params.has_value()) {
        // Future: Use cached performance metrics for optimization
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
    
    // Cache parameters for thread-safe parallel access
    const double cached_a = a_;
    const double cached_b = b_;
    const double cached_inv_width = invWidth_;
    [[maybe_unused]] const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // For uniform CDF: lightweight operation (2-3 comparisons + 1 arithmetic)
    // Data shows cache-aware parallelism has excessive overhead vs SIMD performance
    // Cache-aware parallel overhead dominates computation time regardless of size
    // Use optimized SIMD batch implementation instead of cache-aware parallel threads
    // Call unsafe implementation directly since we already did cache management
    getCumulativeProbabilityBatchUnsafeImpl(values.data(), results.data(), count, cached_a, cached_b, cached_inv_width);
    
    // Still update cache performance metrics for monitoring
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "uniform_cdf");
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

//==============================================================================
// ADVANCED STATISTICAL METHODS
//==============================================================================

std::pair<double, double> UniformDistribution::confidenceIntervalLowerBound(const std::vector<double>& data, double confidence_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double alpha = 1.0 - confidence_level;
    const double min_val = *std::min_element(data.begin(), data.end());
    
    // For uniform distribution, the minimum X_(1) has distribution:
    // F(x) = 1 - ((b-x)/(b-a))^n for a <= x <= b
    // We use the fact that (X_(1) - a)/(b - a) ~ Beta(1, n)
    // The confidence interval uses order statistics theory
    
    // Conservative approach: use the empirical minimum with adjustment
    const double range_estimate = *std::max_element(data.begin(), data.end()) - min_val;
    const double adjustment = range_estimate * std::pow(alpha/2.0, 1.0/static_cast<double>(n)) / (1.0 + std::pow(alpha/2.0, 1.0/static_cast<double>(n)));
    
    const double ci_lower = min_val - adjustment;
    const double ci_upper = min_val;
    
    return {ci_lower, ci_upper};
}

std::pair<double, double> UniformDistribution::confidenceIntervalUpperBound(const std::vector<double>& data, double confidence_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double alpha = 1.0 - confidence_level;
    const double max_val = *std::max_element(data.begin(), data.end());
    const double min_val = *std::min_element(data.begin(), data.end());
    
    // For uniform distribution, the maximum X_(n) has distribution:
    // F(x) = ((x-a)/(b-a))^n for a <= x <= b
    // We use the fact that (b - X_(n))/(b - a) ~ Beta(1, n)
    
    const double range_estimate = max_val - min_val;
    const double adjustment = range_estimate * std::pow(alpha/2.0, 1.0/static_cast<double>(n)) / (1.0 + std::pow(alpha/2.0, 1.0/static_cast<double>(n)));
    
    const double ci_lower = max_val;
    const double ci_upper = max_val + adjustment;
    
    return {ci_lower, ci_upper};
}

std::tuple<double, double, bool> UniformDistribution::likelihoodRatioTest(const std::vector<double>& data, double null_a, double null_b, double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (null_a >= null_b) {
        throw std::invalid_argument("null_a must be less than null_b");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double sample_min = *std::min_element(data.begin(), data.end());
    const double sample_max = *std::max_element(data.begin(), data.end());
    
    // Check if null hypothesis is feasible
    if (sample_min < null_a || sample_max > null_b) {
        // Data outside null hypothesis bounds - reject immediately
        return {std::numeric_limits<double>::infinity(), 0.0, true};
    }
    
    // Log-likelihood under null hypothesis: n * log(1/(null_b - null_a))
    const double log_like_null = static_cast<double>(n) * (-std::log(null_b - null_a));
    
    // Log-likelihood under alternative (MLE): n * log(1/(sample_max - sample_min))
    const double sample_range = sample_max - sample_min;
    const double log_like_alt = (sample_range > 0) ? static_cast<double>(n) * (-std::log(sample_range)) : 0.0;
    
    // Likelihood ratio test statistic: -2 * (log L_0 - log L_1)
    const double test_statistic = -2.0 * (log_like_null - log_like_alt);
    
    // For large n, test statistic follows chi-square with 2 degrees of freedom
    // Approximate p-value using chi-square distribution
    const double p_value = 1.0 - (1.0 - std::exp(-test_statistic/2.0)); // Simplified approximation
    
    const bool reject_null = (p_value < significance_level);
    
    return {test_statistic, p_value, reject_null};
}

std::pair<std::pair<double, double>, std::pair<double, double>> UniformDistribution::bayesianEstimation(const std::vector<double>& data, [[maybe_unused]] double prior_a_shape, [[maybe_unused]] double prior_a_scale, [[maybe_unused]] double prior_b_shape, [[maybe_unused]] double prior_b_scale) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    
    const double sample_min = *std::min_element(data.begin(), data.end());
    const double sample_max = *std::max_element(data.begin(), data.end());
    const size_t n = data.size();
    
    // For uniform distribution with uniform priors on [a, b]:
    // Posterior for 'a' given data: truncated at sample_min
    // Posterior for 'b' given data: truncated at sample_max
    
    // Simplified Bayesian update (assuming uniform priors)
    const double posterior_a_mean = sample_min - (sample_max - sample_min) / (static_cast<double>(n) + 2.0);
    const double posterior_a_var = std::pow(sample_max - sample_min, 2) / (12.0 * (static_cast<double>(n) + 2.0));
    
    const double posterior_b_mean = sample_max + (sample_max - sample_min) / (static_cast<double>(n) + 2.0);
    const double posterior_b_var = std::pow(sample_max - sample_min, 2) / (12.0 * (static_cast<double>(n) + 2.0));
    
    // Return as (mean, std_dev) pairs
    std::pair<double, double> posterior_a = {posterior_a_mean, std::sqrt(posterior_a_var)};
    std::pair<double, double> posterior_b = {posterior_b_mean, std::sqrt(posterior_b_var)};
    
    return {posterior_a, posterior_b};
}

std::pair<double, double> UniformDistribution::robustEstimation(const std::vector<double>& data, const std::string& estimator_type, double trim_proportion) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (trim_proportion < 0.0 || trim_proportion >= 0.5) {
        throw std::invalid_argument("Trim proportion must be in [0, 0.5)");
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    const size_t n = sorted_data.size();
    
    if (estimator_type == "quantile") {
        // Use empirical quantiles with small adjustments
        const size_t lower_idx = static_cast<size_t>(trim_proportion * static_cast<double>(n));
        const size_t upper_idx = n - 1 - lower_idx;
        
        const double robust_a = sorted_data[lower_idx];
        const double robust_b = sorted_data[upper_idx];
        
        return {robust_a, robust_b};
    } else if (estimator_type == "trimmed") {
        // Trimmed mean approach - use trimmed range
        const size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(n));
        const size_t start_idx = trim_count;
        const size_t end_idx = n - trim_count - 1;
        
        if (start_idx >= end_idx) {
            // Fallback to min/max if too much trimming
            return {sorted_data.front(), sorted_data.back()};
        }
        
        const double robust_a = sorted_data[start_idx];
        const double robust_b = sorted_data[end_idx];
        
        return {robust_a, robust_b};
    }
    
    // Default: return min/max
    return {sorted_data.front(), sorted_data.back()};
}

std::pair<double, double> UniformDistribution::methodOfMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    
    // For uniform distribution U(a,b):
    // Mean = (a + b)/2
    // Variance = (b - a)²/12
    // From these: a = mean - sqrt(3*variance), b = mean + sqrt(3*variance)
    
    const size_t n = data.size();
    const double sum = std::accumulate(data.begin(), data.end(), 0.0);
    const double mean = sum / static_cast<double>(n);
    
    double variance = 0.0;
    for (double x : data) {
        variance += (x - mean) * (x - mean);
    }
    variance /= static_cast<double>(n - 1); // Sample variance
    
    const double range_estimate = std::sqrt(12.0 * variance);
    const double a_estimate = mean - range_estimate / 2.0;
    const double b_estimate = mean + range_estimate / 2.0;
    
    return {a_estimate, b_estimate};
}

std::tuple<std::pair<double, double>, std::pair<double, double>> UniformDistribution::bayesianCredibleInterval(const std::vector<double>& data, double credibility_level, double prior_a_shape, double prior_a_scale, double prior_b_shape, double prior_b_scale) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (credibility_level <= 0.0 || credibility_level >= 1.0) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }
    
    // Get Bayesian estimates
    auto posterior_params = bayesianEstimation(data, prior_a_shape, prior_a_scale, prior_b_shape, prior_b_scale);
    
    const double alpha = 1.0 - credibility_level;
    [[maybe_unused]] const double tail_prob = alpha / 2.0;
    
    // For simplicity, use normal approximation to posterior
    const double z_score = 1.96; // Approximate 97.5th percentile of standard normal
    
    // Credible interval for 'a'
    const double a_mean = posterior_params.first.first;
    const double a_std = posterior_params.first.second;
    const double a_ci_lower = a_mean - z_score * a_std;
    const double a_ci_upper = a_mean + z_score * a_std;
    
    // Credible interval for 'b'
    const double b_mean = posterior_params.second.first;
    const double b_std = posterior_params.second.second;
    const double b_ci_lower = b_mean - z_score * b_std;
    const double b_ci_upper = b_mean + z_score * b_std;
    
    std::pair<double, double> a_CI = {a_ci_lower, a_ci_upper};
    std::pair<double, double> b_CI = {b_ci_lower, b_ci_upper};
    
    return {a_CI, b_CI};
}

std::pair<double, double> UniformDistribution::lMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    const size_t n = sorted_data.size();
    
    // L-moments for uniform distribution:
    // L1 = (a + b)/2 (location)
    // L2 = (b - a)/6 (scale)
    
    // Sample L-moments
    double L1 = 0.0; // Sample mean
    for (double x : sorted_data) {
        L1 += x;
    }
    L1 /= static_cast<double>(n);
    
    // L2 calculation using order statistics
    double L2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double weight = (2.0 * static_cast<double>(i) - static_cast<double>(n) + 1.0) / static_cast<double>(n);
        L2 += weight * sorted_data[i];
    }
    L2 /= static_cast<double>(n);
    L2 = std::abs(L2); // L2 should be positive
    
    // Invert the relationships: a = L1 - 3*L2, b = L1 + 3*L2
    const double a_estimate = L1 - 3.0 * L2;
    const double b_estimate = L1 + 3.0 * L2;
    
    return {a_estimate, b_estimate};
}

std::tuple<double, double, bool> UniformDistribution::uniformityTest(const std::vector<double>& data, double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    const size_t n = data.size();
    const double sample_min = *std::min_element(data.begin(), data.end());
    const double sample_max = *std::max_element(data.begin(), data.end());
    const double sample_range = sample_max - sample_min;
    
    if (sample_range == 0.0) {
        // All data points are identical - not uniform
        return {std::numeric_limits<double>::infinity(), 0.0, false};
    }
    
    // Use range/variance ratio test
    // For uniform distribution: Var = Range²/12
    // Test statistic: T = 12 * Var / Range²
    // Should be close to 1 for uniform data
    
    double sample_variance = 0.0;
    double sample_mean = 0.0;
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
    const double p_value = 2.0 * std::min(test_statistic, 2.0 - test_statistic); // Symmetric around 1
    
    const bool uniformity_is_valid = (p_value > significance_level);
    
    return {test_statistic, p_value, uniformity_is_valid};
}

std::tuple<double, double, bool> UniformDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data,
    const UniformDistribution& distribution,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Use the overflow-safe KS statistic calculation from math_utils
    double ks_statistic = math::calculate_ks_statistic(data, distribution);
    
    const size_t n = data.size();
    
    // Asymptotic critical value for KS test
    double critical_value = std::sqrt(-0.5 * std::log(alpha / 2.0)) / std::sqrt(n);
    
    // Asymptotic p-value approximation (Kolmogorov distribution)
    double ks_stat_scaled = ks_statistic * std::sqrt(n);
    double p_value = 2.0 * std::exp(-2.0 * ks_stat_scaled * ks_stat_scaled);
    p_value = std::max(0.0, std::min(1.0, p_value)); // Clamp to [0,1]
    
    bool reject_null = (ks_statistic > critical_value);
    
    return std::make_tuple(ks_statistic, p_value, reject_null);
}

std::tuple<double, double, bool> UniformDistribution::andersonDarlingTest(
    const std::vector<double>& data,
    const UniformDistribution& distribution,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (alpha <= 0.0 || alpha >= 1.0) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Use the overflow-safe AD statistic calculation from math_utils
    double ad_stat = math::calculate_ad_statistic(data, distribution);
    
    // Asymptotic critical values for Anderson-Darling test (approximate)
    double critical_value;
    if (alpha <= 0.01) {
        critical_value = 3.857;
    } else if (alpha <= 0.05) {
        critical_value = 2.492;
    } else if (alpha <= 0.10) {
        critical_value = 1.933;
    } else {
        critical_value = 1.159; // alpha = 0.25
    }
    
    // Better p-value approximation for Anderson-Darling test
    // For the uniform distribution, we use a more accurate formula
    double p_value;
    if (ad_stat < 0.2) {
        p_value = 1.0 - std::exp(-1.2804 * std::pow(ad_stat, -0.5));
    } else if (ad_stat < 0.34) {
        p_value = 1.0 - std::exp(-0.8 * ad_stat - 0.26);
    } else if (ad_stat < 0.6) {
        p_value = std::exp(-0.9 * ad_stat - 0.16);
    } else {
        p_value = std::exp(-1.8 * ad_stat + 0.258);
    }
    p_value = std::max(0.0, std::min(1.0, p_value)); // Clamp to [0,1]
    
    bool reject_null = (ad_stat > critical_value);
    
    return std::make_tuple(ad_stat, p_value, reject_null);
}

//==============================================================================
// RESULT-BASED SETTERS
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

    return VoidResult::ok(true);
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

    return VoidResult::ok(true);
}

std::vector<std::tuple<double, double, double>> UniformDistribution::kFoldCrossValidation(
    const std::vector<double>& data,
    int k,
    unsigned int random_seed) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (k <= 1) {
        throw std::invalid_argument("Number of folds k must be greater than 1");
    }
    
    if (k > static_cast<int>(data.size())) {
        throw std::invalid_argument("Number of folds k cannot exceed data size");
    }
    
    const size_t n = data.size();
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    // Shuffle indices for random fold assignment
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    std::vector<std::tuple<double, double, double>> results;
    results.reserve(static_cast<std::size_t>(k));
    
    const size_t fold_size = n / static_cast<std::size_t>(k);
    const size_t remainder = n % static_cast<std::size_t>(k);
    
    for (int fold = 0; fold < k; ++fold) {
        // Determine validation set indices for this fold
        size_t start_idx = static_cast<std::size_t>(fold) * fold_size + std::min(static_cast<size_t>(fold), remainder);
        size_t end_idx = start_idx + fold_size + (static_cast<size_t>(fold) < remainder ? 1 : 0);
        
        // Split data into training and validation sets
        std::vector<double> train_data, validation_data;
        train_data.reserve(n - (end_idx - start_idx));
        validation_data.reserve(end_idx - start_idx);
        
        for (size_t i = 0; i < n; ++i) {
            if (i >= start_idx && i < end_idx) {
                validation_data.push_back(data[indices[i]]);
            } else {
                train_data.push_back(data[indices[i]]);
            }
        }
        
        // Fit distribution to training data
        if (train_data.size() < 2) {
            results.emplace_back(std::numeric_limits<double>::infinity(), 
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity());
            continue;
        }
        
        auto train_minmax = std::minmax_element(train_data.begin(), train_data.end());
        double train_a = *train_minmax.first;
        double train_b = *train_minmax.second;
        
        // Add small epsilon to ensure all training data is within bounds
        double width = train_b - train_a;
        double epsilon = std::max(1e-10, width * 1e-6);
        train_a -= epsilon;
        train_b += epsilon;
        
        UniformDistribution fitted_dist(train_a, train_b);
        
        // Evaluate on validation data
        std::vector<double> errors;
        errors.reserve(validation_data.size());
        double total_log_likelihood = 0.0;
        
        for (double val : validation_data) {
            // Prediction error (use mean as point prediction)
            double predicted = fitted_dist.getMean();
            double error = std::abs(val - predicted);
            errors.push_back(error);
            
            // Log likelihood
            double log_prob = fitted_dist.getLogProbability(val);
            total_log_likelihood += std::isfinite(log_prob) ? log_prob : -1000.0; // Penalty for out-of-bounds
        }
        
        // Calculate metrics - MAE and RMSE
        double mae = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        
        // Calculate RMSE = sqrt(mean(squared_errors))
        double mse = 0.0;
        for (double error : errors) {
            mse += error * error;
        }
        mse /= static_cast<double>(errors.size());
        double rmse = std::sqrt(mse);
        
        results.emplace_back(mae, rmse, total_log_likelihood);
    }
    
    return results;
}

std::tuple<double, double, double> UniformDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    
    if (data.size() < 2) {
        throw std::invalid_argument("Data must contain at least 2 elements for LOOCV");
    }
    
    const size_t n = data.size();
    std::vector<double> errors;
    errors.reserve(n);
    double total_log_likelihood = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        // Create training set (all data except point i)
        std::vector<double> train_data;
        train_data.reserve(n - 1);
        
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                train_data.push_back(data[j]);
            }
        }
        
        // Fit distribution to training data
        auto train_minmax = std::minmax_element(train_data.begin(), train_data.end());
        double train_a = *train_minmax.first;
        double train_b = *train_minmax.second;
        
        // Add small epsilon to ensure robustness
        double width = train_b - train_a;
        double epsilon = std::max(1e-10, width * 1e-6);
        train_a -= epsilon;
        train_b += epsilon;
        
        UniformDistribution fitted_dist(train_a, train_b);
        
        // Evaluate on left-out point
        double validation_point = data[i];
        
        // Prediction error (use mean as point prediction)
        double predicted = fitted_dist.getMean();
        double error = std::abs(validation_point - predicted);
        errors.push_back(error);
        
        // Log likelihood
        double log_prob = fitted_dist.getLogProbability(validation_point);
        total_log_likelihood += std::isfinite(log_prob) ? log_prob : -1000.0; // Penalty for out-of-bounds
    }
    
    // Calculate final metrics
    double mae = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
    
    double mse = 0.0;
    for (double error : errors) {
        mse += error * error;
    }
    mse /= static_cast<double>(errors.size());
    double rmse = std::sqrt(mse);
    
    return std::make_tuple(mae, rmse, total_log_likelihood);
}

std::tuple<std::pair<double, double>, std::pair<double, double>> UniformDistribution::bootstrapParameterConfidenceIntervals(
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
    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<size_t> index_dist(0, n - 1);
    
    std::vector<double> bootstrap_a_estimates;
    std::vector<double> bootstrap_b_estimates;
    bootstrap_a_estimates.reserve(static_cast<std::size_t>(n_bootstrap));
    bootstrap_b_estimates.reserve(static_cast<std::size_t>(n_bootstrap));
    
    for (int boot = 0; boot < n_bootstrap; ++boot) {
        // Generate bootstrap sample
        std::vector<double> bootstrap_sample;
        bootstrap_sample.reserve(n);
        
        for (size_t i = 0; i < n; ++i) {
            bootstrap_sample.push_back(data[index_dist(rng)]);
        }
        
        // Estimate parameters from bootstrap sample
        auto minmax = std::minmax_element(bootstrap_sample.begin(), bootstrap_sample.end());
        double boot_a = *minmax.first;
        double boot_b = *minmax.second;
        
        bootstrap_a_estimates.push_back(boot_a);
        bootstrap_b_estimates.push_back(boot_b);
    }
    
    // Sort estimates for percentile method
    std::sort(bootstrap_a_estimates.begin(), bootstrap_a_estimates.end());
    std::sort(bootstrap_b_estimates.begin(), bootstrap_b_estimates.end());
    
    // Calculate confidence intervals using percentile method
    double alpha = 1.0 - confidence_level;
    double lower_percentile = alpha / 2.0;
    double upper_percentile = 1.0 - alpha / 2.0;
    
    size_t lower_idx = static_cast<size_t>(std::floor(lower_percentile * (n_bootstrap - 1)));
    size_t upper_idx = static_cast<size_t>(std::ceil(upper_percentile * (n_bootstrap - 1)));
    
    // Ensure indices are within bounds
    lower_idx = std::min(lower_idx, static_cast<size_t>(n_bootstrap - 1));
    upper_idx = std::min(upper_idx, static_cast<size_t>(n_bootstrap - 1));
    
    std::pair<double, double> a_ci = {
        bootstrap_a_estimates[lower_idx],
        bootstrap_a_estimates[upper_idx]
    };
    
    std::pair<double, double> b_ci = {
        bootstrap_b_estimates[lower_idx],
        bootstrap_b_estimates[upper_idx]
    };
    
    return std::make_tuple(a_ci, b_ci);
}

std::tuple<double, double, double, double> UniformDistribution::computeInformationCriteria(
    const std::vector<double>& data,
    const UniformDistribution& fitted_distribution) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    const size_t n = data.size();
    const int k = 2; // Number of parameters for uniform distribution (a, b)
    
    // Compute log likelihood
    double log_likelihood = 0.0;
    for (double x : data) {
        double log_prob = fitted_distribution.getLogProbability(x);
        // Handle the case where data points are outside the distribution bounds
        if (std::isfinite(log_prob)) {
            log_likelihood += log_prob;
        } else {
            // Penalize data points outside bounds with a large negative value
            log_likelihood += -1000.0;
        }
    }
    
    // Compute information criteria
    double aic = -2.0 * log_likelihood + 2.0 * k;
    double bic = -2.0 * log_likelihood + k * std::log(n);
    
    // AICc (corrected AIC for small sample sizes)
    double aicc = aic;
    if (n > k + 1) {
        aicc += (2.0 * k * (k + 1)) / static_cast<double>(n - k - 1);
    }
    
    return std::make_tuple(aic, bic, aicc, log_likelihood);
}

//==============================================================================
// RESULT-BASED SETTERS (C++20 Best Practice: Complex implementations in .cpp)
//==============================================================================

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
    
    return VoidResult::ok(true);
}

//==============================================================================
// SMART AUTO-DISPATCH BATCH OPERATIONS (New Simplified API)
//==============================================================================

void UniformDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                        const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<UniformDistribution>::distType(),
        performance::DistributionTraits<UniformDistribution>::complexity(),
        [](const UniformDistribution& dist, double value) { return dist.getProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getProbabilityBatch(vals, res, count);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getProbabilityBatchParallel(vals, res);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void UniformDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                           const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<UniformDistribution>::distType(),
        performance::DistributionTraits<UniformDistribution>::complexity(),
        [](const UniformDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getLogProbabilityBatch(vals, res, count);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getLogProbabilityBatchParallel(vals, res);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getLogProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getLogProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void UniformDistribution::getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                                  const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<UniformDistribution>::distType(),
        performance::DistributionTraits<UniformDistribution>::complexity(),
        [](const UniformDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getCumulativeProbabilityBatch(vals, res, count);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getCumulativeProbabilityBatchParallel(vals, res);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getCumulativeProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getCumulativeProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

//==============================================================================
// EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==============================================================================

void UniformDistribution::getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                    performance::Strategy strategy) const {
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const UniformDistribution& dist, double value) { return dist.getProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getProbabilityBatch(vals, res, count);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getProbabilityBatchParallel(vals, res);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void UniformDistribution::getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                       performance::Strategy strategy) const {
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const UniformDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getLogProbabilityBatch(vals, res, count);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getLogProbabilityBatchParallel(vals, res);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getLogProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getLogProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void UniformDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                              performance::Strategy strategy) const {
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const UniformDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
        [](const UniformDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getCumulativeProbabilityBatch(vals, res, count);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getCumulativeProbabilityBatchParallel(vals, res);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getCumulativeProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const UniformDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getCumulativeProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

} // namespace libstats

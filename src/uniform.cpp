#include "../include/distributions/uniform.h"
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
    // For uniform distribution, the computation is extremely simple (just bounds checking)
    // so SIMD overhead is not beneficial - use direct scalar implementation
    const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                 (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    if (is_unit_interval) {
        // Unit interval case: result is 1 for x in [0,1], 0 otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            results[i] = (x >= constants::math::ZERO_DOUBLE && x <= constants::math::ONE) ? 
                        constants::math::ONE : constants::math::ZERO_DOUBLE;
        }
    } else {
        // General case: result is inv_width for x in [a,b], 0 otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            results[i] = (x >= a && x <= b) ? inv_width : constants::math::ZERO_DOUBLE;
        }
    }
}

void UniformDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                           double a, double b, double log_inv_width) const noexcept {
    // For uniform distribution, the computation is extremely simple (just bounds checking)
    // so SIMD overhead is not beneficial - use direct scalar implementation
    const bool is_unit_interval = (std::abs(a - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                                 (std::abs(b - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
    
    if (is_unit_interval) {
        // Unit interval case: result is 0 for x in [0,1], -∞ otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            results[i] = (x >= constants::math::ZERO_DOUBLE && x <= constants::math::ONE) ? 
                        constants::math::ZERO_DOUBLE : constants::probability::NEGATIVE_INFINITY;
        }
    } else {
        // General case: result is log_inv_width for x in [a,b], -∞ otherwise
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            results[i] = (x >= a && x <= b) ? log_inv_width : constants::probability::NEGATIVE_INFINITY;
        }
    }
}

void UniformDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                  double a, double b, double inv_width) const noexcept {
    // For uniform distribution CDF, the computation is simple (bounds checking + linear interpolation)
    // so SIMD overhead is not beneficial - use direct scalar implementation like PDF and LogPDF
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use much higher threshold for simple distribution operations to avoid thread pool overhead
    // Simple operations like uniform PDF have minimal computation per element
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element in parallel
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ONE : cached_inv_width;
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ONE : cached_inv_width;
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }
    }
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use much higher threshold for simple distribution operations to avoid thread pool overhead
    // Simple operations like uniform log PDF have minimal computation per element
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element in parallel
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ZERO_DOUBLE : cached_log_inv_width;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ZERO_DOUBLE : cached_log_inv_width;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }
    }
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use much higher threshold for simple distribution operations to avoid thread pool overhead
    // Simple operations like uniform CDF have minimal computation per element
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element in parallel
            if (values[i] < cached_a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] > cached_b) {
                results[i] = constants::math::ONE;
            } else if (cached_is_unit_interval) {
                results[i] = values[i];  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (values[i] - cached_a) * cached_inv_width;
            }
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < cached_a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] > cached_b) {
                results[i] = constants::math::ONE;
            } else if (cached_is_unit_interval) {
                results[i] = values[i];  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (values[i] - cached_a) * cached_inv_width;
            }
        }
    }
}

void UniformDistribution::getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                         WorkStealingPool& pool) const {
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use work-stealing pool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element with work stealing
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ONE : cached_inv_width;
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        });
    } else {
        // Fall back to regular parallel processing
        getProbabilityBatchParallel(values, results);
    }
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use cache-aware processing with adaptive batch sizes
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "uniform_pdf");
    
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        // Use single parallel region with cache-aware grain size
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PDF for each element with cache awareness
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ONE : cached_inv_width;
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }, optimal_grain_size);
        
        // Update cache performance metrics
        cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ONE : cached_inv_width;
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }
    }
}

void UniformDistribution::getLogProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                            WorkStealingPool& pool) const {
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use work-stealing pool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element with work stealing
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ZERO_DOUBLE : cached_log_inv_width;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        });
        
        pool.waitForAll();
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ZERO_DOUBLE : cached_log_inv_width;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }
    }
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "uniform_log_pdf");
    
    // Use cache-aware parallel processing with same threshold as regular parallel operations
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PDF for each element with cache-aware access patterns
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ZERO_DOUBLE : cached_log_inv_width;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }, optimal_grain_size);
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] >= cached_a && values[i] <= cached_b) {
                results[i] = cached_is_unit_interval ? constants::math::ZERO_DOUBLE : cached_log_inv_width;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }
    }
    
    // Update cache manager with performance metrics
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void UniformDistribution::getCumulativeProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                                   WorkStealingPool& pool) const {
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Use work-stealing pool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with work stealing
            if (values[i] < cached_a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] > cached_b) {
                results[i] = constants::math::ONE;
            } else if (cached_is_unit_interval) {
                results[i] = values[i];  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (values[i] - cached_a) * cached_inv_width;
            }
        });
        
        pool.waitForAll();
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < cached_a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] > cached_b) {
                results[i] = constants::math::ONE;
            } else if (cached_is_unit_interval) {
                results[i] = values[i];  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (values[i] - cached_a) * cached_inv_width;
            }
        }
    }
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
    const bool cached_is_unit_interval = isUnitInterval_;
    
    lock.unlock();
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "uniform_cdf");
    
    // Use cache-aware parallel processing with same threshold as regular parallel operations
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with cache-aware access patterns
            if (values[i] < cached_a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] > cached_b) {
                results[i] = constants::math::ONE;
            } else if (cached_is_unit_interval) {
                results[i] = values[i];  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (values[i] - cached_a) * cached_inv_width;
            }
        }, optimal_grain_size);
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < cached_a) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] > cached_b) {
                results[i] = constants::math::ONE;
            } else if (cached_is_unit_interval) {
                results[i] = values[i];  // CDF(x) = x for U(0,1)
            } else {
                results[i] = (values[i] - cached_a) * cached_inv_width;
            }
        }
    }
    
    // Update cache manager with performance metrics
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

//==============================================================================
// ADVANCED STATISTICAL METHODS
//==============================================================================

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
    
    const size_t n = data.size();
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Compute KS statistic: max difference between empirical and theoretical CDF
    double max_diff = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double empirical_cdf = static_cast<double>(i + 1) / n;
        double theoretical_cdf = distribution.getCumulativeProbability(sorted_data[i]);
        
        // Check both F_n(x) - F(x) and F(x) - F_{n-1}(x)
        double diff1 = std::abs(empirical_cdf - theoretical_cdf);
        double diff2 = (i > 0) ? std::abs(theoretical_cdf - static_cast<double>(i) / n) : theoretical_cdf;
        
        max_diff = std::max({max_diff, diff1, diff2});
    }
    
    // Asymptotic critical value for KS test
    double critical_value = std::sqrt(-0.5 * std::log(alpha / 2.0)) / std::sqrt(n);
    
    // Asymptotic p-value approximation (Kolmogorov distribution)
    double ks_stat_scaled = max_diff * std::sqrt(n);
    double p_value = 2.0 * std::exp(-2.0 * ks_stat_scaled * ks_stat_scaled);
    p_value = std::max(0.0, std::min(1.0, p_value)); // Clamp to [0,1]
    
    bool reject_null = (max_diff > critical_value);
    
    return std::make_tuple(max_diff, p_value, reject_null);
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
    
    const size_t n = data.size();
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Compute Anderson-Darling statistic
    double ad_stat = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        double cdf_val = distribution.getCumulativeProbability(sorted_data[i]);
        
        // Clamp CDF to avoid log(0) - be more conservative with clamping
        cdf_val = std::max(1e-15, std::min(1.0 - 1e-15, cdf_val));
        
        // Anderson-Darling formula: A² = -n - (1/n) * Σ[(2i-1)*ln(F(xi)) + (2(n-i)+1)*ln(1-F(xi))]
        double i_plus_1 = static_cast<double>(i + 1);
        double n_double = static_cast<double>(n);
        
        double term1 = (2.0 * i_plus_1 - 1.0) * std::log(cdf_val);
        double term2 = (2.0 * (n_double - i_plus_1) + 1.0) * std::log(1.0 - cdf_val);
        
        // Check for numerical issues
        if (std::isfinite(term1) && std::isfinite(term2)) {
            ad_stat += term1 + term2;
        }
    }
    
    ad_stat = -static_cast<double>(n) - ad_stat / static_cast<double>(n);
    
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
    results.reserve(k);
    
    const size_t fold_size = n / k;
    const size_t remainder = n % k;
    
    for (int fold = 0; fold < k; ++fold) {
        // Determine validation set indices for this fold
        size_t start_idx = fold * fold_size + std::min(static_cast<size_t>(fold), remainder);
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
        
        // Calculate metrics
        double mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        
        double variance = 0.0;
        for (double error : errors) {
            variance += (error - mean_error) * (error - mean_error);
        }
        double std_error = std::sqrt(variance / errors.size());
        
        results.emplace_back(mean_error, std_error, total_log_likelihood);
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
    double mae = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
    
    double mse = 0.0;
    for (double error : errors) {
        mse += error * error;
    }
    mse /= errors.size();
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
    bootstrap_a_estimates.reserve(n_bootstrap);
    bootstrap_b_estimates.reserve(n_bootstrap);
    
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
        aicc += (2.0 * k * (k + 1)) / (n - k - 1);
    }
    
    return std::make_tuple(aic, bic, aicc, log_likelihood);
}

} // namespace libstats

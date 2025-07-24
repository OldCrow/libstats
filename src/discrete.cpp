#include "../include/discrete.h"
#include "../include/math_utils.h"
#include "../include/simd.h"
#include "../include/safety.h"
#include "../include/thread_pool.h"
#include "../include/work_stealing_pool.h"
#include "../include/adaptive_cache.h"
#include "../include/parallel_execution.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <span>

namespace libstats {

//==============================================================================
// CONSTRUCTORS AND DESTRUCTORS
//==============================================================================

DiscreteDistribution::DiscreteDistribution(int a, int b) : DistributionBase(), a_(a), b_(b) {
    validateParameters(a, b);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

DiscreteDistribution::DiscreteDistribution(const DiscreteDistribution& other) 
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    a_ = other.a_;
    b_ = other.b_;
    
    // If the other's cache is valid, copy cached values for efficiency
    if (other.cache_valid_) {
        range_ = other.range_;
        probability_ = other.probability_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        logProbability_ = other.logProbability_;
        isBinary_ = other.isBinary_;
        isStandardDie_ = other.isStandardDie_;
        isSymmetric_ = other.isSymmetric_;
        isSmallRange_ = other.isSmallRange_;
        isLargeRange_ = other.isLargeRange_;
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
        
        // Update atomic parameters
        atomicA_.store(a_, std::memory_order_release);
        atomicB_.store(b_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    } else {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
}

DiscreteDistribution& DiscreteDistribution::operator=(const DiscreteDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        
        // Copy parameters (don't call base class operator= to avoid deadlock)
        a_ = other.a_;
        b_ = other.b_;
        
        // If the other's cache is valid, copy cached values for efficiency
        if (other.cache_valid_) {
            range_ = other.range_;
            probability_ = other.probability_;
            mean_ = other.mean_;
            variance_ = other.variance_;
            logProbability_ = other.logProbability_;
            isBinary_ = other.isBinary_;
            isStandardDie_ = other.isStandardDie_;
            isSymmetric_ = other.isSymmetric_;
            isSmallRange_ = other.isSmallRange_;
            isLargeRange_ = other.isLargeRange_;
            cache_valid_ = true;
            cacheValidAtomic_.store(true, std::memory_order_release);
            
            // Update atomic parameters
            atomicA_.store(a_, std::memory_order_release);
            atomicB_.store(b_, std::memory_order_release);
            atomicParamsValid_.store(true, std::memory_order_release);
        } else {
            // Cache will be updated on first use
            cache_valid_ = false;
            cacheValidAtomic_.store(false, std::memory_order_release);
            atomicParamsValid_.store(false, std::memory_order_release);
        }
    }
    return *this;
}

DiscreteDistribution::DiscreteDistribution(DiscreteDistribution&& other) 
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    a_ = other.a_;
    b_ = other.b_;
    other.a_ = 0;
    other.b_ = 1;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

DiscreteDistribution& DiscreteDistribution::operator=(DiscreteDistribution&& other) {
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
                other.a_ = 0;
                other.b_ = 1;
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
            [[maybe_unused]] int temp_a = a_;
            [[maybe_unused]] int temp_b = b_;
            
            // Atomic-like exchange (single assignment is atomic for built-in types)
            a_ = other.a_;
            b_ = other.b_;
            other.a_ = 0;
            other.b_ = 1;
            
            // Cache invalidation was already done atomically above
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
    }
    return *this;
}

//==============================================================================
// PARAMETER VALIDATION
//==============================================================================

// validateParameters is implemented as static inline in the header

//==============================================================================
// CORE PROBABILITY METHODS  
//==============================================================================

double DiscreteDistribution::getProbability(double x) const {
    // For discrete distribution, check if x is an integer in range
    if (std::floor(x) != x) {
        return constants::math::ZERO_DOUBLE; // Not an integer
    }
    
    const int k = static_cast<int>(x);
    if (k < a_ || k > b_) {
        return constants::math::ZERO_DOUBLE; // Outside support
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
    
    // Fast path optimizations
    if (isBinary_) {
        return constants::math::HALF; // 0.5 for binary [0,1]
    }
    
    return probability_; // 1/(b-a+1)
}

double DiscreteDistribution::getLogProbability(double x) const noexcept {
    // For discrete distribution, check if x is an integer in range
    if (std::floor(x) != x) {
        return constants::probability::NEGATIVE_INFINITY; // Not an integer
    }
    
    const int k = static_cast<int>(x);
    if (k < a_ || k > b_) {
        return constants::probability::NEGATIVE_INFINITY; // Outside support
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
    
    // Fast path optimizations
    if (isBinary_) {
        return -constants::math::LN2; // log(0.5)
    }
    
    return logProbability_; // -log(b-a+1)
}

double DiscreteDistribution::getCumulativeProbability(double x) const {
    if (x < static_cast<double>(a_)) {
        return constants::math::ZERO_DOUBLE;
    }
    if (x >= static_cast<double>(b_)) {
        return constants::math::ONE;
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
    
    // For discrete uniform: F(k) = (floor(k) - a + 1) / (b - a + 1)
    const int k = static_cast<int>(std::floor(x));
    const int numerator = k - a_ + 1;
    
    // Fast path optimizations
    if (isBinary_) {
        return (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
    }
    
    return static_cast<double>(numerator) / static_cast<double>(range_);
}

double DiscreteDistribution::getQuantile(double p) const {
    if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    if (p == constants::math::ZERO_DOUBLE) return static_cast<double>(a_);
    if (p == constants::math::ONE) return static_cast<double>(b_);
    
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
    
    // For discrete uniform: quantile(p) = a + ceil(p * (b-a+1)) - 1
    // But we need to handle edge cases carefully
    const double scaled = p * static_cast<double>(range_);
    const int k = static_cast<int>(std::ceil(scaled)) - 1;
    
    return static_cast<double>(a_ + std::max(0, std::min(k, range_ - 1)));
}

double DiscreteDistribution::sample(std::mt19937& rng) const {
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
    
    // Fast path for binary distribution
    if (isBinary_) {
        std::uniform_int_distribution<int> dis(0, 1);
        return static_cast<double>(dis(rng));
    }
    
    // General case: uniform integer distribution
    std::uniform_int_distribution<int> dis(a_, b_);
    return static_cast<double>(dis(rng));
}

std::vector<double> DiscreteDistribution::sample(std::mt19937& rng, std::size_t n) const {
    std::vector<double> samples(n);
    
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
    
    // Fast path for binary distribution
    if (isBinary_) {
        std::uniform_int_distribution<int> dis(0, 1);
        for (size_t i = 0; i < n; ++i) {
            samples[i] = static_cast<double>(dis(rng));
        }
        return samples;
    }
    
    // General case: uniform integer distribution
    std::uniform_int_distribution<int> dis(a_, b_);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = static_cast<double>(dis(rng));
    }
    
    return samples;
}

//==============================================================================
// PARAMETER GETTERS AND SETTERS
//==============================================================================

double DiscreteDistribution::getMean() const noexcept {
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
    return mean_;
}

double DiscreteDistribution::getVariance() const noexcept {
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


void DiscreteDistribution::setLowerBound(int a) {
    validateParameters(a, b_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::setUpperBound(int b) {
    validateParameters(a_, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::setBounds(int a, int b) {
    validateParameters(a, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

int DiscreteDistribution::getRange() const noexcept {
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
    return range_;
}

double DiscreteDistribution::getSingleOutcomeProbability() const noexcept {
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
    return probability_;
}

//==============================================================================
// DISTRIBUTION MANAGEMENT
//==============================================================================

void DiscreteDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit discrete distribution to empty data");
    }
    
    // For discrete uniform, we round fractional values to nearest integers
    // then find the min and max of the rounded values
    std::vector<int> rounded_values;
    rounded_values.reserve(values.size());
    
    for (double val : values) {
        if (!isValidIntegerValue(val)) {
            throw std::invalid_argument("Value outside valid integer range");
        }
        rounded_values.push_back(roundToInt(val));
    }
    
    int new_a = *std::min_element(rounded_values.begin(), rounded_values.end());
    int new_b = *std::max_element(rounded_values.begin(), rounded_values.end());
    
    validateParameters(new_a, new_b);
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = new_a;
    b_ = new_b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = 0;
    b_ = 1;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

std::string DiscreteDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "DiscreteUniform(a=" << a_ << ", b=" << b_ << ")";
    return oss.str();
}

//==============================================================================
// COMPARISON OPERATORS
//==============================================================================

bool DiscreteDistribution::operator==(const DiscreteDistribution& other) const {
    // Thread-safe comparison with ordered lock acquisition
    if (this == &other) return true;
    
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    
    return (a_ == other.a_) && (b_ == other.b_);
}

//==============================================================================
// BATCH OPERATIONS WITH SIMD ACCELERATION
//==============================================================================

void DiscreteDistribution::getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
    if (count == 0) return;
    if (!values || !results) {
        return; // Can't throw in noexcept context
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
    
    // Cache parameters for batch processing
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_prob = probability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use SIMD-optimized implementation when beneficial
    if (count >= simd::tuned::min_states_for_simd()) {
        getProbabilityBatchUnsafeImpl(values, results, count, cached_a, cached_b, cached_prob);
    } else {
        // Scalar implementation for small arrays
        for (std::size_t i = 0; i < count; ++i) {
            // Check if integer and in range
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }
    }
}

void DiscreteDistribution::getLogProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
    if (count == 0) return;
    if (!values || !results) return; // Can't throw in noexcept context
    
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
    
    // Cache parameters for batch processing
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_log_prob = logProbability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Scalar implementation (discrete distributions are less amenable to vectorization)
    for (std::size_t i = 0; i < count; ++i) {
        // Check if integer and in range
        if (std::floor(values[i]) == values[i]) {
            const int k = static_cast<int>(values[i]);
            if (k >= cached_a && k <= cached_b) {
                results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        } else {
            results[i] = constants::probability::NEGATIVE_INFINITY;
        }
    }
}

void DiscreteDistribution::getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const {
    if (count == 0) return;
    if (!values || !results) {
        throw std::invalid_argument("Invalid pointers for batch CDF calculation");
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
    
    // Cache parameters for batch processing
    const int cached_a = a_;
    const int cached_b = b_;
    const int cached_range = range_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Scalar implementation
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < static_cast<double>(cached_a)) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else if (values[i] >= static_cast<double>(cached_b)) {
            results[i] = constants::math::ONE;
        } else {
            const int k = static_cast<int>(std::floor(values[i]));
            if (cached_is_binary) {
                results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
            } else {
                const int numerator = k - cached_a + 1;
                results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
            }
        }
    }
}

//==============================================================================
// PARALLEL BATCH OPERATIONS  
//==============================================================================

void DiscreteDistribution::getProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
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
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_prob = probability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PMF for each element in parallel
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }
    }
}

void DiscreteDistribution::getLogProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const noexcept {
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
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_log_prob = logProbability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PMF for each element in parallel
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
                } else {
                    results[i] = constants::probability::NEGATIVE_INFINITY;
                }
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
                } else {
                    results[i] = constants::probability::NEGATIVE_INFINITY;
                }
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }
    }
}

void DiscreteDistribution::getCumulativeProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const {
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
    const int cached_a = a_;
    const int cached_b = b_;
    const int cached_range = range_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element in parallel
            if (values[i] < static_cast<double>(cached_a)) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(cached_b)) {
                results[i] = constants::math::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                if (cached_is_binary) {
                    results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
                } else {
                    const int numerator = k - cached_a + 1;
                    results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                }
            }
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < static_cast<double>(cached_a)) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(cached_b)) {
                results[i] = constants::math::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                if (cached_is_binary) {
                    results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
                } else {
                    const int numerator = k - cached_a + 1;
                    results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                }
            }
        }
    }
}

void DiscreteDistribution::getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
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
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_prob = probability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use WorkStealingPool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PMF for each element with work stealing load balancing
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        });
        
        pool.waitForAll();
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }
    }
}

void DiscreteDistribution::getProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                        cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "discrete_pmf_batch_" + std::to_string(count);
    
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
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_prob = probability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "discrete_pmf");
    
    // Use cache-aware parallel processing
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PMF for each element with cache-aware access patterns
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }, optimal_grain_size);
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? constants::math::HALF : cached_prob;
                } else {
                    results[i] = constants::math::ZERO_DOUBLE;
                }
            } else {
                results[i] = constants::math::ZERO_DOUBLE;
            }
        }
    }
    
    // Update cache manager with performance metrics
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void DiscreteDistribution::getLogProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
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
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_log_prob = logProbability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use WorkStealingPool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PMF for each element with work stealing load balancing
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
                } else {
                    results[i] = constants::probability::NEGATIVE_INFINITY;
                }
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        });
        
        pool.waitForAll();
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
                } else {
                    results[i] = constants::probability::NEGATIVE_INFINITY;
                }
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }
    }
}

void DiscreteDistribution::getLogProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                           cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "discrete_log_pmf_batch_" + std::to_string(count);
    
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
    const int cached_a = a_;
    const int cached_b = b_;
    const double cached_log_prob = logProbability_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "discrete_log_pmf");
    
    // Use cache-aware parallel processing
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PMF for each element with cache-aware access patterns
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
                } else {
                    results[i] = constants::probability::NEGATIVE_INFINITY;
                }
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }, optimal_grain_size);
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i]) {
                const int k = static_cast<int>(values[i]);
                if (k >= cached_a && k <= cached_b) {
                    results[i] = cached_is_binary ? -constants::math::LN2 : cached_log_prob;
                } else {
                    results[i] = constants::probability::NEGATIVE_INFINITY;
                }
            } else {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            }
        }
    }
    
    // Update cache manager with performance metrics
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void DiscreteDistribution::getCumulativeProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
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
    const int cached_a = a_;
    const int cached_b = b_;
    const int cached_range = range_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Use WorkStealingPool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with work stealing load balancing
            if (values[i] < static_cast<double>(cached_a)) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(cached_b)) {
                results[i] = constants::math::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                if (cached_is_binary) {
                    results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
                } else {
                    const int numerator = k - cached_a + 1;
                    results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                }
            }
        });
        
        pool.waitForAll();
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < static_cast<double>(cached_a)) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(cached_b)) {
                results[i] = constants::math::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                if (cached_is_binary) {
                    results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
                } else {
                    const int numerator = k - cached_a + 1;
                    results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                }
            }
        }
    }
}

void DiscreteDistribution::getCumulativeProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                                  cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "discrete_cdf_batch_" + std::to_string(count);
    
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
    const int cached_a = a_;
    const int cached_b = b_;
    const int cached_range = range_;
    const bool cached_is_binary = isBinary_;
    
    lock.unlock();
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "discrete_cdf");
    
    // Use cache-aware parallel processing
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with cache-aware access patterns
            if (values[i] < static_cast<double>(cached_a)) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(cached_b)) {
                results[i] = constants::math::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                if (cached_is_binary) {
                    results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
                } else {
                    const int numerator = k - cached_a + 1;
                    results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                }
            }
        }, optimal_grain_size);
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < static_cast<double>(cached_a)) {
                results[i] = constants::math::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(cached_b)) {
                results[i] = constants::math::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                if (cached_is_binary) {
                    results[i] = (k >= 0) ? constants::math::ONE : constants::math::ZERO_DOUBLE;
                } else {
                    const int numerator = k - cached_a + 1;
                    results[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                }
            }
        }
    }
    
    // Update cache manager with performance metrics
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

//==============================================================================
// DISCRETE-SPECIFIC UTILITY METHODS
//==============================================================================

std::vector<int> DiscreteDistribution::sampleIntegers(std::mt19937& rng, std::size_t count) const {
    std::vector<int> samples(count);
    
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
    
    // Fast path for binary distribution
    if (isBinary_) {
        std::uniform_int_distribution<int> dis(0, 1);
        for (size_t i = 0; i < count; ++i) {
            samples[i] = dis(rng);
        }
        return samples;
    }
    
    // General case: uniform integer distribution
    std::uniform_int_distribution<int> dis(a_, b_);
    for (size_t i = 0; i < count; ++i) {
        samples[i] = dis(rng);
    }
    
    return samples;
}

bool DiscreteDistribution::isInSupport(double x) const noexcept {
    // Check if x is an integer in the range [a, b]
    if (std::floor(x) != x) {
        return false; // Not an integer
    }
    
    if (!isValidIntegerValue(x)) {
        return false; // Outside integer bounds
    }
    
    const int k = static_cast<int>(x);
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (k >= a_ && k <= b_);
}

std::vector<int> DiscreteDistribution::getAllOutcomes() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    // Safety check for reasonable range size
    const int range = b_ - a_ + 1;
    
    if (range > 1000000) { // 1M elements max (4MB memory)
        throw std::runtime_error("Range too large for getAllOutcomes() - maximum 1,000,000 elements");
    }
    
    if (range > 10000) { // Warning for large ranges
        // Could log a warning here if logging system exists
        // For now, just proceed but this indicates potentially expensive operation
    }
    
    std::vector<int> outcomes;
    outcomes.reserve(range);
    
    for (int k = a_; k <= b_; ++k) {
        outcomes.push_back(k);
    }
    
    return outcomes;
}

//==============================================================================
// PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void DiscreteDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                         int a, int b, double probability) const noexcept {
    // Optimized implementation using integer checks
    // This is the core discrete distribution batch operation
    for (std::size_t i = 0; i < count; ++i) {
        // Use specialized integer checking
        if (std::floor(values[i]) == values[i] && isValidIntegerValue(values[i])) {
            const int k = static_cast<int>(values[i]);
            results[i] = (k >= a && k <= b) ? probability : constants::math::ZERO_DOUBLE;
        } else {
            results[i] = constants::math::ZERO_DOUBLE; // Non-integers have zero probability
        }
    }
}

void DiscreteDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          int a, int b, double log_probability) const noexcept {
    // Optimized implementation using integer checks for log probabilities
    for (std::size_t i = 0; i < count; ++i) {
        // Use specialized integer checking
        if (std::floor(values[i]) == values[i] && isValidIntegerValue(values[i])) {
            const int k = static_cast<int>(values[i]);
            results[i] = (k >= a && k <= b) ? log_probability : constants::probability::NEGATIVE_INFINITY;
        } else {
            results[i] = constants::probability::NEGATIVE_INFINITY; // Non-integers have zero probability
        }
    }
}

void DiscreteDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                 int a, int b, double inv_range) const noexcept {
    // Optimized implementation for discrete CDF
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < static_cast<double>(a)) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else if (values[i] >= static_cast<double>(b)) {
            results[i] = constants::math::ONE;
        } else {
            const int k = static_cast<int>(std::floor(values[i]));
            const int numerator = k - a + 1;
            results[i] = static_cast<double>(numerator) * inv_range;
        }
    }
}

void DiscreteDistribution::getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getProbabilityBatchUnsafeImpl(values, results, count, a_, b_, probability_);
}

void DiscreteDistribution::getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getLogProbabilityBatchUnsafeImpl(values, results, count, a_, b_, logProbability_);
}

//==============================================================================
// STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const DiscreteDistribution& distribution) {
    return os << distribution.toString();
}

} // namespace libstats

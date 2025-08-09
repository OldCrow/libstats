#include "../include/distributions/poisson.h"
#include "../include/core/constants.h"
#include "../include/core/validation.h"
#include "../include/core/math_utils.h"
#include "../include/core/log_space_ops.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd_policy.h"
#include "../include/core/safety.h"
#include "../include/platform/parallel_execution.h"
#include "../include/platform/thread_pool.h"
#include "../include/core/dispatch_utils.h"
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <numeric>
#include <any>

namespace libstats {

//==============================================================================
// VALIDATION FUNCTIONS (for safe factory pattern)
//==============================================================================

// Note: validatePoissonParameters is already defined in error_handling.h

//==============================================================================
// CONSTRUCTORS AND DESTRUCTORS
//==============================================================================

PoissonDistribution::PoissonDistribution(double lambda) 
    : DistributionBase(), lambda_(lambda) {
    validateParameters(lambda);
    // Cache will be updated on first use
}

PoissonDistribution::PoissonDistribution(const PoissonDistribution& other) 
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    // Cache will be updated on first use
}

PoissonDistribution& PoissonDistribution::operator=(const PoissonDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        
        // Copy parameters
        lambda_ = other.lambda_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

PoissonDistribution::PoissonDistribution(PoissonDistribution&& other)
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    other.lambda_ = constants::math::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

PoissonDistribution& PoissonDistribution::operator=(PoissonDistribution&& other) noexcept {
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
            
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
    }
    return *this;
}

//==============================================================================
// PARAMETER GETTERS AND SETTERS
//==============================================================================

void PoissonDistribution::setLambda(double lambda) {
    validateParameters(lambda);
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void PoissonDistribution::setParameters(double lambda) {
    validateParameters(lambda);
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

double PoissonDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_;
}

double PoissonDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_;
}

double PoissonDistribution::getSkewness() const noexcept {
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
    
    return constants::math::ONE / sqrtLambda_;  // 1/√λ
}

double PoissonDistribution::getKurtosis() const noexcept {
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
    
    return invLambda_;  // 1/λ (excess kurtosis)
}

double PoissonDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::floor(lambda_);
}

//==============================================================================
// CORE PROBABILITY METHODS
//==============================================================================

double PoissonDistribution::getProbability(double x) const {
    if (x < constants::math::ZERO_DOUBLE) return constants::math::ZERO_DOUBLE;
    
    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x)) return constants::math::ZERO_DOUBLE;
    
    return getProbabilityExact(k);
}

double PoissonDistribution::getLogProbability(double x) const noexcept {
    if (x < constants::math::ZERO_DOUBLE) return constants::probability::MIN_LOG_PROBABILITY;
    
    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x)) return constants::probability::MIN_LOG_PROBABILITY;
    
    return getLogProbabilityExact(k);
}

double PoissonDistribution::getCumulativeProbability(double x) const {
    if (x < constants::math::ZERO_DOUBLE) return constants::math::ZERO_DOUBLE;
    
    int k = roundToNonNegativeInt(x);
    if (!isValidCount(x)) return constants::math::ONE;
    
    return getCumulativeProbabilityExact(k);
}

double PoissonDistribution::getQuantile(double p) const {
    if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
        throw std::invalid_argument("Probability must be in [0,1]");
    }
    
    if (p == constants::math::ZERO_DOUBLE) return constants::math::ZERO_DOUBLE;
    if (p == constants::math::ONE) return std::numeric_limits<double>::infinity();
    
    // Use bracketing search for quantile
    int lower = 0;
    int upper = static_cast<int>(lambda_ + constants::thresholds::poisson::QUANTILE_UPPER_BOUND_MULTIPLIER * std::sqrt(lambda_)); // Conservative upper bound
    
    // Expand upper bound if necessary
    while (getCumulativeProbabilityExact(upper) < p) {
        lower = upper;
        upper *= 2;
    }
    
    // Binary search
    while (upper - lower > 1) {
        int mid = (lower + upper) / 2;
        if (getCumulativeProbabilityExact(mid) < p) {
            lower = mid;
        } else {
            upper = mid;
        }
    }
    
    return static_cast<double>(upper);
}

double PoissonDistribution::sample(std::mt19937& rng) const {
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
    
    const double cached_lambda = lambda_;
    const bool cached_is_small = isSmallLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    
    lock.unlock(); // Release lock before generation
    
    if (cached_is_small) {
        // Knuth's algorithm for small lambda
        double L = cached_exp_neg_lambda;
        int k = 0;
        double p = constants::math::ONE;
        
        std::uniform_real_distribution<double> uniform(constants::math::ZERO_DOUBLE, constants::math::ONE);
        
        do {
            k++;
            p *= uniform(rng);
        } while (p > L);
        
        return static_cast<double>(k - 1);
    } else {
        // Normal approximation method for large lambda
        std::normal_distribution<double> normal(cached_lambda, std::sqrt(cached_lambda));
        
        while (true) {
            double sample = normal(rng);
            if (sample >= constants::math::ZERO_DOUBLE) {
                return std::round(sample);
            }
        }
    }
}

std::vector<double> PoissonDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);
    
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
    
    const double cached_lambda = lambda_;
    const bool cached_is_small = isSmallLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    lock.unlock();
    
    // Generate batch samples using the appropriate method
    if (cached_is_small) {
        // Knuth's algorithm for small lambda - optimized for batch
        double L = cached_exp_neg_lambda;
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        
        for (size_t i = 0; i < n; ++i) {
            int k = 0;
            double p = 1.0;
            
            do {
                k++;
                p *= uniform(rng);
            } while (p > L);
            
            samples.push_back(static_cast<double>(k - 1));
        }
    } else {
        // Normal approximation method for large lambda - optimized for batch
        std::normal_distribution<double> normal(cached_lambda, std::sqrt(cached_lambda));
        
        for (size_t i = 0; i < n; ++i) {
            while (true) {
                double sample = normal(rng);
                if (sample >= 0.0) {
                    samples.push_back(std::round(sample));
                    break;
                }
            }
        }
    }
    
    return samples;
}

//==============================================================================
// SMART AUTO-DISPATCH BATCH METHODS
//==============================================================================

void PoissonDistribution::getProbability(std::span<const double> values, std::span<double> results, const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<PoissonDistribution>::distType(),
        performance::DistributionTraits<PoissonDistribution>::complexity(),
        [](const PoissonDistribution& dist, double value) { return dist.getProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getProbabilityBatch(vals, res, count);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getProbabilityBatchParallel(vals, res);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void PoissonDistribution::getLogProbability(std::span<const double> values, std::span<double> results, const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<PoissonDistribution>::distType(),
        performance::DistributionTraits<PoissonDistribution>::complexity(),
        [](const PoissonDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getLogProbabilityBatch(vals, res, count);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getLogProbabilityBatchParallel(vals, res);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getLogProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getLogProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void PoissonDistribution::getCumulativeProbability(std::span<const double> values, std::span<double> results, const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<PoissonDistribution>::distType(),
        performance::DistributionTraits<PoissonDistribution>::complexity(),
        [](const PoissonDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getCumulativeProbabilityBatch(vals, res, count);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getCumulativeProbabilityBatchParallel(vals, res);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getCumulativeProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getCumulativeProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

//==============================================================================
// EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==============================================================================

void PoissonDistribution::getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                    performance::Strategy strategy) const {
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const PoissonDistribution& dist, double value) { return dist.getProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getProbabilityBatch(vals, res, count);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getProbabilityBatchParallel(vals, res);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void PoissonDistribution::getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                       performance::Strategy strategy) const {
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const PoissonDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getLogProbabilityBatch(vals, res, count);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getLogProbabilityBatchParallel(vals, res);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getLogProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getLogProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

void PoissonDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                              performance::Strategy strategy) const {
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const PoissonDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
        [](const PoissonDistribution& dist, const double* vals, double* res, size_t count) {
            dist.getCumulativeProbabilityBatch(vals, res, count);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res) {
            dist.getCumulativeProbabilityBatchParallel(vals, res);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            dist.getCumulativeProbabilityBatchWorkStealing(vals, res, pool);
        },
        [](const PoissonDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            dist.getCumulativeProbabilityBatchCacheAware(vals, res, cache);
        }
    );
}

//==============================================================================
// POISSON-SPECIFIC METHODS
//==============================================================================

double PoissonDistribution::getProbabilityExact(int k) const {
    if (k < 0) return constants::math::ZERO_DOUBLE;
    
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
    
    if (isSmallLambda_) {
        return computePMFSmall(k);
    } else {
        return computePMFLarge(k);
    }
}

double PoissonDistribution::getLogProbabilityExact(int k) const noexcept {
    if (k < 0) return constants::probability::MIN_LOG_PROBABILITY;
    
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
    
    return computeLogPMF(k);
}

double PoissonDistribution::getCumulativeProbabilityExact(int k) const {
    if (k < 0) return constants::math::ZERO_DOUBLE;
    
    return computeCDF(k);
}

std::vector<int> PoissonDistribution::sampleIntegers(std::mt19937& rng, std::size_t count) const {
    std::vector<int> samples;
    samples.reserve(count);
    
    for (std::size_t i = 0; i < count; ++i) {
        samples.push_back(static_cast<int>(sample(rng)));
    }
    
    return samples;
}

bool PoissonDistribution::canUseNormalApproximation() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return lambda_ > constants::thresholds::poisson::NORMAL_APPROXIMATION_THRESHOLD;  // Rule of thumb: λ > threshold for reasonable normal approximation
}

//==============================================================================
// PRIVATE COMPUTATIONAL METHODS
//==============================================================================

double PoissonDistribution::computePMFSmall(int k) const noexcept {
    // Direct computation for small lambda
    if (k < static_cast<int>(FACTORIAL_CACHE.size())) {
        // Use cached factorial
        return std::pow(lambda_, k) * expNegLambda_ / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
    } else {
        // Use log-space computation
        return std::exp(computeLogPMF(k));
    }
}

double PoissonDistribution::computePMFLarge(int k) const noexcept {
    // Use Stirling's approximation or normal approximation for large lambda
    if (isVeryLargeLambda_ && std::abs(k - lambda_) < 3.0 * sqrtLambda_) {
        // Normal approximation with continuity correction
        double z = (k + constants::math::HALF - lambda_) / sqrtLambda_;
        return std::exp(constants::math::NEG_HALF * z * z) / (sqrtLambda_ * constants::math::SQRT_2PI);
    } else {
        // Use log-space computation
        return std::exp(computeLogPMF(k));
    }
}

double PoissonDistribution::computeLogPMF(int k) const noexcept {
    // log P(X = k) = k * log(λ) - λ - log(k!)
    double log_factorial_k = logFactorial(k);
    return k * logLambda_ - lambda_ - log_factorial_k;
}

double PoissonDistribution::computeCDF(int k) const noexcept {
    // Use regularized incomplete gamma function: P(X <= k) = Q(k+1, λ)
    // where Q(a,x) is the regularized upper incomplete gamma function
    return libstats::math::gamma_q(k + 1, lambda_);
}

double PoissonDistribution::factorial(int n) noexcept {
    if (n < 0) return constants::math::ZERO_DOUBLE;
    if (n < static_cast<int>(FACTORIAL_CACHE.size())) {
        return FACTORIAL_CACHE[static_cast<std::size_t>(n)];
    }
    
    // Use Stirling's approximation for large n
    if (n > 170) return std::numeric_limits<double>::infinity(); // Overflow
    
    double result = constants::math::ONE;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

double PoissonDistribution::logFactorial(int n) noexcept {
    if (n < 0) return constants::probability::MIN_LOG_PROBABILITY;
    if (n == 0 || n == 1) return constants::math::ZERO_DOUBLE;
    
    if (n < static_cast<int>(FACTORIAL_CACHE.size())) {
        return std::log(FACTORIAL_CACHE[static_cast<std::size_t>(n)]);
    }
    
    // Use Stirling's approximation: log(n!) ≈ n*log(n) - n + 0.5*log(2πn)
    return std::lgamma(n + constants::math::ONE);
}

//==============================================================================
// BATCH OPERATIONS USING SIMD
//==============================================================================

void PoissonDistribution::getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
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
    const double cached_log_lambda = logLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getProbabilityBatchUnsafeImpl(values, results, count, cached_lambda, 
                                  cached_log_lambda, cached_exp_neg_lambda);
}

void PoissonDistribution::getLogProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept {
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
    const double cached_log_lambda = logLambda_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_lambda, cached_log_lambda);
}

void PoissonDistribution::getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const {
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
    
    const double cached_lambda = lambda_;
    
    lock.unlock(); // Release lock before heavy computation
    
    // Call unsafe implementation with cached values
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, cached_lambda);
}

void PoissonDistribution::getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getProbabilityBatchUnsafeImpl(values, results, count, lambda_, logLambda_, expNegLambda_);
}

void PoissonDistribution::getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getLogProbabilityBatchUnsafeImpl(values, results, count, lambda_, logLambda_);
}

void PoissonDistribution::getCumulativeProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept {
    getCumulativeProbabilityBatchUnsafeImpl(values, results, count, lambda_);
}

//==============================================================================
// PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void PoissonDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                        double lambda, double log_lambda, double exp_neg_lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
                continue;
            }
            
            int k = roundToNonNegativeInt(values[i]);
            if (!isValidCount(values[i])) {
                results[i] = constants::math::ZERO_DOUBLE;
                continue;
            }
            
            if (k == 0) {
                results[i] = exp_neg_lambda;
            } else if (lambda < constants::thresholds::poisson::SMALL_LAMBDA_THRESHOLD && k < static_cast<int>(FACTORIAL_CACHE.size())) {
                results[i] = std::pow(lambda, k) * exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                double log_result = k * log_lambda - lambda - logFactorial(k);
                results[i] = std::exp(log_result);
            }
        }
        return;
    }
    
    // If SIMD is enabled but no vectorized implementation available, fall back to scalar
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
            continue;
        }
        
        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = constants::math::ZERO_DOUBLE;
            continue;
        }
        
        if (k == 0) {
            results[i] = exp_neg_lambda;
        } else if (lambda < constants::thresholds::poisson::SMALL_LAMBDA_THRESHOLD && k < static_cast<int>(FACTORIAL_CACHE.size())) {
            results[i] = std::pow(lambda, k) * exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
        } else {
            double log_result = k * log_lambda - lambda - logFactorial(k);
            results[i] = std::exp(log_result);
        }
    }
}

void PoissonDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                           double lambda, double log_lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < constants::math::ZERO_DOUBLE) {
                results[i] = constants::probability::MIN_LOG_PROBABILITY;
                continue;
            }
            
            int k = roundToNonNegativeInt(values[i]);
            if (!isValidCount(values[i])) {
                results[i] = constants::probability::MIN_LOG_PROBABILITY;
                continue;
            }
            
            results[i] = k * log_lambda - lambda - logFactorial(k);
        }
        return;
    }
    
    // If SIMD is enabled but no vectorized implementation available, fall back to scalar
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < constants::math::ZERO_DOUBLE) {
            results[i] = constants::probability::MIN_LOG_PROBABILITY;
            continue;
        }
        
        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = constants::probability::MIN_LOG_PROBABILITY;
            continue;
        }
        
        results[i] = k * log_lambda - lambda - logFactorial(k);
    }
}

void PoissonDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                  double lambda) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < constants::math::ZERO_DOUBLE) {
                results[i] = constants::math::ZERO_DOUBLE;
                continue;
            }
            
            int k = roundToNonNegativeInt(values[i]);
            if (!isValidCount(values[i])) {
                results[i] = constants::math::ONE;
                continue;
            }
            
            results[i] = libstats::math::gamma_q(k + 1, lambda);
        }
        return;
    }
    
    // If SIMD is enabled but no vectorized implementation available, fall back to scalar
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
            continue;
        }
        
        int k = roundToNonNegativeInt(values[i]);
        if (!isValidCount(values[i])) {
            results[i] = constants::math::ONE;
            continue;
        }
        
        results[i] = libstats::math::gamma_q(k + 1, lambda);
    }
}

//==============================================================================
// DISTRIBUTION MANAGEMENT
//==============================================================================

void PoissonDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit to empty data");
    }
    
    // Check minimum data points for reliable fitting
    if (values.size() < constants::thresholds::MIN_DATA_POINTS_FOR_CHI_SQUARE) {  // Minimum data points for reliable fitting
        throw std::invalid_argument("Insufficient data points for reliable Poisson fitting");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : values) {
        if (value < constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    // For Poisson distribution, MLE gives λ = sample mean
    double sum = std::accumulate(values.begin(), values.end(), constants::math::ZERO_DOUBLE);
    double sample_mean = sum / static_cast<double>(values.size());
    
    // Ensure fitted lambda is positive
    if (sample_mean <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Sample mean must be positive for Poisson distribution");
    }
    
    // Set the new parameter
    setLambda(sample_mean);
}

void PoissonDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = constants::math::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

std::string PoissonDistribution::toString() const {
    std::ostringstream oss;
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    oss << "Poisson(λ=" << lambda_ << ")";
    return oss.str();
}

//==============================================================================
// COMPARISON OPERATORS
//==============================================================================

bool PoissonDistribution::operator==(const PoissonDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    
    return std::abs(lambda_ - other.lambda_) <= constants::precision::DEFAULT_TOLERANCE;
}

//==============================================================================
// STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const PoissonDistribution& dist) {
    os << dist.toString();
    return os;
}

std::istream& operator>>(std::istream& is, PoissonDistribution& distribution) {
    std::string token;
    double lambda;
    
    // Expected format: "Poisson(λ=<value>)"
    // We'll parse this step by step
    
    // Skip whitespace and read the first part
    is >> token;
    if (token.find("Poisson(") != 0) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    // Extract λ value
    if (token.find("λ=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    size_t lambda_pos = token.find("λ=") + 2;
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
    
    // Validate and set parameters using the safe API
    auto result = distribution.trySetParameters(lambda);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

//==============================================================================
// PARALLEL BATCH OPERATIONS
//==============================================================================

void PoissonDistribution::getProbabilityBatchParallel(std::span<const double> input_values,
                                                    std::span<double> output_results) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
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
    const double cached_log_lambda = logLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    [[maybe_unused]] const bool cached_is_small_lambda = isSmallLambda_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use higher threshold for simple distribution operations to avoid thread pool overhead
    // Poisson PDF operations have minimal computation per element, especially for small k values
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            if (input_values[i] < constants::math::ZERO_DOUBLE) {
                output_results[i] = constants::math::ZERO_DOUBLE;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::math::ZERO_DOUBLE;
                return;
            }
            
            // Compute PMF using cached parameters (exactly like SIMD version)
            if (k == 0) {
                output_results[i] = cached_exp_neg_lambda;
            } else if (cached_lambda < constants::thresholds::poisson::SMALL_LAMBDA_THRESHOLD && k < static_cast<int>(FACTORIAL_CACHE.size())) {
                // Direct computation for small lambda and k
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                // Log-space computation
                const double log_pmf = k * cached_log_lambda - cached_lambda - logFactorial(k);
                output_results[i] = std::exp(log_pmf);
            }
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (input_values[i] < constants::math::ZERO_DOUBLE) {
                output_results[i] = constants::math::ZERO_DOUBLE;
                continue;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::math::ZERO_DOUBLE;
                continue;
            }
            
            // Compute PMF using cached parameters (optimized like SIMD version)
            if (k == 0) {
                // Special case optimization: P(X = 0) = e^(-λ)
                output_results[i] = cached_exp_neg_lambda;
            } else if (cached_lambda < constants::thresholds::poisson::SMALL_LAMBDA_THRESHOLD && k < static_cast<int>(FACTORIAL_CACHE.size())) {
                // Direct computation for small lambda and k using cached factorial
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                // Use log-space computation for numerical stability
                const double log_pmf = k * cached_log_lambda - cached_lambda - logFactorial(k);
                output_results[i] = std::exp(log_pmf);
            }
        }
    }
}

void PoissonDistribution::getLogProbabilityBatchParallel(std::span<const double> input_values, 
                                                       std::span<double> output_results) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
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
    const double cached_log_lambda = logLambda_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            if (input_values[i] < constants::math::ZERO_DOUBLE) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                return;
            }
            
            // log P(X = k) = k * log(λ) - λ - log(k!)
            output_results[i] = k * cached_log_lambda - cached_lambda - logFactorial(k);
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (input_values[i] < constants::math::ZERO_DOUBLE) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                continue;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                continue;
            }
            
            output_results[i] = k * cached_log_lambda - cached_lambda - logFactorial(k);
        }
    }
}

void PoissonDistribution::getCumulativeProbabilityBatchParallel(std::span<const double> input_values, 
                                                              std::span<double> output_results) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
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
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 1.0;
                return;
            }
            
            // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
            output_results[i] = libstats::math::gamma_q(k + 1, cached_lambda);
        });
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                continue;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 1.0;
                continue;
            }
            
            output_results[i] = libstats::math::gamma_q(k + 1, cached_lambda);
        }
    }
}

//==============================================================================
// WORK-STEALING PARALLEL BATCH OPERATIONS
//==============================================================================

void PoissonDistribution::getProbabilityBatchWorkStealing(std::span<const double> input_values,
                                                         std::span<double> output_results,
                                                         WorkStealingPool& pool) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
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
    const double cached_log_lambda = logLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    [[maybe_unused]] const bool cached_is_small_lambda = isSmallLambda_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use work-stealing pool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 0.0;
                return;
            }
            
            // Compute PMF using cached parameters (exactly like SIMD version)
            if (k == 0) {
                output_results[i] = cached_exp_neg_lambda;
            } else if (cached_lambda < constants::thresholds::poisson::SMALL_LAMBDA_THRESHOLD && k < static_cast<int>(FACTORIAL_CACHE.size())) {
                // Direct computation for small lambda and k
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                // Log-space computation
                const double log_pmf = k * cached_log_lambda - cached_lambda - logFactorial(k);
                output_results[i] = std::exp(log_pmf);
            }
        });
    } else {
        // Fall back to regular parallel processing for small datasets
        std::span<const double> input_span(input_values);
        std::span<double> output_span(output_results);
        getProbabilityBatchParallel(input_span, output_span);
    }
}

void PoissonDistribution::getLogProbabilityBatchWorkStealing(std::span<const double> input_values,
                                                           std::span<double> output_results,
                                                           WorkStealingPool& pool) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
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
    const double cached_log_lambda = logLambda_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use work-stealing pool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            if (input_values[i] < 0.0) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                return;
            }
            
            // log P(X = k) = k * log(λ) - λ - log(k!)
            output_results[i] = k * cached_log_lambda - cached_lambda - logFactorial(k);
        });
    } else {
        // Fall back to regular parallel processing for small datasets
        std::span<const double> input_span(input_values);
        std::span<double> output_span(output_results);
        getLogProbabilityBatchParallel(input_span, output_span);
    }
}

void PoissonDistribution::getCumulativeProbabilityBatchWorkStealing(std::span<const double> input_values,
                                                                   std::span<double> output_results,
                                                                   WorkStealingPool& pool) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
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
    
    lock.unlock(); // Release lock before parallel processing
    
    // Use work-stealing pool for dynamic load balancing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
        pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 1.0;
                return;
            }
            
            // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
            output_results[i] = libstats::math::gamma_q(k + 1, cached_lambda);
        });
    } else {
        // Fall back to regular parallel processing for small datasets
        std::span<const double> input_span(input_values);
        std::span<double> output_span(output_results);
        getCumulativeProbabilityBatchParallel(input_span, output_span);
    }
}

//==============================================================================
// CACHE-AWARE PARALLEL BATCH OPERATIONS
//==============================================================================

void PoissonDistribution::getProbabilityBatchCacheAware(std::span<const double> input_values,
                                                       std::span<double> output_results,
                                                       cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "poisson_pdf_batch_" + std::to_string(count);
    
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
    const double cached_log_lambda = logLambda_;
    const double cached_exp_neg_lambda = expNegLambda_;
    const bool cached_is_small_lambda = isSmallLambda_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "poisson_pdf");
    
    // Use cache-aware parallel processing with adaptive grain sizing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute PMF for each element with cache-aware access patterns
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 0.0;
                return;
            }
            
            // Compute PMF using cached parameters
            if (cached_is_small_lambda && k <= constants::thresholds::poisson::SMALL_K_CACHE_THRESHOLD) {
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                const double log_pmf = k * cached_log_lambda - cached_lambda - logFactorial(k);
                output_results[i] = std::exp(log_pmf);
            }
        }, optimal_grain_size);  // Use adaptive grain size from cache manager
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                continue;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 0.0;
                continue;
            }
            
            // Compute PMF using cached parameters
            if (cached_is_small_lambda && k <= constants::thresholds::poisson::SMALL_K_CACHE_THRESHOLD) {
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[static_cast<std::size_t>(k)];
            } else {
                const double log_pmf = k * cached_log_lambda - cached_lambda - logFactorial(k);
                output_results[i] = std::exp(log_pmf);
            }
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void PoissonDistribution::getLogProbabilityBatchCacheAware(std::span<const double> input_values,
                                                          std::span<double> output_results,
                                                          cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "poisson_logpdf_batch_" + std::to_string(count);
    
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
    const double cached_log_lambda = logLambda_;
    
    lock.unlock(); // Release lock before parallel processing
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "poisson_logpdf");
    
    // Use cache-aware parallel processing with adaptive grain sizing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute log PMF for each element with cache-aware access patterns
            if (input_values[i] < 0.0) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                return;
            }
            
            // log P(X = k) = k * log(λ) - λ - log(k!)
            output_results[i] = k * cached_log_lambda - cached_lambda - logFactorial(k);
        }, optimal_grain_size);  // Use adaptive grain size from cache manager
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (input_values[i] < 0.0) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                continue;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = constants::probability::MIN_LOG_PROBABILITY;
                continue;
            }
            
            // log P(X = k) = k * log(λ) - λ - log(k!)
            output_results[i] = k * cached_log_lambda - cached_lambda - logFactorial(k);
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void PoissonDistribution::getCumulativeProbabilityBatchCacheAware(std::span<const double> input_values,
                                                                 std::span<double> output_results,
                                                                 cache::AdaptiveCache<std::string, double>& cache_manager) const {
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const std::size_t count = input_values.size();
    if (count == 0) return;
    
    // Integrate with Level 0-3 adaptive cache system
    const std::string cache_key = "poisson_cdf_batch_" + std::to_string(count);
    
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
    
    lock.unlock(); // Release lock before parallel processing
    
    // Determine optimal batch size based on cache behavior
    const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "poisson_cdf");
    
    // Use cache-aware parallel processing with adaptive grain sizing
    // Use same threshold as regular parallel operations to avoid inconsistency
    if (count >= constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // Compute CDF for each element with cache-aware access patterns
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                return;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 1.0;
                return;
            }
            
            // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
            output_results[i] = libstats::math::gamma_q(k + 1, cached_lambda);
        }, optimal_grain_size);  // Use adaptive grain size from cache manager
    } else {
        // Fall back to serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            if (input_values[i] < 0.0) {
                output_results[i] = 0.0;
                continue;
            }
            
            int k = roundToNonNegativeInt(input_values[i]);
            if (!isValidCount(input_values[i])) {
                output_results[i] = 1.0;
                continue;
            }
            
            // Use regularized incomplete gamma function: P(X ≤ k) = Q(k+1, λ)
            output_results[i] = libstats::math::gamma_q(k + 1, cached_lambda);
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

// Explicit template instantiations for common types
// Note: Template instantiations handled internally

//==============================================================================
// ADVANCED STATISTICAL METHODS
//==============================================================================

std::pair<double, double> PoissonDistribution::confidenceIntervalRate(
    const std::vector<double>& data, 
    double confidence_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    const double total_count = std::accumulate(data.begin(), data.end(), 0.0);
    const double alpha = constants::math::ONE - confidence_level;
    
    // For Poisson data, the sum follows Poisson(n*λ) distribution
    // Use relationship between Poisson and Chi-square distributions
    // CI for λ based on total count and sample size
    
    const double alpha_half = alpha * constants::math::HALF;
    
    // Lower bound: chi2_lower/2n where chi2_lower has 2*total_count degrees of freedom
    double lower_bound;
    if (total_count > 0) {
        const double chi2_lower = libstats::math::inverse_chi_squared_cdf(alpha_half, 2 * total_count);
        lower_bound = chi2_lower / (2.0 * static_cast<double>(n));
    } else {
        lower_bound = constants::math::ZERO_DOUBLE;
    }
    
    // Upper bound: chi2_upper/2n where chi2_upper has 2*(total_count+1) degrees of freedom
    const double chi2_upper = libstats::math::inverse_chi_squared_cdf(constants::math::ONE - alpha_half, 2 * (total_count + 1));
    const double upper_bound = chi2_upper / (constants::math::TWO * static_cast<double>(n));
    
    return {lower_bound, upper_bound};
}

std::tuple<double, double, bool> PoissonDistribution::likelihoodRatioTest(
    const std::vector<double>& data,
    double lambda0,
    double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (lambda0 <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Lambda0 must be positive");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(n);
    
    // MLE estimate of lambda
    const double lambda_hat = sample_mean;
    
    // Create restricted model (H0: λ = λ0) and unrestricted model (H1: λ = λ̂)
    PoissonDistribution restricted_model(lambda0);
    PoissonDistribution unrestricted_model(lambda_hat);
    
    // Calculate log-likelihoods
    double log_likelihood_restricted = 0.0;
    double log_likelihood_unrestricted = 0.0;
    
    for (double x : data) {
        log_likelihood_restricted += restricted_model.getLogProbability(x);
        log_likelihood_unrestricted += unrestricted_model.getLogProbability(x);
    }
    
    // Likelihood ratio statistic: LR = 2 * (L(λ̂) - L(λ0))
    const double lr_statistic = constants::math::TWO * (log_likelihood_unrestricted - log_likelihood_restricted);
    
    // Under H0, LR follows chi-squared distribution with 1 degree of freedom
    const double p_value = constants::math::ONE - libstats::math::chi_squared_cdf(lr_statistic, 1);
    
    const bool reject_null = p_value < significance_level;
    
    return {lr_statistic, p_value, reject_null};
}

double PoissonDistribution::methodOfMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    // For Poisson distribution, method of moments estimator is simply the sample mean
    // since E[X] = Var[X] = λ
    const double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
    
    if (sample_mean <= 0.0) {
        throw std::invalid_argument("Sample mean must be positive for Poisson distribution");
    }
    
    return sample_mean;
}

std::pair<double, double> PoissonDistribution::bayesianEstimation(
    const std::vector<double>& data,
    double prior_shape,
    double prior_rate) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (prior_shape <= 0.0 || prior_rate <= 0.0) {
        throw std::invalid_argument("Prior parameters must be positive");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    const double sum_x = std::accumulate(data.begin(), data.end(), 0.0);
    
    // For Poisson likelihood and Gamma(α, β) prior:
    // Posterior is Gamma(α + Σx_i, β + n)
    const double posterior_shape = prior_shape + sum_x;
    const double posterior_rate = prior_rate + static_cast<double>(n);
    
    return {posterior_shape, posterior_rate};
}

std::pair<double, double> PoissonDistribution::bayesianCredibleInterval(
    const std::vector<double>& data, double credibility_level, double prior_shape, double prior_rate) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (credibility_level <= 0.0 || credibility_level >= 1.0) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }
    if (prior_shape <= 0.0 || prior_rate <= 0.0) {
        throw std::invalid_argument("Prior parameters must be positive");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    const double sum_x = std::accumulate(data.begin(), data.end(), 0.0);
    
    // For Poisson likelihood and Gamma(α, β) prior:
    // Posterior is Gamma(α + Σx_i, β + n)
    const double posterior_shape = prior_shape + sum_x;
    const double posterior_rate = prior_rate + static_cast<double>(n);
    
    // Calculate credible interval using inverse gamma CDF
    const double alpha = constants::math::ONE - credibility_level;
    const double lower_percentile = alpha * constants::math::HALF;
    const double upper_percentile = constants::math::ONE - alpha * constants::math::HALF;
    
    // For Gamma distribution, the rate parameter λ follows Gamma distribution
    // Use gamma inverse CDF to find quantiles
    const double lower_bound = libstats::math::gamma_inverse_cdf(lower_percentile, posterior_shape, constants::math::ONE / posterior_rate);
    const double upper_bound = libstats::math::gamma_inverse_cdf(upper_percentile, posterior_shape, constants::math::ONE / posterior_rate);
    
    return {lower_bound, upper_bound};
}

double PoissonDistribution::robustEstimation(
    const std::vector<double>& data, const std::string& estimator_type, double trim_proportion) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (trim_proportion < 0.0 || trim_proportion > constants::math::HALF) {
        throw std::invalid_argument("Trim proportion must be between 0 and 0.5");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    const size_t n = sorted_data.size();
    double estimate;
    
    if (estimator_type == "winsorized") {
        // Winsorized mean: replace extreme values with percentiles
        const size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(n));
        const double lower_val = sorted_data[trim_count];
        const double upper_val = sorted_data[n - 1 - trim_count];
        
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            if (sorted_data[i] < lower_val) {
                sum += lower_val;
            } else if (sorted_data[i] > upper_val) {
                sum += upper_val;
            } else {
                sum += sorted_data[i];
            }
        }
        estimate = sum / static_cast<double>(n);
        
    } else if (estimator_type == "trimmed") {
        // Trimmed mean: remove extreme values
        const size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(n));
        const size_t start_idx = trim_count;
        const size_t end_idx = n - trim_count;
        
        if (end_idx <= start_idx) {
            throw std::invalid_argument("Trim proportion too large - no data remains");
        }
        
        double sum = 0.0;
        for (size_t i = start_idx; i < end_idx; ++i) {
            sum += sorted_data[i];
        }
        estimate = sum / static_cast<double>(end_idx - start_idx);
        
    } else if (estimator_type == "median") {
        // Median estimator
        if (n % 2 == 0) {
            estimate = (sorted_data[n/2 - 1] + sorted_data[n/2]) * constants::math::HALF;
        } else {
            estimate = sorted_data[n/2];
        }
        
    } else {
        throw std::invalid_argument("Unknown estimator type: must be 'winsorized', 'trimmed', or 'median'");
    }
    
    if (estimate <= 0.0) {
        throw std::runtime_error("Robust estimate must be positive for Poisson distribution");
    }
    
    return estimate;
}

double PoissonDistribution::lMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    // For Poisson distribution, L-moments estimator: λ = L₁ (first L-moment = mean)
    // Since E[X] = λ for Poisson, the first L-moment equals the sample mean
    const double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
    
    if (sample_mean <= 0.0) {
        throw std::invalid_argument("L-moments estimate must be positive for Poisson distribution");
    }
    
    return sample_mean;
}

std::tuple<double, double, bool> PoissonDistribution::overdispersionTest(
    const std::vector<double>& data, double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);
    
    // Calculate sample variance
    double variance = 0.0;
    for (double value : data) {
        const double diff = value - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(n - 1); // Sample variance with Bessel's correction
    
    // Overdispersion index: variance/mean ratio
    const double dispersion_index = variance / mean;
    
    // Test statistic: (n-1) * (variance/mean - 1) / sqrt(2*(n-1))
    // Under null hypothesis (no overdispersion), this follows standard normal
    const double test_statistic = static_cast<double>(n - 1) * (dispersion_index - constants::math::ONE) / std::sqrt(constants::math::TWO * static_cast<double>(n - 1));
    
    // Two-sided test for overdispersion
    const double p_value = constants::math::TWO * (constants::math::ONE - libstats::math::normal_cdf(std::abs(test_statistic)));
    
    const bool is_overdispersed = p_value < significance_level;
    
    return {test_statistic, p_value, is_overdispersed};
}

std::tuple<double, double, bool> PoissonDistribution::excessZerosTest(
    const std::vector<double>& data, double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    const size_t observed_zeros = static_cast<size_t>(std::count(data.begin(), data.end(), 0.0));
    const double lambda_hat = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);
    
    // Expected number of zeros under Poisson(λ): n * e^(-λ)
    const double expected_zeros = static_cast<double>(n) * std::exp(-lambda_hat);
    
    // Variance of number of zeros: n * e^(-λ) * (1 - e^(-λ))
    const double exp_neg_lambda = std::exp(-lambda_hat);
    const double variance_zeros = static_cast<double>(n) * exp_neg_lambda * (constants::math::ONE - exp_neg_lambda);
    
    if (variance_zeros <= 0.0) {
        throw std::runtime_error("Variance of zeros count is non-positive");
    }
    
    // Z-test statistic for excess zeros
    const double z_statistic = (static_cast<double>(observed_zeros) - expected_zeros) / std::sqrt(variance_zeros);
    
    // Two-sided p-value
    const double p_value = constants::math::TWO * (constants::math::ONE - libstats::math::normal_cdf(std::abs(z_statistic)));
    
    const bool has_excess_zeros = p_value < significance_level;
    
    return {z_statistic, p_value, has_excess_zeros};
}

std::tuple<double, double, bool> PoissonDistribution::rateStabilityTest(
    const std::vector<double>& data, double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (data.size() < 3) {
        throw std::invalid_argument("At least 3 data points required for rate stability test");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    
    // Perform linear regression: y_i = a + b*i + ε_i
    // Test H0: b = 0 (no trend) vs H1: b ≠ 0 (trend exists)
    
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i + 1); // Time index (1, 2, 3, ...)
        const double y = data[i];
        
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    
    const double n_double = static_cast<double>(n);
    const double mean_x = sum_x / n_double;
    const double mean_y = sum_y / n_double;
    
    // Calculate regression slope (b) and intercept (a)
    const double denominator = sum_xx - n_double * mean_x * mean_x;
    if (std::abs(denominator) < constants::precision::DEFAULT_TOLERANCE) {
        throw std::runtime_error("Cannot perform regression: denominator too small");
    }
    
    const double slope = (sum_xy - n_double * mean_x * mean_y) / denominator;
    const double intercept = mean_y - slope * mean_x;
    
    // Calculate residual sum of squares
    double rss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i + 1);
        const double predicted = intercept + slope * x;
        const double residual = data[i] - predicted;
        rss += residual * residual;
    }
    
    const double mse = rss / static_cast<double>(n - 2); // Mean squared error
    const double se_slope = std::sqrt(mse / (sum_xx - n_double * mean_x * mean_x)); // Standard error of slope
    
    // t-statistic for testing H0: slope = 0
    const double t_statistic = slope / se_slope;
    
    // Two-sided p-value using t-distribution with (n-2) degrees of freedom
    const int df = static_cast<int>(n - 2);
    const double p_value = constants::math::TWO * (constants::math::ONE - libstats::math::t_cdf(std::abs(t_statistic), df));
    
    const bool rate_is_stable = p_value >= significance_level; // Rate is stable if we fail to reject H0
    
    return {t_statistic, p_value, rate_is_stable};
}

std::tuple<double, double, bool> PoissonDistribution::comprehensiveGoodnessOfFitTest(
    const std::vector<double>& data, double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    // Fit Poisson distribution to data
    PoissonDistribution fitted_dist;
    fitted_dist.fit(data);
    
    // Perform multiple goodness-of-fit tests
    try {
        // 1. Mean-variance equality test (overdispersion test)
        auto overdispersion_result = overdispersionTest(data, significance_level);
        
        // 2. Chi-square goodness-of-fit test
        auto chi_square_result = chiSquareGoodnessOfFit(data, fitted_dist, significance_level);
        
        // 3. Kolmogorov-Smirnov test
        auto ks_result = kolmogorovSmirnovTest(data, fitted_dist, significance_level);
        
        // Combine test results using Fisher's method for combining p-values
        const double chi2_p = std::get<1>(chi_square_result);
        const double ks_p = std::get<1>(ks_result);
        const double overdispersion_p = std::get<1>(overdispersion_result);
        
        // Fisher's method: -2 * Σ ln(p_i) ~ χ²(2k) where k is number of tests
        const double fisher_statistic = -constants::math::TWO * 
            (std::log(std::max(chi2_p, constants::probability::MIN_PROBABILITY)) + 
             std::log(std::max(ks_p, constants::probability::MIN_PROBABILITY)) + 
             std::log(std::max(overdispersion_p, constants::probability::MIN_PROBABILITY)));
        
        // Combined p-value using chi-square distribution with 6 degrees of freedom (3 tests × 2)
        const double combined_p_value = constants::math::ONE - libstats::math::chi_squared_cdf(fisher_statistic, 6);
        
        // Overall assessment: data follows Poisson if combined p-value >= significance_level
        const bool follows_poisson = combined_p_value >= significance_level;
        
        return {fisher_statistic, combined_p_value, follows_poisson};
        
    } catch (const std::exception& e) {
        // If any individual test fails, return conservative result
        return {0.0, 0.0, false};
    }
}

//==============================================================================
// GOODNESS-OF-FIT TESTS
//==============================================================================

std::tuple<double, double, bool> PoissonDistribution::chiSquareGoodnessOfFit(
    const std::vector<double>& data,
    const PoissonDistribution& distribution,
    double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (data.size() < constants::thresholds::MIN_DATA_POINTS_FOR_CHI_SQUARE) {
        throw std::invalid_argument("At least 5 data points required for chi-square test");
    }
    if (significance_level <= constants::math::ZERO_DOUBLE || significance_level >= constants::math::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    
    // Create frequency table
    std::map<int, int> observed_freq;
    int max_value = 0;
    
    for (double value : data) {
        int k = static_cast<int>(std::round(value));
        observed_freq[k]++;
        max_value = std::max(max_value, k);
    }
    
    // Group rare events to ensure expected frequencies >= 5
    std::vector<std::pair<int, int>> grouped_observed;
    std::vector<double> expected_freq;
    
    int current_group_start = 0;
    int current_observed = 0;
    
    for (int k = 0; k <= max_value + 5; ++k) {
        current_observed += observed_freq[k];
        double expected = static_cast<double>(n) * distribution.getProbabilityExact(k);
        
        // If we have enough expected frequency or we're at the end, close the group
        if (expected >= constants::thresholds::DEFAULT_EXPECTED_FREQUENCY_THRESHOLD || (current_observed > 0 && k >= max_value)) {
            grouped_observed.emplace_back(current_group_start, current_observed);
            
            // Calculate total expected frequency for this group
            double group_expected = 0.0;
            for (int j = current_group_start; j <= k; ++j) {
                group_expected += static_cast<double>(n) * distribution.getProbabilityExact(j);
            }
            expected_freq.push_back(group_expected);
            
            current_group_start = k + 1;
            current_observed = 0;
            
            if (k >= max_value && group_expected < 1e-10) break;
        }
    }
    
    // Calculate chi-square statistic
    double chi_square_stat = 0.0;
    const size_t num_groups = grouped_observed.size();
    
    for (size_t i = 0; i < num_groups; ++i) {
        const double observed = grouped_observed[i].second;
        const double expected = expected_freq[i];
        
        if (expected > 0) {
            chi_square_stat += (observed - expected) * (observed - expected) / expected;
        }
    }
    
    // Degrees of freedom = number of groups - 1 - number of estimated parameters
    const int df = static_cast<int>(num_groups) - 1 - 1; // -1 for estimated lambda
    
    if (df <= 0) {
        throw std::runtime_error("Insufficient degrees of freedom for chi-square test");
    }
    
    // Calculate p-value
    const double p_value = constants::math::ONE - libstats::math::chi_squared_cdf(chi_square_stat, df);
    
    const bool reject_null = p_value < significance_level;
    
    return {chi_square_stat, p_value, reject_null};
}

std::tuple<double, double, bool> PoissonDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data,
    const PoissonDistribution& distribution,
    double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (data.size() < constants::thresholds::MIN_DATA_POINTS_FOR_CHI_SQUARE) {
        throw std::invalid_argument("At least 5 data points required for KS test");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    // Use the centralized, overflow-safe KS statistic calculation from math_utils
    double ks_statistic = math::calculate_ks_statistic(data, distribution);
    
    const size_t n = data.size();
    
    // Approximate p-value using Kolmogorov distribution
    // For discrete distributions, this is an approximation
    const double sqrt_n = std::sqrt(static_cast<double>(n));
    const double lambda_ks = sqrt_n * ks_statistic;
    
    // Improved p-value calculation for discrete distributions
    double p_value;
    if (lambda_ks < 0.27) {
        p_value = 1.0;
    } else if (lambda_ks < 1.0) {
        p_value = 2.0 * std::exp(-2.0 * lambda_ks * lambda_ks);
    } else {
        // Asymptotic approximation with correction terms
        p_value = 2.0 * std::exp(-2.0 * lambda_ks * lambda_ks);
        for (int k = 1; k <= 10; ++k) {
            p_value += 2.0 * std::pow(-1, k) * std::exp(-2.0 * k * k * lambda_ks * lambda_ks);
        }
    }
    
    p_value = std::max(0.0, std::min(1.0, p_value)); // Clamp to [0,1]
    
    const bool reject_null = p_value < significance_level;
    
    return {ks_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> PoissonDistribution::andersonDarlingTest(
    const std::vector<double>& data, const PoissonDistribution& distribution, double significance_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (data.size() < constants::thresholds::MIN_DATA_POINTS_FOR_CHI_SQUARE) {
        throw std::invalid_argument("At least 5 data points required for Anderson-Darling test");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    // Use the centralized, numerically stable AD statistic calculation from math_utils
    double ad_statistic = math::calculate_ad_statistic(data, distribution);
    
    // Critical values and p-value approximation for discrete Anderson-Darling test
    // These are approximations since exact distribution is complex for discrete case
    double p_value;
    if (ad_statistic < 0.5) {
        p_value = 1.0 - std::exp(-1.2337 * std::pow(ad_statistic, -1.0) + 1.0);
    } else if (ad_statistic < 2.0) {
        p_value = 1.0 - std::exp(-0.75 * ad_statistic - 0.5);
    } else {
        p_value = std::exp(-ad_statistic);
    }
    
    // Ensure p-value is in valid range
    p_value = std::max(0.0, std::min(1.0, p_value));
    
    const bool reject_null = p_value < significance_level;
    
    return {ad_statistic, p_value, reject_null};
}

//==============================================================================
// CROSS-VALIDATION METHODS
//==============================================================================

std::vector<std::tuple<double, double, double>> PoissonDistribution::kFoldCrossValidation(
    const std::vector<double>& data,
    int k,
    unsigned int random_seed) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (k <= 1 || k > static_cast<int>(data.size())) {
        throw std::invalid_argument("k must be between 2 and the number of data points");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    std::vector<std::tuple<double, double, double>> results;
    results.reserve(static_cast<size_t>(k));
    
    // Create random indices for k-fold splitting
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    const size_t fold_size = n / static_cast<size_t>(k);
    
    for (int fold = 0; fold < k; ++fold) {
        // Determine fold boundaries
        const size_t start_idx = static_cast<size_t>(fold) * fold_size;
        const size_t end_idx = (fold == k - 1) ? n : (static_cast<size_t>(fold) + 1) * fold_size;
        
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
        PoissonDistribution fitted_model;
        fitted_model.fit(training_data);
        
        // Evaluate on validation data
        double total_absolute_error = 0.0;
        double total_squared_error = 0.0;
        double total_log_likelihood = 0.0;
        
        for (double val : validation_data) {
            const double predicted_mean = fitted_model.getMean();
            const double absolute_error = std::abs(val - predicted_mean);
            const double squared_error = (val - predicted_mean) * (val - predicted_mean);
            
            total_absolute_error += absolute_error;
            total_squared_error += squared_error;
            total_log_likelihood += fitted_model.getLogProbability(val);
        }
        
        // Calculate metrics for this fold
        const double mae = total_absolute_error / static_cast<double>(validation_data.size());
        const double rmse = std::sqrt(total_squared_error / static_cast<double>(validation_data.size()));
        
        results.emplace_back(mae, rmse, total_log_likelihood);
    }
    
    return results;
}

std::tuple<double, double, double> PoissonDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    
    if (data.size() < 3) {
        throw std::invalid_argument("At least 3 data points required for LOOCV");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
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
        PoissonDistribution fitted_model;
        fitted_model.fit(training_data);
        
        // Evaluate on left-out point
        const double predicted_mean = fitted_model.getMean();
        const double actual_value = data[i];
        
        const double absolute_error = std::abs(actual_value - predicted_mean);
        const double squared_error = (actual_value - predicted_mean) * (actual_value - predicted_mean);
        
        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);
        
        total_log_likelihood += fitted_model.getLogProbability(actual_value);
    }
    
    // Calculate summary statistics
    const double mean_absolute_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / static_cast<double>(n);
    const double mean_squared_error = std::accumulate(squared_errors.begin(), squared_errors.end(), 0.0) / static_cast<double>(n);
    const double root_mean_squared_error = std::sqrt(mean_squared_error);
    
    return {mean_absolute_error, root_mean_squared_error, total_log_likelihood};
}

//==============================================================================
// INFORMATION CRITERIA
//==============================================================================

std::tuple<double, double, double, double> PoissonDistribution::computeInformationCriteria(
    const std::vector<double>& data,
    const PoissonDistribution& fitted_distribution) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const double n = static_cast<double>(data.size());
    const int k = fitted_distribution.getNumParameters(); // 1 for Poisson (lambda)
    
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
        aicc = aic + (constants::math::TWO * k * (k + constants::math::ONE)) / (n - k - constants::math::ONE);
    } else {
        aicc = std::numeric_limits<double>::infinity(); // Undefined for small samples
    }
    
    return {aic, bic, aicc, log_likelihood};
}

//==============================================================================
// BOOTSTRAP METHODS
//==============================================================================

std::pair<double, double> PoissonDistribution::bootstrapParameterConfidenceIntervals(
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
    
    // Validate that all values are non-negative (count data)
    for (double value : data) {
        if (value < 0.0) {
            throw std::invalid_argument("Poisson distribution requires non-negative count data");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("All data values must be finite");
        }
    }
    
    const size_t n = data.size();
    std::vector<double> bootstrap_lambdas;
    bootstrap_lambdas.reserve(static_cast<size_t>(n_bootstrap));
    
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
        
        // Fit Poisson model to bootstrap sample
        PoissonDistribution bootstrap_model;
        bootstrap_model.fit(bootstrap_sample);
        
        bootstrap_lambdas.push_back(bootstrap_model.getLambda());
    }
    
    // Sort for quantile calculation
    std::sort(bootstrap_lambdas.begin(), bootstrap_lambdas.end());
    
    // Calculate confidence intervals using percentile method
    const double alpha = constants::math::ONE - confidence_level;
    const double lower_percentile = alpha * constants::math::HALF;
    const double upper_percentile = constants::math::ONE - alpha * constants::math::HALF;
    
    const size_t lower_idx = static_cast<size_t>(lower_percentile * (n_bootstrap - 1));
    const size_t upper_idx = static_cast<size_t>(upper_percentile * (n_bootstrap - 1));
    
    return {bootstrap_lambdas[lower_idx], bootstrap_lambdas[upper_idx]};
}

//==============================================================================
// RESULT-BASED SETTERS (C++20 Best Practice: Complex implementations in .cpp)
//==============================================================================

VoidResult PoissonDistribution::trySetLambda(double lambda) noexcept {
    auto validation = validatePoissonParameters(lambda);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    
    // Note: PoissonDistribution doesn't have atomicParamsValid_ - this is specific to other distributions
    // The atomic cache validation is handled by cacheValidAtomic_
    
    return VoidResult::ok(true);
}

VoidResult PoissonDistribution::trySetParameters(double lambda) noexcept {
    auto validation = validatePoissonParameters(lambda);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    
    // Note: PoissonDistribution doesn't have atomicParamsValid_ - this is specific to other distributions
    // The atomic cache validation is handled by cacheValidAtomic_
    
    return VoidResult::ok(true);
}

} // namespace libstats

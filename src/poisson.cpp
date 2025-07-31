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
        return std::pow(lambda_, k) * expNegLambda_ / FACTORIAL_CACHE[k];
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
        return FACTORIAL_CACHE[n];
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
        return std::log(FACTORIAL_CACHE[n]);
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
                results[i] = std::pow(lambda, k) * exp_neg_lambda / FACTORIAL_CACHE[k];
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
            results[i] = std::pow(lambda, k) * exp_neg_lambda / FACTORIAL_CACHE[k];
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
    double sample_mean = sum / values.size();
    
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
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[k];
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
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[k];
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
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[k];
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
// CACHE-AWARE PARALLEL BATCH OPERATIONS (Template Implementation)
//==============================================================================

template<typename KeyType, typename ValueType>
void PoissonDistribution::getProbabilityBatchCacheAware(std::span<const double> input_values,
                                                       std::span<double> output_results,
                                                       cache::AdaptiveCache<KeyType, ValueType>& cache_manager) const {
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
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[k];
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
                output_results[i] = std::pow(cached_lambda, k) * cached_exp_neg_lambda / FACTORIAL_CACHE[k];
            } else {
                const double log_pmf = k * cached_log_lambda - cached_lambda - logFactorial(k);
                output_results[i] = std::exp(log_pmf);
            }
        }
    }
    
    // Update cache manager with performance metrics for future optimizations
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

template<typename KeyType, typename ValueType>
void PoissonDistribution::getLogProbabilityBatchCacheAware(std::span<const double> input_values,
                                                          std::span<double> output_results,
                                                          cache::AdaptiveCache<KeyType, ValueType>& cache_manager) const {
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

template<typename KeyType, typename ValueType>
void PoissonDistribution::getCumulativeProbabilityBatchCacheAware(std::span<const double> input_values,
                                                                 std::span<double> output_results,
                                                                 cache::AdaptiveCache<KeyType, ValueType>& cache_manager) const {
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
template void PoissonDistribution::getProbabilityBatchCacheAware<std::string, double>(
    std::span<const double>, std::span<double>, cache::AdaptiveCache<std::string, double>&) const;
template void PoissonDistribution::getLogProbabilityBatchCacheAware<std::string, double>(
    std::span<const double>, std::span<double>, cache::AdaptiveCache<std::string, double>&) const;
template void PoissonDistribution::getCumulativeProbabilityBatchCacheAware<std::string, double>(
    std::span<const double>, std::span<double>, cache::AdaptiveCache<std::string, double>&) const;

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
        lower_bound = chi2_lower / (2.0 * n);
    } else {
        lower_bound = constants::math::ZERO_DOUBLE;
    }
    
    // Upper bound: chi2_upper/2n where chi2_upper has 2*(total_count+1) degrees of freedom
    const double chi2_upper = libstats::math::inverse_chi_squared_cdf(constants::math::ONE - alpha_half, 2 * (total_count + 1));
    const double upper_bound = chi2_upper / (constants::math::TWO * n);
    
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
    const double sample_mean = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE) / n;
    
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
    const double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    
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
    const double posterior_rate = prior_rate + n;
    
    return {posterior_shape, posterior_rate};
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
        double expected = n * distribution.getProbabilityExact(k);
        
        // If we have enough expected frequency or we're at the end, close the group
        if (expected >= constants::thresholds::DEFAULT_EXPECTED_FREQUENCY_THRESHOLD || (current_observed > 0 && k >= max_value)) {
            grouped_observed.emplace_back(current_group_start, current_observed);
            
            // Calculate total expected frequency for this group
            double group_expected = 0.0;
            for (int j = current_group_start; j <= k; ++j) {
                group_expected += n * distribution.getProbabilityExact(j);
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
    
    // Sort data for empirical CDF calculation
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    const size_t n = sorted_data.size();
    double max_diff = 0.0;
    
    // Calculate maximum difference between empirical and theoretical CDFs
    for (size_t i = 0; i < n; ++i) {
        const double x = sorted_data[i];
        
        // Empirical CDF at x
        const double emp_cdf = static_cast<double>(i + 1) / n;
        
        // Theoretical CDF at x
        const double theo_cdf = distribution.getCumulativeProbability(x);
        
        // Check both F(x) - Fn(x) and Fn(x) - F(x)
        const double diff1 = std::abs(emp_cdf - theo_cdf);
        const double diff2 = (i > 0) ? std::abs(static_cast<double>(i) / n - theo_cdf) : theo_cdf;
        
        max_diff = std::max({max_diff, diff1, diff2});
    }
    
    const double ks_statistic = max_diff;
    
    // Approximate p-value using Kolmogorov distribution
    // For discrete distributions, this is an approximation
    const double sqrt_n = std::sqrt(n);
    const double lambda_ks = sqrt_n * ks_statistic;
    
    // Approximation for p-value (simplified)
    double p_value;
    if (lambda_ks < 0.27) {
        p_value = 1.0;
    } else if (lambda_ks < 1.0) {
        p_value = 2.0 * std::exp(-2.0 * lambda_ks * lambda_ks);
    } else {
        // Asymptotic approximation
        p_value = 2.0 * std::exp(-2.0 * lambda_ks * lambda_ks);
        for (int k = 1; k <= 10; ++k) {
            p_value += 2.0 * std::pow(-1, k) * std::exp(-2.0 * k * k * lambda_ks * lambda_ks);
        }
    }
    
    p_value = std::max(0.0, std::min(1.0, p_value)); // Clamp to [0,1]
    
    const bool reject_null = p_value < significance_level;
    
    return {ks_statistic, p_value, reject_null};
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
    results.reserve(k);
    
    // Create random indices for k-fold splitting
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    const size_t fold_size = n / k;
    
    for (int fold = 0; fold < k; ++fold) {
        // Determine fold boundaries
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
        const double mae = total_absolute_error / validation_data.size();
        const double rmse = std::sqrt(total_squared_error / validation_data.size());
        
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
    const double mean_absolute_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / n;
    const double mean_squared_error = std::accumulate(squared_errors.begin(), squared_errors.end(), 0.0) / n;
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
    bootstrap_lambdas.reserve(n_bootstrap);
    
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

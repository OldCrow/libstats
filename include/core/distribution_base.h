#ifndef LIBSTATS_DISTRIBUTION_BASE_H_
#define LIBSTATS_DISTRIBUTION_BASE_H_

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <concepts>
#include <span>
#include <optional>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <cstdlib>
#include <cstdint>
#include "constants.h"
#include "../platform/platform_constants.h"

// Forward compatibility - include the advanced cache for type aliases
#include "../platform/adaptive_cache.h"

namespace libstats {

/**
 * @brief Enhanced base class for probability distributions with comprehensive statistical interface
 * 
 * Modern C++20 implementation providing:
 * - Complete Rule of Five
 * - Thread-safe statistical property access
 * - Comprehensive statistical interface (PDF, CDF, quantiles, moments)
 * - Random number generation
 * - Statistical validation and diagnostics
 * - Parameter estimation with diagnostics
 * - C++20 concepts for type safety
 * - Enhanced caching with performance metrics
 * - SIMD-optimized batch operations
 * - Memory optimization features
 * - Adaptive cache management
 * 
 * @section derived_class_guide Guide for Derived Class Implementation
 * 
 * This base class provides a robust foundation for statistical distributions with
 * thread safety, performance optimizations, and comprehensive error handling.
 * When implementing derived classes, follow these guidelines:
 * 
 * @subsection level0_utilities Using Level 0 Utilities
 * 
 * Level 0 utilities provide essential building blocks for distribution implementations:
 * 
 * **Constants (constants.h):**
 * @code
 * // Use predefined mathematical constants
 * #include "constants.h"
 * using namespace libstats::constants;
 * 
 * double result = mathematical::PI * 2.0;  // Instead of hardcoded 6.28318...
 * double log_val = mathematical::LOG_2;    // Instead of std::log(2.0)
 * 
 * // Use numerical tolerances
 * if (std::abs(diff) < numerical::EPSILON) {
 *     // Values are approximately equal
 * }
 * 
 * // Use probability bounds
 * double safe_prob = std::clamp(prob, probability::MIN_PROBABILITY, 
 *                              probability::MAX_PROBABILITY);
 * @endcode
 * 
 * **Error Handling (error_handling.h):**
 * @code
 * #include "error_handling.h"
 * 
 * // Use safe factory pattern (recommended)
 * static Result<MyDistribution> create(double param1, double param2) {
 *     if (param1 <= 0.0) {
 *         return Result<MyDistribution>::error(ValidationError::InvalidParameter);
 *     }
 *     return Result<MyDistribution>::success(MyDistribution(param1, param2));
 * }
 * 
 * // Use validation functions
 * auto validation_result = validation::validateNormalParameters(mu, sigma);
 * if (!validation_result.is_success()) {
 *     return validation_result.get_error();
 * }
 * @endcode
 * 
 * **CPU Detection (cpu_detection.h):**
 * @code
 * #include "cpu_detection.h"
 * 
 * // Check SIMD support at runtime
 * if (cpu::supports_avx2() && data.size() >= simd::optimal_vector_size()) {
 *     // Use SIMD optimized path
 *     return computeVectorized(data);
 * } else {
 *     // Use scalar fallback
 *     return computeScalar(data);
 * }
 * @endcode
 * 
 * **SIMD Operations (simd.h):**
 * @code
 * #include "simd.h"
 * 
 * // Use vectorized operations for bulk computations
 * void computeBatchProbabilities(const std::vector<double>& x_values,
 *                               std::vector<double>& results) {
 *     if (simd::should_use_simd(x_values.size())) {
 *         simd::vector_exp(x_values.data(), results.data(), x_values.size());
 *     } else {
 *         // Scalar fallback
 *         for (size_t i = 0; i < x_values.size(); ++i) {
 *             results[i] = std::exp(x_values[i]);
 *         }
 *     }
 * }
 * @endcode
 * 
 * @subsection level1_utilities Using Level 1 Utilities
 * 
 * Level 1 utilities provide advanced mathematical and statistical functions:
 * 
 * **Safety Functions (safety.h):**
 * @code
 * #include "safety.h"
 * 
 * // Use safe mathematical operations
 * double safe_result = safety::safe_log(value);     // Handles value <= 0
 * double safe_exp = safety::safe_exp(log_value);    // Prevents overflow
 * double safe_sqrt = safety::safe_sqrt(value);      // Handles value < 0
 * 
 * // Use bounds checking
 * safety::check_bounds(index, 0, container.size(), "array access");
 * 
 * // Use convergence detection
 * safety::ConvergenceDetector detector(1e-8, 1000);
 * while (!detector.hasConverged(current_value)) {
 *     current_value = iterate(current_value);
 *     if (detector.hasReachedMaxIterations()) break;
 * }
 * @endcode
 * 
 * **Mathematical Utilities (math_utils.h):**
 * @code
 * #include "math_utils.h"
 * 
 * // Use special functions
 * double gamma_val = math_utils::gamma_function(x);
 * double beta_val = math_utils::beta_function(a, b);
 * double erf_val = math_utils::error_function(x);
 * 
 * // Use numerical integration
 * auto pdf_func = [this](double x) { return this->getProbability(x); };
 * double cdf_val = math_utils::adaptive_simpson(pdf_func, lower, upper, 1e-10);
 * 
 * // Use root finding
 * auto cdf_diff = [this, target](double x) { 
 *     return this->getCumulativeProbability(x) - target; 
 * };
 * double quantile = math_utils::newton_raphson(cdf_diff, initial_guess, 1e-10);
 * @endcode
 * 
 * **Log-Space Operations (log_space_ops.h):**
 * @code
 * #include "log_space_ops.h"
 * 
 * // Use log-space arithmetic for numerical stability
 * double log_sum = log_space_ops::logSumExp({log_a, log_b, log_c});
 * double log_product = log_a + log_b;  // log(a * b) = log(a) + log(b)
 * 
 * // Use vectorized log-space operations
 * std::vector<double> log_probs = {log_p1, log_p2, log_p3};
 * double total_log_prob = log_space_ops::logSumExpVector(log_probs);
 * @endcode
 * 
 * **Validation Functions (validation.h):**
 * @code
 * #include "validation.h"
 * 
 * // Use statistical tests
 * auto ks_result = validation::kolmogorov_smirnov_test(data, *this);
 * auto ad_result = validation::anderson_darling_test(data, *this);
 * 
 * // Use model selection criteria
 * double aic_score = validation::calculate_aic(log_likelihood, num_parameters);
 * double bic_score = validation::calculate_bic(log_likelihood, num_parameters, 
 *                                            sample_size);
 * @endcode
 * 
 * @subsection thread_safety_patterns Thread Safety Implementation Patterns
 * 
 * The base class provides thread-safe caching infrastructure. Follow these patterns:
 * 
 * **Parameter Access Pattern:**
 * @code
 * class MyDistribution : public DistributionBase {
 * private:
 *     mutable std::shared_mutex param_mutex_;  // Separate mutex for parameters
 *     double param1_;
 *     double param2_;
 *     
 *     // Cached values (protected by cache_mutex_ from base)
 *     mutable double cached_mean_;
 *     mutable double cached_variance_;
 *     
 * public:
 *     // Thread-safe parameter access
 *     double getParam1() const {
 *         std::shared_lock lock(param_mutex_);
 *         return param1_;
 *     }
 *     
 *     // Thread-safe parameter modification
 *     void setParam1(double value) {
 *         // Validate first (no locks held)
 *         if (value <= 0.0) {
 *             throw std::invalid_argument("param1 must be positive");
 *         }
 *         
 *         // Update parameter and invalidate cache
 *         {
 *             std::unique_lock param_lock(param_mutex_);
 *             param1_ = value;
 *         }
 *         invalidateCache();  // Base class method
 *     }
 * };
 * @endcode
 * 
 * **Cache Update Pattern:**
 * @code
 * void MyDistribution::updateCacheUnsafe() const override {
 *     // This method is called under unique lock from base class
 *     // Safe to access parameters with shared lock
 *     std::shared_lock param_lock(param_mutex_);
 *     
 *     // Compute expensive values once
 *     cached_mean_ = param1_ + param2_;
 *     cached_variance_ = param1_ * param2_;
 *     
 *     // Mark cache as valid (required)
 *     cache_valid_ = true;
 *     cacheValidAtomic_.store(true, std::memory_order_release);
 * }
 * @endcode
 * 
 * **Thread-Safe Property Access:**
 * @code
 * double MyDistribution::getMean() const override {
 *     return getCachedValue([this]() { return cached_mean_; });
 * }
 * 
 * double MyDistribution::getVariance() const override {
 *     return getCachedValue([this]() { return cached_variance_; });
 * }
 * @endcode
 * 
 * **Bulk Operations with SIMD:**
 * @code
 * std::vector<double> MyDistribution::getBatchProbabilities(
 *     const std::vector<double>& x_values) const {
 *     std::vector<double> results(x_values.size());
 *     
 *     // Get parameters once under shared lock
 *     double p1, p2;
 *     {
 *         std::shared_lock lock(param_mutex_);
 *         p1 = param1_;
 *         p2 = param2_;
 *     }
 *     
 *     // Use SIMD if beneficial
 *     if (cpu::supports_avx2() && x_values.size() >= 8) {
 *         computeBatchSIMD(x_values, results, p1, p2);
 *     } else {
 *         computeBatchScalar(x_values, results, p1, p2);
 *     }
 *     
 *     return results;
 * }
 * @endcode
 * 
 * @subsection performance_considerations Performance Considerations
 * 
 * **Lock Ordering to Prevent Deadlocks:**
 * - Always acquire locks in consistent order: parameters first, then cache
 * - Use std::lock() for multiple lock acquisition when necessary
 * - Keep lock scope minimal
 * 
 * **Cache Efficiency:**
 * - Cache expensive computations (log-gamma, special functions)
 * - Use atomic flags for lock-free cache validity checks
 * - Invalidate cache only when parameters actually change
 * 
 * **SIMD Usage Guidelines:**
 * - Check CPU support at runtime, not compile time
 * - Use SIMD for bulk operations (typically n >= 8 for doubles)
 * - Provide scalar fallbacks for all SIMD operations
 * - Align memory for SIMD operations when possible
 * 
 * **Memory Management:**
 * - Prefer stack allocation for temporary objects
 * - Use std::vector for dynamic arrays (automatic SIMD alignment)
 * - Consider memory layout for cache efficiency
 * - Use thread-local memory pools for temporary allocations
 * - Leverage SIMD-aligned vectors for bulk operations
 * - Use SmallVector for small temporary collections
 * 
 * **Enhanced Caching Strategies:**
 * - Configure cache limits based on available memory
 * - Use TTL-based expiration for memory-constrained environments
 * - Monitor cache metrics to optimize performance
 * - Implement adaptive cache sizing based on usage patterns
 * - Use memory-aware eviction policies
 * 
 * **Cache Performance Optimization:**
 * @code
 * // Configure cache for memory-constrained environments
 * CacheConfig config;
 * config.max_memory_usage = 512 * 1024;  // 512KB limit
 * config.cache_ttl = std::chrono::milliseconds(2000);  // 2 second TTL
 * config.memory_pressure_aware = true;
 * distribution.configureCacheSettings(config);
 * 
 * // Monitor cache performance
 * auto metrics = distribution.getCacheMetrics();
 * if (metrics.hitRate() < 0.7) {
 *     // Consider adjusting cache parameters
 *     config.cache_ttl = std::chrono::milliseconds(5000);
 *     distribution.configureCacheSettings(config);
 * }
 * @endcode
 * 
 * **Memory Pool Usage:**
 * @code
 * // Use thread-local memory pool for temporary allocations
 * auto& pool = DistributionBase::getThreadPool();
 * if (pool.hasSpace(needed_size)) {
 *     void* temp_memory = pool.allocate(needed_size);
 *     // Use temp_memory for computations
 *     // Memory is automatically managed by pool
 * }
 * 
 * // Use SIMD-aligned vectors for bulk operations
 * simd_vector<double> aligned_data(1000);
 * auto results = distribution.getBatchProbabilities(aligned_data);
 * 
 * // Use SmallVector for small temporary collections
 * SmallVector<double, 16> small_results;
 * for (double x : small_input_range) {
 *     small_results.push_back(distribution.getProbability(x));
 * }
 * @endcode
 * 
 * @subsection error_handling_best_practices Error Handling Best Practices
 * 
 * **Parameter Validation:**
 * - Validate parameters before acquiring locks
 * - Use specific error types from ValidationError enum
 * - Provide descriptive error messages
 * - Use safe factory pattern for construction
 * 
 * **Numerical Stability:**
 * - Use log-space operations for small probabilities
 * - Use safe mathematical functions from safety.h
 * - Check for overflow/underflow conditions
 * - Handle edge cases explicitly
 * 
 * **Exception Safety:**
 * - Use RAII for resource management
 * - Provide strong exception guarantee when possible
 * - Validate inputs before modifying state
 * - Use noexcept for operations that cannot throw
 */
class DistributionBase {
public:
    // =============================================================================
    // RULE OF FIVE - Modern C++20 Implementation
    // =============================================================================
    
    /**
     * @brief Default constructor
     */
    DistributionBase() = default;
    
    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     */
    virtual ~DistributionBase() = default;
    
    /**
     * @brief Copy constructor with thread-safe cache handling
     */
    DistributionBase(const DistributionBase& other);
    
    /**
     * @brief Move constructor with thread-safe state transfer
     */
    DistributionBase(DistributionBase&& other) noexcept;
    
    /**
     * @brief Copy assignment operator with thread-safe implementation
     */
    DistributionBase& operator=(const DistributionBase& other);
    
    /**
     * @brief Move assignment operator with thread-safe implementation  
     */
    DistributionBase& operator=(DistributionBase&& other) noexcept;

    // =============================================================================
    // CORE PROBABILITY INTERFACE - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Probability density/mass function evaluation
     * @param x Value at which to evaluate the distribution
     * @return Probability density (continuous) or mass (discrete) at x
     */
    virtual double getProbability(double x) const = 0;
    
    /**
     * @brief Log probability density/mass function evaluation
     * @param x Value at which to evaluate the log distribution
     * @return Log probability density/mass at x
     * @note Override for numerical stability when possible
     */
    virtual double getLogProbability(double x) const;
    
    /**
     * @brief Cumulative distribution function evaluation
     * @param x Value at which to evaluate the CDF
     * @return P(X <= x)
     */
    virtual double getCumulativeProbability(double x) const = 0;
    
    /**
     * @brief Quantile function (inverse CDF)
     * @param p Probability value in [0,1]
     * @return x such that P(X <= x) = p
     * @throws std::invalid_argument if p not in [0,1]
     */
    virtual double getQuantile(double p) const = 0;

    // =============================================================================
    // STATISTICAL MOMENTS - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Distribution mean (first moment)
     * @return Expected value E[X]
     */
    virtual double getMean() const = 0;
    
    /**
     * @brief Distribution variance (second central moment)
     * @return Variance Var(X) = E[(X - μ)²]
     * @note Prefer variance over standard deviation as fundamental property
     */
    virtual double getVariance() const = 0;
    
    /**
     * @brief Distribution skewness (third standardized moment)
     * @return Skewness coefficient
     * @note Return NaN if undefined for the distribution
     */
    virtual double getSkewness() const = 0;
    
    /**
     * @brief Distribution kurtosis (fourth standardized moment)
     * @return Excess kurtosis (kurtosis - 3)
     * @note Return NaN if undefined for the distribution
     */
    virtual double getKurtosis() const = 0;

    // =============================================================================
    // DERIVED STATISTICAL PROPERTIES - Implemented in Base Class
    // =============================================================================
    
    /**
     * @brief Standard deviation (square root of variance)
     * @return σ = √Var(X)
     */
    double getStandardDeviation() const {
        double var = getVariance();
        return var >= 0.0 ? std::sqrt(var) : std::numeric_limits<double>::quiet_NaN();
    }
    
    /**
     * @brief Coefficient of variation (relative standard deviation)
     * @return CV = σ/μ (if μ ≠ 0)
     */
    double getCoefficientOfVariation() const {
        double mean = getMean();
        return (mean != 0.0) ? getStandardDeviation() / std::abs(mean) : 
                               std::numeric_limits<double>::infinity();
    }

    // =============================================================================
    // RANDOM NUMBER GENERATION - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Generate single random sample from distribution
     * @param rng Random number generator
     * @return Single random sample
     */
    virtual double sample(std::mt19937& rng) const = 0;
    
    /**
     * @brief Generate multiple random samples from distribution
     * @param rng Random number generator
     * @param n Number of samples to generate
     * @return Vector of random samples
     * @note Base implementation calls sample() n times; override for efficiency
     */
    virtual std::vector<double> sample(std::mt19937& rng, size_t n) const;
    
    // =============================================================================
    // SIMD-OPTIMIZED BATCH OPERATIONS - Virtual (Override for Performance)
    // =============================================================================
    
    /**
     * @brief Batch probability density/mass function evaluation
     * @param x_values Vector of values to evaluate
     * @return Vector of probability densities/masses
     * @note Base implementation calls getProbability() for each value; override for SIMD optimization
     */
    virtual std::vector<double> getBatchProbabilities(const std::vector<double>& x_values) const;
    
    /**
     * @brief Batch log probability density/mass function evaluation
     * @param x_values Vector of values to evaluate
     * @return Vector of log probability densities/masses
     * @note Base implementation calls getLogProbability() for each value; override for SIMD optimization
     */
    virtual std::vector<double> getBatchLogProbabilities(const std::vector<double>& x_values) const;
    
    /**
     * @brief Batch cumulative distribution function evaluation
     * @param x_values Vector of values to evaluate
     * @return Vector of cumulative probabilities P(X <= x)
     * @note Base implementation calls getCumulativeProbability() for each value; override for SIMD optimization
     */
    virtual std::vector<double> getBatchCumulativeProbabilities(const std::vector<double>& x_values) const;
    
    /**
     * @brief Batch quantile function evaluation
     * @param p_values Vector of probability values in [0,1]
     * @return Vector of quantile values
     * @note Base implementation calls getQuantile() for each value; override for SIMD optimization
     * @throws std::invalid_argument if any p_value not in [0,1]
     */
    virtual std::vector<double> getBatchQuantiles(const std::vector<double>& p_values) const;
    
    /**
     * @brief Check if SIMD batch operations should be used for given size
     * @param vector_size Size of the input vector
     * @return True if SIMD operations would be beneficial
     */
    static bool shouldUseSIMDBatch(size_t vector_size) noexcept {
        return vector_size >= constants::simd::DEFAULT_BLOCK_SIZE;
    }

    // =============================================================================
    // PARAMETER ESTIMATION - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Fit distribution parameters to data using Maximum Likelihood Estimation
     * @param data Vector of observations
     * @throws std::invalid_argument if data is empty or contains invalid values
     */
    virtual void fit(const std::vector<double>& data) = 0;
    
    /**
     * @brief Reset distribution to default parameter values
     */
    virtual void reset() noexcept = 0;

    // =============================================================================
    // DISTRIBUTION METADATA - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Get number of parameters for this distribution
     * @return Number of free parameters
     * @note Used for AIC/BIC calculations
     */
    virtual int getNumParameters() const = 0;
    
    /**
     * @brief Get distribution name
     * @return Human-readable distribution name
     */
    virtual std::string getDistributionName() const = 0;
    
    /**
     * @brief Get string representation of distribution with current parameters
     * @return Formatted string description
     */
    virtual std::string toString() const = 0;

    // =============================================================================
    // DISTRIBUTION PROPERTIES - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Check if distribution is discrete
     * @return true if discrete, false if continuous
     */
    virtual bool isDiscrete() const = 0;
    
    /**
     * @brief Get distribution support lower bound
     * @return Minimum possible value (or -infinity)
     */
    virtual double getSupportLowerBound() const = 0;
    
    /**
     * @brief Get distribution support upper bound  
     * @return Maximum possible value (or +infinity)
     */
    virtual double getSupportUpperBound() const = 0;

    // =============================================================================
    // STATISTICAL VALIDATION AND DIAGNOSTICS - Virtual (Override Optional)
    // =============================================================================
    
    /**
     * @brief Validation result structure
     */
    struct ValidationResult {
        double ks_statistic;          ///< Kolmogorov-Smirnov test statistic
        double ks_p_value;            ///< KS test p-value
        double ad_statistic;          ///< Anderson-Darling test statistic  
        double ad_p_value;            ///< AD test p-value
        bool distribution_adequate;   ///< Overall assessment
        std::string recommendations;  ///< Improvement suggestions
    };
    
    /**
     * @brief Enhanced fitting result structure
     */
    struct FitResults {
        double log_likelihood;        ///< Log-likelihood of fitted parameters
        double aic;                   ///< Akaike Information Criterion
        double bic;                   ///< Bayesian Information Criterion
        ValidationResult validation;  ///< Goodness-of-fit assessment
        std::vector<double> residuals; ///< Standardized residuals
        bool fit_successful;          ///< Whether fitting converged
        std::string fit_diagnostics;  ///< Detailed fitting information
    };
    
    /**
     * @brief Fit distribution with comprehensive diagnostics
     * @param data Vector of observations
     * @return Detailed fitting results and validation
     * @note Base implementation calls fit() and validates; override for efficiency
     */
    virtual FitResults fitWithDiagnostics(const std::vector<double>& data);
    
    /**
     * @brief Validate distribution fit against data
     * @param data Vector of observations to validate against
     * @return Validation results
     * @note Base implementation performs KS and AD tests
     */
    virtual ValidationResult validate(const std::vector<double>& data) const;

    // =============================================================================
    // INFORMATION THEORY METRICS - Virtual (Override Optional)
    // =============================================================================
    
    /**
     * @brief Calculate entropy of the distribution
     * @return Differential entropy (continuous) or entropy (discrete)
     * @note Return NaN if not analytically computable
     */
    virtual double getEntropy() const {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    /**
     * @brief Calculate Kullback-Leibler divergence from another distribution
     * @param other Distribution to compare against
     * @return KL divergence D(this||other)
     * @note Base implementation uses numerical integration; override for efficiency
     */
    virtual double getKLDivergence(const DistributionBase& other) const;

    // =============================================================================
    // DISTRIBUTION COMPARISON - Virtual (Override Optional)
    // =============================================================================
    
    /**
     * @brief Check if two distributions are approximately equal
     * @param other Distribution to compare
     * @param tolerance Numerical tolerance for comparison
     * @return true if distributions are approximately equal
     */
    virtual bool isApproximatelyEqual(const DistributionBase& other, 
                                    double tolerance = 1e-10) const;
    
    // =============================================================================
    // FORWARD-COMPATIBILITY TYPES - Public Interface for Level 4 Integration
    // =============================================================================
    
    /**
     * @brief Forward-compatibility type aliases for Level 4 integration
     * @note These will become the primary cache types in the next phase
     */
    template<typename Key, typename Value>
    using AdvancedAdaptiveCache = libstats::cache::AdaptiveCache<Key, Value>;
    using AdvancedCacheConfig = libstats::cache::AdaptiveCacheConfig;

protected:
    // =============================================================================
    // THREAD-SAFE CACHE MANAGEMENT - Protected Interface
    // =============================================================================
    
    /**
     * @brief Thread-safe cache management infrastructure
     */
    mutable std::shared_mutex cache_mutex_;
    mutable bool cache_valid_{false};
    
    /**
     * @brief Atomic cache validity flag for C++20 compliant lock-free fast paths
     * @note This provides lock-free cache validity checking for high-performance scenarios
     */
    mutable std::atomic<bool> cacheValidAtomic_{false};
    
    /**
     * @brief Enhanced cache performance metrics with adaptive management
     */
    struct CacheMetrics {
        std::atomic<size_t> hits{0};
        std::atomic<size_t> misses{0};
        std::atomic<size_t> invalidations{0};
        std::atomic<size_t> updates{0};
        std::atomic<size_t> memory_usage{0};  // Memory usage in bytes
        std::atomic<size_t> cache_evictions{0};
        std::atomic<std::chrono::steady_clock::time_point::rep> last_access_time{0};
        
        double hitRate() const noexcept {
            size_t total_accesses = hits.load() + misses.load();
            return total_accesses > 0 ? static_cast<double>(hits.load()) / total_accesses : 0.0;
        }
        
        double averageUpdateTime() const noexcept {
            size_t total_updates = updates.load();
            return total_updates > 0 ? static_cast<double>(last_access_time.load()) / total_updates : 0.0;
        }
        
        void resetMetrics() noexcept {
            hits.store(0);
            misses.store(0);
            invalidations.store(0);
            updates.store(0);
            memory_usage.store(0);
            cache_evictions.store(0);
            last_access_time.store(0);
        }
    };
    
    /**
     * @brief Enhanced snapshot of cache metrics for external access
     */
    struct CacheMetricsSnapshot {
        size_t hits;
        size_t misses;
        size_t invalidations;
        size_t updates;
        size_t memory_usage;
        size_t cache_evictions;
        double average_update_time;
        
        double hitRate() const noexcept {
            size_t total_accesses = hits + misses;
            return total_accesses > 0 ? static_cast<double>(hits) / total_accesses : 0.0;
        }
        
        double memoryEfficiency() const noexcept {
            return memory_usage > 0 ? static_cast<double>(hits) / memory_usage : 0.0;
        }
    };
    
    /**
     * @brief Legacy cache configuration (DEPRECATED)
     * @deprecated Use libstats::cache::AdaptiveCacheConfig for new implementations
     * @note This will be replaced by libstats::cache::AdaptiveCacheConfig in derived classes
     */
    struct CacheConfig {
        size_t max_memory_usage = 1024 * 1024;  // 1MB default
        size_t min_cache_size = 64;             // Minimum cache entries
        size_t max_cache_size = 1024;           // Maximum cache entries
        double eviction_threshold = 0.8;        // Evict when 80% full
        std::chrono::milliseconds cache_ttl{5000}; // 5 second TTL
        bool adaptive_sizing = true;            // Enable adaptive sizing
        bool memory_pressure_aware = true;      // React to memory pressure
    };
    
    
    mutable CacheMetrics cache_metrics_;
    mutable CacheConfig cache_config_;
    
    /**
     * @brief Cache entry with timestamp and priority
     */
    template<typename T>
    struct CacheEntry {
        T value;
        std::chrono::steady_clock::time_point timestamp;
        size_t access_count = 0;
        size_t memory_size = sizeof(T);
        
        bool isExpired(std::chrono::milliseconds ttl) const noexcept {
            auto now = std::chrono::steady_clock::now();
            return (now - timestamp) > ttl;
        }
        
        double priority() const noexcept {
            auto age = std::chrono::steady_clock::now() - timestamp;
            auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(age).count();
            return static_cast<double>(access_count) / (1.0 + age_ms / 1000.0);
        }
    };
    
    /**
     * @brief Memory-aware cache with adaptive sizing
     */
    template<typename Key, typename Value>
    class AdaptiveCache {
    private:
        mutable std::unordered_map<Key, CacheEntry<Value>> cache_;
        mutable std::shared_mutex cache_mutex_;
        CacheConfig config_;
        mutable CacheMetrics* metrics_;
        
    public:
        explicit AdaptiveCache(const CacheConfig& config, CacheMetrics* metrics) 
            : config_(config), metrics_(metrics) {}
        
        std::optional<Value> get(const Key& key) const {
            std::shared_lock lock(cache_mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                if (!it->second.isExpired(config_.cache_ttl)) {
                    it->second.access_count++;
                    metrics_->hits.fetch_add(1, std::memory_order_relaxed);
                    return it->second.value;
                }
            }
            metrics_->misses.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }
        
        void put(const Key& key, const Value& value) {
            std::unique_lock lock(cache_mutex_);
            
            // Check memory pressure and evict if necessary
            if (shouldEvict()) {
                evictLeastUseful();
            }
            
            CacheEntry<Value> entry;
            entry.value = value;
            entry.timestamp = std::chrono::steady_clock::now();
            entry.access_count = 1;
            
            cache_[key] = std::move(entry);
            updateMemoryUsage();
        }
        
        void clear() {
            std::unique_lock lock(cache_mutex_);
            cache_.clear();
            metrics_->memory_usage.store(0, std::memory_order_relaxed);
        }
        
        size_t size() const {
            std::shared_lock lock(cache_mutex_);
            return cache_.size();
        }
        
    private:
        bool shouldEvict() const {
            if (!config_.memory_pressure_aware) return false;
            
            size_t current_memory = metrics_->memory_usage.load();
            return current_memory > config_.max_memory_usage * config_.eviction_threshold;
        }
        
        void evictLeastUseful() {
            if (cache_.empty()) return;
            
            // Find entry with lowest priority
            auto least_useful = std::min_element(cache_.begin(), cache_.end(),
                [](const auto& a, const auto& b) {
                    return a.second.priority() < b.second.priority();
                });
            
            if (least_useful != cache_.end()) {
                cache_.erase(least_useful);
                metrics_->cache_evictions.fetch_add(1, std::memory_order_relaxed);
            }
        }
        
        void updateMemoryUsage() {
            size_t total_memory = 0;
            for (const auto& entry : cache_) {
                total_memory += entry.second.memory_size;
            }
            metrics_->memory_usage.store(total_memory, std::memory_order_relaxed);
        }
    };
    
    /**
     * @brief Update cached statistical properties (must be overridden)
     * @note Called under unique lock; implementation should set cache_valid_ = true
     */
    virtual void updateCacheUnsafe() const = 0;
    
    /**
     * @brief Invalidate cache when parameters change
     * @note Thread-safe; call whenever parameters are modified
     */
    void invalidateCache() noexcept;
    
    /**
     * @brief Get shared lock for reading cached values
     */
    std::shared_lock<std::shared_mutex> getSharedLock() const noexcept;
    
    /**
     * @brief Get unique lock for modifying cache/parameters
     */
    std::unique_lock<std::shared_mutex> getUniqueLock() const noexcept;
    
    /**
     * @brief Thread-safe cached value access with double-checked locking
     * @param accessor Function to access cached value
     * @return Cached value
     */
    template<typename Func>
    auto getCachedValue(Func&& accessor) const -> decltype(accessor());
    
    /**
     * @brief Get cache performance metrics
     * @return Current cache performance metrics
     */
    CacheMetricsSnapshot getCacheMetrics() const noexcept {
        CacheMetricsSnapshot snapshot;
        snapshot.hits = cache_metrics_.hits.load();
        snapshot.misses = cache_metrics_.misses.load();
        snapshot.invalidations = cache_metrics_.invalidations.load();
        snapshot.updates = cache_metrics_.updates.load();
        snapshot.memory_usage = cache_metrics_.memory_usage.load();
        snapshot.cache_evictions = cache_metrics_.cache_evictions.load();
        snapshot.average_update_time = cache_metrics_.averageUpdateTime();
        return snapshot;
    }
    
    /**
     * @brief Reset cache performance metrics
     */
    void resetCacheMetrics() noexcept {
        cache_metrics_.resetMetrics();
    }
    
    /**
     * @brief Configure cache settings for memory optimization
     * @param config New cache configuration
     */
    void configureCacheSettings(const CacheConfig& config) noexcept {
        cache_config_ = config;
    }
    
    /**
     * @brief Get current cache configuration
     * @return Current cache configuration
     */
    const CacheConfig& getCacheConfig() const noexcept {
        return cache_config_;
    }
    
    // =============================================================================
    // MEMORY OPTIMIZATION FEATURES
    // =============================================================================
    
    /**
     * @brief Memory pool for efficient allocation of temporary objects
     */
    class MemoryPool {
    private:
        static constexpr size_t POOL_SIZE = 1024 * 1024; // 1MB pool
        static constexpr size_t ALIGNMENT = 64; // Cache line alignment
        
        alignas(ALIGNMENT) std::byte pool_[POOL_SIZE];
        std::atomic<size_t> offset_{0};
        mutable std::mutex pool_mutex_;
        
    public:
        /**
         * @brief Allocate aligned memory from pool
         * @param size Size in bytes
         * @param alignment Required alignment (default: 64 bytes for SIMD)
         * @return Pointer to allocated memory or nullptr if insufficient space
         */
        void* allocate(size_t size, size_t alignment = ALIGNMENT) noexcept {
            // Ensure size is aligned
            size = (size + alignment - 1) & ~(alignment - 1);
            
            size_t current_offset = offset_.load(std::memory_order_acquire);
            size_t new_offset = current_offset + size;
            
            if (new_offset > POOL_SIZE) {
                return nullptr; // Pool exhausted
            }
            
            // Try to atomically update offset
            while (!offset_.compare_exchange_weak(current_offset, new_offset,
                                                 std::memory_order_release,
                                                 std::memory_order_acquire)) {
                new_offset = current_offset + size;
                if (new_offset > POOL_SIZE) {
                    return nullptr;
                }
            }
            
            return &pool_[current_offset];
        }
        
        /**
         * @brief Reset pool (not thread-safe, use with caution)
         */
        void reset() noexcept {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            offset_.store(0, std::memory_order_release);
        }
        
        /**
         * @brief Get current pool usage
         * @return Bytes used in pool
         */
        size_t getUsage() const noexcept {
            return offset_.load(std::memory_order_acquire);
        }
        
        /**
         * @brief Check if pool has sufficient space
         * @param size Required size
         * @return true if space available
         */
        bool hasSpace(size_t size) const noexcept {
            return offset_.load(std::memory_order_acquire) + size <= POOL_SIZE;
        }
    };
    
    /**
     * @brief SIMD-aligned vector allocator
     */
    template<typename T>
    class SIMDAllocator {
    public:
        using value_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        
        template<typename U>
        struct rebind {
            using other = SIMDAllocator<U>;
        };
        
        static constexpr size_t SIMD_ALIGNMENT = 64; // 64-byte alignment for AVX-512
        
        SIMDAllocator() = default;
        
        template<typename U>
        SIMDAllocator(const SIMDAllocator<U>&) noexcept {}
        
        pointer allocate(size_type n) {
            if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
                throw std::bad_alloc();
            }
            
            size_type size = n * sizeof(T);
            void* ptr = std::aligned_alloc(SIMD_ALIGNMENT, size);
            
            if (!ptr) {
                throw std::bad_alloc();
            }
            
            return static_cast<pointer>(ptr);
        }
        
        void deallocate(pointer p, size_type) noexcept {
            std::free(p);
        }
        
        template<typename U>
        bool operator==(const SIMDAllocator<U>&) const noexcept {
            return true;
        }
        
        template<typename U>
        bool operator!=(const SIMDAllocator<U>&) const noexcept {
            return false;
        }
    };
    
    /**
     * @brief SIMD-optimized vector type
     */
    template<typename T>
    using simd_vector = std::vector<T, SIMDAllocator<T>>;
    
    /**
     * @brief Memory-efficient small vector optimization
     */
    template<typename T, size_t N = 8>
    class SmallVector {
    private:
        alignas(T) std::byte storage_[N * sizeof(T)];
        std::unique_ptr<T[]> heap_data_;
        size_t size_ = 0;
        size_t capacity_ = N;
        
        T* data() noexcept {
            return capacity_ == N ? reinterpret_cast<T*>(storage_) : heap_data_.get();
        }
        
        const T* data() const noexcept {
            return capacity_ == N ? reinterpret_cast<const T*>(storage_) : heap_data_.get();
        }
        
    public:
        SmallVector() = default;
        
        ~SmallVector() {
            clear();
        }
        
        SmallVector(const SmallVector& other) : size_(other.size_), capacity_(other.capacity_) {
            if (capacity_ > N) {
                heap_data_ = std::make_unique<T[]>(capacity_);
            }
            
            for (size_t i = 0; i < size_; ++i) {
                new (data() + i) T(other.data()[i]);
            }
        }
        
        SmallVector(SmallVector&& other) noexcept
            : size_(other.size_), capacity_(other.capacity_) {
            if (capacity_ > N) {
                heap_data_ = std::move(other.heap_data_);
            } else {
                for (size_t i = 0; i < size_; ++i) {
                    new (data() + i) T(std::move(other.data()[i]));
                }
            }
            other.size_ = 0;
            other.capacity_ = N;
        }
        
        void push_back(const T& value) {
            if (size_ >= capacity_) {
                reserve(capacity_ * 2);
            }
            new (data() + size_) T(value);
            ++size_;
        }
        
        void push_back(T&& value) {
            if (size_ >= capacity_) {
                reserve(capacity_ * 2);
            }
            new (data() + size_) T(std::move(value));
            ++size_;
        }
        
        void reserve(size_t new_capacity) {
            if (new_capacity <= capacity_) return;
            
            auto new_data = std::make_unique<T[]>(new_capacity);
            
            for (size_t i = 0; i < size_; ++i) {
                new (new_data.get() + i) T(std::move(data()[i]));
                data()[i].~T();
            }
            
            heap_data_ = std::move(new_data);
            capacity_ = new_capacity;
        }
        
        void clear() {
            for (size_t i = 0; i < size_; ++i) {
                data()[i].~T();
            }
            size_ = 0;
        }
        
        size_t size() const noexcept { return size_; }
        size_t capacity() const noexcept { return capacity_; }
        bool empty() const noexcept { return size_ == 0; }
        
        T& operator[](size_t index) noexcept { return data()[index]; }
        const T& operator[](size_t index) const noexcept { return data()[index]; }
        
        T* begin() noexcept { return data(); }
        T* end() noexcept { return data() + size_; }
        const T* begin() const noexcept { return data(); }
        const T* end() const noexcept { return data() + size_; }
    };
    
    /**
     * @brief Stack-based memory allocator for temporary computations
     */
    template<size_t StackSize = 4096>
    class StackAllocator {
    private:
        alignas(std::max_align_t) char stack_[StackSize];
        char* current_ = stack_;
        
    public:
        template<typename T>
        T* allocate(size_t count) {
            size_t size = count * sizeof(T);
            size_t alignment = alignof(T);
            
            // Align current pointer
            uintptr_t ptr = reinterpret_cast<uintptr_t>(current_);
            ptr = (ptr + alignment - 1) & ~(alignment - 1);
            current_ = reinterpret_cast<char*>(ptr);
            
            if (current_ + size > stack_ + StackSize) {
                throw std::bad_alloc();
            }
            
            T* result = reinterpret_cast<T*>(current_);
            current_ += size;
            return result;
        }
        
        void reset() noexcept {
            current_ = stack_;
        }
        
        size_t used() const noexcept {
            return current_ - stack_;
        }
        
        size_t available() const noexcept {
            return StackSize - used();
        }
    };
    
    // Thread-local memory pool for performance
    static thread_local MemoryPool thread_pool_;
    
    /**
     * @brief Get thread-local memory pool
     * @return Reference to thread-local memory pool
     */
    static MemoryPool& getThreadPool() noexcept {
        return thread_pool_;
    }

    // =============================================================================
    // NUMERICAL UTILITIES - Protected Static Methods
    // =============================================================================
    
    /**
     * @brief Numerical integration for CDF calculation
     * @param pdf_func PDF function to integrate
     * @param lower_bound Integration lower bound
     * @param upper_bound Integration upper bound
     * @param tolerance Numerical tolerance
     * @return Integral approximation
     */
    static double numericalIntegration(std::function<double(double)> pdf_func,
                                     double lower_bound, double upper_bound,
                                     double tolerance = 1e-8);
    
    /**
     * @brief Newton-Raphson method for quantile calculation
     * @param cdf_func CDF function
     * @param target_probability Target probability value
     * @param initial_guess Initial guess for root
     * @param tolerance Numerical tolerance
     * @return Quantile approximation
     */
    static double newtonRaphsonQuantile(std::function<double(double)> cdf_func,
                                      double target_probability,
                                      double initial_guess,
                                      double tolerance = 1e-10);
    
    /**
     * @brief Validate data for fitting
     * @param data Data to validate
     * @throws std::invalid_argument if data is invalid
     */
    static void validateFittingData(const std::vector<double>& data);
    
    /**
     * @brief Calculate empirical CDF from data
     * @param data Sorted data vector
     * @return Vector of empirical CDF values
     */
    static std::vector<double> calculateEmpiricalCDF(const std::vector<double>& data);

    // =============================================================================
    // SPECIAL MATHEMATICAL FUNCTIONS - Protected Static Methods
    // =============================================================================
    
    /**
     * @brief Error function erf(x)
     */
    static double erf(double x) noexcept;
    
    /**
     * @brief Complementary error function erfc(x)
     */
    static double erfc(double x) noexcept;
    
    /**
     * @brief Log gamma function ln(Γ(x))
     */
    static double lgamma(double x) noexcept;
    
    /**
     * @brief Regularized incomplete gamma function P(a,x)
     */
    static double gammaP(double a, double x) noexcept;
    
    /**
     * @brief Regularized incomplete gamma function Q(a,x) = 1 - P(a,x)
     */
    static double gammaQ(double a, double x) noexcept;
    
    /**
     * @brief Regularized incomplete beta function I_x(a,b)
     */
    static double betaI(double x, double a, double b) noexcept;

private:
    // =============================================================================
    // INTERNAL IMPLEMENTATION DETAILS
    // =============================================================================
    
    
    /**
     * @brief Helper function for incomplete beta function continued fraction
     */
    static double betaI_continued_fraction(double x, double a, double b) noexcept;
    
    /**
     * @brief Helper function for adaptive Simpson's rule integration
     */
    static double adaptiveSimpsonIntegration(std::function<double(double)> func,
                                           double a, double b, double tolerance,
                                           int depth, int max_depth);
};

// =============================================================================
// TEMPLATE SPECIALIZATIONS FOR EFFICIENT IMPLEMENTATION
// =============================================================================

/**
 * @brief Template helper for cached statistical properties
 * @tparam PropertyType Type of cached property
 */
template<typename PropertyType>
class CachedProperty {
private:
    mutable PropertyType value_;
    mutable bool valid_{false};
    
public:
    template<typename ComputeFunc>
    PropertyType get(ComputeFunc&& compute_func) const {
        if (!valid_) {
            value_ = compute_func();
            valid_ = true;
        }
        return value_;
    }
    
    void invalidate() noexcept {
        valid_ = false;
    }
};

// =============================================================================
// IMPLEMENTATION DETAILS
// =============================================================================

// Template method implementation
template<typename Func>
auto DistributionBase::getCachedValue(Func&& accessor) const -> decltype(accessor()) {
    // Fast path: check if cache is valid under shared lock
    {
        auto lock = getSharedLock();
        if (cache_valid_) {
            cache_metrics_.hits.fetch_add(1, std::memory_order_relaxed);
            return accessor();
        }
    }
    
    // Slow path: update cache under unique lock
    {
        auto lock = getUniqueLock();
        if (!cache_valid_) {  // Double-check pattern
            cache_metrics_.misses.fetch_add(1, std::memory_order_relaxed);
            updateCacheUnsafe();
            cache_metrics_.updates.fetch_add(1, std::memory_order_relaxed);
        } else {
            // Another thread updated the cache while we were waiting
            cache_metrics_.hits.fetch_add(1, std::memory_order_relaxed);
        }
        return accessor();
    }
}

} // namespace libstats

#endif // LIBSTATS_DISTRIBUTION_BASE_H_

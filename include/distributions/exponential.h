#ifndef LIBSTATS_EXPONENTIAL_H_
#define LIBSTATS_EXPONENTIAL_H_

#include "../core/distribution_base.h"
#include "../core/constants.h"
#include "../platform/simd.h" // Ensure SIMD operations are available
#include "../core/error_handling.h" // Safe error handling without exceptions
#include "../platform/adaptive_cache.h" // For adaptive cache integration
#include "../platform/parallel_execution.h" // For parallel execution policies
#include "../platform/work_stealing_pool.h" // For WorkStealingPool
#include "../core/performance_dispatcher.h" // For smart auto-dispatch
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation
#include <span>        // C++20 std::span for type-safe array access

namespace libstats {

/**
 * @brief Thread-safe Exponential Distribution for modeling waiting times and decay processes.
 * 
 * @details The Exponential distribution is a continuous probability distribution that describes
 * the time between events in a Poisson point process. It's commonly used to model
 * lifetimes, waiting times, and decay processes with the key memoryless property.
 * 
 * @par Mathematical Definition:
 * - PDF: f(x; λ) = λ * exp(-λx) for x ≥ 0, 0 otherwise
 * - CDF: F(x; λ) = 1 - exp(-λx) for x ≥ 0, 0 otherwise
 * - Parameters: λ > 0 (rate parameter)
 * - Support: x ∈ [0, ∞)
 * - Mean: 1/λ
 * - Variance: 1/λ²
 * - Mode: 0 (distribution is monotonically decreasing)
 * - Median: ln(2)/λ ≈ 0.693/λ
 * 
 * @par Key Properties:
 * - **Memoryless Property**: P(X > s+t | X > s) = P(X > t)
 * - **Scale Parameter**: 1/λ is the scale parameter (mean waiting time)
 * - **Relationship to Poisson**: Inter-arrival times in Poisson process
 * - **Conjugate Prior**: Gamma distribution is conjugate prior for λ
 * - **Maximum Entropy**: Among all distributions with given mean on [0,∞)
 * 
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - Concurrent reads are optimized with std::shared_mutex
 * - Cache invalidation uses atomic operations for lock-free fast paths
 * - Deadlock prevention via ordered lock acquisition with std::lock()
 * 
 * @par Performance Features:
 * - Atomic cache validity flags for lock-free fast path access
 * - Extensive caching of computed values (log(λ), 1/λ, -λ, 1/λ²)
 * - Optimized PDF/CDF computation avoiding repeated calculations
 * - Fast parameter validation with IEEE 754 compliance
 * - Branch-free computation paths for common operations
 * 
 * @par Usage Examples:
 * @code
 * // Standard exponential distribution (λ=1, mean=1)
 * auto result = ExponentialDistribution::create(1.0);
 * if (result.isOk()) {
 *     auto standard = std::move(result.value);
 *     
 *     // Fast decay process (λ=5, mean=0.2)
 *     auto fastResult = ExponentialDistribution::create(5.0);
 *     if (fastResult.isOk()) {
 *         auto fastDecay = std::move(fastResult.value);
 *         
 *         // Fit to inter-arrival time data
 *         std::vector<double> waitTimes = {0.1, 0.3, 0.7, 0.2, 0.5};
 *         fastDecay.fit(waitTimes);
 *         
 *         // Compute survival probability P(X > t)
 *         double survivalProb = 1.0 - fastDecay.getCumulativeProbability(2.0);
 *     }
 * }
 * @endcode
 * 
 * @par Applications:
 * - Reliability engineering (failure time analysis)
 * - Queuing theory (service time modeling)
 * - Radioactive decay modeling
 * - Network packet inter-arrival times
 * - Financial risk modeling (time to default)
 * - Biological processes (cell division times)
 * - Machine maintenance scheduling
 * 
 * @par Statistical Properties:
 * - Skewness: 2 (always right-skewed)
 * - Kurtosis: 6 (heavy-tailed distribution)
 * - Entropy: 1 - ln(λ)
 * - Moment generating function: λ/(λ-t) for t < λ
 * 
 * @par Implementation Details (C++20 Best Practices):
 * - Complex constructors/operators moved to .cpp for faster compilation
 * - Exception-safe design with RAII principles
 * - Optimized parameter validation with comprehensive error messages
 * - Lock-free fast paths using atomic operations
 * - IEEE 754 compliant floating-point handling
 * 
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
class ExponentialDistribution : public DistributionBase
{   
private:
    //==========================================================================
    // DISTRIBUTION PARAMETERS
    //==========================================================================
    
    /** @brief Rate parameter λ - must be positive */
    double lambda_{constants::math::ONE};
    
    /** @brief C++20 atomic copy of parameter for lock-free access */
    mutable std::atomic<double> atomicLambda_{constants::math::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // PERFORMANCE CACHE
    //==========================================================================
    
    /** @brief Cached value of ln(λ) for efficiency in log probability calculations */
    mutable double logLambda_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of 1/λ (mean and scale parameter) for efficiency */
    mutable double invLambda_{constants::math::ONE};
    
    /** @brief Cached value of -λ for efficiency in PDF and log-PDF calculations */
    mutable double negLambda_{-constants::math::ONE};
    
    /** @brief Cached value of 1/λ² for variance calculation efficiency */
    mutable double invLambdaSquared_{constants::math::ONE};
    
    //==========================================================================
    // OPTIMIZATION FLAGS
    //==========================================================================
    
    /** @brief Atomic cache validity flag for lock-free fast path optimization */
    mutable std::atomic<bool> cacheValidAtomic_{false};
    
    /** @brief True if λ = 1 for unit exponential optimizations */
    mutable bool isUnitRate_{true};
    
    /** @brief True if λ is very large (> 1000) for numerical stability */
    mutable bool isHighRate_{false};
    
    /** @brief True if λ is very small (< 0.001) for numerical stability */
    mutable bool isLowRate_{false};

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        invLambda_ = constants::math::ONE / lambda_;
        invLambdaSquared_ = invLambda_ * invLambda_;
        
        // Core cached values
        logLambda_ = std::log(lambda_);
        negLambda_ = -lambda_;
        
        // Optimization flags
        isUnitRate_ = (std::abs(lambda_ - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        isHighRate_ = (lambda_ > constants::math::THOUSAND);
        isLowRate_ = (lambda_ < constants::math::THOUSANDTH);
        
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
        
        // Update atomic parameters for lock-free access
        atomicLambda_.store(lambda_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    
    /**
     * Validates parameters for the Exponential distribution
     * @param lambda Rate parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double lambda) {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");
        }
    }

    friend std::istream& operator>>(std::istream& is,
            libstats::ExponentialDistribution& distribution);

public:
    //==========================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    
    /**
     * Constructs an Exponential distribution with given rate parameter.
     * 
     * @param lambda Rate parameter λ (must be positive)
     * @throws std::invalid_argument if lambda is invalid
     * 
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit ExponentialDistribution(double lambda = constants::math::ONE);
    
    /**
     * Thread-safe copy constructor
     * 
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    ExponentialDistribution(const ExponentialDistribution& other);
    
    /**
     * Copy assignment operator
     * 
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    ExponentialDistribution& operator=(const ExponentialDistribution& other);
    
    /**
     * Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    ExponentialDistribution(ExponentialDistribution&& other);
    
    /**
     * Move assignment operator (C++11 COMPLIANT)
     * Implementation in .cpp: Thread-safe move with atomic operations
     * @note noexcept compliant using atomic state management
     */
    ExponentialDistribution& operator=(ExponentialDistribution&& other) noexcept;

    /**
     * @brief Destructor - explicitly defaulted to satisfy Rule of Five
     * Implementation inline: Trivial destruction, kept for performance
     * 
     * Note: C++20 Best Practice - Rule of Five uses complexity-based placement:
     * - Simple operations (destructor) stay inline for performance
     * - Complex operations (copy/move) moved to .cpp for maintainability
     */
    ~ExponentialDistribution() override = default;
    
    //==========================================================================
    // SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================
    
    /**
     * @brief Safely create an Exponential distribution without throwing exceptions
     * 
     * This factory method provides exception-free construction to work around
     * ABI compatibility issues with Homebrew LLVM libc++ on macOS where
     * exceptions thrown from the library cause segfaults during unwinding.
     * 
     * @param lambda Rate parameter λ (must be positive)
     * @return Result containing either a valid ExponentialDistribution or error info
     * 
     * @par Usage Example:
     * @code
     * auto result = ExponentialDistribution::create(2.0);
     * if (result.isOk()) {
     *     auto distribution = std::move(result.value);
     *     // Use distribution safely...
     * } else {
     *     std::cout << "Error: " << result.message << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] static Result<ExponentialDistribution> create(double lambda = 1.0) noexcept {
        auto validation = validateExponentialParameters(lambda);
        if (validation.isError()) {
            return Result<ExponentialDistribution>::makeError(validation.error_code, validation.message);
        }
        
        // Use private factory to bypass validation
        return Result<ExponentialDistribution>::ok(createUnchecked(lambda));
    }
    
    /**
     * @brief Safely try to set parameters without throwing exceptions
     * 
     * @param lambda New rate parameter
     * @return VoidResult indicating success or failure
     * 
     * Implementation in .cpp: Complex thread-safe parameter validation and atomic state management
     */
    [[nodiscard]] VoidResult trySetParameters(double lambda) noexcept;
    
    /**
     * @brief Check if current parameters are valid
     * @return VoidResult indicating validity
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return validateExponentialParameters(lambda_);
    }

    //==========================================================================
    // CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * Computes the probability density function for the Exponential distribution.
     * 
     * @param x The value at which to evaluate the PDF
     * @return Probability density (or approximated probability for discrete sampling)
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * Computes the logarithm of the probability density function for numerical stability.
     * 
     * For exponential distribution: log(f(x)) = log(λ) - λx for x ≥ 0
     * 
     * @param x The value at which to evaluate the log-PDF
     * @return Natural logarithm of the probability density, or -∞ for invalid values
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * Evaluates the CDF at x using the standard exponential CDF formula
     * For exponential distribution: F(x) = 1 - exp(-λx) for x ≥ 0, 0 otherwise
     * 
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;
    
    /**
     * @brief Computes the quantile function (inverse CDF)
     * For exponential distribution: F^(-1)(p) = -ln(1-p)/λ
     * 
     * @param p Probability value in [0,1]
     * @return x such that P(X ≤ x) = p
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;
    
    /**
     * @brief Generate single random sample from distribution
     * Uses inverse transform method: X = -ln(U)/λ where U ~ Uniform(0,1)
     * 
     * @param rng Random number generator
     * @return Single random sample
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;
    
    /**
     * @brief Generate multiple random samples from distribution
     * Optimized batch sampling using inverse transform method
     * 
     * @param rng Random number generator
     * @param n Number of samples to generate
     * @return Vector of random samples
     */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /**
     * Gets the rate parameter λ.
     * Thread-safe: acquires shared lock to protect lambda_
     * 
     * @return Current rate parameter value
     */
    [[nodiscard]] double getLambda() const noexcept { 
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return lambda_; 
    }
    
    /**
     * @brief Fast lock-free atomic getter for rate parameter λ
     * 
     * Provides high-performance access to the rate parameter using atomic operations
     * for lock-free fast path. Falls back to locked getter if atomic parameters
     * are not valid (e.g., during parameter updates).
     * 
     * @return Current rate parameter value
     * 
     * @note This method is optimized for high-frequency access patterns where
     *       the distribution parameters are relatively stable. It uses atomic
     *       loads with acquire semantics for proper memory synchronization.
     * 
     * @par Performance Characteristics:
     * - Lock-free fast path: ~2-5ns per call
     * - Fallback to locked path: ~50-100ns per call
     * - Thread-safe without blocking other readers
     * 
     * @par Usage Example:
     * @code
     * // High-frequency parameter access in performance-critical loops
     * for (size_t i = 0; i < large_dataset.size(); ++i) {
     *     double lambda = dist.getLambdaAtomic();  // Lock-free access
     *     results[i] = compute_something(data[i], lambda);
     * }
     * @endcode
     */
    [[nodiscard]] double getLambdaAtomic() const noexcept {
        // Fast path: check if atomic parameters are valid
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            // Lock-free atomic access with proper memory ordering
            return atomicLambda_.load(std::memory_order_acquire);
        }
        
        // Fallback: use traditional locked getter if atomic parameters are stale
        return getLambda();
    }
    
    /**
     * Sets the rate parameter λ (exception-based API).
     * 
     * @param lambda New rate parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    void setLambda(double lambda);
    
    /**
     * @brief Safely set the rate parameter λ without throwing exceptions (Result-based API).
     * 
     * @param lambda New rate parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetLambda(double lambda) noexcept;
    
    /**
     * Gets the mean of the distribution.
     * For Exponential distribution, mean = 1/λ
     * Uses cached value to eliminate division.
     * 
     * @return Mean value
     */
    [[nodiscard]] double getMean() const noexcept override;
    
    /**
     * Gets the variance of the distribution.
     * For Exponential distribution, variance = 1/λ²
     * Uses cached value to eliminate divisions and multiplications.
     * 
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const noexcept override;
    
    /**
     * @brief Gets the skewness of the distribution.
     * For Exponential distribution, skewness = 2 (always right-skewed)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Skewness value (always 2)
     */
    [[nodiscard]] double getSkewness() const noexcept override {
        return 2.0;  // Exponential distribution is always right-skewed
    }
    
    /**
     * @brief Gets the kurtosis of the distribution.
     * For Exponential distribution, excess kurtosis = 6
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Excess kurtosis value (always 6)
     */
    [[nodiscard]] double getKurtosis() const noexcept override {
        return 6.0;  // Exponential distribution has high kurtosis
    }
    
    /**
     * @brief Gets the number of parameters for this distribution.
     * For Exponential distribution, there is 1 parameter: lambda (rate)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Number of parameters (always 1)
     */
    [[nodiscard]] int getNumParameters() const noexcept override {
        return 1;
    }
    
    /**
     * @brief Gets the distribution name.
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Distribution name
     */
    [[nodiscard]] std::string getDistributionName() const override {
        return "Exponential";
    }
    
    /**
     * @brief Checks if the distribution is discrete.
     * For Exponential distribution, it's continuous
     * Inline for performance - no thread safety needed for constant
     * 
     * @return false (always continuous)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override {
        return false;
    }
    
    /**
     * @brief Gets the lower bound of the distribution support.
     * For Exponential distribution, support is [0, ∞)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Lower bound (0)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return 0.0;
    }
    
    /**
     * @brief Gets the upper bound of the distribution support.
     * For Exponential distribution, support is [0, ∞)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Upper bound (+infinity)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }
    
    /**
     * Gets the scale parameter (reciprocal of rate parameter).
     * This is equivalent to the mean for exponential distributions.
     * Uses cached value to eliminate division.
     * 
     * @return Scale parameter (1/λ)
     */
    [[nodiscard]] double getScale() const noexcept;

    //==========================================================================
    // DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Exponential distribution, MLE gives λ = 1/sample_mean.
     * 
     * @param values Vector of observed data
     */
    void fit(const std::vector<double>& values) override;

    /**
     * Resets the distribution to default parameters (λ = 1.0).
     * This corresponds to the standard exponential distribution.
     */
    void reset() noexcept override;

    /**
     * Returns a string representation of the distribution.
     * 
     * @return String describing the distribution parameters
     */
    std::string toString() const override;
    
    //==========================================================================
    // ADVANCED STATISTICAL METHODS
    //==========================================================================
    
    /**
     * @brief Confidence interval for rate parameter λ
     * 
     * Calculates confidence interval for the population rate parameter using
     * the chi-squared distribution (since 2nλ/λ̂ follows χ²(2n) distribution).
     * 
     * @param data Sample data
     * @param confidence_level Confidence level (e.g., 0.95 for 95%)
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<double, double> confidenceIntervalRate(
        const std::vector<double>& data, 
        double confidence_level = 0.95);
    
    /**
     * @brief Confidence interval for scale parameter (mean waiting time)
     * 
     * Calculates confidence interval for population scale parameter (1/λ)
     * using the relationship with the rate parameter confidence interval.
     * 
     * @param data Sample data  
     * @param confidence_level Confidence level (e.g., 0.95 for 95%)
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<double, double> confidenceIntervalScale(
        const std::vector<double>& data,
        double confidence_level = 0.95);
    
    /**
     * @brief Likelihood ratio test for exponential parameter
     * 
     * Tests the null hypothesis H₀: λ = λ₀ against H₁: λ ≠ λ₀
     * using the likelihood ratio statistic -2ln(Λ) ~ χ²(1).
     * 
     * @param data Sample data
     * @param null_lambda Hypothesized rate parameter under H₀
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (LR_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> likelihoodRatioTest(
        const std::vector<double>& data,
        double null_lambda,
        double alpha = 0.05);
    
    /**
     * @brief Bayesian parameter estimation with Gamma conjugate prior
     * 
     * Performs Bayesian estimation of exponential rate parameter using
     * Gamma conjugate prior. For exponential likelihood with Gamma(α,β) prior,
     * the posterior is Gamma(α + n, β + Σxᵢ).
     * 
     * @param data Observed data
     * @param prior_shape Prior shape parameter α (default: 1)
     * @param prior_rate Prior rate parameter β (default: 1)
     * @return Pair of (posterior_shape, posterior_rate)
     */
    static std::pair<double, double> bayesianEstimation(
        const std::vector<double>& data,
        double prior_shape = 1.0,
        double prior_rate = 1.0);
    
    /**
     * @brief Credible interval from Bayesian posterior
     * 
     * Calculates Bayesian credible interval for rate parameter
     * from posterior Gamma distribution.
     * 
     * @param data Observed data
     * @param credibility_level Credibility level (e.g., 0.95 for 95%)
     * @param prior_shape Prior shape parameter α (default: 1)
     * @param prior_rate Prior rate parameter β (default: 1)
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<double, double> bayesianCredibleInterval(
        const std::vector<double>& data,
        double credibility_level = 0.95,
        double prior_shape = 1.0,
        double prior_rate = 1.0);
    
    /**
     * @brief Robust parameter estimation using M-estimators
     * 
     * Provides robust estimation of rate parameter that is less
     * sensitive to outliers than maximum likelihood. Uses truncated
     * likelihood or Winsorized estimation.
     * 
     * @param data Sample data
     * @param estimator_type Type of robust estimator ("winsorized", "trimmed")
     * @param trim_proportion Proportion to trim/winsorize (default: 0.1)
     * @return Robust rate parameter estimate
     */
    static double robustEstimation(
        const std::vector<double>& data,
        const std::string& estimator_type = "winsorized",
        double trim_proportion = 0.1);
    
    /**
     * @brief Method of moments parameter estimation
     * 
     * Estimates rate parameter by matching sample moments with
     * theoretical distribution moments. For exponential: λ = 1/sample_mean.
     * 
     * @param data Sample data
     * @return Rate parameter estimate
     */
    static double methodOfMomentsEstimation(
        const std::vector<double>& data);
    
    /**
     * @brief L-moments parameter estimation
     * 
     * Uses L-moments (linear combinations of order statistics)
     * for robust parameter estimation. L₁ = mean, λ = 1/L₁.
     * 
     * @param data Sample data
     * @return Rate parameter estimate from L-moments
     */
    static double lMomentsEstimation(
        const std::vector<double>& data);
    
    /**
     * @brief Exponentiality test using coefficient of variation
     * 
     * Tests the null hypothesis that data follows an exponential distribution.
     * Uses the fact that CV = 1 exactly for exponential distributions.
     * 
     * @param data Sample data
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (CV_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> coefficientOfVariationTest(
        const std::vector<double>& data,
        double alpha = 0.05);
    
    /**
     * @brief Kolmogorov-Smirnov goodness-of-fit test
     * 
     * Tests the null hypothesis that data follows the specified exponential distribution.
     * Compares empirical CDF with theoretical exponential CDF.
     * 
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (KS_statistic, p_value, reject_null)
     * @note p_value approximation using asymptotic distribution
     */
    static std::tuple<double, double, bool> kolmogorovSmirnovTest(
        const std::vector<double>& data,
        const ExponentialDistribution& distribution,
        double alpha = 0.05);
    
    /**
     * @brief Anderson-Darling goodness-of-fit test
     * 
     * Tests the null hypothesis that data follows the specified exponential distribution.
     * More sensitive to deviations in the tails than KS test.
     * 
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (AD_statistic, p_value, reject_null)
     * @note Uses asymptotic p-value approximation for exponential case
     */
    static std::tuple<double, double, bool> andersonDarlingTest(
        const std::vector<double>& data,
        const ExponentialDistribution& distribution,
        double alpha = 0.05);
    
    //==========================================================================
    // CROSS-VALIDATION AND MODEL SELECTION
    //==========================================================================
    
    /**
     * @brief K-fold cross-validation for parameter estimation
     * 
     * Performs k-fold cross-validation to assess parameter estimation quality
     * and model stability. Splits data into k folds, trains on k-1 folds,
     * and validates on the remaining fold.
     * 
     * @param data Sample data for cross-validation
     * @param k Number of folds (default: 5)
     * @param random_seed Seed for random fold assignment (default: 42)
     * @return Vector of k validation results: (rate_error, scale_error, log_likelihood)
     */
    static std::vector<std::tuple<double, double, double>> kFoldCrossValidation(
        const std::vector<double>& data,
        int k = 5,
        unsigned int random_seed = 42);
    
    /**
     * @brief Leave-one-out cross-validation for parameter estimation
     * 
     * Performs leave-one-out cross-validation (LOOCV) to assess parameter
     * estimation quality. For each data point, trains on all other points
     * and validates on the left-out point.
     * 
     * @param data Sample data for cross-validation
     * @return Tuple of (mean_absolute_error, root_mean_squared_error, total_log_likelihood)
     */
    static std::tuple<double, double, double> leaveOneOutCrossValidation(
        const std::vector<double>& data);
    
    /**
     * @brief Bootstrap parameter confidence intervals
     * 
     * Uses bootstrap resampling to estimate confidence intervals for
     * the rate parameter λ.
     * 
     * @param data Sample data for bootstrap resampling
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @param n_bootstrap Number of bootstrap samples (default: 1000)
     * @param random_seed Seed for random sampling (default: 42)
     * @return Pair of (rate_CI_lower, rate_CI_upper)
     */
    static std::pair<double, double> bootstrapParameterConfidenceInterval(
        const std::vector<double>& data,
        double confidence_level = 0.95,
        int n_bootstrap = 1000,
        unsigned int random_seed = 42);
    
    /**
     * @brief Model comparison using information criteria
     * 
     * Computes various information criteria (AIC, BIC, AICc) for model selection.
     * Lower values indicate better model fit while penalizing complexity.
     * 
     * @param data Sample data used for fitting
     * @param fitted_distribution The fitted exponential distribution
     * @return Tuple of (AIC, BIC, AICc, log_likelihood)
     */
    static std::tuple<double, double, double, double> computeInformationCriteria(
        const std::vector<double>& data,
        const ExponentialDistribution& fitted_distribution);
    
    //==========================================================================
    // SAFE BATCH OPERATIONS WITH SIMD ACCELERATION
    // 
    // 🛡️  DEVELOPER SAFETY GUIDE FOR BATCH OPERATIONS 🛡️
    // 
    // These methods provide high-performance batch probability calculations with
    // comprehensive safety guarantees. While they internally use optimized SIMD
    // operations with raw pointers, all unsafe operations are encapsulated behind
    // thoroughly validated public interfaces.
    // 
    // SAFETY FEATURES PROVIDED:
    // ✅ Automatic input validation and bounds checking
    // ✅ Thread-safe parameter caching with proper locking
    // ✅ Runtime CPU feature detection for SIMD compatibility
    // ✅ Graceful fallback to scalar operations when SIMD unavailable
    // ✅ Memory alignment handled automatically
    // 
    // RECOMMENDED USAGE PATTERNS:
    // 1. For basic users: Use these raw pointer interfaces with pre-allocated arrays
    // 2. For C++20 users: Prefer the std::span interfaces below for type safety
    // 3. For parallel processing: Use the ParallelUtils-integrated methods
    // 4. For maximum safety: Use the cache-aware methods with additional validation
    // 
    // PERFORMANCE CHARACTERISTICS:
    // - Small arrays (\u003c ~64 elements): Uses scalar loops, no SIMD overhead
    // - Large arrays (≥ ~64 elements): Uses SIMD vectorization (2-8x speedup)
    // - Thread-safe caching minimizes parameter validation overhead
    // - Zero-copy operation on properly sized input arrays
    //==========================================================================
    
    /**
     * @brief Safe SIMD-optimized batch probability calculation
     * 
     * Computes PDF for multiple values simultaneously using vectorized operations.
     * This method automatically handles input validation, thread safety, and
     * optimal algorithm selection based on array size and CPU capabilities.
     * 
     * @param values Pointer to array of input values (must be valid for 'count' elements)
     * @param results Pointer to pre-allocated array for results (must be valid for 'count' elements)
     * @param count Number of values to process (must be \u003e 0)
     * 
     * @throws std::invalid_argument if count is 0 or pointers are invalid
     * 
     * @note Arrays do not need special alignment - alignment is handled internally
     * @note For arrays \u003c ~64 elements, uses optimized scalar loops
     * @note For arrays ≥ ~64 elements, uses SIMD vectorization when available
     * 
     * @par Example Usage:
     * @code
     * std::vector\u003cdouble\u003e inputs = {0.0, 1.0, 2.0, -1.0};
     * std::vector\u003cdouble\u003e outputs(inputs.size());
     * distribution.getProbabilityBatch(inputs.data(), outputs.data(), inputs.size());
     * @endcode
     */
    void getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept;
    
    /**
     * SIMD-optimized batch log probability calculation
     * Computes log PDF for multiple values simultaneously using vectorized operations
     * @param values Array of input values
     * @param results Array to store log probability results (must be pre-allocated)
     * @param count Number of values to process
     * @warning Arrays must be aligned to SIMD_ALIGNMENT for optimal performance
     */
    void getLogProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept;
    
    /**
     * SIMD-optimized batch cumulative probability calculation
     * Computes CDF for multiple values simultaneously using vectorized operations
     * @param values Array of input values
     * @param results Array to store cumulative probability results (must be pre-allocated)
     * @param count Number of values to process
     * @warning Arrays must be aligned to SIMD_ALIGNMENT for optimal performance
     */
    void getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const;
    
    /**
     * Ultra-high performance lock-free batch operations
     * These methods assume cache is valid and skip all locking - use with extreme care
     * @warning Only call when you can guarantee cache validity and thread safety
     */
    void getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept;
    void getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept;
    void getCumulativeProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const noexcept;
    
    //==========================================================================
    // THREAD POOL PARALLEL BATCH OPERATIONS
    //==========================================================================
    
    /**
     * Advanced parallel batch probability calculation using ParallelUtils::parallelFor
     * Leverages Level 0-3 thread pool infrastructure for optimal work distribution
     * Combines SIMD vectorization with multi-core parallelism for maximum performance
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @throws std::invalid_argument if span sizes don't match
     */
    void getProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const;
    
    /**
     * Advanced parallel batch log probability calculation using ParallelUtils::parallelFor
     * Leverages Level 0-3 thread pool infrastructure for optimal work distribution
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @throws std::invalid_argument if span sizes don't match
     */
    void getLogProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const noexcept;
    
    /**
     * Advanced parallel batch CDF calculation using ParallelUtils::parallelFor
     * Leverages Level 0-3 thread pool infrastructure for optimal work distribution
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @throws std::invalid_argument if span sizes don't match
     */
    void getCumulativeProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const;
    
    /**
     * Work-stealing parallel batch probability calculation for heavy computational loads
     * Uses WorkStealingPool for dynamic load balancing across uneven workloads
     * Optimal for large datasets where work distribution may be irregular
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @param pool Reference to WorkStealingPool for load balancing
     * @throws std::invalid_argument if span sizes don't match
     */
    void getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                        WorkStealingPool& pool) const;
    
    /**
     * Cache-aware batch processing using adaptive cache management
     * Integrates with Level 0-3 adaptive cache system for predictive cache warming
     * Automatically determines optimal batch sizes based on cache behavior
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @param cache_manager Reference to adaptive cache manager
     * @throws std::invalid_argument if span sizes don't match
     */
    void getProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                      cache::AdaptiveCache<std::string, double>& cache_manager) const;
    
    /**
     * Work-stealing parallel batch log probability calculation for heavy computational loads
     * Uses WorkStealingPool for dynamic load balancing across uneven workloads
     * Optimal for large datasets where work distribution may be irregular
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @param pool Reference to WorkStealingPool for load balancing
     * @throws std::invalid_argument if span sizes don't match
     */
    void getLogProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                           WorkStealingPool& pool) const;
    
    /**
     * Cache-aware batch log probability processing using adaptive cache management
     * Integrates with Level 0-3 adaptive cache system for predictive cache warming
     * Automatically determines optimal batch sizes based on cache behavior
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @param cache_manager Reference to adaptive cache manager
     * @throws std::invalid_argument if span sizes don't match
     */
    void getLogProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                         cache::AdaptiveCache<std::string, double>& cache_manager) const;
    
    /**
     * Work-stealing parallel batch CDF calculation for heavy computational loads
     * Uses WorkStealingPool for dynamic load balancing across uneven workloads
     * Optimal for large datasets where work distribution may be irregular
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @param pool Reference to WorkStealingPool for load balancing
     * @throws std::invalid_argument if span sizes don't match
     */
    void getCumulativeProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                                  WorkStealingPool& pool) const;
    
    /**
     * Cache-aware batch CDF processing using adaptive cache management
     * Integrates with Level 0-3 adaptive cache system for predictive cache warming
     * Automatically determines optimal batch sizes based on cache behavior
     * @param values C++20 span of input values for type-safe array access
     * @param results C++20 span of output results (must be same size as values)
     * @param cache_manager Reference to adaptive cache manager
     * @throws std::invalid_argument if span sizes don't match
     */
    void getCumulativeProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                                cache::AdaptiveCache<std::string, double>& cache_manager) const;
    
    //==========================================================================
    // SMART AUTO-DISPATCH BATCH OPERATIONS (C++20 Simplified API)
    //==========================================================================
    
    /**
     * @brief Smart auto-dispatch batch probability calculation with performance hints
     * 
     * Automatically selects the optimal execution strategy (SCALAR, SIMD, PARALLEL, etc.)
     * based on batch size, system capabilities, and user hints. Provides a unified
     * interface that adapts to different hardware and workload characteristics.
     * 
     * @param values Input values as C++20 span for type safety
     * @param results Output results as C++20 span (must be same size as values)
     * @param hint Performance optimization hints (default: AUTO selection)
     * 
     * @throws std::invalid_argument if spans have different sizes
     * 
     * @par Strategy Selection Logic:
     * - Tiny batches (≤8): SCALAR for minimal overhead
     * - Small batches (9-63): SIMD_BATCH for vectorization benefits
     * - Medium batches (64-4095): PARALLEL_SIMD for multi-core + vectorization
     * - Large batches (≥4096): WORK_STEALING or CACHE_AWARE for load balancing
     * 
     * @par Performance Characteristics:
     * - AUTO mode: ~5-10ns overhead per batch for strategy selection
     * - SCALAR: Optimal for tiny batches, ~1-2ns per element
     * - SIMD: 2-4x speedup for compatible operations
     * - PARALLEL: Near-linear scaling with core count
     * 
     * @par Usage Examples:
     * @code
     * std::vector<double> inputs = {0.1, 0.5, 1.0, 2.0};
     * std::vector<double> outputs(inputs.size());
     * 
     * // Auto-dispatch (recommended)
     * dist.getProbability(inputs, outputs);
     * 
     * // Force specific strategy
     * performance::PerformanceHint hint;
     * hint.strategy = performance::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
     * dist.getProbability(inputs, outputs, hint);
     * @endcode
     */
    void getProbability(std::span<const double> values, std::span<double> results,
                       const performance::PerformanceHint& hint = {}) const;
    
    /**
     * @brief Smart auto-dispatch batch log probability calculation with performance hints
     * 
     * Automatically selects the optimal execution strategy for log PDF computation
     * based on batch size, system capabilities, and user performance hints.
     * 
     * @param values Input values as C++20 span for type safety
     * @param results Output log probability results as C++20 span
     * @param hint Performance optimization hints (default: AUTO selection)
     * 
     * @throws std::invalid_argument if spans have different sizes
     * 
     * @note Log probability calculations are typically more computationally intensive
     *       than regular PDF, so the auto-dispatcher may prefer parallel strategies
     *       at smaller batch sizes compared to getProbability().
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                          const performance::PerformanceHint& hint = {}) const;
    
    /**
     * @brief Smart auto-dispatch batch cumulative probability calculation with performance hints
     * 
     * Automatically selects the optimal execution strategy for CDF computation
     * based on batch size, system capabilities, and user performance hints.
     * 
     * @param values Input values as C++20 span for type safety
     * @param results Output cumulative probability results as C++20 span
     * @param hint Performance optimization hints (default: AUTO selection)
     * 
     * @throws std::invalid_argument if spans have different sizes
     * 
     * @note CDF calculations for exponential distribution (involving exp function)
     *       benefit significantly from SIMD vectorization and parallel processing.
     */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                 const performance::PerformanceHint& hint = {}) const;
    
    //==========================================================================
    // COMPARISON OPERATORS
    //==========================================================================
    
    /**
     * Equality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const ExponentialDistribution& other) const;
    
    /**
     * Inequality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const ExponentialDistribution& other) const { return !(*this == other); }
    

private:
    //==========================================================================
    // PRIVATE FACTORY METHODS
    //==========================================================================
    
    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param lambda Rate parameter (assumed valid)
     * @return ExponentialDistribution with the given parameter
     */
    static ExponentialDistribution createUnchecked(double lambda) noexcept {
        ExponentialDistribution dist(lambda, true); // bypass validation
        return dist;
    }
    
    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param lambda Rate parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    ExponentialDistribution(double lambda, bool /*bypassValidation*/) noexcept
        : DistributionBase(), lambda_(lambda) {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        
        // Initialize atomic parameters to invalid state
        atomicLambda_.store(lambda, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    
    //==========================================================================
    // PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================
    
    /** @brief Internal implementation for batch PDF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double lambda, double neg_lambda) const noexcept;
    
    /** @brief Internal implementation for batch log PDF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double log_lambda, double neg_lambda) const noexcept;
    
    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                 double neg_lambda) const noexcept;
    
    // Note: Redundant SIMD methods removed - SIMD optimization is handled
    // internally within the *UnsafeImpl methods above
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const ExponentialDistribution& dist);

} // namespace libstats

#endif // LIBSTATS_EXPONENTIAL_H_

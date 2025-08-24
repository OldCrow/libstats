#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "../common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools, adaptive
// caching, etc.)
#include "../common/distribution_platform_common.h"

// Additional standard headers specific to Gaussian (C++20 showcase)
#include <algorithm>  // C++20 ranges algorithms

namespace stats {

/**
 * @brief Modern C++20 Gaussian (Normal) distribution for modeling continuous data.
 *
 * The Gaussian distribution is a continuous probability distribution that is
 * symmetric around the mean, showing that data near the mean are more frequent
 * in occurrence than data far from the mean.
 *
 * @par Mathematical Definition:
 * PDF: f(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)
 * where μ is the mean and σ is the standard deviation
 *
 * @par Distribution Properties:
 * - Mean: μ
 * - Variance: σ²
 * - Support: x ∈ (-∞, ∞)
 * - Symmetry: Symmetric around the mean
 *
 * @par Performance Features:
 * - Thread-safe concurrent access using std::shared_mutex
 * - Aggressive caching for repeated calculations
 * - SIMD-optimized batch operations (AVX/AVX-512)
 * - Special optimizations for standard normal (μ=0, σ=1)
 *
 * @par Implementation Notes (C++20 Best Practices):
 * - Complex constructors/operators moved to .cpp for better build times
 * - Simple destructor kept inline (= default) for performance
 * - Thread-safe operations use std::shared_mutex for concurrent reads
 * - Deadlock prevention via std::lock() with std::defer_lock
 *
 * @par Thread Safety:
 * All methods are thread-safe. Multiple threads can safely read from the same
 * instance concurrently, while parameter modifications are properly synchronized.
 *
 * @par Usage Examples:
 * @code
 * // Create standard normal distribution
 * GaussianDistribution stdNormal(0.0, 1.0);
 *
 * // Evaluate probability density
 * double pdf = stdNormal.getProbability(1.0);
 *
 * // Fit to data
 * std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
 * GaussianDistribution fitted;
 * fitted.fit(data);
 *
 * // Batch operations for performance
 * std::vector<double> values(1000);
 * std::vector<double> results(1000);
 * fitted.getProbabilityBatch(values.data(), results.data(), 1000);
 * @endcode
 *
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
class GaussianDistribution : public DistributionBase {
   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    /**
     * @brief Constructs a Gaussian distribution with given parameters.
     *
     * @param mean Mean parameter μ (any finite value)
     * @param standardDeviation Standard deviation parameter σ (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     *
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit GaussianDistribution(double mean = detail::ZERO_DOUBLE,
                                  double standardDeviation = detail::ONE);

    /**
     * @brief Thread-safe copy constructor
     *
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    GaussianDistribution(const GaussianDistribution& other);

    /**
     * @brief Copy assignment operator
     *
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    GaussianDistribution& operator=(const GaussianDistribution& other);

    /**
     * @brief Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    GaussianDistribution(GaussianDistribution&& other);

    /**
     * @brief Move assignment operator (C++11 COMPLIANT)
     * Implementation in .cpp: Thread-safe move with atomic operations
     * @note noexcept compliant using atomic state management
     */
    GaussianDistribution& operator=(GaussianDistribution&& other) noexcept;

    /**
     * @brief Destructor - explicitly defaulted to satisfy Rule of Five
     * Implementation inline: Trivial destruction, kept for performance
     *
     * Note: C++20 Best Practice - Rule of Five uses complexity-based placement:
     * - Simple operations (destructor) stay inline for performance
     * - Complex operations (copy/move) moved to .cpp for maintainability
     */
    ~GaussianDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Gaussian distribution without throwing exceptions
     *
     * This factory method provides exception-free construction to work around
     * ABI compatibility issues with Homebrew LLVM libc++ on macOS where
     * exceptions thrown from the library cause segfaults during unwinding.
     *
     * @param mean Mean parameter μ (any finite value)
     * @param standardDeviation Standard deviation parameter σ (must be positive)
     * @return Result containing either a valid GaussianDistribution or error info
     *
     * @par Usage Example:
     * @code
     * auto result = GaussianDistribution::create(0.0, 1.0);
     * if (result.isOk()) {
     *     auto distribution = std::move(result.value);
     *     // Use distribution safely...
     * } else {
     *     std::cout << "Error: " << result.message << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] static Result<GaussianDistribution> create(
        double mean = 0.0, double standardDeviation = 1.0) noexcept {
        auto validation = validateGaussianParameters(mean, standardDeviation);
        if (validation.isError()) {
            return Result<GaussianDistribution>::makeError(validation.error_code,
                                                           validation.message);
        }

        // Use private factory to bypass validation
        return Result<GaussianDistribution>::ok(createUnchecked(mean, standardDeviation));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /**
     * @brief Gets the mean parameter μ.
     * Thread-safe: acquires lock to protect mean_
     *
     * @return Current mean value
     */
    [[nodiscard]] double getMean() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mean_;
    }

    /**
     * Fast lock-free getter for mean parameter using atomic copy.
     * PERFORMANCE: Uses atomic load - no locking overhead
     * WARNING: May return stale value if parameters are being updated
     *
     * @return Atomic copy of mean parameter (may be slightly stale)
     */
    [[nodiscard]] double getMeanAtomic() const noexcept {
        // Fast path: check if atomic parameters are valid
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            // Lock-free atomic access with proper memory ordering
            return atomicMean_.load(std::memory_order_acquire);
        }

        // Fallback: use traditional locked getter if atomic parameters are stale
        return getMean();
    }

    /**
     * @brief Sets the mean parameter μ (exception-based API).
     * Thread-safe: validates first, then locks and sets
     *
     * @param mean New mean parameter (any finite value)
     * @throws std::invalid_argument if mean is not finite
     */
    void setMean(double mean);

    /**
     * @brief Gets the standard deviation parameter σ.
     * Thread-safe: acquires lock to protect standardDeviation_
     *
     * @return Current standard deviation value
     */
    [[nodiscard]] double getStandardDeviation() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return standardDeviation_;
    }

    /**
     * Fast lock-free getter for standard deviation parameter using atomic copy.
     * PERFORMANCE: Uses atomic load - no locking overhead
     * WARNING: May return stale value if parameters are being updated
     *
     * @return Atomic copy of standard deviation parameter (may be slightly stale)
     */
    [[nodiscard]] double getStandardDeviationAtomic() const noexcept {
        // Fast path: check if atomic parameters are valid
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            // Lock-free atomic access with proper memory ordering
            return atomicStandardDeviation_.load(std::memory_order_acquire);
        }

        // Fallback: use traditional locked getter if atomic parameters are stale
        return getStandardDeviation();
    }

    /**
     * @brief Sets the standard deviation parameter σ (exception-based API).
     * Thread-safe: validates first, then locks and sets
     *
     * @param stdDev New standard deviation parameter (must be positive)
     * @throws std::invalid_argument if stdDev <= 0 or is not finite
     */
    void setStandardDeviation(double stdDev);

    /**
     * @brief Sets both parameters simultaneously.
     * Thread-safe: acquires unique lock for cache invalidation
     *
     * @param mean New mean parameter
     * @param stdDev New standard deviation parameter
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(double mean, double stdDev);

    /**
     * @brief Gets the variance of the distribution.
     * For Gaussian distribution, variance = σ²
     * Thread-safe: acquires lock to protect standardDeviation_
     *
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return standardDeviation_ * standardDeviation_;
    }

    /**
     * @brief Gets the skewness of the distribution.
     * For Gaussian distribution, skewness = 0 (symmetric)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Skewness value (always 0)
     */
    [[nodiscard]] double getSkewness() const noexcept override { return detail::ZERO_DOUBLE; }

    /**
     * @brief Gets the kurtosis of the distribution.
     * For Gaussian distribution, excess kurtosis = 0
     * Inline for performance - no thread safety needed for constant
     *
     * @return Excess kurtosis value (always 0)
     */
    [[nodiscard]] double getKurtosis() const noexcept override { return detail::ZERO_DOUBLE; }

    /**
     * @brief Gets the distribution name.
     * Inline for performance - no thread safety needed for constant
     *
     * @return Distribution name
     */
    [[nodiscard]] std::string getDistributionName() const override { return "Gaussian"; }

    /**
     * @brief Gets the number of parameters for this distribution.
     * For Gaussian distribution, there are 2 parameters: mean and standard deviation
     * Inline for performance - no thread safety needed for constant
     *
     * @return Number of parameters (always 2)
     */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /**
     * @brief Checks if the distribution is discrete.
     * For Gaussian distribution, it's continuous
     * Inline for performance - no thread safety needed for constant
     *
     * @return false (always continuous)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /**
     * @brief Gets the lower bound of the distribution support.
     * For Gaussian distribution, support is (-∞, ∞)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Lower bound (-infinity)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return -std::numeric_limits<double>::infinity();
    }

    /**
     * @brief Gets the upper bound of the distribution support.
     * For Gaussian distribution, support is (-∞, ∞)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Upper bound (+infinity)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==============================================================================
    // 4. RESULT-BASED SETTERS
    //==============================================================================

    /**
     * @brief Safely set the mean parameter μ without throwing exceptions (Result-based API).
     * Thread-safe: validates first, then locks and sets
     *
     * @param mean New mean parameter (any finite value)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetMean(double mean) noexcept;

    /**
     * @brief Safely set the standard deviation parameter σ without throwing exceptions
     * (Result-based API). Thread-safe: validates first, then locks and sets
     *
     * @param stdDev New standard deviation parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetStandardDeviation(double stdDev) noexcept;

    /**
     * @brief Safely set both parameters simultaneously without throwing exceptions (Result-based
     * API). Thread-safe: validates first, then locks and sets both parameters atomically
     *
     * @param mean New mean parameter (any finite value)
     * @param standardDeviation New standard deviation parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(double mean, double standardDeviation) noexcept;

    /**
     * @brief Check if current parameters are valid
     * @return VoidResult indicating validity
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return validateGaussianParameters(mean_, standardDeviation_);
    }

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief Computes the probability density function for the Gaussian distribution.
     * Formula: PDF(x) = (1/(σ√(2π))) * exp(-½((x-μ)/σ)²)
     *
     * @param x The value at which to evaluate the PDF
     * @return Probability density
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Evaluates the logarithm of the probability density function
     * Formula: log PDF(x) = -½log(2π) - log(σ) - ½((x-μ)/σ)²
     * More numerically stable for small probabilities
     *
     * @param x The value at which to evaluate the log PDF
     * @return Log probability density
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief Evaluates the CDF at x using the error function
     * Formula: CDF(x) = (1/2) * (1 + erf((x-μ)/(σ√2)))
     *
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Computes the quantile function (inverse CDF)
     * Uses inverse error function for accurate quantile calculation
     *
     * @param p Probability value in [0,1]
     * @return x such that P(X ≤ x) = p
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate single random sample from distribution
     * Uses Box-Muller transform for high-quality Gaussian samples
     *
     * @param rng Random number generator
     * @return Single random sample
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /**
     * @brief Generate multiple random samples from distribution
     * Optimized batch sampling using Box-Muller transform
     *
     * @param rng Random number generator
     * @param n Number of samples to generate
     * @return Vector of random samples
     */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fits the distribution parameters to the given data using maximum likelihood
     * estimation. For Gaussian distribution, MLE gives sample mean and sample standard deviation.
     * Uses parallel execution for large datasets when available.
     *
     * @param values Vector of observed data
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple datasets
     * Efficiently fits Gaussian parameters to multiple independent datasets in parallel
     *
     * @param datasets Vector of datasets, each representing independent observations
     * @param results Vector to store fitted GaussianDistribution objects
     * @throws std::invalid_argument if datasets is empty or results size doesn't match
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<GaussianDistribution>& results);

    /**
     * @brief Resets the distribution to default parameters (μ = 0.0, σ = 1.0).
     * This corresponds to the standard normal distribution.
     */
    void reset() noexcept override;

    /**
     * @brief Returns a string representation of the distribution.
     *
     * @return String describing the distribution parameters
     */
    std::string toString() const override;

    //==========================================================================
    // 7. ADVANCED STATISTICAL METHODS
    //==========================================================================

    /**
     * @brief Confidence interval for mean parameter μ
     *
     * Calculates confidence interval for the population mean using the
     * t-distribution when population variance is unknown, or normal
     * distribution when population variance is known.
     *
     * @param data Sample data
     * @param confidence_level Confidence level (e.g., 0.95 for 95%)
     * @param population_variance_known If true, uses known population variance
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<double, double> confidenceIntervalMean(const std::vector<double>& data,
                                                            double confidence_level = 0.95,
                                                            bool population_variance_known = false);

    /**
     * @brief Confidence interval for variance parameter σ²
     *
     * Calculates confidence interval for population variance using
     * chi-squared distribution.
     *
     * @param data Sample data
     * @param confidence_level Confidence level (e.g., 0.95 for 95%)
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<double, double> confidenceIntervalVariance(const std::vector<double>& data,
                                                                double confidence_level = 0.95);

    /**
     * @brief One-sample t-test for population mean
     *
     * Tests the null hypothesis that the population mean equals
     * the specified value against a two-tailed alternative.
     *
     * @param data Sample data
     * @param hypothesized_mean Null hypothesis mean value
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (t_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> oneSampleTTest(const std::vector<double>& data,
                                                           double hypothesized_mean,
                                                           double alpha = 0.05);

    /**
     * @brief Two-sample t-test for equal means
     *
     * Tests whether two independent samples have equal population means.
     * Assumes equal variances by default, with Welch's test option.
     *
     * @param data1 First sample
     * @param data2 Second sample
     * @param equal_variances If true, assumes equal population variances
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (t_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> twoSampleTTest(const std::vector<double>& data1,
                                                           const std::vector<double>& data2,
                                                           bool equal_variances = true,
                                                           double alpha = 0.05);

    /**
     * @brief Paired t-test for matched samples
     *
     * Tests whether the mean difference between paired observations
     * is significantly different from zero.
     *
     * @param data1 First sample (paired with data2)
     * @param data2 Second sample (paired with data1)
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (t_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> pairedTTest(const std::vector<double>& data1,
                                                        const std::vector<double>& data2,
                                                        double alpha = 0.05);

    /**
     * @brief Bayesian parameter estimation with conjugate prior
     *
     * Performs Bayesian estimation of Gaussian parameters using
     * normal-inverse-gamma conjugate prior. Returns posterior
     * distribution parameters.
     *
     * @param data Observed data
     * @param prior_mean Prior mean for μ (default: 0)
     * @param prior_precision Prior precision for μ (default: 0.001)
     * @param prior_shape Prior shape parameter for precision (default: 1)
     * @param prior_rate Prior rate parameter for precision (default: 1)
     * @return Tuple of (posterior_mean, posterior_precision, posterior_shape, posterior_rate)
     */
    static std::tuple<double, double, double, double> bayesianEstimation(
        const std::vector<double>& data, double prior_mean = 0.0, double prior_precision = 0.001,
        double prior_shape = 1.0, double prior_rate = 1.0);

    /**
     * @brief Credible interval from Bayesian posterior
     *
     * Calculates Bayesian credible interval for mean parameter
     * from posterior distribution.
     *
     * @param data Observed data
     * @param credibility_level Credibility level (e.g., 0.95 for 95%)
     * @param prior_mean Prior mean for μ (default: 0)
     * @param prior_precision Prior precision for μ (default: 0.001)
     * @param prior_shape Prior shape parameter for precision (default: 1)
     * @param prior_rate Prior rate parameter for precision (default: 1)
     * @return Pair of (lower_bound, upper_bound)
     */
    static std::pair<double, double> bayesianCredibleInterval(
        const std::vector<double>& data, double credibility_level = 0.95, double prior_mean = 0.0,
        double prior_precision = 0.001, double prior_shape = 1.0, double prior_rate = 1.0);

    /**
     * @brief Robust parameter estimation using M-estimators
     *
     * Provides robust estimation of location and scale parameters
     * that are less sensitive to outliers than maximum likelihood.
     *
     * @param data Sample data
     * @param estimator_type Type of robust estimator ("huber", "tukey", "hampel")
     * @param tuning_constant Tuning constant for the estimator
     * @return Pair of (robust_mean, robust_scale)
     */
    static std::pair<double, double> robustEstimation(const std::vector<double>& data,
                                                      const std::string& estimator_type = "huber",
                                                      double tuning_constant = 1.345);

    /**
     * @brief Method of moments parameter estimation
     *
     * Estimates parameters by matching sample moments with
     * theoretical distribution moments.
     *
     * @param data Sample data
     * @return Pair of (mean_estimate, stddev_estimate)
     */
    static std::pair<double, double> methodOfMomentsEstimation(const std::vector<double>& data);

    /**
     * @brief L-moments parameter estimation
     *
     * Uses L-moments (linear combinations of order statistics)
     * for robust parameter estimation, particularly useful
     * for extreme value analysis.
     *
     * @param data Sample data
     * @return Pair of (location_parameter, scale_parameter)
     */
    static std::pair<double, double> lMomentsEstimation(const std::vector<double>& data);

    /**
     * @brief Advanced moment calculations (up to 6th moment)
     *
     * Calculates higher-order sample moments for detailed
     * distributional analysis.
     *
     * @param data Sample data
     * @param center_on_mean If true, calculates central moments
     * @return Vector of moments [1st, 2nd, 3rd, 4th, 5th, 6th]
     */
    static std::vector<double> calculateHigherMoments(const std::vector<double>& data,
                                                      bool center_on_mean = true);

    /**
     * @brief Jarque-Bera normality test
     *
     * Tests the null hypothesis that data follows a normal distribution
     * using skewness and kurtosis statistics.
     *
     * @param data Sample data
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (JB_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> jarqueBeraTest(const std::vector<double>& data,
                                                           double alpha = 0.05);

    /**
     * @brief Shapiro-Wilk normality test
     *
     * Tests the null hypothesis that data follows a normal distribution.
     * Generally more powerful than Jarque-Bera for small samples.
     *
     * @param data Sample data (works best for n <= 5000)
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (W_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> shapiroWilkTest(const std::vector<double>& data,
                                                            double alpha = 0.05);

    /**
     * @brief Likelihood ratio test for nested models
     *
     * Tests whether the simpler model (restricted) is adequate
     * compared to the more complex model (unrestricted).
     *
     * @param data Sample data
     * @param restricted_model Simpler model (e.g., fixed mean)
     * @param unrestricted_model More complex model
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (LR_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> likelihoodRatioTest(
        const std::vector<double>& data, const GaussianDistribution& restricted_model,
        const GaussianDistribution& unrestricted_model, double alpha = 0.05);

    //==========================================================================
    // 8. GOODNESS-OF-FIT TESTS
    //==========================================================================

    /**
     * @brief Kolmogorov-Smirnov goodness-of-fit test
     *
     * Tests the null hypothesis that data follows the specified Gaussian distribution.
     * More general than JB/SW tests as it doesn't assume any particular alternative.
     *
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (KS_statistic, p_value, reject_null)
     * @note p_value approximation using asymptotic distribution
     */
    static std::tuple<double, double, bool> kolmogorovSmirnovTest(
        const std::vector<double>& data, const GaussianDistribution& distribution,
        double alpha = 0.05);

    /**
     * @brief Anderson-Darling goodness-of-fit test
     *
     * Tests the null hypothesis that data follows the specified Gaussian distribution.
     * More sensitive to deviations in the tails than KS test.
     *
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (AD_statistic, p_value, reject_null)
     * @note Uses asymptotic p-value approximation for Gaussian case
     */
    static std::tuple<double, double, bool> andersonDarlingTest(
        const std::vector<double>& data, const GaussianDistribution& distribution,
        double alpha = 0.05);

    //==========================================================================
    // 9. CROSS-VALIDATION METHODS
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
     * @return Vector of k validation results: (mean_error, std_error, log_likelihood)
     */
    static std::vector<std::tuple<double, double, double>> kFoldCrossValidation(
        const std::vector<double>& data, int k = 5, unsigned int random_seed = 42);

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

    //==========================================================================
    // 10. INFORMATION CRITERIA
    //==========================================================================

    /**
     * @brief Model comparison using information criteria
     *
     * Computes various information criteria (AIC, BIC, AICc) for model selection.
     * Lower values indicate better model fit while penalizing complexity.
     *
     * @param data Sample data used for fitting
     * @param fitted_distribution The fitted Gaussian distribution
     * @return Tuple of (AIC, BIC, AICc, log_likelihood)
     */
    static std::tuple<double, double, double, double> computeInformationCriteria(
        const std::vector<double>& data, const GaussianDistribution& fitted_distribution);

    //==========================================================================
    // 11. BOOTSTRAP METHODS
    //==========================================================================

    /**
     * @brief Bootstrap parameter confidence intervals
     *
     * Uses bootstrap resampling to estimate confidence intervals for
     * the distribution parameters (mean and standard deviation).
     *
     * @param data Sample data for bootstrap resampling
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @param n_bootstrap Number of bootstrap samples (default: 1000)
     * @param random_seed Seed for random sampling (default: 42)
     * @return Tuple of ((mean_CI_lower, mean_CI_upper), (std_CI_lower, std_CI_upper))
     */
    static std::tuple<std::pair<double, double>, std::pair<double, double>>
    bootstrapParameterConfidenceIntervals(const std::vector<double>& data,
                                          double confidence_level = 0.95, int n_bootstrap = 1000,
                                          unsigned int random_seed = 42);

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Compute the standardized value (z-score) for a given x
     *
     * Z-score transforms a value to the number of standard deviations from the mean:
     * z = (x - μ) / σ
     *
     * @param x Value to standardize
     * @return Standardized value (z-score)
     */
    [[nodiscard]] double getStandardizedValue(double x) const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (x - mean_) * invStandardDeviation_;  // Use cached 1/σ for efficiency
    }

    /**
     * @brief Convert a standardized value (z-score) back to the original scale
     *
     * Transforms a z-score back to the original distribution scale:
     * x = μ + σ * z
     *
     * @param z Standardized value (z-score)
     * @return Value in original scale
     */
    [[nodiscard]] double getValueFromStandardized(double z) const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mean_ + standardDeviation_ * z;
    }

    /**
     * @brief Check if this is a standard normal distribution
     *
     * Tests whether μ = 0 and σ = 1 within numerical tolerance.
     * Standard normal distribution is N(0,1).
     *
     * @return true if μ ≈ 0 and σ ≈ 1, false otherwise
     */
    [[nodiscard]] bool isStandardNormal() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (std::abs(mean_) <= detail::DEFAULT_TOLERANCE) &&
               (std::abs(standardDeviation_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
    }

    /**
     * @brief Compute the entropy of the distribution
     *
     * For Gaussian distribution: H(X) = 0.5 * ln(2πeσ²) = 0.5 * (ln(2π) + 1 + 2*ln(σ))
     * Entropy measures the average information content.
     * Uses efficient computation with precomputed constants.
     *
     * @return Entropy value
     */
    [[nodiscard]] double getEntropy() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        // H(X) = 0.5 * (ln(2π) + 1 + 2*ln(σ))
        return detail::HALF_LN_2PI + detail::HALF + std::log(standardDeviation_);
    }

    /**
     * @brief Get the median of the distribution
     *
     * For Gaussian distribution, the median equals the mean.
     * This is a property of symmetric distributions.
     *
     * @return Median value (equals μ)
     */
    [[nodiscard]] double getMedian() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mean_;  // For Gaussian, median = mean
    }

    /**
     * @brief Get the mode of the distribution
     *
     * For Gaussian distribution, the mode equals the mean.
     * This is where the PDF achieves its maximum value.
     *
     * @return Mode value (equals μ)
     */
    [[nodiscard]] double getMode() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mean_;  // For Gaussian, mode = mean
    }

#ifdef DEBUG
    /**
     * Debug method to check if standard normal optimization is active
     * Only available in debug builds
     * @return true if this distribution is detected as standard normal
     */
    bool isUsingStandardNormalOptimization() const;
#endif

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================

    /**
     * @brief Smart auto-dispatch batch probability calculation
     *
     * This method automatically selects the optimal execution strategy (SCALAR, SIMD, PARALLEL,
     * etc.) based on:
     * - Batch size and system capabilities
     * - Available CPU features (SIMD support)
     * - Threading overhead characteristics
     *
     * Users should prefer this method over manual strategy selection.
     *
     * @param values Input values to evaluate
     * @param results Output array for probability densities
     * @param hint Optional performance hints for advanced users
     */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Smart auto-dispatch batch log probability calculation
     *
     * Automatically selects optimal execution strategy for log probability computation.
     *
     * @param values Input values to evaluate
     * @param results Output array for log probability densities
     * @param hint Optional performance hints for advanced users
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Smart auto-dispatch batch cumulative probability calculation
     *
     * Automatically selects optimal execution strategy for CDF computation.
     *
     * @param values Input values to evaluate
     * @param results Output array for cumulative probabilities
     * @param hint Optional performance hints for advanced users
     */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 14. EXPLICIT STRATEGY BATCH OPERATIONS
    //==========================================================================

    /**
     * @brief Explicit strategy batch probability calculation for power users
     *
     * Allows explicit selection of execution strategy, bypassing auto-dispatch.
     * Use when you have specific performance requirements or want deterministic execution.
     *
     * @param values Input values to evaluate
     * @param results Output array for probability densities
     * @param strategy Explicit execution strategy to use
     * @throws std::invalid_argument if strategy is not supported
     */
    void getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                    detail::Strategy strategy) const;

    /**
     * @brief Explicit strategy batch log probability calculation for power users
     *
     * Allows explicit selection of execution strategy, bypassing auto-dispatch.
     * Use when you have specific performance requirements or want deterministic execution.
     *
     * @param values Input values to evaluate
     * @param results Output array for log probability densities
     * @param strategy Explicit execution strategy to use
     * @throws std::invalid_argument if strategy is not supported
     */
    void getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                       detail::Strategy strategy) const;

    /**
     * @brief Explicit strategy batch cumulative probability calculation for power users
     *
     * Allows explicit selection of execution strategy, bypassing auto-dispatch.
     * Use when you have specific performance requirements or want deterministic execution.
     *
     * @param values Input values to evaluate
     * @param results Output array for cumulative probabilities
     * @param strategy Explicit execution strategy to use
     * @throws std::invalid_argument if strategy is not supported
     */
    void getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                              std::span<double> results,
                                              detail::Strategy strategy) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    /**
     * @brief Equality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const GaussianDistribution& other) const;

    /**
     * @brief Inequality comparison operator
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const GaussianDistribution& other) const { return !(*this == other); }

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    /**
     * @brief Stream input operator
     * @param is Input stream
     * @param dist Distribution to input
     * @return Reference to the input stream
     */
    friend std::istream& operator>>(std::istream&, stats::GaussianDistribution& dist);

    /**
     * @brief Stream output operator
     * @param os Output stream
     * @param dist Distribution to output
     * @return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream&, const stats::GaussianDistribution& dist);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param mean Mean parameter (assumed valid)
     * @param standardDeviation Standard deviation parameter (assumed valid)
     * @return GaussianDistribution with the given parameters
     */
    static GaussianDistribution createUnchecked(double mean, double standardDeviation) noexcept {
        GaussianDistribution dist(mean, standardDeviation, true);  // bypass validation
        return dist;
    }

    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param mean Mean parameter (assumed valid)
     * @param standardDeviation Standard deviation parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    GaussianDistribution(double mean, double standardDeviation, bool /*bypassValidation*/) noexcept
        : DistributionBase(), mean_(mean), standardDeviation_(standardDeviation) {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief Internal implementation for batch PDF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double mean, double norm_constant, double neg_half_inv_var,
                                       bool is_standard_normal) const noexcept;

    /** @brief Internal implementation for batch log PDF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double mean, double log_std, double neg_half_inv_var,
                                          bool is_standard_normal) const noexcept;

    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double mean, double sigma_sqrt2,
                                                 bool is_standard_normal) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     * Marked inline for performance optimization
     */
    inline void updateCacheUnsafe() const noexcept override;

    /**
     * Validates parameters for the Gaussian distribution
     * @param mean Mean parameter (any finite value)
     * @param stdDev Standard deviation parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double mean, double stdDev);

    //==========================================================================
    // 20. PRIVATE UTILITY METHODS
    //==========================================================================

    // Note: Currently no private utility methods needed for Gaussian distribution
    // This section maintained for template compliance

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Mean parameter μ - can be any finite real number */
    double mean_{detail::ZERO_DOUBLE};

    /** @brief Standard deviation parameter σ - must be positive */
    double standardDeviation_{detail::ONE};

    /** @brief C++20 atomic copies of parameters for lock-free access */
    mutable std::atomic<double> atomicMean_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicStandardDeviation_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief Cached normalization constant 1/(σ√(2π)) for PDF calculations */
    mutable double normalizationConstant_{detail::ZERO_DOUBLE};

    /** @brief Cached value -1/(2σ²) for efficient exponent calculation */
    mutable double negHalfSigmaSquaredInv_{detail::ZERO_DOUBLE};

    /** @brief Cached log(σ) for log-probability calculations */
    mutable double logStandardDeviation_{detail::ZERO_DOUBLE};

    /** @brief Cached σ√2 for CDF calculations using error function */
    mutable double sigmaSqrt2_{detail::ZERO_DOUBLE};

    /** @brief Cached 1/σ for efficient reciprocal operations */
    mutable double invStandardDeviation_{detail::ZERO_DOUBLE};

    /** @brief Cached σ² to avoid repeated multiplication */
    mutable double cachedSigmaSquared_{detail::ONE};

    /** @brief Cached 2σ² for CDF optimizations */
    mutable double cachedTwoSigmaSquared_{detail::TWO};

    /** @brief Cached log(2σ²) for log-space operations */
    mutable double cachedLogTwoSigmaSquared_{detail::ZERO_DOUBLE};

    /** @brief Cached 1/σ² for direct exponent calculation */
    mutable double cachedInvSigmaSquared_{detail::ONE};

    /** @brief Cached √(2π) constant for direct normalization */
    mutable double cachedSqrtTwoPi_{detail::SQRT_2PI};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if this is standard normal (μ=0, σ=1) for ultra-fast path */
    mutable bool isStandardNormal_{false};

    /** @brief True if σ² = 1 for additional fast path optimizations */
    mutable bool isUnitVariance_{true};

    /** @brief True if μ = 0 for additional fast path optimizations */
    mutable bool isZeroMean_{true};

    /** @brief True if parameters require high precision arithmetic */
    mutable bool isHighPrecision_{false};

    /** @brief True if σ² < 0.0625 for numerical stability path */
    mutable bool isLowVariance_{false};

    /** @brief Atomic cache validity flag for lock-free fast path */
    mutable std::atomic<bool> cacheValidAtomic_{false};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Gaussian distribution uses standard caching only
    // This section maintained for template compliance
};

}  // namespace stats

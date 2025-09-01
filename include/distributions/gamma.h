#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "../common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools, adaptive
// caching, etc.)
#include "../common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Gamma Distribution for modeling continuous positive-valued data.
 *
 * @details The Gamma distribution is a continuous probability distribution that generalizes
 * the exponential distribution and is widely used in statistics, engineering, and science.
 * It's the conjugate prior for the precision parameter of the normal distribution and
 * for the rate parameter of the Poisson distribution. The distribution is particularly
 * useful for modeling waiting times, lifetimes, and other positive continuous phenomena.
 *
 * @par Mathematical Definition:
 * - PDF: f(x; α, β) = (β^α / Γ(α)) * x^(α-1) * e^(-βx) for x ≥ 0, 0 otherwise
 * - CDF: F(x; α, β) = γ(α, βx) / Γ(α) (regularized incomplete gamma function)
 * - Parameters: α > 0 (shape), β > 0 (rate), or θ = 1/β (scale)
 * - Support: x ∈ [0, ∞)
 * - Mean: α/β = αθ
 * - Variance: α/β² = αθ²
 * - Mode: (α-1)/β = (α-1)θ for α ≥ 1, 0 for α < 1
 * - Median: No closed form, computed numerically
 *
 * @par Parameterization:
 * This implementation uses the **shape-rate parameterization** (α, β) by default:
 * - α = shape parameter (dimensionless)
 * - β = rate parameter (1/time or 1/scale)
 * - Alternative: shape-scale (α, θ) where θ = 1/β
 *
 * @par Key Properties:
 * - **Exponential Family**: Member of the exponential family of distributions
 * - **Conjugate Prior**: For Poisson rate and normal precision parameters
 * - **Reproductive Property**: Sum of independent Gamma(αᵢ, β) is Gamma(Σαᵢ, β)
 * - **Relationship to Chi-squared**: χ²(ν) = Gamma(ν/2, 1/2)
 * - **Limiting Cases**: Gamma(1, β) = Exponential(β), Gamma(α, β) → Normal as α → ∞
 * - **Scale Family**: Gamma(α, β) = (1/β) * Gamma(α, 1)
 *
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - Concurrent reads are optimized with std::shared_mutex
 * - Cache invalidation uses atomic operations for lock-free fast paths
 * - Deadlock prevention via ordered lock acquisition with std::lock()
 *
 * @par Performance Features:
 * - Atomic cache validity flags for lock-free fast path access
 * - Extensive caching of computed values (log(Γ(α)), log(β), digamma(α), etc.)
 * - Optimized PDF/CDF computation with precomputed gamma functions
 * - Fast parameter validation with IEEE 754 compliance
 * - Special algorithms for integer α, α < 1, and large α cases
 *
 * @par Usage Examples:
 * @code
 * // Reliability analysis: equipment lifetime (shape=2, rate=0.5)
 * auto result = GammaDistribution::create(2.0, 0.5);
 * if (result.isOk()) {
 *     auto lifetime = std::move(result.value);
 *
 *     // Bayesian analysis: prior for Poisson rate (shape=1, rate=1)
 *     auto priorResult = GammaDistribution::create(1.0, 1.0);
 *     if (priorResult.isOk()) {
 *         auto prior = std::move(priorResult.value);
 *
 *         // Fit to observed positive data
 *         std::vector<double> lifetimes = {1.2, 2.1, 0.8, 3.4, 1.9, 2.7};
 *         lifetime.fit(lifetimes);
 *
 *         // Probability of failure before time t=2
 *         double failureProb = lifetime.getCumulativeProbability(2.0);
 *
 *         // 95th percentile for maintenance scheduling
 *         double percentile95 = lifetime.getQuantile(0.95);
 *
 *         // Generate random failure time
 *         std::mt19937 rng(42);
 *         double failureTime = lifetime.sample(rng);
 *     }
 * }
 * @endcode
 *
 * @par Applications:
 * - **Reliability Engineering**: Component lifetimes, failure analysis
 * - **Bayesian Statistics**: Conjugate priors for rates and precisions
 * - **Queueing Theory**: Service time distributions
 * - **Meteorology**: Rainfall amounts, wind speeds
 * - **Finance**: Loss distributions, operational risk
 * - **Biology**: Gene expression levels, enzyme kinetics
 * - **Epidemiology**: Disease duration, recovery times
 * - **Quality Control**: Process variation, defect rates
 *
 * @par Statistical Properties:
 * - Skewness: 2/√α (right-skewed, approaches 0 as α increases)
 * - Kurtosis: 6/α (excess kurtosis, approaches 0 as α increases)
 * - Entropy: α - log(β) + log(Γ(α)) + (1-α)ψ(α) where ψ is digamma function
 * - Moment generating function: (1 - t/β)^(-α) for t < β
 * - Characteristic function: (1 - it/β)^(-α)
 *
 * @par Computational Algorithms:
 * - **PDF**: Direct computation with cached log(Γ(α)) and log(β)
 * - **CDF**: Regularized incomplete gamma function using continued fractions
 * - **Quantile**: Newton-Raphson iteration with bracketing for robustness
 * - **Sampling**: Marsaglia-Tsang squeeze method for α ≥ 1, Ahrens-Dieter for α < 1
 * - **Special Cases**: Optimized algorithms for integer α and exponential (α=1)
 *
 * @par Numerical Considerations:
 * - Robust handling of extreme parameter values using log-space computation
 * - Accurate gamma function computation using Lanczos approximation
 * - Efficient incomplete gamma function using continued fractions
 * - Special handling for α near 0, α = 1, and very large α
 * - IEEE 754 compliant boundary handling for numerical stability
 *
 * @par Implementation Details (C++20 Best Practices):
 * - Complex constructors/operators moved to .cpp for faster compilation
 * - Exception-safe design with RAII principles
 * - Optimized parameter validation with comprehensive error messages
 * - Lock-free fast paths using atomic operations
 * - Specialized algorithms for different parameter regimes
 *
 * @author libstats Development Team
 * @version 2.9.1
 * @since 1.0.0
 */
class GammaDistribution : public DistributionBase {
   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructs a Gamma distribution with given shape and rate parameters.
     *
     * @param alpha Shape parameter α (must be positive, default: 1.0)
     * @param beta Rate parameter β (must be positive, default: 1.0)
     * @throws std::invalid_argument if parameters are invalid
     *
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit GammaDistribution(double alpha = detail::ONE, double beta = detail::ONE);

    /**
     * @brief Thread-safe copy constructor
     *
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    GammaDistribution(const GammaDistribution& other);

    /**
     * @brief Copy assignment operator
     *
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    GammaDistribution& operator=(const GammaDistribution& other);

    /**
     * @brief Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    GammaDistribution(GammaDistribution&& other);

    /**
     * @brief Move assignment operator (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with deadlock prevention
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    GammaDistribution& operator=(GammaDistribution&& other);

    /**
     * @brief Destructor - explicitly defaulted to satisfy Rule of Five
     * Implementation inline: Trivial destruction, kept for performance
     *
     * Note: C++20 Best Practice - Rule of Five uses complexity-based placement:
     * - Simple operations (destructor) stay inline for performance
     * - Complex operations (copy/move) moved to .cpp for maintainability
     */
    ~GammaDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Gamma distribution without throwing exceptions
     *
     * This factory method provides exception-free construction to work around
     * ABI compatibility issues with Homebrew LLVM libc++ on macOS where
     * exceptions thrown from the library cause segfaults during unwinding.
     *
     * @param alpha Shape parameter α (must be positive)
     * @param beta Rate parameter β (must be positive)
     * @return Result containing either a valid GammaDistribution or error info
     *
     * @par Usage Example:
     * @code
     * auto result = GammaDistribution::create(2.0, 0.5);
     * if (result.isOk()) {
     *     auto distribution = std::move(result.value);
     *     // Use distribution safely...
     * } else {
     *     std::cout << "Error: " << result.message << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] static Result<GammaDistribution> create(double alpha = 1.0,
                                                          double beta = 1.0) noexcept {
        auto validation = validateGammaParameters(alpha, beta);
        if (validation.isError()) {
            return Result<GammaDistribution>::makeError(validation.error_code, validation.message);
        }

        // Use private factory to bypass validation
        return Result<GammaDistribution>::ok(createUnchecked(alpha, beta));
    }

    /**
     * @brief Safely create a Gamma distribution using shape-scale parameterization
     *
     * @param alpha Shape parameter α (must be positive)
     * @param scale Scale parameter θ = 1/β (must be positive)
     * @return Result containing either a valid GammaDistribution or error info
     */
    [[nodiscard]] static Result<GammaDistribution> createWithScale(double alpha,
                                                                   double scale) noexcept {
        if (scale <= 0.0) {
            return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter,
                                                        "Scale parameter must be positive");
        }
        return create(alpha, 1.0 / scale);
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /**
     * Gets the shape parameter α.
     * Thread-safe: acquires shared lock to protect alpha_
     *
     * @return Current shape parameter value
     */
    [[nodiscard]] double getAlpha() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return alpha_;
    }

    /**
     * @brief Fast lock-free atomic getter for shape parameter α
     *
     * Provides high-performance access to the shape parameter using atomic operations
     * for lock-free fast path. Falls back to locked getter if atomic parameters
     * are not valid (e.g., during parameter updates).
     *
     * @return Current shape parameter value
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
     *     double alpha = dist.getAlphaAtomic();  // Lock-free access
     *     results[i] = compute_something(data[i], alpha);
     * }
     * @endcode
     */
    [[nodiscard]] double getAlphaAtomic() const noexcept;

    /**
     * @brief Set the shape parameter α
     *
     * @param alpha New shape parameter (must be positive)
     * @throws std::invalid_argument if alpha <= 0
     */
    void setAlpha(double alpha);

    /**
     * Gets the rate parameter β.
     * Thread-safe: acquires shared lock to protect beta_
     *
     * @return Current rate parameter value
     */
    [[nodiscard]] double getBeta() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return beta_;
    }

    /**
     * @brief Fast lock-free atomic getter for rate parameter β
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
     *     double beta = dist.getBetaAtomic();  // Lock-free access
     *     results[i] = compute_something(data[i], beta);
     * }
     * @endcode
     */
    [[nodiscard]] double getBetaAtomic() const noexcept;

    /**
     * @brief Set the rate parameter β
     *
     * @param beta New rate parameter (must be positive)
     * @throws std::invalid_argument if beta <= 0
     */
    void setBeta(double beta);

    /**
     * @brief Set both parameters simultaneously
     *
     * @param alpha New shape parameter (must be positive)
     * @param beta New rate parameter (must be positive)
     * @throws std::invalid_argument if either parameter <= 0
     */
    void setParameters(double alpha, double beta);

    /**
     * Gets the scale parameter θ = 1/β.
     * Uses cached value to eliminate division.
     *
     * @return Scale parameter value
     */
    [[nodiscard]] double getScale() const noexcept;

    /**
     * Gets the mean of the distribution.
     * For Gamma distribution, mean = α/β = αθ
     * Uses cached value to eliminate division.
     *
     * @return Mean value
     */
    [[nodiscard]] double getMean() const noexcept override;

    /**
     * Gets the variance of the distribution.
     * For Gamma distribution, variance = α/β² = αθ²
     * Uses cached value to eliminate divisions and multiplications.
     *
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const noexcept override;

    /**
     * @brief Gets the skewness of the distribution.
     * For Gamma distribution, skewness = 2/√α
     * Uses cached value to eliminate square root computation.
     *
     * @return Skewness value (2/√α)
     */
    [[nodiscard]] double getSkewness() const noexcept override;

    /**
     * @brief Gets the kurtosis of the distribution.
     * For Gamma distribution, excess kurtosis = 6/α
     * Uses direct computation for efficiency.
     *
     * @return Excess kurtosis value (6/α)
     */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /**
     * @brief Gets the number of parameters for this distribution.
     * For Gamma distribution, there are 2 parameters: alpha (shape) and beta (rate)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Number of parameters (always 2)
     */
    [[nodiscard]] int getNumParameters() const noexcept override;

    /**
     * @brief Gets the distribution name.
     * Inline for performance - no thread safety needed for constant
     *
     * @return Distribution name
     */
    [[nodiscard]] std::string getDistributionName() const override;

    /**
     * @brief Checks if the distribution is discrete.
     * For Gamma distribution, it's continuous
     * Inline for performance - no thread safety needed for constant
     *
     * @return false (always continuous)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override;

    /**
     * @brief Gets the lower bound of the distribution support.
     * For Gamma distribution, support is [0, ∞)
     *
     * @return Lower bound (0)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override;

    /**
     * @brief Gets the upper bound of the distribution support.
     * For Gamma distribution, support is [0, ∞)
     *
     * @return Upper bound (+infinity)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override;

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    /**
     * @brief Safely set the shape parameter α without throwing exceptions (Result-based API).
     *
     * @param alpha New shape parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetAlpha(double alpha) noexcept;

    /**
     * @brief Safely set the rate parameter β without throwing exceptions (Result-based API).
     *
     * @param beta New rate parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetBeta(double beta) noexcept;

    /**
     * @brief Safely try to set all parameters without throwing exceptions
     *
     * @param alpha New shape parameter
     * @param beta New rate parameter
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(double alpha, double beta) noexcept;

    /**
     * @brief Check if current parameters are valid
     * @return VoidResult indicating validity
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * Computes the probability density function for the Gamma distribution.
     *
     * For Gamma distribution: f(x) = (β^α / Γ(α)) * x^(α-1) * e^(-βx) for x ≥ 0
     * Uses log-space computation for numerical stability.
     *
     * @param x The value at which to evaluate the PDF
     * @return Probability density for the given value, 0 for x < 0
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * Computes the logarithm of the probability density function for numerical stability.
     *
     * For Gamma distribution: log(f(x)) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - βx for x > 0
     *
     * @param x The value at which to evaluate the log-PDF
     * @return Natural logarithm of the probability density, or -∞ for x ≤ 0
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * Evaluates the CDF at x using the regularized incomplete gamma function.
     *
     * For Gamma distribution: F(x) = γ(α, βx) / Γ(α) where γ is the lower incomplete gamma function
     *
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Computes the quantile function (inverse CDF)
     *
     * For Gamma distribution: F^(-1)(p) computed using Newton-Raphson iteration
     *
     * @param p Probability value in [0,1]
     * @return x such that P(X ≤ x) = p
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate single random sample from distribution
     *
     * Uses Marsaglia-Tsang squeeze method for α ≥ 1, Ahrens-Dieter for α < 1
     *
     * @param rng Random number generator
     * @return Single random sample
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /**
     * @brief Generate multiple random samples from distribution
     * Optimized batch sampling using appropriate algorithm for λ size
     *
     * @param rng Random number generator
     * @param n Number of samples to generate
     * @return Vector of random samples (integer values as doubles)
     */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Gamma distribution, uses method of moments as initial guess, then Newton-Raphson.
     *
     * @param values Vector of observed positive data
     * @throws std::invalid_argument if values is empty or contains non-positive values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple datasets
     * Efficiently fits gamma distribution parameters to multiple independent datasets in parallel
     *
     * @param datasets Vector of datasets, each representing independent observations
     * @param results Vector to store fitted GammaDistribution objects
     * @throws std::invalid_argument if datasets is empty or results size doesn't match
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<GammaDistribution>& results);

    /**
     * Resets the distribution to default parameters (α = 1.0, β = 1.0).
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
    // 7. ADVANCED STATISTICAL METHODS
    //==========================================================================

    /**
     * @brief Confidence interval for shape parameter α
     *
     * Computes confidence interval for the shape parameter using profile likelihood method.
     * Uses iterative root-finding to determine bounds where log-likelihood drops by χ²(1)/2.
     *
     * @param data Vector of observed positive data
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @return Pair of (lower_bound, upper_bound) for α
     * @throws std::invalid_argument if confidence_level not in (0,1) or data empty/invalid
     */
    [[nodiscard]] static std::pair<double, double> confidenceIntervalShape(
        const std::vector<double>& data, double confidence_level = 0.95);

    /**
     * @brief Confidence interval for rate parameter β
     *
     * Computes confidence interval for the rate parameter using profile likelihood method.
     * Uses iterative root-finding to determine bounds where log-likelihood drops by χ²(1)/2.
     *
     * @param data Vector of observed positive data
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @return Pair of (lower_bound, upper_bound) for β
     * @throws std::invalid_argument if confidence_level not in (0,1) or data empty/invalid
     */
    [[nodiscard]] static std::pair<double, double> confidenceIntervalRate(
        const std::vector<double>& data, double confidence_level = 0.95);

    /**
     * @brief Likelihood ratio test for Gamma parameters
     *
     * Tests H0: (α, β) = (α₀, β₀) vs H1: (α, β) ≠ (α₀, β₀) using likelihood ratio statistic.
     * The test statistic -2ln(Λ) follows χ²(2) distribution under H0.
     *
     * @param data Vector of observed positive data
     * @param null_shape Null hypothesis value for α
     * @param null_rate Null hypothesis value for β
     * @param significance_level Significance level for test
     * @return Tuple of (test_statistic, p_value, reject_null)
     */
    [[nodiscard]] static std::tuple<double, double, bool> likelihoodRatioTest(
        const std::vector<double>& data, double null_shape, double null_rate,
        double significance_level = 0.05);

    /**
     * @brief Bayesian estimation with conjugate priors
     *
     * Uses conjugate priors: α ~ Gamma(α_α, β_α) and β ~ Gamma(α_β, β_β).
     * Returns posterior parameters for both shape and rate parameters.
     * Note: Full conjugacy requires more complex prior structure.
     *
     * @param data Vector of observed positive data
     * @param prior_shape_shape Prior shape for α parameter (default: 1.0)
     * @param prior_shape_rate Prior rate for α parameter (default: 1.0)
     * @param prior_rate_shape Prior shape for β parameter (default: 1.0)
     * @param prior_rate_rate Prior rate for β parameter (default: 1.0)
     * @return Tuple of (posterior_shape_shape, posterior_shape_rate, posterior_rate_shape,
     * posterior_rate_rate)
     */
    [[nodiscard]] static std::tuple<double, double, double, double> bayesianEstimation(
        const std::vector<double>& data, double prior_shape_shape = 1.0,
        double prior_shape_rate = 1.0, double prior_rate_shape = 1.0, double prior_rate_rate = 1.0);

    /**
     * @brief Robust parameter estimation using M-estimators
     *
     * Provides robust estimation of Gamma parameters that is less sensitive to outliers.
     * Uses trimmed/winsorized data or quantile-based methods.
     *
     * @param data Vector of observed positive data
     * @param estimator_type Type of robust estimator ("winsorized", "trimmed", "quantile")
     * @param trim_proportion Proportion to trim/winsorize (default: 0.1)
     * @return Pair of (robust_shape_estimate, robust_rate_estimate)
     */
    [[nodiscard]] static std::pair<double, double> robustEstimation(
        const std::vector<double>& data, const std::string& estimator_type = "winsorized",
        double trim_proportion = 0.1);

    /**
     * @brief Method of moments estimation
     *
     * Estimates Gamma parameters by matching sample moments with theoretical moments:
     * α = (sample_mean)² / sample_variance
     * β = sample_mean / sample_variance
     *
     * @param data Vector of observed positive data
     * @return Pair of (shape_estimate, rate_estimate)
     * @throws std::invalid_argument if data is empty or has zero variance
     */
    [[nodiscard]] static std::pair<double, double> methodOfMomentsEstimation(
        const std::vector<double>& data);

    /**
     * @brief Bayesian credible interval from posterior distributions
     *
     * Calculates Bayesian credible intervals for shape and rate parameters
     * from their posterior distributions after observing data.
     *
     * @param data Vector of observed positive data
     * @param credibility_level Credibility level (e.g., 0.95 for 95%)
     * @param prior_shape_shape Prior shape for α parameter (default: 1.0)
     * @param prior_shape_rate Prior rate for α parameter (default: 1.0)
     * @param prior_rate_shape Prior shape for β parameter (default: 1.0)
     * @param prior_rate_rate Prior rate for β parameter (default: 1.0)
     * @return Tuple of ((shape_CI_lower, shape_CI_upper), (rate_CI_lower, rate_CI_upper))
     */
    [[nodiscard]] static std::tuple<std::pair<double, double>, std::pair<double, double>>
    bayesianCredibleInterval(const std::vector<double>& data, double credibility_level = 0.95,
                             double prior_shape_shape = 1.0, double prior_shape_rate = 1.0,
                             double prior_rate_shape = 1.0, double prior_rate_rate = 1.0);

    /**
     * @brief L-moments parameter estimation
     *
     * Uses L-moments (linear combinations of order statistics) for robust
     * parameter estimation. More robust than ordinary moments for extreme distributions.
     *
     * @param data Vector of observed positive data
     * @return Pair of (shape_estimate, rate_estimate)
     */
    [[nodiscard]] static std::pair<double, double> lMomentsEstimation(
        const std::vector<double>& data);

    /**
     * @brief Normal approximation validity test for large shape parameter
     *
     * Tests whether the Gamma distribution can be well-approximated by a normal distribution.
     * For large α, Gamma(α,β) ≈ N(α/β, α/β²). Tests goodness of this approximation.
     *
     * @param data Vector of observed positive data
     * @param significance_level Significance level for test
     * @return Tuple of (test_statistic, p_value, approximation_is_valid)
     */
    [[nodiscard]] static std::tuple<double, double, bool> normalApproximationTest(
        const std::vector<double>& data, double significance_level = 0.05);

    //==========================================================================
    // 8. GOODNESS-OF-FIT TESTS
    //==========================================================================

    /**
     * @brief Kolmogorov-Smirnov goodness-of-fit test
     *
     * Tests the null hypothesis that data follows the specified Gamma distribution.
     * Compares empirical CDF with theoretical Gamma CDF using KS statistic.
     *
     * @param data Sample data to test
     * @param distribution Theoretical Gamma distribution to test against
     * @param significance_level Significance level (default: 0.05)
     * @return Tuple of (KS_statistic, p_value, reject_null)
     * @note Uses asymptotic p-value approximation for large samples
     */
    [[nodiscard]] static std::tuple<double, double, bool> kolmogorovSmirnovTest(
        const std::vector<double>& data, const GammaDistribution& distribution,
        double significance_level = 0.05);

    /**
     * @brief Anderson-Darling goodness-of-fit test
     *
     * Tests the null hypothesis that data follows the specified Gamma distribution.
     * More sensitive to deviations in the tails than KS test, especially effective
     * for detecting departures from the Gamma family.
     *
     * @param data Sample data to test
     * @param distribution Theoretical Gamma distribution to test against
     * @param significance_level Significance level (default: 0.05)
     * @return Tuple of (AD_statistic, p_value, reject_null)
     * @note Uses asymptotic p-value approximation for Gamma distributions
     */
    [[nodiscard]] static std::tuple<double, double, bool> andersonDarlingTest(
        const std::vector<double>& data, const GammaDistribution& distribution,
        double significance_level = 0.05);

    //==========================================================================
    // 9. CROSS-VALIDATION METHODS
    //==========================================================================

    /**
     * @brief K-fold cross-validation for parameter estimation
     *
     * Performs k-fold cross-validation to assess parameter estimation quality
     * and model stability. Splits data into k folds, trains on k-1 folds,
     * and validates on the remaining fold. Useful for assessing overfitting
     * and parameter estimation robustness.
     *
     * @param data Sample data for cross-validation
     * @param k Number of folds (default: 5)
     * @param random_seed Seed for random fold assignment (default: 42)
     * @return Vector of k validation results: (log_likelihood, shape_error, rate_error)
     *         where shape_error and rate_error are squared errors from true parameters
     * @throws std::invalid_argument if data is empty, k < 2, or k > data.size()
     */
    [[nodiscard]] static std::vector<std::tuple<double, double, double>> kFoldCrossValidation(
        const std::vector<double>& data, int k = 5, unsigned int random_seed = 42);

    /**
     * @brief Leave-one-out cross-validation for parameter estimation
     *
     * Performs leave-one-out cross-validation (LOOCV) to assess parameter
     * estimation quality. For each data point, trains on all other points
     * and validates on the left-out point. Provides nearly unbiased estimate
     * of model performance but is computationally expensive.
     *
     * @param data Sample data for cross-validation
     * @return Tuple of (mean_log_likelihood, variance_log_likelihood, total_computation_time_ms)
     * @throws std::invalid_argument if data size < 3 (insufficient for meaningful LOOCV)
     */
    [[nodiscard]] static std::tuple<double, double, double> leaveOneOutCrossValidation(
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
     * @param fitted_distribution The fitted Gamma distribution
     * @return Tuple of (AIC, BIC, AICc, log_likelihood)
     */
    [[nodiscard]] static std::tuple<double, double, double, double> computeInformationCriteria(
        const std::vector<double>& data, const GammaDistribution& fitted_distribution);

    //==========================================================================
    // 11. BOOTSTRAP METHODS
    //==========================================================================

    /**
     * @brief Bootstrap parameter confidence intervals
     *
     * Uses bootstrap resampling to estimate confidence intervals for
     * the distribution parameters (shape α and rate β).
     *
     * @param data Sample data for bootstrap resampling
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @param n_bootstrap Number of bootstrap samples (default: 1000)
     * @param random_seed Seed for random sampling (default: 42)
     * @return Tuple of ((shape_CI_lower, shape_CI_upper), (rate_CI_lower, rate_CI_upper))
     */
    [[nodiscard]] static std::tuple<std::pair<double, double>, std::pair<double, double>>
    bootstrapParameterConfidenceIntervals(const std::vector<double>& data,
                                          double confidence_level = 0.95, int n_bootstrap = 1000,
                                          unsigned int random_seed = 42);

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /**
     * @brief Check if this is an exponential distribution (α = 1)
     *
     * @return true if α = 1 within tolerance
     */
    [[nodiscard]] bool isExponentialDistribution() const noexcept;

    /**
     * @brief Check if this is a chi-squared distribution (β = 0.5)
     *
     * @return true if β = 0.5 within tolerance
     */
    [[nodiscard]] bool isChiSquaredDistribution() const noexcept;

    /**
     * @brief Get the degrees of freedom if this is a chi-squared distribution
     *
     * @return 2α if β = 0.5, otherwise throws exception
     * @throws std::logic_error if not a chi-squared distribution
     */
    [[nodiscard]] double getDegreesOfFreedom() const;

    /**
     * @brief Compute the entropy of the distribution
     *
     * H(X) = α - log(β) + log(Γ(α)) + (1-α)ψ(α)
     *
     * @return Entropy value
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief Get the median of the distribution
     * For Gamma distribution, median is approximated using the quantile function
     *
     * @return Median value (quantile at p=0.5)
     */
    [[nodiscard]] double getMedian() const noexcept;

    /**
     * Gets the mode of the distribution.
     * For Gamma distribution, mode = (α-1)/β = (α-1)θ for α ≥ 1, 0 for α < 1
     *
     * @return Mode value
     */
    [[nodiscard]] double getMode() const noexcept;

    /**
     * @brief Check if the distribution is suitable for normal approximation
     *
     * Returns true if α is large enough (typically α > 100) for normal approximation
     *
     * @return true if normal approximation is accurate
     */
    [[nodiscard]] bool canUseNormalApproximation() const noexcept;

    /**
     * @brief Create a gamma distribution from mean and variance
     *
     * Uses method of moments: α = mean²/variance, β = mean/variance
     *
     * @param mean Desired mean (must be positive)
     * @param variance Desired variance (must be positive)
     * @return Result containing GammaDistribution or error
     */
    [[nodiscard]] static Result<GammaDistribution> createFromMoments(double mean,
                                                                     double variance) noexcept;

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
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
     */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

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
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /**
     * @brief Smart auto-dispatch batch cumulative probability calculation with performance hints
     *
     * Automatically selects the optimal execution strategy for CDF computation
     * based on batch size, system capabilities, and user performance hints.
     *
     *
     * @param values Input values as C++20 span for type safety
     * @param results Output cumulative probability results as C++20 span
     * @param hint Performance optimization hints (default: AUTO selection)
     *
     * @throws std::invalid_argument if spans have different sizes
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
     *
     * @deprecated Consider migrating to auto-dispatch with hints for better portability
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
     *
     * @deprecated Consider migrating to auto-dispatch with hints for better portability
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
     *
     * @deprecated Consider migrating to auto-dispatch with hints for better portability
     */
    void getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                              std::span<double> results,
                                              detail::Strategy strategy) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    /**
     * Equality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are equal within tolerance
     */
    bool operator==(const GammaDistribution& other) const;

    /**
     * Inequality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const GammaDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    /**
     * @brief Stream input operator
     * @param is Input stream
     * @param dist Distribution to input
     * @return Reference to the input stream
     */
    friend std::istream& operator>>(std::istream& is, stats::GammaDistribution&);

    /**
     * @brief Stream output operator
     * @param os Output stream
     * @param dist Distribution to output
     * @return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const stats::GammaDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param alpha Shape parameter (assumed valid)
     * @param beta Rate parameter (assumed valid)
     * @return GammaDistribution with the given parameters
     */
    static GammaDistribution createUnchecked(double alpha, double beta) noexcept;

    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param alpha Shape parameter (assumed valid)
     * @param beta Rate parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    GammaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief Internal implementation for batch PDF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double alpha, double beta, double log_gamma_alpha,
                                       double alpha_log_beta,
                                       double alpha_minus_one) const noexcept;

    /** @brief Internal implementation for batch log PDF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double alpha, double beta, double log_gamma_alpha,
                                          double alpha_log_beta,
                                          double alpha_minus_one) const noexcept;

    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double alpha,
                                                 double beta) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    /** @brief Compute incomplete gamma function using continued fractions */
    [[nodiscard]] static double incompleteGamma(double a, double x) noexcept;

    /** @brief Compute regularized incomplete gamma function P(a,x) */
    [[nodiscard]] static double regularizedIncompleteGamma(double a, double x) noexcept;

    /** @brief Compute quantile using Newton-Raphson with bracketing */
    [[nodiscard]] double computeQuantile(double p) const noexcept;

    /** @brief Sample using Marsaglia-Tsang method for α ≥ 1 */
    [[nodiscard]] double sampleMarsagliaTsang(std::mt19937& rng) const noexcept;

    /** @brief Sample using Ahrens-Dieter method for α < 1 */
    [[nodiscard]] double sampleAhrensDieter(std::mt19937& rng) const noexcept;

    /** @brief Fit parameters using method of moments */
    void fitMethodOfMoments(const std::vector<double>& values);

    /** @brief Fit parameters using maximum likelihood estimation */
    void fitMaximumLikelihood(const std::vector<double>& values);

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        logGammaAlpha_ = std::lgamma(alpha_);
        logBeta_ = std::log(beta_);
        alphaLogBeta_ = alpha_ * logBeta_;
        alphaMinusOne_ = alpha_ - detail::ONE;

        // Derived parameters
        scale_ = detail::ONE / beta_;
        mean_ = alpha_ * scale_;
        variance_ = mean_ * scale_;

        // Advanced functions
        digammaAlpha_ = GammaDistribution::computeDigamma(alpha_);
        sqrtAlpha_ = std::sqrt(alpha_);

        // Optimization flags
        isExponential_ = (std::abs(alpha_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
        isIntegerAlpha_ = (std::abs(alpha_ - std::round(alpha_)) <= detail::DEFAULT_TOLERANCE);
        isSmallAlpha_ = (alpha_ < detail::ONE);
        isLargeAlpha_ = (alpha_ > 100.0);
        isStandardGamma_ = (std::abs(beta_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
        isChiSquared_ = (std::abs(beta_ - detail::HALF) <= detail::DEFAULT_TOLERANCE);

        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);

        // Update atomic parameters for lock-free access
        atomicAlpha_.store(alpha_, std::memory_order_release);
        atomicBeta_.store(beta_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }

    /**
     * Validates parameters for the Gamma distribution
     * @param alpha Shape parameter (must be positive)
     * @param beta Rate parameter (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double alpha, double beta) {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Alpha (shape parameter) must be a positive finite number");
        }
        if (std::isnan(beta) || std::isinf(beta) || beta <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Beta (rate parameter) must be a positive finite number");
        }
    }

    //==========================================================================
    // 20. PRIVATE UTILITY METHODS
    //==========================================================================

    /**
     * Computes the digamma function ψ(x) = d/dx log(Γ(x))
     * Uses series expansion and asymptotic approximation
     */
    static double computeDigamma(double x) noexcept;

    /**
     * Computes the trigamma function ψ'(x) = d²/dx² log(Γ(x))
     * Uses series expansion and asymptotic approximation
     */
    static double computeTrigamma(double x) noexcept;

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Shape parameter α - must be positive */
    double alpha_{detail::ONE};

    /** @brief Rate parameter β - must be positive (β = 1/scale) */
    double beta_{detail::ONE};

    /** @brief C++20 atomic copies of parameters for lock-free access */
    mutable std::atomic<double> atomicAlpha_{detail::ONE};
    mutable std::atomic<double> atomicBeta_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief Cached value of log(Γ(α)) for efficiency in PDF calculations */
    mutable double logGammaAlpha_{detail::ZERO_DOUBLE};

    /** @brief Cached value of log(β) for efficiency in PDF calculations */
    mutable double logBeta_{detail::ZERO_DOUBLE};

    /** @brief Cached value of α*log(β) for efficiency in PDF calculations */
    mutable double alphaLogBeta_{detail::ZERO_DOUBLE};

    /** @brief Cached value of α-1 for efficiency in PDF calculations */
    mutable double alphaMinusOne_{detail::ZERO_DOUBLE};

    /** @brief Cached value of 1/β (scale parameter θ) for efficiency */
    mutable double scale_{detail::ONE};

    /** @brief Cached value of α/β (mean) for efficiency */
    mutable double mean_{detail::ONE};

    /** @brief Cached value of α/β² (variance) for efficiency */
    mutable double variance_{detail::ONE};

    /** @brief Cached value of digamma(α) for efficiency in various calculations */
    mutable double digammaAlpha_{detail::ZERO_DOUBLE};

    /** @brief Cached value of √α for efficiency in normal approximation */
    mutable double sqrtAlpha_{detail::ONE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if α = 1 (exponential distribution) for optimization */
    mutable bool isExponential_{true};

    /** @brief True if α is an integer for optimization */
    mutable bool isIntegerAlpha_{true};

    /** @brief True if α < 1 for special sampling algorithm */
    mutable bool isSmallAlpha_{false};

    /** @brief True if α is large (> 100) for normal approximation */
    mutable bool isLargeAlpha_{false};

    /** @brief True if β = 1 (standard gamma) for optimization */
    mutable bool isStandardGamma_{true};

    /** @brief True if this is a chi-squared distribution (β = 0.5) */
    mutable bool isChiSquared_{false};

    /** @brief Atomic cache validity flag for lock-free fast path optimization */
    mutable std::atomic<bool> cacheValidAtomic_{false};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Gamma distribution uses standard caching only
    // This section maintained for template compliance
};

}  // namespace stats

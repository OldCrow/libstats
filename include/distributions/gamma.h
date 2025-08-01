#pragma once

#include "../core/distribution_base.h"
#include "../core/constants.h"
#include "../core/error_handling.h" // Safe error handling without exceptions
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation

namespace libstats {

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
class GammaDistribution : public DistributionBase
{   
private:
    //==========================================================================
    // DISTRIBUTION PARAMETERS
    //==========================================================================
    
    /** @brief Shape parameter α - must be positive */
    double alpha_{constants::math::ONE};
    
    /** @brief Rate parameter β - must be positive (β = 1/scale) */
    double beta_{constants::math::ONE};

    //==========================================================================
    // PERFORMANCE CACHE
    //==========================================================================
    
    /** @brief Cached value of log(Γ(α)) for efficiency in PDF calculations */
    mutable double logGammaAlpha_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of log(β) for efficiency in PDF calculations */
    mutable double logBeta_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of α*log(β) for efficiency in PDF calculations */
    mutable double alphaLogBeta_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of α-1 for efficiency in PDF calculations */
    mutable double alphaMinusOne_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of 1/β (scale parameter θ) for efficiency */
    mutable double scale_{constants::math::ONE};
    
    /** @brief Cached value of α/β (mean) for efficiency */
    mutable double mean_{constants::math::ONE};
    
    /** @brief Cached value of α/β² (variance) for efficiency */
    mutable double variance_{constants::math::ONE};
    
    /** @brief Cached value of digamma(α) for efficiency in various calculations */
    mutable double digammaAlpha_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of √α for efficiency in normal approximation */
    mutable double sqrtAlpha_{constants::math::ONE};
    
    //==========================================================================
    // OPTIMIZATION FLAGS
    //==========================================================================
    
    /** @brief Atomic cache validity flag for lock-free fast path optimization */
    mutable std::atomic<bool> cacheValidAtomic_{false};
    
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

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        logGammaAlpha_ = std::lgamma(alpha_);
        logBeta_ = std::log(beta_);
        alphaLogBeta_ = alpha_ * logBeta_;
        alphaMinusOne_ = alpha_ - constants::math::ONE;
        
        // Derived parameters
        scale_ = constants::math::ONE / beta_;
        mean_ = alpha_ * scale_;
        variance_ = mean_ * scale_;
        
        // Advanced functions
        digammaAlpha_ = computeDigamma(alpha_);
        sqrtAlpha_ = std::sqrt(alpha_);
        
        // Optimization flags
        isExponential_ = (std::abs(alpha_ - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        isIntegerAlpha_ = (std::abs(alpha_ - std::round(alpha_)) <= constants::precision::DEFAULT_TOLERANCE);
        isSmallAlpha_ = (alpha_ < constants::math::ONE);
        isLargeAlpha_ = (alpha_ > 100.0);
        isStandardGamma_ = (std::abs(beta_ - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        isChiSquared_ = (std::abs(beta_ - constants::math::HALF) <= constants::precision::DEFAULT_TOLERANCE);
        
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
    }
    
    /**
     * Validates parameters for the Gamma distribution
     * @param alpha Shape parameter (must be positive)
     * @param beta Rate parameter (must be positive)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double alpha, double beta) {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Alpha (shape parameter) must be a positive finite number");
        }
        if (std::isnan(beta) || std::isinf(beta) || beta <= constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Beta (rate parameter) must be a positive finite number");
        }
    }
    
    /**
     * Computes the digamma function ψ(x) = d/dx log(Γ(x))
     * Uses series expansion and asymptotic approximation
     */
    static double computeDigamma(double x) noexcept;

    friend std::istream& operator>>(std::istream& is,
            libstats::GammaDistribution& distribution);

public:
    //==========================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    
    /**
     * Constructs a Gamma distribution with given shape and rate parameters.
     * 
     * @param alpha Shape parameter α (must be positive, default: 1.0)
     * @param beta Rate parameter β (must be positive, default: 1.0)
     * @throws std::invalid_argument if parameters are invalid
     * 
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit GammaDistribution(double alpha = constants::math::ONE, 
                              double beta = constants::math::ONE);
    
    /**
     * Thread-safe copy constructor
     * 
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    GammaDistribution(const GammaDistribution& other);
    
    /**
     * Copy assignment operator
     * 
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    GammaDistribution& operator=(const GammaDistribution& other);
    
    /**
     * Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    GammaDistribution(GammaDistribution&& other);
    
    /**
     * Move assignment operator (DEFENSIVE THREAD SAFETY)
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
    // SAFE FACTORY METHODS (Exception-free construction)
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
    [[nodiscard]] static Result<GammaDistribution> create(double alpha = 1.0, double beta = 1.0) noexcept {
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
    [[nodiscard]] static Result<GammaDistribution> createWithScale(double alpha, double scale) noexcept {
        if (scale <= 0.0) {
            return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter, 
                                                        "Scale parameter must be positive");
        }
        return create(alpha, 1.0 / scale);
    }
    
    /**
     * @brief Safely try to set parameters without throwing exceptions
     * 
     * @param alpha New shape parameter
     * @param beta New rate parameter
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(double alpha, double beta) noexcept {
        auto validation = validateGammaParameters(alpha, beta);
        if (validation.isError()) {
            return validation;
        }
        
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        alpha_ = alpha;
        beta_ = beta;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        
        return VoidResult::ok(true);
    }
    
    /**
     * @brief Check if current parameters are valid
     * @return VoidResult indicating validity
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return validateGammaParameters(alpha_, beta_);
    }

    //==========================================================================
    // CORE PROBABILITY METHODS
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

    //==========================================================================
    // PARAMETER GETTERS AND SETTERS
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
     * Gets the scale parameter θ = 1/β.
     * Uses cached value to eliminate division.
     * 
     * @return Scale parameter value
     */
    [[nodiscard]] double getScale() const noexcept;
    
    /**
     * Sets the shape parameter α (exception-based API).
     * 
     * @param alpha New shape parameter (must be positive)
     * @throws std::invalid_argument if alpha <= 0 or is not finite
     */
    void setAlpha(double alpha);
    
    /**
     * @brief Safely set the shape parameter α without throwing exceptions (Result-based API).
     * 
     * @param alpha New shape parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetAlpha(double alpha) noexcept;
    
    /**
     * Sets the rate parameter β (exception-based API).
     * 
     * @param beta New rate parameter (must be positive)
     * @throws std::invalid_argument if beta <= 0 or is not finite
     */
    void setBeta(double beta);
    
    /**
     * @brief Safely set the rate parameter β without throwing exceptions (Result-based API).
     * 
     * @param beta New rate parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetBeta(double beta) noexcept;
    
    /**
     * Sets the scale parameter θ = 1/β.
     * 
     * @param scale New scale parameter (must be positive)
     * @throws std::invalid_argument if scale <= 0 or is not finite
     */
    void setScale(double scale);
    
    /**
     * Sets both parameters simultaneously.
     * 
     * @param alpha New shape parameter
     * @param beta New rate parameter
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(double alpha, double beta);
    
    /**
     * Sets parameters using shape-scale parameterization.
     * 
     * @param alpha New shape parameter
     * @param scale New scale parameter
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParametersWithScale(double alpha, double scale);
    
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
    [[nodiscard]] int getNumParameters() const noexcept override {
        return 2;
    }
    
    /**
     * @brief Gets the distribution name.
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Distribution name
     */
    [[nodiscard]] std::string getDistributionName() const override {
        return "Gamma";
    }
    
    /**
     * @brief Checks if the distribution is discrete.
     * For Gamma distribution, it's continuous
     * Inline for performance - no thread safety needed for constant
     * 
     * @return false (always continuous)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override {
        return false;
    }
    
    /**
     * @brief Gets the lower bound of the distribution support.
     * For Gamma distribution, support is [0, ∞)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Lower bound (0)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return 0.0;
    }
    
    /**
     * @brief Gets the upper bound of the distribution support.
     * For Gamma distribution, support is [0, ∞)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Upper bound (+infinity)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }
    
    /**
     * Gets the mode of the distribution.
     * For Gamma distribution, mode = (α-1)/β = (α-1)θ for α ≥ 1, 0 for α < 1
     * 
     * @return Mode value
     */
    [[nodiscard]] double getMode() const noexcept;

    //==========================================================================
    // DISTRIBUTION MANAGEMENT
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
    // COMPARISON OPERATORS
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
    bool operator!=(const GammaDistribution& other) const { return !(*this == other); }
    
    //==========================================================================
    // SIMD BATCH OPERATIONS
    //==========================================================================
    
    /**
     * Batch computation of probability densities using SIMD optimization.
     * Processes multiple values simultaneously for improved performance.
     * 
     * @param values Input array of values to evaluate
     * @param results Output array for probability densities (must be same size as values)
     * @param count Number of values to process
     * @note This method is optimized for large batch sizes where SIMD overhead is justified
     */
    void getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept;
    
    /**
     * Batch computation of log probability densities using SIMD optimization.
     * Processes multiple values simultaneously for improved performance.
     * 
     * @param values Input array of values to evaluate
     * @param results Output array for log probability densities (must be same size as values)
     * @param count Number of values to process
     * @note This method is optimized for large batch sizes where SIMD overhead is justified
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

    //==========================================================================
    // GAMMA-SPECIFIC UTILITY METHODS
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
    [[nodiscard]] static Result<GammaDistribution> createFromMoments(double mean, double variance) noexcept;

private:
    //==========================================================================
    // PRIVATE FACTORY METHODS
    //==========================================================================
    
    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param alpha Shape parameter (assumed valid)
     * @param beta Rate parameter (assumed valid)
     * @return GammaDistribution with the given parameters
     */
    static GammaDistribution createUnchecked(double alpha, double beta) noexcept {
        GammaDistribution dist(alpha, beta, true); // bypass validation
        return dist;
    }
    
    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param alpha Shape parameter (assumed valid)
     * @param beta Rate parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    GammaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept
        : DistributionBase(), alpha_(alpha), beta_(beta) {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    
    //==========================================================================
    // PRIVATE COMPUTATIONAL METHODS
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
    
    //==========================================================================
    // PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================
    
    /** @brief Internal implementation for batch PDF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double alpha, double beta, double log_gamma_alpha, 
                                       double alpha_log_beta, double alpha_minus_one) const noexcept;
    
    /** @brief Internal implementation for batch log PDF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double alpha, double beta, double log_gamma_alpha, 
                                          double alpha_log_beta, double alpha_minus_one) const noexcept;
    
    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                 double alpha, double beta) const noexcept;
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const GammaDistribution& dist);

} // namespace libstats

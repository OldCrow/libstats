#ifndef LIBSTATS_EXPONENTIAL_H_
#define LIBSTATS_EXPONENTIAL_H_

#include "distribution_base.h"
#include "constants.h"
#include "simd.h" // Ensure SIMD operations are available
#include "error_handling.h" // Safe error handling without exceptions
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation

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
     */
    [[nodiscard]] VoidResult trySetParameters(double lambda) noexcept {
        auto validation = validateExponentialParameters(lambda);
        if (validation.isError()) {
            return validation;
        }
        
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        lambda_ = lambda;
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
     * Sets the rate parameter λ.
     * 
     * @param lambda New rate parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    void setLambda(double lambda);
    
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
    
    //==========================================================================
    // PRIVATE SIMD IMPLEMENTATION METHODS
    //==========================================================================
    
    /** @brief SIMD implementation for batch PDF calculation */
    void getProbabilityBatchSIMD(const double* values, double* results, std::size_t count,
                                 double lambda, double neg_lambda) const noexcept;
    
    /** @brief SIMD implementation for batch log PDF calculation */
    void getLogProbabilityBatchSIMD(const double* values, double* results, std::size_t count,
                                    double log_lambda, double neg_lambda) const noexcept;
    
    /** @brief SIMD implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchSIMD(const double* values, double* results, std::size_t count,
                                           double neg_lambda) const noexcept;
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

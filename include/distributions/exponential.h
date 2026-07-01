#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Common platform headers for distributions (consolidates shared platform dependencies)

namespace stats {

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
 * @version 2.0.0
 * @since 2.0.0
 */
class ExponentialDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<ExponentialDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::EXPONENTIAL;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Constructs an Exponential distribution with given rate parameter.
     *
     * @param lambda Rate parameter λ (must be positive)
     * @throws std::invalid_argument if lambda is invalid
     *
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit ExponentialDistribution(double lambda = detail::ONE);

    /**
     * @brief Thread-safe copy constructor
     *
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    ExponentialDistribution(const ExponentialDistribution& other);

    /**
     * @brief Copy assignment operator
     *
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    ExponentialDistribution& operator=(const ExponentialDistribution& other);

    /**
     * @brief Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     *
     */
    ExponentialDistribution(ExponentialDistribution&& other) noexcept;

    /**
     * @brief Move assignment operator (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with deadlock prevention
     *
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
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create an Exponential distribution without throwing exceptions
     *
     * This factory method provides exception-free construction.
     * See `error_handling.h` for the Result<T> design rationale.
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
    [[nodiscard]] static Result<ExponentialDistribution> create(double lambda = 1.0) {
        auto validation = validateExponentialParameters(lambda);
        if (validation.isError()) {
            return Result<ExponentialDistribution>::makeError(validation.error_code,
                                                              validation.message);
        }

        // Use private factory to bypass validation
        return Result<ExponentialDistribution>::ok(createUnchecked(lambda));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
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
    [[nodiscard]] double getLambdaAtomic() const noexcept;

    /**
     * Sets the rate parameter λ (exception-based API).
     *
     * @param lambda New rate parameter (must be positive)
     * @throws std::invalid_argument if lambda <= 0 or is not finite
     */
    void setLambda(double lambda);

    /**
     * @brief Sets the rate parameter (exception-based API).
     * Thread-safe: acquires unique lock for cache invalidation
     *
     * @param lambda New rate parameter λ (must be positive)
     * @throws std::invalid_argument if lambda is invalid
     */
    void setParameters(double lambda);

    /**
     * Gets the scale parameter (reciprocal of rate parameter).
     * This is equivalent to the mean for exponential distributions.
     * Uses cached value to eliminate division.
     *
     * @return Scale parameter (1/λ)
     */
    [[nodiscard]] double getScale() const;

    /**
     * Gets the mean of the distribution.
     * For Exponential distribution, mean = 1/λ
     * Uses cached value to eliminate division.
     *
     * @return Mean value
     */
    [[nodiscard]] double getMean() const override;

    /**
     * Gets the variance of the distribution.
     * For Exponential distribution, variance = 1/λ²
     * Uses cached value to eliminate divisions and multiplications.
     *
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const override;

    /**
     * @brief Gets the skewness of the distribution.
     * For Exponential distribution, skewness = 2 (always right-skewed)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Skewness value (always 2)
     */
    [[nodiscard]] double getSkewness() const override;

    /**
     * @brief Gets the kurtosis of the distribution.
     * For Exponential distribution, excess kurtosis = 6
     * Inline for performance - no thread safety needed for constant
     *
     * @return Excess kurtosis value (always 6)
     */
    [[nodiscard]] double getKurtosis() const override;

    /**
     * @brief Gets the number of parameters for this distribution.
     * For Exponential distribution, there is 1 parameter: lambda (rate)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Number of parameters (always 1)
     */
    [[nodiscard]] int getNumParameters() const noexcept override;

    /**
     * @brief Gets the distribution name.
     * Inline for performance - no thread safety needed for constant
     *
     * @return Distribution name
     */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override { return "Exponential"; }

    /**
     * @brief Checks if the distribution is discrete.
     * For Exponential distribution, it's continuous
     * Inline for performance - no thread safety needed for constant
     *
     * @return false (always continuous)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override;

    /**
     * @brief Gets the lower bound of the distribution support.
     * For Exponential distribution, support is [0, ∞)
     *
     * @return Lower bound (0)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override;

    /**
     * @brief Gets the upper bound of the distribution support.
     * For Exponential distribution, support is [0, ∞)
     *
     * @return Upper bound (+infinity)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override;

    //==============================================================================
    // 4. RESULT-BASED SETTERS
    //==============================================================================

    /**
     * @brief Safely set the rate parameter λ without throwing exceptions (Result-based API).
     *
     * @param lambda New rate parameter (must be positive)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetLambda(double lambda) noexcept;

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
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
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
    [[nodiscard]] double getLogProbability(double x) const override;

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
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Exponential distribution, MLE gives λ = 1/sample_mean.
     *
     * @param values Vector of observed data
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple datasets
     * Efficiently fits exponential distribution parameters to multiple independent datasets in
     * parallel
     *
     * @param datasets Vector of datasets, each representing independent observations
     * @param results Vector to store fitted ExponentialDistribution objects
     * @throws std::invalid_argument if datasets is empty or results size doesn't match
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<ExponentialDistribution>& results);

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
    // 7. ADVANCED STATISTICAL METHODS
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
    // coefficientOfVariationTest moved to stats::analysis::exponential in v2.0.0.

    //==========================================================================
    // 8. GOODNESS-OF-FIT TESTS
    //==========================================================================

    

    

    

    

    

    

    /**
     * @brief Compute the half-life of the exponential process
     *
     * Half-life is the time required for the quantity to reduce to half its initial value.
     * For exponential distribution: half_life = ln(2) / rate
     *
     * @return Half-life value
     */
    [[nodiscard]] double getHalfLife() const;

    /**
     * @brief Check if the distribution has the memoryless property
     *
     * The exponential distribution is memoryless: P(X > s+t | X > s) = P(X > t)
     * This method always returns true for exponential distributions.
     *
     * @return true (exponential distribution is always memoryless)
     */
    [[nodiscard]] bool isMemoryless() const noexcept;

    /**
     * @brief Get the median of the distribution
     *
     * For exponential distribution: median = ln(2) / λ
     * This is the value where P(X ≤ median) = 0.5
     *
     * @return Median value
     */
    [[nodiscard]] double getMedian() const override;

    /**
     * @brief Compute the entropy of the distribution
     *
     * For exponential distribution: H(X) = 1 - ln(λ)
     * Entropy measures the average information content.
     *
     * @return Entropy value
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief Get the mode of the distribution
     *
     * For exponential distribution, the mode is always 0.
     * This is where the PDF achieves its maximum value.
     *
     * @return Mode value (always 0.0)
     */
    [[nodiscard]] double getMode() const;

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
     *
     * @par Strategy Selection Logic:
     * - Tiny batches (≤8): SCALAR for minimal overhead
     * - Small batches (9-63): SIMD_BATCH for vectorization benefits
     * - Medium batches (64-4095): PARALLEL_SIMD for multi-core + vectorization
     * - Large batches (≥4096): WORK_STEALING for load balancing
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
     * detail::PerformanceHint hint;
     * hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
     * dist.getProbability(inputs, outputs, hint);
     * @endcode
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
     *
     * @note Log probability calculations are typically more computationally intensive
     *       than regular PDF, so the auto-dispatcher may prefer parallel strategies
     *       at smaller batch sizes compared to getProbability().
     */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

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
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
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
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    /**
     * @brief Stream input operator
     * @param is Input stream
     * @param dist Distribution to input
     * @return Reference to the input stream
     */
    friend std::istream& operator>>(std::istream& is, stats::ExponentialDistribution& dist);

    /**
     * @brief Stream output operator
     * @param os Output stream
     * @param dist Distribution to output
     * @return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const stats::ExponentialDistribution& dist);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param lambda Rate parameter (assumed valid)
     * @return ExponentialDistribution with the given parameter
     */
    static ExponentialDistribution createUnchecked(double lambda) noexcept {
        ExponentialDistribution dist(lambda, true);  // bypass validation
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
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief Internal implementation for batch PDF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double lambda, double neg_lambda) const noexcept;

    /** @brief Internal implementation for batch log PDF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double log_lambda, double neg_lambda) const noexcept;

    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count,
                                                 double neg_lambda) const noexcept;

    // Note: Redundant SIMD methods removed - SIMD optimization is handled
    // internally within the *UnsafeImpl methods above

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        invLambda_ = detail::ONE / lambda_;
        invLambdaSquared_ = invLambda_ * invLambda_;

        // Core cached values
        logLambda_ = std::log(lambda_);
        negLambda_ = -lambda_;

        // Optimization flags
        isUnitRate_ = (std::abs(lambda_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
        isHighRate_ = (lambda_ > detail::THOUSAND);
        isLowRate_ = (lambda_ < detail::THOUSANDTH);

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
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");
        }
    }

    //==========================================================================
    // 20. PRIVATE UTILITY METHODS
    //==========================================================================

    // Note: Currently no private utility methods needed for Exponential distribution
    // This section maintained for template compliance

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Rate parameter λ - must be positive */
    double lambda_{detail::ONE};

    /** @brief C++20 atomic copy of parameter for lock-free access */
    mutable std::atomic<double> atomicLambda_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief Cached value of ln(λ) for efficiency in log probability calculations */
    mutable double logLambda_{detail::ZERO_DOUBLE};

    /** @brief Cached value of 1/λ (mean and scale parameter) for efficiency */
    mutable double invLambda_{detail::ONE};

    /** @brief Cached value of -λ for efficiency in PDF and log-PDF calculations */
    mutable double negLambda_{-detail::ONE};

    /** @brief Cached value of 1/λ² for variance calculation efficiency */
    mutable double invLambdaSquared_{detail::ONE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if λ = 1 for unit exponential optimizations */
    mutable bool isUnitRate_{true};

    /** @brief True if λ is very large (> 1000) for numerical stability */
    mutable bool isHighRate_{false};

    /** @brief True if λ is very small (< 0.001) for numerical stability */
    mutable bool isLowRate_{false};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Exponential distribution uses standard caching only
    // This section maintained for template compliance
};

}  // namespace stats

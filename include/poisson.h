#ifndef LIBSTATS_POISSON_H_
#define LIBSTATS_POISSON_H_

#include "distribution_base.h"
#include "constants.h"
#include "error_handling.h" // Safe error handling without exceptions
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation

namespace libstats {

/**
 * @brief Thread-safe Poisson Distribution for modeling count data and rare events.
 * 
 * @details The Poisson distribution is a discrete probability distribution that models
 * the number of events occurring in a fixed interval of time or space, given that
 * these events occur with a known constant average rate and independently of the
 * time since the last event. It's the limiting case of the binomial distribution
 * when n → ∞ and p → 0 while np remains constant.
 * 
 * @par Mathematical Definition:
 * - PMF: P(X = k) = (λ^k * e^(-λ)) / k! for k = 0, 1, 2, ..., 0 otherwise
 * - CDF: F(k) = Σ(i=0 to k) (λ^i * e^(-λ)) / i! = Q(k+1, λ) (regularized gamma function)
 * - Parameter: λ > 0 (rate parameter, mean number of events)
 * - Support: k ∈ {0, 1, 2, 3, ...} (non-negative integers)
 * - Mean: λ
 * - Variance: λ
 * - Mode: ⌊λ⌋ (floor of λ, or λ-1 if λ is integer)
 * - Median: ≈ λ + 1/3 - 0.02/λ (approximation for large λ)
 * 
 * @par Key Properties:
 * - **Mean = Variance**: Unique property where E[X] = Var[X] = λ
 * - **Additive**: Sum of independent Poisson(λ₁) and Poisson(λ₂) is Poisson(λ₁ + λ₂)
 * - **Memoryless**: Inter-arrival times follow exponential distribution
 * - **Limiting Distribution**: Binomial(n,p) → Poisson(np) as n→∞, p→0
 * - **Conjugate Prior**: Gamma distribution is conjugate prior for λ
 * - **Overdispersion**: When variance > mean, suggests negative binomial may be better
 * 
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - Concurrent reads are optimized with std::shared_mutex
 * - Cache invalidation uses atomic operations for lock-free fast paths
 * - Deadlock prevention via ordered lock acquisition with std::lock()
 * 
 * @par Performance Features:
 * - Atomic cache validity flags for lock-free fast path access
 * - Extensive caching of computed values (λ, log(λ), e^(-λ), √λ, log(Γ(λ+1)))
 * - Optimized PMF computation with precomputed factorials and Stirling's approximation
 * - Fast parameter validation with IEEE 754 compliance
 * - Special algorithms for small λ, large λ, and integer λ cases
 * 
 * @par Usage Examples:
 * @code
 * // Customer arrivals (average 3 per hour)
 * auto result = PoissonDistribution::create(3.0);
 * if (result.isOk()) {
 *     auto arrivals = std::move(result.value);
 *     
 *     // Network packet errors (average 0.1 per second)
 *     auto errorResult = PoissonDistribution::create(0.1);
 *     if (errorResult.isOk()) {
 *         auto errors = std::move(errorResult.value);
 *         
 *         // Fit to observed count data
 *         std::vector<double> counts = {2, 1, 4, 3, 2, 5, 1, 3, 2, 4};
 *         arrivals.fit(counts);
 *         
 *         // Probability of exactly 5 arrivals
 *         double exactFive = arrivals.getProbability(5);
 *         
 *         // Probability of 3 or fewer arrivals
 *         double atMostThree = arrivals.getCumulativeProbability(3);
 *         
 *         // Generate random arrival count
 *         std::mt19937 rng(42);
 *         int count = static_cast<int>(arrivals.sample(rng));
 *     }
 * }
 * @endcode
 * 
 * @par Applications:
 * - **Queuing Theory**: Customer arrivals, service requests
 * - **Reliability Engineering**: Equipment failures, defects
 * - **Telecommunications**: Call arrivals, packet errors
 * - **Biology**: Mutations, cell counts, bacterial colonies
 * - **Economics**: Rare market events, insurance claims
 * - **Physics**: Radioactive decay, particle detection
 * - **Web Analytics**: Page views, clicks, conversions
 * - **Quality Control**: Defects per unit, accidents per period
 * 
 * @par Statistical Properties:
 * - Skewness: 1/√λ (right-skewed, approaches 0 as λ increases)
 * - Kurtosis: 1/λ (excess kurtosis, approaches 0 as λ increases)
 * - Entropy: λ(1 - log(λ)) + e^(-λ) * Σ(k=0 to ∞) (λ^k * log(k!)) / k!
 * - Moment generating function: e^(λ(e^t - 1))
 * - Characteristic function: e^(λ(e^(it) - 1))
 * - Probability generating function: e^(λ(z - 1))
 * 
 * @par Computational Algorithms:
 * - **Small λ (< 10)**: Direct computation with cached factorials
 * - **Medium λ (10-100)**: Stirling's approximation for factorials
 * - **Large λ (> 100)**: Normal approximation with continuity correction
 * - **CDF**: Regularized incomplete gamma function Q(k+1, λ)
 * - **Quantile**: Inverse regularized gamma function with bracketing
 * - **Sampling**: Knuth's algorithm for small λ, acceptance-rejection for large λ
 * 
 * @par Numerical Considerations:
 * - Robust handling of factorial overflow using log-space computation
 * - Efficient computation of e^(-λ) for large λ using series expansion
 * - Accurate CDF computation using continued fractions for gamma function
 * - Special handling for λ near machine epsilon and very large λ
 * - IEEE 754 compliant boundary handling for extreme values
 * 
 * @par Implementation Details (C++20 Best Practices):
 * - Complex constructors/operators moved to .cpp for faster compilation
 * - Exception-safe design with RAII principles
 * - Optimized parameter validation with comprehensive error messages
 * - Lock-free fast paths using atomic operations
 * - Specialized algorithms for different λ ranges
 * 
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
class PoissonDistribution : public DistributionBase
{   
private:
    //==========================================================================
    // DISTRIBUTION PARAMETERS
    //==========================================================================
    
    /** @brief Rate parameter λ (mean number of events) - must be positive */
    double lambda_{constants::math::ONE};

    //==========================================================================
    // PERFORMANCE CACHE
    //==========================================================================
    
    /** @brief Cached value of log(λ) for efficiency in PMF calculations */
    mutable double logLambda_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of e^(-λ) for efficiency in PMF calculations */
    mutable double expNegLambda_{constants::math::E_INV};
    
    /** @brief Cached value of √λ for efficiency in normal approximation */
    mutable double sqrtLambda_{constants::math::ONE};
    
    /** @brief Cached value of log(Γ(λ+1)) for Stirling's approximation */
    mutable double logGammaLambdaPlus1_{constants::math::ZERO_DOUBLE};
    
    /** @brief Cached value of 1/λ for efficiency in various calculations */
    mutable double invLambda_{constants::math::ONE};
    
    //==========================================================================
    // OPTIMIZATION FLAGS
    //==========================================================================
    
    /** @brief Atomic cache validity flag for lock-free fast path optimization */
    mutable std::atomic<bool> cacheValidAtomic_{false};
    
    /** @brief True if λ is small (< 10) for direct computation algorithm */
    mutable bool isSmallLambda_{true};
    
    /** @brief True if λ is large (> 100) for normal approximation */
    mutable bool isLargeLambda_{false};
    
    /** @brief True if λ is very large (> 1000) for asymptotic approximations */
    mutable bool isVeryLargeLambda_{false};
    
    /** @brief True if λ is an integer for optimization */
    mutable bool isIntegerLambda_{true};
    
    /** @brief True if λ is very small (< 0.1) for series expansion */
    mutable bool isTinyLambda_{false};

    //==========================================================================
    // COMPUTATIONAL CACHE FOR SMALL LAMBDA
    //==========================================================================
    
    /** @brief Pre-computed factorials for small integers (0! to 20!) */
    static constexpr std::array<double, 21> FACTORIAL_CACHE = {
        1.0,                    // 0!
        1.0,                    // 1!
        2.0,                    // 2!
        6.0,                    // 3!
        24.0,                   // 4!
        120.0,                  // 5!
        720.0,                  // 6!
        5040.0,                 // 7!
        40320.0,                // 8!
        362880.0,               // 9!
        3628800.0,              // 10!
        39916800.0,             // 11!
        479001600.0,            // 12!
        6227020800.0,           // 13!
        87178291200.0,          // 14!
        1307674368000.0,        // 15!
        20922789888000.0,       // 16!
        355687428096000.0,      // 17!
        6402373705728000.0,     // 18!
        121645100408832000.0,   // 19!
        2432902008176640000.0   // 20!
    };

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        logLambda_ = std::log(lambda_);
        expNegLambda_ = std::exp(-lambda_);
        sqrtLambda_ = std::sqrt(lambda_);
        invLambda_ = constants::math::ONE / lambda_;
        
        // Stirling's approximation for log(Γ(λ+1)) = log(λ!)
        logGammaLambdaPlus1_ = std::lgamma(lambda_ + 1.0);
        
        // Optimization flags
        isSmallLambda_ = (lambda_ < 10.0);
        isLargeLambda_ = (lambda_ > 100.0);
        isVeryLargeLambda_ = (lambda_ > 1000.0);
        isIntegerLambda_ = (std::abs(lambda_ - std::round(lambda_)) <= constants::precision::DEFAULT_TOLERANCE);
        isTinyLambda_ = (lambda_ < 0.1);
        
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
    }
    
    /**
     * Validates parameters for the Poisson distribution
     * @param lambda Rate parameter (must be positive and finite)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double lambda) {
        if (std::isnan(lambda) || std::isinf(lambda) || lambda <= constants::math::ZERO_DOUBLE) {
            throw std::invalid_argument("Lambda (rate parameter) must be a positive finite number");
        }
        if (lambda > constants::math::MAX_POISSON_LAMBDA) {
            throw std::invalid_argument("Lambda too large for accurate Poisson computation");
        }
    }

    friend std::istream& operator>>(std::istream& is,
            libstats::PoissonDistribution& distribution);

public:
    //==========================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    
    /**
     * Constructs a Poisson distribution with given rate parameter.
     * 
     * @param lambda Rate parameter λ (must be positive, default: 1.0)
     * @throws std::invalid_argument if lambda is invalid
     * 
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit PoissonDistribution(double lambda = constants::math::ONE);
    
    /**
     * Thread-safe copy constructor
     * 
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    PoissonDistribution(const PoissonDistribution& other);
    
    /**
     * Copy assignment operator
     * 
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    PoissonDistribution& operator=(const PoissonDistribution& other);
    
    /**
     * Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    PoissonDistribution(PoissonDistribution&& other);
    
    /**
     * Move assignment operator (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with deadlock prevention
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    PoissonDistribution& operator=(PoissonDistribution&& other);

    /**
     * @brief Destructor - explicitly defaulted to satisfy Rule of Five
     * Implementation inline: Trivial destruction, kept for performance
     * 
     * Note: C++20 Best Practice - Rule of Five uses complexity-based placement:
     * - Simple operations (destructor) stay inline for performance
     * - Complex operations (copy/move) moved to .cpp for maintainability
     */
    ~PoissonDistribution() override = default;
    
    //==========================================================================
    // SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================
    
    /**
     * @brief Safely create a Poisson distribution without throwing exceptions
     * 
     * This factory method provides exception-free construction to work around
     * ABI compatibility issues with Homebrew LLVM libc++ on macOS where
     * exceptions thrown from the library cause segfaults during unwinding.
     * 
     * @param lambda Rate parameter λ (must be positive)
     * @return Result containing either a valid PoissonDistribution or error info
     * 
     * @par Usage Example:
     * @code
     * auto result = PoissonDistribution::create(2.5);
     * if (result.isOk()) {
     *     auto distribution = std::move(result.value);
     *     // Use distribution safely...
     * } else {
     *     std::cout << "Error: " << result.message << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] static Result<PoissonDistribution> create(double lambda = 1.0) noexcept {
        auto validation = validatePoissonParameters(lambda);
        if (validation.isError()) {
            return Result<PoissonDistribution>::makeError(validation.error_code, validation.message);
        }
        
        // Use private factory to bypass validation
        return Result<PoissonDistribution>::ok(createUnchecked(lambda));
    }
    
    /**
     * @brief Safely try to set parameters without throwing exceptions
     * 
     * @param lambda New rate parameter
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(double lambda) noexcept {
        auto validation = validatePoissonParameters(lambda);
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
        return validatePoissonParameters(lambda_);
    }

    //==========================================================================
    // CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * Computes the probability mass function for the Poisson distribution.
     * 
     * For Poisson distribution: P(X = k) = (λ^k * e^(-λ)) / k! for k = 0, 1, 2, ...
     * Uses different algorithms based on λ size for optimal performance and accuracy.
     * 
     * @param x The value at which to evaluate the PMF (will be rounded to nearest non-negative integer)
     * @return Probability mass for the given value, 0 for negative values
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * Computes the logarithm of the probability mass function for numerical stability.
     * 
     * For Poisson distribution: log(P(X = k)) = k*log(λ) - λ - log(k!) for k = 0, 1, 2, ...
     * Uses log-space computation to avoid overflow for large k or λ.
     * 
     * @param x The value at which to evaluate the log-PMF (will be rounded to nearest non-negative integer)
     * @return Natural logarithm of the probability mass, or -∞ for negative values
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * Evaluates the CDF at x using the regularized incomplete gamma function.
     * 
     * For Poisson distribution: F(k) = Q(k+1, λ) where Q is the regularized gamma function
     * Uses efficient algorithms for different λ ranges.
     * 
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;
    
    /**
     * @brief Computes the quantile function (inverse CDF)
     * 
     * For Poisson distribution: F^(-1)(p) = min{k : F(k) ≥ p}
     * Uses bracketing search with efficient CDF computation.
     * 
     * @param p Probability value in [0,1]
     * @return k such that P(X ≤ k) ≥ p (smallest integer satisfying this)
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;
    
    /**
     * @brief Generate single random sample from distribution
     * 
     * Uses Knuth's algorithm for small λ, acceptance-rejection for large λ
     * 
     * @param rng Random number generator
     * @return Single random sample (integer value as double)
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
     * For Poisson distribution, mean = λ
     * Uses direct parameter access for efficiency.
     * 
     * @return Mean value (equals λ)
     */
    [[nodiscard]] double getMean() const noexcept override;
    
    /**
     * Gets the variance of the distribution.
     * For Poisson distribution, variance = λ
     * Uses direct parameter access for efficiency.
     * 
     * @return Variance value (equals λ)
     */
    [[nodiscard]] double getVariance() const noexcept override;
    
    /**
     * @brief Gets the skewness of the distribution.
     * For Poisson distribution, skewness = 1/√λ
     * Uses cached value to eliminate square root computation.
     * 
     * @return Skewness value (1/√λ)
     */
    [[nodiscard]] double getSkewness() const noexcept override;
    
    /**
     * @brief Gets the kurtosis of the distribution.
     * For Poisson distribution, excess kurtosis = 1/λ
     * Uses cached value to eliminate division.
     * 
     * @return Excess kurtosis value (1/λ)
     */
    [[nodiscard]] double getKurtosis() const noexcept override;
    
    /**
     * @brief Gets the number of parameters for this distribution.
     * For Poisson distribution, there is 1 parameter: lambda (rate)
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
        return "Poisson";
    }
    
    /**
     * @brief Checks if the distribution is discrete.
     * For Poisson distribution, it's discrete
     * Inline for performance - no thread safety needed for constant
     * 
     * @return true (always discrete)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override {
        return true;
    }
    
    /**
     * @brief Gets the lower bound of the distribution support.
     * For Poisson distribution, support is {0, 1, 2, ...}
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Lower bound (0)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return 0.0;
    }
    
    /**
     * @brief Gets the upper bound of the distribution support.
     * For Poisson distribution, support is {0, 1, 2, ...}
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Upper bound (+infinity)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }
    
    /**
     * Gets the mode of the distribution.
     * For Poisson distribution, mode = ⌊λ⌋ (floor of λ)
     * 
     * @return Mode value
     */
    [[nodiscard]] double getMode() const noexcept;

    //==========================================================================
    // DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * Fits the distribution parameters to the given data using maximum likelihood estimation.
     * For Poisson distribution, MLE gives λ = sample_mean.
     * 
     * @param values Vector of observed count data (should be non-negative integers)
     * @throws std::invalid_argument if values is empty or contains negative values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * Resets the distribution to default parameters (λ = 1.0).
     * This corresponds to a standard Poisson distribution.
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
    bool operator==(const PoissonDistribution& other) const;
    
    /**
     * Inequality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const PoissonDistribution& other) const { return !(*this == other); }
    
    //==========================================================================
    // SIMD BATCH OPERATIONS
    //==========================================================================
    
    /**
     * Batch computation of probability masses using SIMD optimization.
     * Processes multiple values simultaneously for improved performance.
     * 
     * @param values Input array of values to evaluate (will be rounded to non-negative integers)
     * @param results Output array for probability masses (must be same size as values)
     * @param count Number of values to process
     * @note This method is optimized for large batch sizes where SIMD overhead is justified
     */
    void getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept;
    
    /**
     * Batch computation of log probability masses using SIMD optimization.
     * Processes multiple values simultaneously for improved performance.
     * 
     * @param values Input array of values to evaluate (will be rounded to non-negative integers)
     * @param results Output array for log probability masses (must be same size as values)
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
    // POISSON-SPECIFIC UTILITY METHODS
    //==========================================================================
    
    /**
     * @brief Generate multiple random integer samples efficiently
     * 
     * Optimized for multiple samples with reduced RNG overhead
     * 
     * @param rng Random number generator
     * @param count Number of samples to generate
     * @return Vector of random integer samples
     */
    [[nodiscard]] std::vector<int> sampleIntegers(std::mt19937& rng, std::size_t count) const;
    
    /**
     * @brief Compute probability of exactly k events
     * 
     * Convenience method for integer values without rounding
     * 
     * @param k Non-negative integer count
     * @return P(X = k)
     */
    [[nodiscard]] double getProbabilityExact(int k) const;
    
    /**
     * @brief Compute log probability of exactly k events
     * 
     * Convenience method for integer values without rounding
     * 
     * @param k Non-negative integer count
     * @return log(P(X = k))
     */
    [[nodiscard]] double getLogProbabilityExact(int k) const noexcept;
    
    /**
     * @brief Compute cumulative probability up to k events
     * 
     * Convenience method for integer values
     * 
     * @param k Non-negative integer count
     * @return P(X ≤ k)
     */
    [[nodiscard]] double getCumulativeProbabilityExact(int k) const;
    
    /**
     * @brief Check if the distribution is suitable for normal approximation
     * 
     * Returns true if λ is large enough (typically λ > 10) for normal approximation
     * 
     * @return true if normal approximation is accurate
     */
    [[nodiscard]] bool canUseNormalApproximation() const noexcept;

private:
    //==========================================================================
    // PRIVATE FACTORY METHODS
    //==========================================================================
    
    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param lambda Rate parameter (assumed valid)
     * @return PoissonDistribution with the given parameter
     */
    static PoissonDistribution createUnchecked(double lambda) noexcept {
        PoissonDistribution dist(lambda, true); // bypass validation
        return dist;
    }
    
    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param lambda Rate parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    PoissonDistribution(double lambda, bool /*bypassValidation*/) noexcept
        : DistributionBase(), lambda_(lambda) {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    
    //==========================================================================
    // PRIVATE COMPUTATIONAL METHODS
    //==========================================================================
    
    /** @brief Compute PMF for small λ using direct method */
    [[nodiscard]] double computePMFSmall(int k) const noexcept;
    
    /** @brief Compute PMF for large λ using Stirling's approximation */
    [[nodiscard]] double computePMFLarge(int k) const noexcept;
    
    /** @brief Compute log PMF for any λ using log-space arithmetic */
    [[nodiscard]] double computeLogPMF(int k) const noexcept;
    
    /** @brief Compute CDF using regularized incomplete gamma function */
    [[nodiscard]] double computeCDF(int k) const noexcept;
    
    /** @brief Fast factorial computation with caching */
    [[nodiscard]] static double factorial(int n) noexcept;
    
    /** @brief Fast log factorial computation using Stirling's approximation */
    [[nodiscard]] static double logFactorial(int n) noexcept;
    
    //==========================================================================
    // PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================
    
    /** @brief Internal implementation for batch PMF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double lambda, double log_lambda, double exp_neg_lambda) const noexcept;
    
    /** @brief Internal implementation for batch log PMF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double lambda, double log_lambda) const noexcept;
    
    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                 double lambda) const noexcept;
    
    //==========================================================================
    // PRIVATE UTILITY METHODS
    //==========================================================================
    
    /** @brief Round double to nearest non-negative integer */
    static int roundToNonNegativeInt(double x) noexcept {
        if (x < 0.0) return 0;
        return static_cast<int>(std::round(x));
    }
    
    /** @brief Check if rounded value is a valid count (non-negative integer) */
    static bool isValidCount(double x) noexcept {
        return (x >= 0.0 && x <= static_cast<double>(INT_MAX));
    }
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const PoissonDistribution& dist);

} // namespace libstats

#endif // LIBSTATS_POISSON_H_

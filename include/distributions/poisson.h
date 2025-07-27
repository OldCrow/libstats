#ifndef LIBSTATS_POISSON_H_
#define LIBSTATS_POISSON_H_

#include "../core/distribution_base.h"
#include "../core/constants.h"
#include "../core/error_handling.h" // Safe error handling without exceptions
#include "../platform/work_stealing_pool.h" // For parallel work-stealing operations
#include "../platform/adaptive_cache.h" // For cache-aware operations
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation
#include <span>        // For std::span interface in parallel operations
#include <tuple>       // For statistical test results
#include <vector>      // For batch operations and data handling

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
        logGammaLambdaPlus1_ = std::lgamma(lambda_ + constants::math::ONE);
        
        // Optimization flags
        isSmallLambda_ = (lambda_ < constants::thresholds::poisson::SMALL_LAMBDA_THRESHOLD);
        isLargeLambda_ = (lambda_ > constants::math::HUNDRED);
        isVeryLargeLambda_ = (lambda_ > constants::math::THOUSAND);
        isIntegerLambda_ = (std::abs(lambda_ - std::round(lambda_)) <= constants::precision::DEFAULT_TOLERANCE);
        isTinyLambda_ = (lambda_ < constants::math::TENTH);
        
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
        if (lambda > constants::thresholds::poisson::MAX_POISSON_LAMBDA) {
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
    PoissonDistribution& operator=(PoissonDistribution&& other) noexcept;

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
    [[nodiscard]] static Result<PoissonDistribution> create(double lambda = constants::math::ONE) noexcept {
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
    // PARALLEL BATCH OPERATIONS
    //==========================================================================
    
    /**
     * @brief Parallel batch PDF computation using std::span interface
     * 
     * Computes probabilities for multiple values using parallel processing.
     * Automatically chooses optimal parallelization strategy based on input size.
     * 
     * @param input_values Input values (std::span for zero-copy)
     * @param output_results Output probabilities (std::span for zero-copy)
     * @throws std::invalid_argument if span sizes don't match
     */
    void getProbabilityBatchParallel(std::span<const double> input_values, 
                                   std::span<double> output_results) const;
    
    /**
     * @brief Parallel batch log-PDF computation using std::span interface
     * 
     * @param input_values Input values (std::span for zero-copy)
     * @param output_results Output log-probabilities (std::span for zero-copy)
     * @throws std::invalid_argument if span sizes don't match
     */
    void getLogProbabilityBatchParallel(std::span<const double> input_values, 
                                      std::span<double> output_results) const;
    
    /**
     * @brief Parallel batch CDF computation using std::span interface
     * 
     * @param input_values Input values (std::span for zero-copy)
     * @param output_results Output cumulative probabilities (std::span for zero-copy)
     * @throws std::invalid_argument if span sizes don't match
     */
    void getCumulativeProbabilityBatchParallel(std::span<const double> input_values, 
                                             std::span<double> output_results) const;
    
    /**
     * @brief Work-stealing parallel batch PDF computation
     * 
     * Uses work-stealing thread pool for dynamic load balancing across threads.
     * Optimal for irregular computational loads or NUMA architectures.
     * 
     * @param input_values Input values
     * @param output_results Output probabilities
     * @param pool Work-stealing thread pool
     */
    void getProbabilityBatchWorkStealing(std::span<const double> input_values,
                                       std::span<double> output_results,
                                       WorkStealingPool& pool) const;
    
    /**
     * @brief Work-stealing parallel batch log-PDF computation
     * 
     * @param input_values Input values
     * @param output_results Output log-probabilities  
     * @param pool Work-stealing thread pool
     */
    void getLogProbabilityBatchWorkStealing(std::span<const double> input_values,
                                          std::span<double> output_results,
                                          WorkStealingPool& pool) const;
    
    /**
     * @brief Work-stealing parallel batch CDF computation
     * 
     * @param input_values Input values
     * @param output_results Output cumulative probabilities
     * @param pool Work-stealing thread pool
     */
    void getCumulativeProbabilityBatchWorkStealing(std::span<const double> input_values,
                                                  std::span<double> output_results,
                                                  WorkStealingPool& pool) const;
    
    /**
     * @brief Cache-aware parallel batch PDF computation
     * 
     * Uses adaptive caching to minimize redundant computations for repeated values.
     * Optimal for datasets with many duplicate or similar values.
     * 
     * @param input_values Input values
     * @param output_results Output probabilities
     * @param cache_manager Adaptive cache for memoization
     */
    template<typename KeyType, typename ValueType>
    void getProbabilityBatchCacheAware(std::span<const double> input_values,
                                     std::span<double> output_results,
                                     cache::AdaptiveCache<KeyType, ValueType>& cache_manager) const;
    
    /**
     * @brief Cache-aware parallel batch log-PDF computation
     * 
     * @param input_values Input values
     * @param output_results Output log-probabilities
     * @param cache_manager Adaptive cache for memoization
     */
    template<typename KeyType, typename ValueType>
    void getLogProbabilityBatchCacheAware(std::span<const double> input_values,
                                        std::span<double> output_results,
                                        cache::AdaptiveCache<KeyType, ValueType>& cache_manager) const;
    
    /**
     * @brief Cache-aware parallel batch CDF computation
     * 
     * @param input_values Input values
     * @param output_results Output cumulative probabilities
     * @param cache_manager Adaptive cache for memoization
     */
    template<typename KeyType, typename ValueType>
    void getCumulativeProbabilityBatchCacheAware(std::span<const double> input_values,
                                                std::span<double> output_results,
                                                cache::AdaptiveCache<KeyType, ValueType>& cache_manager) const;
    
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
    
    //==========================================================================
    // ADVANCED STATISTICAL METHODS
    //==========================================================================
    
    /**
     * @brief Compute confidence interval for rate parameter λ
     * 
     * Uses exact method based on the relationship between Poisson and Chi-square distributions.
     * For observed count data, constructs confidence interval for the underlying rate parameter.
     * 
     * @param data Vector of observed count data
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @return Pair of (lower_bound, upper_bound) for λ
     * @throws std::invalid_argument if confidence_level not in (0,1) or data empty
     */
    [[nodiscard]] static std::pair<double, double> confidenceIntervalRate(
        const std::vector<double>& data, double confidence_level = 0.95);
    
    /**
     * @brief Likelihood ratio test for rate parameter
     * 
     * Tests H0: λ = λ0 vs H1: λ ≠ λ0 using likelihood ratio statistic.
     * 
     * @param data Vector of observed count data
     * @param lambda0 Null hypothesis value for λ
     * @param significance_level Significance level for test
     * @return Tuple of (test_statistic, p_value, reject_null)
     */
    [[nodiscard]] static std::tuple<double, double, bool> likelihoodRatioTest(
        const std::vector<double>& data, double lambda0, double significance_level = 0.05);
    
    /**
     * @brief Method of moments estimation
     * 
     * For Poisson distribution, method of moments estimator is simply the sample mean.
     * Included for completeness and consistency with other distributions.
     * 
     * @param data Vector of observed count data
     * @return Estimated λ parameter (sample mean)
     * @throws std::invalid_argument if data is empty
     */
    [[nodiscard]] static double methodOfMomentsEstimation(const std::vector<double>& data);
    
    /**
     * @brief Bayesian estimation with conjugate Gamma prior
     * 
     * Uses Gamma(α, β) prior for λ. Posterior is Gamma(α + Σx_i, β + n).
     * Returns posterior parameters for the conjugate Gamma distribution.
     * 
     * @param data Vector of observed count data
     * @param prior_shape Prior shape parameter α (default: 1.0)
     * @param prior_rate Prior rate parameter β (default: 1.0)
     * @return Pair of (posterior_shape, posterior_rate)
     */
    [[nodiscard]] static std::pair<double, double> bayesianEstimation(
        const std::vector<double>& data, double prior_shape = 1.0, double prior_rate = 1.0);
    
    //==========================================================================
    // GOODNESS-OF-FIT TESTS
    //==========================================================================
    
    /**
     * @brief Chi-square goodness-of-fit test for Poisson distribution
     * 
     * Tests whether observed data follows the specified Poisson distribution.
     * Groups rare events to ensure expected frequencies ≥ 5 for valid chi-square test.
     * 
     * @param data Vector of observed count data
     * @param distribution Hypothesized Poisson distribution
     * @param significance_level Significance level for test
     * @return Tuple of (chi_square_statistic, p_value, reject_null)
     */
    [[nodiscard]] static std::tuple<double, double, bool> chiSquareGoodnessOfFit(
        const std::vector<double>& data, const PoissonDistribution& distribution, 
        double significance_level = 0.05);
    
    /**
     * @brief Kolmogorov-Smirnov test adapted for discrete distributions
     * 
     * Tests goodness-of-fit using the maximum difference between empirical
     * and theoretical CDFs, with adjustments for discrete distributions.
     * 
     * @param data Vector of observed count data
     * @param distribution Hypothesized Poisson distribution
     * @param significance_level Significance level for test
     * @return Tuple of (ks_statistic, p_value, reject_null)
     */
    [[nodiscard]] static std::tuple<double, double, bool> kolmogorovSmirnovTest(
        const std::vector<double>& data, const PoissonDistribution& distribution,
        double significance_level = 0.05);
    
    //==========================================================================
    // CROSS-VALIDATION METHODS
    //==========================================================================
    
    /**
     * @brief K-fold cross-validation for model assessment
     * 
     * Splits data into k folds, fits Poisson distribution to k-1 folds,
     * and evaluates on the remaining fold. Reports performance metrics.
     * 
     * @param data Vector of observed count data
     * @param k Number of folds (default: 5)
     * @param random_seed Seed for random fold assignment
     * @return Vector of (mae, rmse, log_likelihood) for each fold
     */
    [[nodiscard]] static std::vector<std::tuple<double, double, double>> kFoldCrossValidation(
        const std::vector<double>& data, int k = 5, unsigned int random_seed = 42);
    
    /**
     * @brief Leave-one-out cross-validation
     * 
     * Fits Poisson distribution to n-1 data points and evaluates on the left-out point.
     * Repeats for all data points and reports aggregate metrics.
     * 
     * @param data Vector of observed count data
     * @return Tuple of (mean_absolute_error, rmse, total_log_likelihood)
     */
    [[nodiscard]] static std::tuple<double, double, double> leaveOneOutCrossValidation(
        const std::vector<double>& data);
    
    //==========================================================================
    // INFORMATION CRITERIA
    //==========================================================================
    
    /**
     * @brief Compute information criteria for model selection
     * 
     * Calculates AIC, BIC, and AICc for the fitted Poisson distribution.
     * Lower values indicate better model fit with appropriate complexity penalty.
     * 
     * @param data Vector of observed count data
     * @param distribution Fitted Poisson distribution
     * @return Tuple of (AIC, BIC, AICc, log_likelihood)
     */
    [[nodiscard]] static std::tuple<double, double, double, double> computeInformationCriteria(
        const std::vector<double>& data, const PoissonDistribution& distribution);
    
    //==========================================================================
    // BOOTSTRAP METHODS
    //==========================================================================
    
    /**
     * @brief Bootstrap confidence intervals for parameter λ
     * 
     * Uses bootstrap resampling to construct confidence intervals for the rate parameter.
     * More robust than asymptotic methods for small sample sizes.
     * 
     * @param data Vector of observed count data
     * @param confidence_level Confidence level (e.g., 0.95)
     * @param num_bootstrap_samples Number of bootstrap samples
     * @param random_seed Seed for reproducible results
     * @return Confidence interval (lower_bound, upper_bound) for λ
     */
    [[nodiscard]] static std::pair<double, double> bootstrapParameterConfidenceIntervals(
        const std::vector<double>& data, double confidence_level = 0.95,
        int num_bootstrap_samples = 1000, unsigned int random_seed = 42);

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

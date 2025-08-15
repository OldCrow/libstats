#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "../core/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools, adaptive caching, etc.)
#include "distribution_platform_common.h"

namespace libstats {

/**
 * @brief Thread-safe Discrete Uniform Distribution for modeling equiprobable integer outcomes.
 * 
 * @details The Discrete Uniform distribution is a discrete probability distribution where
 * each integer value in a finite range has equal probability. This is the discrete
 * analog of the continuous uniform distribution and models situations where all
 * outcomes are equally likely (fair dice, card draws, etc.).
 * 
 * @par Mathematical Definition:
 * - PMF: P(X = k) = 1/(b-a+1) for k ∈ {a, a+1, ..., b}, 0 otherwise
 * - CDF: F(k) = (⌊k⌋ - a + 1)/(b - a + 1) for k ∈ {a, a+1, ..., b}
 * - Parameters: a, b ∈ ℤ with a ≤ b (lower and upper bounds, inclusive)
 * - Support: k ∈ {a, a+1, a+2, ..., b}
 * - Mean: (a + b)/2
 * - Variance: (b - a + 1)² - 1)/12 = ((b - a)(b - a + 2))/12
 * - Mode: Not unique (any value in {a, a+1, ..., b})
 * - Median: (a + b)/2 (if b - a is even), otherwise either ⌊(a+b)/2⌋ or ⌈(a+b)/2⌉
 * 
 * @par Key Properties:
 * - **Maximum Entropy**: Among all discrete distributions with given support
 * - **Symmetric**: Distribution is symmetric around the midpoint (a + b)/2
 * - **Memoryless on Subsets**: Conditional distribution on subsets remains uniform
 * - **Integer Support**: All values are integers within the specified range
 * - **Finite Support**: Unlike continuous uniform, has finite discrete support
 * - **Equal Probability**: Each outcome has probability 1/(b-a+1)
 * 
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - Concurrent reads are optimized with std::shared_mutex
 * - Cache invalidation uses atomic operations for lock-free fast paths
 * - Deadlock prevention via ordered lock acquisition with std::lock()
 * 
 * @par Performance Features:
 * - Atomic cache validity flags for lock-free fast path access
 * - Extensive caching of computed values (range, probability, midpoint, variance)
 * - Optimized PMF/CDF computation with integer arithmetic
 * - Fast parameter validation with integer bounds checking
 * - Special optimizations for common ranges (dice, cards, binary outcomes)
 * 
 * @par Usage Examples:
 * @code
 * // Standard six-sided die [1,6]
 * auto diceResult = DiscreteDistribution::create(1, 6);
 * if (diceResult.isOk()) {
 *     auto dice = std::move(diceResult.value);
 *     
 *     // Playing card ranks [1,13] (Ace to King)
 *     auto cardResult = DiscreteDistribution::create(1, 13);
 *     if (cardResult.isOk()) {
 *         auto cards = std::move(cardResult.value);
 *         
 *         // Fit to observed roll data
 *         std::vector<double> rolls = {1, 3, 6, 2, 4, 5, 1, 6, 3, 2};
 *         dice.fit(rolls);
 *         
 *         // Probability of rolling 4 or higher
 *         double highRoll = 1.0 - dice.getCumulativeProbability(3);
 *         
 *         // Generate random roll
 *         std::mt19937 rng(42);
 *         int roll = static_cast<int>(dice.sample(rng));
 *     }
 * }
 * @endcode
 * 
 * @par Applications:
 * - Gaming and gambling (dice, cards, roulette)
 * - Survey responses with equal-weighted options
 * - Computer science (array indexing, hash functions)
 * - Quality control (discrete defect categories)
 * - Digital communications (symbol alphabets)
 * - Combinatorics and permutations
 * - Random sampling without replacement
 * - Discrete optimization problems
 * 
 * @par Statistical Properties:
 * - Skewness: 0 (perfectly symmetric)
 * - Kurtosis: -6(b-a+1)²+1)/(5((b-a+1)²-1)) (approximately -1.2 for large ranges)
 * - Entropy: log(b - a + 1)
 * - Moment generating function: (e^(at) - e^((b+1)t))/((b-a+1)(1-e^t)) for t ≠ 0
 * - Probability generating function: (z^a - z^(b+1))/((b-a+1)(1-z)) for z ≠ 1
 * 
 * @par Special Cases:
 * - **Binary/Bernoulli**: DiscreteUniform(0,1) for coin flips
 * - **Standard Die**: DiscreteUniform(1,6) for six-sided dice
 * - **Byte Values**: DiscreteUniform(0,255) for 8-bit integers
 * - **Percent**: DiscreteUniform(0,100) for integer percentages
 * - **Index Range**: DiscreteUniform(0,n-1) for array indexing
 * 
 * @par Numerical Considerations:
 * - Integer arithmetic prevents floating-point precision issues
 * - Efficient handling of large ranges (up to int32 limits)
 * - Optimized floor/ceiling operations for CDF calculations
 * - Robust parameter validation for integer bounds
 * - Fast modular arithmetic for sampling
 * 
 * @par Implementation Details (C++20 Best Practices):
 * - Complex constructors/operators moved to .cpp for faster compilation
 * - Exception-safe design with RAII principles
 * - Optimized parameter validation with comprehensive error messages
 * - Lock-free fast paths using atomic operations
 * - Integer-optimized algorithms for discrete operations
 * 
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
class DiscreteDistribution : public DistributionBase
{
public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    
    /**
     * @brief Constructs a Discrete Uniform distribution with given bounds.
     * 
     * @param a Lower bound (default: 0)
     * @param b Upper bound (default: 1, must be > a)
     * @throws std::invalid_argument if parameters are invalid
     * 
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit DiscreteDistribution(int a = 0, int b = 1);
    
    /**
     * @brief Thread-safe copy constructor
     * 
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    DiscreteDistribution(const DiscreteDistribution& other);
    
    /**
     * @brief Copy assignment operator
     * 
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    DiscreteDistribution& operator=(const DiscreteDistribution& other);
    
    /**
     * @brief Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    DiscreteDistribution(DiscreteDistribution&& other);
    
    /**
     * @brief Move assignment operator (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with deadlock prevention
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    DiscreteDistribution& operator=(DiscreteDistribution&& other);

    /**
     * @brief Destructor - explicitly defaulted to satisfy Rule of Five
     * Implementation inline: Trivial destruction, kept for performance
     * 
     * Note: C++20 Best Practice - Rule of Five uses complexity-based placement:
     * - Simple operations (destructor) stay inline for performance
     * - Complex operations (copy/move) moved to .cpp for maintainability
     */
    ~DiscreteDistribution() override = default;
    
    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================
    
    /**
     * @brief Safely create a Discrete Uniform distribution without throwing exceptions
     * 
     * This factory method provides exception-free construction to work around
     * ABI compatibility issues with Homebrew LLVM libc++ on macOS where
     * exceptions thrown from the library cause segfaults during unwinding.
     * 
     * @param a Lower bound parameter (inclusive)
     * @param b Upper bound parameter (inclusive, must be >= a)
     * @return Result containing either a valid DiscreteDistribution or error info
     * 
     * @par Usage Example:
     * @code
     * auto result = DiscreteDistribution::create(1, 6);  // Standard die
     * if (result.isOk()) {
     *     auto dice = std::move(result.value);
     *     // Use distribution safely...
     * } else {
     *     std::cout << "Error: " << result.message << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] static Result<DiscreteDistribution> create(int a = 0, int b = 1) noexcept {
        auto validation = validateDiscreteParameters(a, b);
        if (validation.isError()) {
            return Result<DiscreteDistribution>::makeError(validation.error_code, validation.message);
        }
        
        // Use private factory to bypass validation
        return Result<DiscreteDistribution>::ok(createUnchecked(a, b));
    }
    
    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /**
     * Gets the lower bound parameter a.
     * Thread-safe: acquires shared lock to protect a_
     *
     * @return Current lower bound value
     */
    [[nodiscard]] int getLowerBound() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return a_;
    }
    
    /**
     * Gets the upper bound parameter b.
     * Thread-safe: acquires shared lock to protect b_
     *
     * @return Current upper bound value
     */
    [[nodiscard]] int getUpperBound() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return b_;
    }
    
    /**
     * @brief Fast lock-free atomic getter for lower bound parameter a
     * 
     * Provides high-performance access to the lower bound parameter using atomic operations
     * for lock-free fast path. Falls back to locked getter if atomic parameters
     * are not valid (e.g., during parameter updates).
     * 
     * @return Current lower bound parameter value
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
     *     int lower = dist.getLowerBoundAtomic();  // Lock-free access
     *     results[i] = compute_something(data[i], lower);
     * }
     * @endcode
     */
    [[nodiscard]] int getLowerBoundAtomic() const noexcept {
        // Fast path: check if atomic parameters are valid
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            // Lock-free atomic access with proper memory ordering
            return atomicA_.load(std::memory_order_acquire);
        }
        
        // Fallback: use traditional locked getter if atomic parameters are stale
        return getLowerBound();
    }
    
    /**
     * @brief Fast lock-free atomic getter for upper bound parameter b
     * 
     * Provides high-performance access to the upper bound parameter using atomic operations
     * for lock-free fast path. Falls back to locked getter if atomic parameters
     * are not valid (e.g., during parameter updates).
     * 
     * @return Current upper bound parameter value
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
     *     int upper = dist.getUpperBoundAtomic();  // Lock-free access
     *     results[i] = compute_something(data[i], upper);
     * }
     * @endcode
     */
    [[nodiscard]] int getUpperBoundAtomic() const noexcept {
        // Fast path: check if atomic parameters are valid
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            // Lock-free atomic access with proper memory ordering
            return atomicB_.load(std::memory_order_acquire);
        }
        
        // Fallback: use traditional locked getter if atomic parameters are stale
        return getUpperBound();
    }
    
    /**
     * Sets the lower bound parameter a.
     *
     * @param a New lower bound (must be <= current upper bound)
     * @throws std::invalid_argument if a > b or parameters are invalid
     */
    void setLowerBound(int a);
    
    /**
     * Sets the upper bound parameter b.
     *
     * @param b New upper bound (must be >= current lower bound)
     * @throws std::invalid_argument if b < a or parameters are invalid
     */
    void setUpperBound(int b);
    
    /**
     * Sets both bounds simultaneously.
     *
     * @param a New lower bound
     * @param b New upper bound (must be >= a)
     * @throws std::invalid_argument if parameters are invalid
     */
    void setBounds(int a, int b);
    
    /**
     * @brief Sets both parameters simultaneously (exception-based API).
     * Thread-safe: acquires unique lock for cache invalidation
     *
     * @param a New lower bound parameter (inclusive)
     * @param b New upper bound parameter (inclusive, must be >= a)
     * @throws std::invalid_argument if parameters are invalid
     */
    void setParameters(int a, int b);
    
    /**
     * Gets the mean of the distribution.
     * For Discrete Uniform distribution, mean = (a + b)/2
     * Uses cached value to eliminate addition and division.
     *
     * @return Mean value
     */
    [[nodiscard]] double getMean() const noexcept override;
    
    /**
     * Gets the variance of the distribution.
     * For Discrete Uniform distribution, variance = ((b-a)(b-a+2))/12
     * Uses cached value to eliminate multiplications and divisions.
     *
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const noexcept override;
    
    /**
     * @brief Gets the skewness of the distribution.
     * For Discrete Uniform distribution, skewness = 0 (perfectly symmetric)
     * Inline for performance - no thread safety needed for constant
     *
     * @return Skewness value (always 0)
     */
    [[nodiscard]] double getSkewness() const noexcept override {
        return 0.0;  // Discrete uniform distribution is perfectly symmetric
    }
    
    /**
     * @brief Gets the kurtosis of the distribution.
     * For Discrete Uniform distribution, excess kurtosis ≈ -1.2 for large ranges
     * Inline for performance - no thread safety needed for constant
     *
     * @return Excess kurtosis value (approximately -1.2)
     */
    [[nodiscard]] double getKurtosis() const noexcept override {
        return -1.2;  // Approximate excess kurtosis for discrete uniform
    }
    
    /**
     * @brief Gets the number of parameters for this distribution.
     * For Discrete Uniform distribution, there are 2 parameters: a (lower) and b (upper)
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
        return "Discrete";
    }
    
    /**
     * @brief Checks if the distribution is discrete.
     * For Discrete Uniform distribution, it's discrete
     * Inline for performance - no thread safety needed for constant
     *
     * @return true (always discrete)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override {
        return true;
    }
    
    /**
     * @brief Gets the lower bound of the distribution support.
     * For Discrete Uniform distribution, support is {a, a+1, ..., b}
     *
     * @return Lower bound (parameter a as double)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return static_cast<double>(a_);
    }
    
    /**
     * @brief Gets the upper bound of the distribution support.
     * For Discrete Uniform distribution, support is {a, a+1, ..., b}
     *
     * @return Upper bound (parameter b as double)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return static_cast<double>(b_);
    }
    
    /**
     * Gets the range of the distribution (b - a + 1).
     * This is the number of possible outcomes.
     * Uses cached value to eliminate arithmetic.
     *
     * @return Range of the distribution
     */
    [[nodiscard]] int getRange() const noexcept;
    
    /**
     * Gets the probability of any single outcome.
     * For discrete uniform, each outcome has probability 1/(b-a+1).
     * Uses cached value to eliminate division.
     *
     * @return Probability of any single outcome
     */
    [[nodiscard]] double getSingleOutcomeProbability() const noexcept;
    
    /**
     * Gets the mode of the distribution.
     * For Discrete Uniform distribution, mode is not unique - any value in {a, a+1, ..., b} is a mode.
     * This method returns the midpoint (a + b) / 2.0 as a representative mode.
     *
     * @return Mode value (midpoint of the range)
     */
    [[nodiscard]] double getMode() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (static_cast<double>(a_) + static_cast<double>(b_)) / 2.0;
    }
    
    /**
     * Gets the median of the distribution.
     * For Discrete Uniform distribution, median is (a + b) / 2.0.
     * If the range has an even number of elements, this is the average of the two middle values.
     *
     * @return Median value
     */
    [[nodiscard]] double getMedian() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return (static_cast<double>(a_) + static_cast<double>(b_)) / 2.0;
    }
    
    //==============================================================================
    // 4. RESULT-BASED SETTERS
    //==============================================================================
    
    /**
     * @brief Safely try to set lower bound without throwing exceptions
     *
     * @param a New lower bound parameter (must be <= current upper bound)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetLowerBound(int a) noexcept;
    
    /**
     * @brief Safely try to set upper bound without throwing exceptions
     *
     * @param b New upper bound parameter (must be >= current lower bound)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetUpperBound(int b) noexcept;
    
    /**
     * @brief Safely try to set both bounds without throwing exceptions
     *
     * @param a New lower bound parameter (inclusive)
     * @param b New upper bound parameter (inclusive, must be >= a)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetBounds(int a, int b) noexcept;
    
    /**
     * @brief Safely try to set parameters without throwing exceptions
     *
     * @param a New lower bound parameter (inclusive)
     * @param b New upper bound parameter (inclusive, must be >= a)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(int a, int b) noexcept;
    
    /**
     * @brief Check if current parameters are valid
     * @return VoidResult indicating validity
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return validateDiscreteParameters(a_, b_);
    }

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * Computes the probability mass function for the Discrete Uniform distribution.
     * 
     * For discrete uniform distribution: P(X = k) = 1/(b-a+1) for k ∈ {a, a+1, ..., b}, 0 otherwise
     * 
     * @param x The value at which to evaluate the PMF (will be rounded to nearest integer)
     * @return Probability mass (constant 1/(b-a+1) for integers in support, 0 otherwise)
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * Computes the logarithm of the probability mass function for numerical stability.
     * 
     * For discrete uniform distribution: log(P(X = k)) = -log(b-a+1) for k ∈ {a, a+1, ..., b}, -∞ otherwise
     * 
     * @param x The value at which to evaluate the log-PMF (will be rounded to nearest integer)
     * @return Natural logarithm of the probability mass, or -∞ for values outside support
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * Evaluates the CDF at x using the discrete uniform CDF formula.
     * 
     * For discrete uniform distribution: F(k) = (⌊k⌋ - a + 1)/(b - a + 1) for k ∈ {a, a+1, ..., b}
     * 
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;
    
    /**
     * @brief Computes the quantile function (inverse CDF)
     * 
     * For discrete uniform distribution: F^(-1)(p) = a + ⌊p(b-a+1)⌋
     * 
     * @param p Probability value in [0,1]
     * @return k such that P(X ≤ k) ≥ p (smallest integer satisfying this)
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;
    
    /**
     * @brief Generate single random sample from distribution
     * 
     * Uses efficient integer arithmetic: X = a + (uniform_int(0, b-a))
     * 
     * @param rng Random number generator
     * @return Single random integer sample (as double for interface compatibility)
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;
    
    /**
     * @brief Generate multiple random samples from distribution
     * 
     * Optimized batch sampling with reduced RNG overhead
     * 
     * @param rng Random number generator
     * @param n Number of samples to generate
     * @return Vector of random integer samples (as doubles for interface compatibility)
     */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * Fits the distribution parameters to the given data.
     * For Discrete Uniform distribution, uses sample minimum and maximum as bounds.
     * 
     * @param values Vector of observed data (will be rounded to nearest integers)
     * @throws std::invalid_argument if values is empty or contains invalid data
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple datasets
     * Efficiently fits discrete distribution parameters to multiple independent datasets in parallel
     * 
     * @param datasets Vector of datasets, each representing independent observations
     * @param results Vector to store fitted DiscreteDistribution objects
     * @throws std::invalid_argument if datasets is empty or results size doesn't match
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                               std::vector<DiscreteDistribution>& results);

    /**
     * Resets the distribution to default parameters (a = 0, b = 1).
     * This corresponds to a binary uniform distribution.
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
     * @brief Confidence interval for lower bound a
     *
     * Computes confidence interval for lower bound using order statistics.
     * Utilizes exact distribution of the minimum for discrete uniform distributions.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @return Pair of (lower_bound, upper_bound) for a
     * @throws std::invalid_argument if confidence_level not in (0,1) or data empty/invalid
     */
    [[nodiscard]] static std::pair<int, int> confidenceIntervalLowerBound(
        const std::vector<double>& data, double confidence_level = 0.95);
    
    /**
     * @brief Confidence interval for upper bound b
     *
     * Computes confidence interval for upper bound using order statistics.
     * Utilizes exact distribution of the maximum for discrete uniform distributions.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @return Pair of (lower_bound, upper_bound) for b
     * @throws std::invalid_argument if confidence_level not in (0,1) or data empty/invalid
     */
    [[nodiscard]] static std::pair<int, int> confidenceIntervalUpperBound(
        const std::vector<double>& data, double confidence_level = 0.95);
    
    /**
     * @brief Likelihood ratio test for Discrete Uniform bounds
     *
     * Tests H0: (a, b) = (a₀, b₀) vs H1: (a, b) ≠ (a₀, b₀) using likelihood ratio statistic.
     * Adapted for discrete case with exact computation of likelihood ratios.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param null_a Null hypothesis value for a lower bound
     * @param null_b Null hypothesis value for b upper bound
     * @param significance_level Significance level for test
     * @return Tuple of (test_statistic, p_value, reject_null)
     */
    [[nodiscard]] static std::tuple<double, double, bool> likelihoodRatioTest(
        const std::vector<double>& data, int null_a, int null_b, double significance_level = 0.05);
    
    /**
     * @brief Bayesian estimation for Discrete Uniform bounds
     *
     * Uses Beta-like conjugate priors for discrete uniform bounds estimation.
     * Returns posterior parameters as distributions for both bounds.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param prior_a_alpha Prior alpha for a (default: 1.0)
     * @param prior_a_beta Prior beta for a (default: 1.0)
     * @param prior_b_alpha Prior alpha for b (default: 1.0)
     * @param prior_b_beta Prior beta for b (default: 1.0)
     * @return Pair of (posterior_a_interval, posterior_b_interval)
     */
    [[nodiscard]] static std::pair<std::pair<double, double>, std::pair<double, double>> bayesianEstimation(
        const std::vector<double>& data, double prior_a_alpha = 1.0, double prior_a_beta = 1.0,
        double prior_b_alpha = 1.0, double prior_b_beta = 1.0);
    
    /**
     * @brief Robust estimation using mode-based methods
     *
     * Provides robust estimation of Discrete Uniform bounds using discrete-specific methods.
     * Uses mode ranges and frequency analysis for outlier resistance.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param estimator_type Type of robust estimator ("mode_range", "frequency_trim")
     * @param trim_proportion Proportion to trim (default: 0.1)
     * @return Pair of (robust_a_estimate, robust_b_estimate)
     */
    [[nodiscard]] static std::pair<int, int> robustEstimation(
        const std::vector<double>& data, const std::string& estimator_type = "mode_range",
        double trim_proportion = 0.1);
    
    /**
     * @brief Method of moments estimation
     *
     * Estimates Discrete Uniform bounds by matching sample moments with theoretical moments:
     * For discrete uniform on {a, a+1, ..., b}:
     * - Mean = (a + b) / 2
     * - Variance = (b - a + 1)² - 1) / 12
     *
     * @param data Vector of observed integer data (as doubles)
     * @return Pair of (a_estimate, b_estimate)
     * @throws std::invalid_argument if data is empty or invalid
     */
    [[nodiscard]] static std::pair<int, int> methodOfMomentsEstimation(
        const std::vector<double>& data);
    
    /**
     * @brief Bayesian credible interval from posterior distributions
     *
     * Calculates Bayesian credible intervals for lower and upper bounds
     * from their posterior distributions after observing discrete data.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param credibility_level Credibility level (e.g., 0.95 for 95%)
     * @param prior_a_alpha Prior alpha for a parameter (default: 1.0)
     * @param prior_a_beta Prior beta for a parameter (default: 1.0)
     * @param prior_b_alpha Prior alpha for b parameter (default: 1.0)
     * @param prior_b_beta Prior beta for b parameter (default: 1.0)
     * @return Tuple of ((a_CI_lower, a_CI_upper), (b_CI_lower, b_CI_upper))
     */
    [[nodiscard]] static std::tuple<std::pair<double, double>, std::pair<double, double>> bayesianCredibleInterval(
        const std::vector<double>& data, double credibility_level = 0.95,
        double prior_a_alpha = 1.0, double prior_a_beta = 1.0,
        double prior_b_alpha = 1.0, double prior_b_beta = 1.0);
    
    /**
     * @brief L-moments parameter estimation
     *
     * Uses L-moments (linear combinations of order statistics) for robust
     * parameter estimation. For discrete uniform on {a,...,b}: uses sample order statistics.
     *
     * @param data Vector of observed integer data (as doubles)
     * @return Pair of (a_estimate, b_estimate)
     */
    [[nodiscard]] static std::pair<int, int> lMomentsEstimation(
        const std::vector<double>& data);
    
    /**
     * @brief Discrete uniformity test using chi-square goodness-of-fit
     *
     * Tests whether the data follows a discrete uniform distribution.
     * Uses exact discrete probability calculations for small ranges.
     *
     * @param data Vector of observed integer data (as doubles)
     * @param significance_level Significance level for test
     * @return Tuple of (test_statistic, p_value, uniformity_is_valid)
     */
    [[nodiscard]] static std::tuple<double, double, bool> discreteUniformityTest(
        const std::vector<double>& data, double significance_level = 0.05);
    
    
    //==========================================================================
    // 8. GOODNESS-OF-FIT TESTS
    //==========================================================================
    
    /**
     * @brief Kolmogorov-Smirnov goodness-of-fit test
     *
     * Tests the null hypothesis that data follows the specified discrete distribution.
     * Note: KS test is generally less appropriate for discrete data than chi-squared.
     *
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (KS_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> kolmogorovSmirnovTest(
        const std::vector<double>& data,
        const DiscreteDistribution& distribution,
        double alpha = 0.05);
    
    /**
     * @brief Anderson-Darling goodness-of-fit test for discrete distributions
     *
     * Tests the null hypothesis that data follows the specified discrete distribution.
     * More sensitive to tail differences than the KS test and adapted for discrete cases.
     *
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (AD_statistic, p_value, reject_null)
     * @note Uses asymptotic p-value approximation adjusted for discrete distributions
     */
    static std::tuple<double, double, bool> andersonDarlingTest(
        const std::vector<double>& data,
        const DiscreteDistribution& distribution,
        double alpha = 0.05);
    
    /**
     * @brief Chi-squared goodness-of-fit test for discrete distributions
     *
     * Tests the null hypothesis that observed data follows the specified discrete distribution.
     * Particularly appropriate for discrete distributions.
     *
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (chi_squared_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> chiSquaredGoodnessOfFitTest(
        const std::vector<double>& data,
        const DiscreteDistribution& distribution,
        double alpha = 0.05);
    
    //==========================================================================
    // 9. CROSS-VALIDATION METHODS
    //==========================================================================
    
    /**
     * @brief K-fold cross-validation for parameter estimation
     *
     * Performs k-fold cross-validation to assess parameter estimation quality
     * and model stability for discrete distributions.
     *
     * @param data Sample data for cross-validation
     * @param k Number of folds (default: 5)
     * @param random_seed Seed for random fold assignment (default: 42)
     * @return Vector of k validation results: (mean_error, std_error, log_likelihood)
     */
    static std::vector<std::tuple<double, double, double>> kFoldCrossValidation(
        const std::vector<double>& data,
        int k = 5,
        unsigned int random_seed = 42);
    
    /**
     * @brief Leave-one-out cross-validation for parameter estimation
     *
     * Performs leave-one-out cross-validation (LOOCV) to assess parameter
     * estimation quality for discrete distributions.
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
     * @param fitted_distribution The fitted discrete distribution
     * @return Tuple of (AIC, BIC, AICc, log_likelihood)
     */
    static std::tuple<double, double, double, double> computeInformationCriteria(
        const std::vector<double>& data,
        const DiscreteDistribution& fitted_distribution);
    
    //==========================================================================
    // 11. BOOTSTRAP METHODS
    //==========================================================================
    
    /**
     * @brief Bootstrap parameter confidence intervals
     *
     * Uses bootstrap resampling to estimate confidence intervals for
     * the discrete distribution parameters (lower and upper bounds).
     *
     * @param data Sample data for bootstrap resampling
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @param n_bootstrap Number of bootstrap samples (default: 1000)
     * @param random_seed Seed for random sampling (default: 42)
     * @return Tuple of ((lower_bound_CI_lower, lower_bound_CI_upper), (upper_bound_CI_lower, upper_bound_CI_upper))
     */
    static std::tuple<std::pair<double, double>, std::pair<double, double>> bootstrapParameterConfidenceIntervals(
        const std::vector<double>& data,
        double confidence_level = 0.95,
        int n_bootstrap = 1000,
        unsigned int random_seed = 42);
    
    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
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
     * @brief Check if a value is in the support of the distribution
     *
     * @param x Value to check (will be rounded to nearest integer)
     * @return true if the rounded value is in {a, a+1, ..., b}
     */
    [[nodiscard]] bool isInSupport(double x) const noexcept;
    
    /**
     * @brief Get all possible outcomes as a vector
     *
     * @return Vector containing all integers from a to b (inclusive)
     * @warning Only call for small ranges to avoid memory issues
     */
    [[nodiscard]] std::vector<int> getAllOutcomes() const;
    
    

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================

    /**
     * @brief Smart auto-dispatch batch probability calculation
     * 
     * This method automatically selects the optimal execution strategy based on:
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
    void getProbability(std::span<const double> values, std::span<double> results, const performance::PerformanceHint& hint = {}) const;

    /**
     * @brief Smart auto-dispatch batch log probability calculation
     * 
     * Automatically selects optimal execution strategy for log probability computation.
     * 
     * @param values Input values to evaluate
     * @param results Output array for log probability densities
     * @param hint Optional performance hints for advanced users
     */
    void getLogProbability(std::span<const double> values, std::span<double> results, const performance::PerformanceHint& hint = {}) const;

    /**
     * @brief Smart auto-dispatch batch cumulative probability calculation
     * 
     * Automatically selects optimal execution strategy for CDF computation.
     * 
     * @param values Input values to evaluate
     * @param results Output array for cumulative probabilities
     * @param hint Optional performance hints for advanced users
     */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results, const performance::PerformanceHint& hint = {}) const;

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
                                   performance::Strategy strategy) const;

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
                                      performance::Strategy strategy) const;

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
    void getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                             performance::Strategy strategy) const;

    
    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================
    
    /**
     * Equality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are equal
     */
    bool operator==(const DiscreteDistribution& other) const;
    
    /**
     * Inequality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const DiscreteDistribution& other) const { return !(*this == other); }

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================
    
    /**
     * @brief Stream input operator
     * @param is Input stream
     * @param dist Distribution to input
     * @return Reference to the input stream
     */
    friend std::istream& operator>>(std::istream& is, libstats::DiscreteDistribution& dist);
    
    /**
     * @brief Stream output operator
     * @param os Output stream
     * @param dist Distribution to output
     * @return Reference to the output stream
     */
    friend std::ostream& operator<<(std::ostream& os, const libstats::DiscreteDistribution& dist);

private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================
    
    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param a Lower bound parameter (assumed valid)
     * @param b Upper bound parameter (assumed valid)
     * @return DiscreteDistribution with the given parameters
     */
    static DiscreteDistribution createUnchecked(int a, int b) noexcept {
        DiscreteDistribution dist(a, b, true); // bypass validation
        return dist;
    }
    
    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param a Lower bound parameter (assumed valid)
     * @param b Upper bound parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    DiscreteDistribution(int a, int b, bool /*bypassValidation*/) noexcept
        : DistributionBase(), a_(a), b_(b) {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    
    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================
    
    /** @brief Internal implementation for batch PMF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       int a, int b, double probability) const noexcept;
    
    /** @brief Internal implementation for batch log PMF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          int a, int b, double log_probability) const noexcept;
    
    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                 int a, int b, double inv_range) const noexcept;
    
    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================
    
    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        range_ = b_ - a_ + 1;
        probability_ = 1.0 / static_cast<double>(range_);
        mean_ = (static_cast<double>(a_) + static_cast<double>(b_)) / 2.0;
        
        // Variance for discrete uniform: ((b-a)(b-a+2))/12
        const double width = static_cast<double>(b_ - a_);
        variance_ = (width * (width + 2.0)) / 12.0;
        
        logProbability_ = -std::log(static_cast<double>(range_));
        
        // Optimization flags
        isBinary_ = (a_ == 0 && b_ == 1);
        isStandardDie_ = (a_ == 1 && b_ == 6);
        isSymmetric_ = (a_ == -b_);
        isSmallRange_ = (range_ <= 10);
        isLargeRange_ = (range_ > 1000);
        
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
        
        // Update atomic parameters for lock-free access
        atomicA_.store(a_, std::memory_order_release);
        atomicB_.store(b_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    
    /**
     * Validates parameters for the Discrete Uniform distribution
     * @param a Lower bound parameter (integer)
     * @param b Upper bound parameter (integer, must be >= a)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(int a, int b) {
        if (a > b) {
            throw std::invalid_argument("Upper bound (b) must be greater than or equal to lower bound (a)");
        }
        // Check for integer overflow in range calculation
        if (b > INT_MAX - 1 || a < INT_MIN + 1) {
            throw std::invalid_argument("Parameter range too large - risk of integer overflow");
        }
        // Additional safety check for very large ranges
        const long long range_check = static_cast<long long>(b) - static_cast<long long>(a) + 1;
        if (range_check > INT_MAX) {
            throw std::invalid_argument("Parameter range exceeds maximum supported size");
        }
    }
    
    //==========================================================================
    // 20. PRIVATE UTILITY METHODS
    //==========================================================================
    
    /** @brief Round double to nearest integer with proper handling of edge cases */
    static int roundToInt(double x) noexcept {
        return static_cast<int>(std::round(x));
    }
    
    /** @brief Check if rounded value is within integer bounds */
    static bool isValidIntegerValue(double x) noexcept {
        return (x >= static_cast<double>(INT_MIN) && x <= static_cast<double>(INT_MAX));
    }
    
    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================
    
    /** @brief Lower bound parameter a (inclusive) */
    int a_{0};
    
    /** @brief Upper bound parameter b (inclusive) */
    int b_{1};
    
    /** @brief C++20 atomic copies of parameters for lock-free access */
    mutable std::atomic<int> atomicA_{0};
    mutable std::atomic<int> atomicB_{1};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================
    
    /** @brief Cached value of (b - a + 1) - the number of possible outcomes */
    mutable int range_{2};
    
    /** @brief Cached value of 1.0/range for efficiency in PMF calculations */
    mutable double probability_{0.5};
    
    /** @brief Cached value of (a + b)/2.0 for efficiency in mean calculations */
    mutable double mean_{0.5};
    
    /** @brief Cached value of ((b - a)(b - a + 2))/12.0 for efficiency in variance calculations */
    mutable double variance_{0.25};
    
    /** @brief Cached value of log(probability_) for efficiency in log-PMF calculations */
    mutable double logProbability_{-constants::math::LN2};
    
    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================
    
    /** @brief Atomic cache validity flag for lock-free fast path optimization */
    mutable std::atomic<bool> cacheValidAtomic_{false};
    
    /** @brief True if this is a binary distribution [0,1] for optimization */
    mutable bool isBinary_{true};
    
    /** @brief True if this is a standard die [1,6] for optimization */
    mutable bool isStandardDie_{false};
    
    /** @brief True if this is a symmetric distribution around zero for optimization */
    mutable bool isSymmetric_{false};
    
    /** @brief True if the range is small (≤ 10) for lookup table optimization */
    mutable bool isSmallRange_{true};
    
    /** @brief True if the range is large (> 1000) for approximation algorithms */
    mutable bool isLargeRange_{false};
    
    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================
    
    // Note: Discrete distribution uses standard caching only
    // This section maintained for template compliance
};

} // namespace libstats

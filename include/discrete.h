#ifndef LIBSTATS_DISCRETE_H_
#define LIBSTATS_DISCRETE_H_

#include "distribution_base.h"
#include "constants.h"
#include "error_handling.h" // Safe error handling without exceptions
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation

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
private:
    //==========================================================================
    // DISTRIBUTION PARAMETERS
    //==========================================================================
    
    /** @brief Lower bound parameter a (inclusive) */
    int a_{0};
    
    /** @brief Upper bound parameter b (inclusive) */
    int b_{1};

    //==========================================================================
    // PERFORMANCE CACHE
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
    // OPTIMIZATION FLAGS
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

    friend std::istream& operator>>(std::istream& is,
            libstats::DiscreteDistribution& distribution);

public:
    //==========================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    
    /**
     * Constructs a Discrete Uniform distribution with given integer bounds.
     * 
     * @param a Lower bound (inclusive, default: 0)
     * @param b Upper bound (inclusive, default: 1, must be >= a)
     * @throws std::invalid_argument if parameters are invalid
     * 
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit DiscreteDistribution(int a = 0, int b = 1);
    
    /**
     * Thread-safe copy constructor
     * 
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    DiscreteDistribution(const DiscreteDistribution& other);
    
    /**
     * Copy assignment operator
     * 
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    DiscreteDistribution& operator=(const DiscreteDistribution& other);
    
    /**
     * Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     * @warning NOT noexcept due to potential lock acquisition exceptions
     */
    DiscreteDistribution(DiscreteDistribution&& other);
    
    /**
     * Move assignment operator (DEFENSIVE THREAD SAFETY)
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
    // SAFE FACTORY METHODS (Exception-free construction)
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
    
    /**
     * @brief Safely try to set parameters without throwing exceptions
     * 
     * @param a New lower bound parameter (inclusive)
     * @param b New upper bound parameter (inclusive, must be >= a)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetParameters(int a, int b) noexcept {
        auto validation = validateDiscreteParameters(a, b);
        if (validation.isError()) {
            return validation;
        }
        
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        a_ = a;
        b_ = b;
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
        return validateDiscreteParameters(a_, b_);
    }

    //==========================================================================
    // CORE PROBABILITY METHODS
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

    //==========================================================================
    // PARAMETER GETTERS AND SETTERS
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

    //==========================================================================
    // DISTRIBUTION MANAGEMENT
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
    // COMPARISON OPERATORS
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
    // SIMD BATCH OPERATIONS
    //==========================================================================
    
    /**
     * Batch computation of probability masses using SIMD optimization.
     * Processes multiple values simultaneously for improved performance.
     * 
     * @param values Input array of values to evaluate (will be rounded to integers)
     * @param results Output array for probability masses (must be same size as values)
     * @param count Number of values to process
     * @note This method is optimized for large batch sizes where SIMD overhead is justified
     */
    void getProbabilityBatch(const double* values, double* results, std::size_t count) const noexcept;
    
    /**
     * Batch computation of log probability masses using SIMD optimization.
     * Processes multiple values simultaneously for improved performance.
     * 
     * @param values Input array of values to evaluate (will be rounded to integers)
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
    // DISCRETE-SPECIFIC UTILITY METHODS
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

private:
    //==========================================================================
    // PRIVATE FACTORY METHODS
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
    // PRIVATE BATCH IMPLEMENTATION METHODS
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
    // PRIVATE UTILITY METHODS
    //==========================================================================
    
    /** @brief Round double to nearest integer with proper handling of edge cases */
    static int roundToInt(double x) noexcept {
        return static_cast<int>(std::round(x));
    }
    
    /** @brief Check if rounded value is within integer bounds */
    static bool isValidIntegerValue(double x) noexcept {
        return (x >= static_cast<double>(INT_MIN) && x <= static_cast<double>(INT_MAX));
    }
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const DiscreteDistribution& dist);

} // namespace libstats

#endif // LIBSTATS_DISCRETE_H_

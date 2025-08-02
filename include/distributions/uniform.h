#pragma once

#include "../core/distribution_base.h"
#include "../core/constants.h"
#include "../core/error_handling.h" // Safe error handling without exceptions
#include "../core/performance_dispatcher.h" // For smart auto-dispatch
#include "../platform/parallel_execution.h" // For parallel batch operations
#include "../platform/thread_pool.h"        // For thread pool integration
#include "../platform/work_stealing_pool.h" // For work-stealing parallel execution
#include "../platform/adaptive_cache.h"     // For cache-aware batch operations
#include <mutex>       // For thread-safe cache updates
#include <shared_mutex> // For shared_mutex and shared_lock
#include <atomic>      // For atomic cache validation
#include <span>        // For C++20 type-safe array access

namespace libstats {

/**
 * @brief Thread-safe Uniform Distribution for modeling equiprobable outcomes over intervals.
 * 
 * @details The Uniform distribution (also known as rectangular distribution) is a continuous
 * probability distribution where all values in a finite interval are equally likely.
 * This is the fundamental assumption of "no prior knowledge" in Bayesian statistics.
 * 
 * @par Mathematical Definition:
 * - PDF: f(x; a, b) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
 * - CDF: F(x; a, b) = (x-a)/(b-a) for a ≤ x ≤ b, 0 for x < a, 1 for x > b
 * - Parameters: a, b ∈ ℝ with a < b (lower and upper bounds)
 * - Support: x ∈ [a, b]
 * - Mean: (a + b)/2
 * - Variance: (b - a)²/12
 * - Mode: Not unique (any value in [a, b])
 * - Median: (a + b)/2
 * 
 * @par Key Properties:
 * - **Maximum Entropy**: Among all distributions with given support [a, b]
 * - **Symmetric**: Distribution is symmetric around the midpoint (a + b)/2
 * - **Memoryless on Intervals**: Conditional distribution on sub-intervals is uniform
 * - **Scale Invariant**: Linear transformations preserve uniform nature
 * - **Conjugate Prior**: For location parameters in some contexts
 * - **Rectangular Shape**: Constant probability density over support
 * 
 * @par Thread Safety:
 * - All methods are fully thread-safe using reader-writer locks
 * - Concurrent reads are optimized with std::shared_mutex
 * - Cache invalidation uses atomic operations for lock-free fast paths
 * - Deadlock prevention via ordered lock acquisition with std::lock()
 * 
 * @par Performance Features:
 * - Atomic cache validity flags for lock-free fast path access
 * - Extensive caching of computed values (width, 1/width, midpoint, width²)
 * - Optimized PDF/CDF computation with branch-free interval checking
 * - Fast parameter validation with IEEE 754 compliance
 * - Special optimizations for unit interval [0,1] and standard interval [-1,1]
 * 
 * @par Usage Examples:
 * @code
 * // Unit interval uniform distribution [0,1]
 * auto result = UniformDistribution::create(0.0, 1.0);
 * if (result.isOk()) {
 *     auto unit = std::move(result.value);
 *     
 *     // Temperature measurement with uncertainty [19.5, 20.5]°C
 *     auto tempResult = UniformDistribution::create(19.5, 20.5);
 *     if (tempResult.isOk()) {
 *         auto temperature = std::move(tempResult.value);
 *         
 *         // Fit to observed data (finds min/max bounds)
 *         std::vector<double> measurements = {19.7, 20.1, 19.9, 20.3, 19.8};
 *         temperature.fit(measurements);
 *         
 *         // Probability of measurement within tolerance
 *         double withinTolerance = temperature.getCumulativeProbability(20.2) - 
 *                                 temperature.getCumulativeProbability(19.8);
 *     }
 * }
 * @endcode
 * 
 * @par Applications:
 * - Monte Carlo simulations (random number generation)
 * - Measurement uncertainty modeling
 * - Non-informative Bayesian priors
 * - Digital signal processing (quantization noise)
 * - Game theory (mixed strategies)
 * - Quality control (tolerance intervals)
 * - Financial modeling (price ranges)
 * - Computer graphics (random sampling)
 * 
 * @par Statistical Properties:
 * - Skewness: 0 (perfectly symmetric)
 * - Kurtosis: -1.2 (platykurtic - lighter tails than normal)
 * - Entropy: ln(b - a)
 * - Moment generating function: (e^(bt) - e^(at))/(t(b-a)) for t ≠ 0
 * - Characteristic function: (e^(ibt) - e^(iat))/(it(b-a)) for t ≠ 0
 * 
 * @par Special Cases:
 * - **Standard Uniform**: U(0,1) - fundamental for random number generation
 * - **Symmetric Uniform**: U(-c,c) - common in error modeling
 * - **Unit Uniform**: U(0,1) - basis for inverse transform sampling
 * - **Integer Uniform**: Discrete approximation for fair dice, cards, etc.
 * 
 * @par Numerical Considerations:
 * - Robust against floating-point precision issues
 * - Special handling for very small intervals (b - a < ε)
 * - Optimized branch-free comparisons for interval membership
 * - IEEE 754 compliant boundary handling
 * 
 * @par Implementation Details (C++20 Best Practices):
 * - Complex constructors/operators moved to .cpp for faster compilation
 * - Exception-safe design with RAII principles
 * - Optimized parameter validation with comprehensive error messages
 * - Lock-free fast paths using atomic operations
 * - Branch-free interval checking for performance
 * 
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
class UniformDistribution : public DistributionBase
{   
private:
    //==========================================================================
    // DISTRIBUTION PARAMETERS
    //==========================================================================
    
    /** @brief Lower bound parameter a */
    double a_{constants::math::ZERO_DOUBLE};
    
    /** @brief Upper bound parameter b */
    double b_{constants::math::ONE};

    //==========================================================================
    // PERFORMANCE CACHE
    //==========================================================================
    
    /** @brief Cached value of (b - a) for efficiency in PDF and variance calculations */
    mutable double width_{constants::math::ONE};
    
    /** @brief Cached value of 1/(b - a) for efficiency in PDF calculations */
    mutable double invWidth_{constants::math::ONE};
    
    /** @brief Cached value of (a + b)/2 for efficiency in mean and median calculations */
    mutable double midpoint_{constants::math::HALF};
    
    /** @brief Cached value of (b - a)² for efficiency in variance calculations */
    mutable double widthSquared_{constants::math::ONE};
    
    /** @brief Cached value of (b - a)²/12 for efficiency in variance calculations */
    mutable double variance_{constants::math::ONE_TWELFTH};
    
    //==========================================================================
    // ATOMIC PARAMETER COPIES (for lock-free access)
    //==========================================================================
    
    /** @brief Atomic copy of lower bound parameter a for lock-free access */
    mutable std::atomic<double> atomicA_{constants::math::ZERO_DOUBLE};
    
    /** @brief Atomic copy of upper bound parameter b for lock-free access */
    mutable std::atomic<double> atomicB_{constants::math::ONE};
    
    /** @brief Atomic validity flag for atomic parameter copies */
    mutable std::atomic<bool> atomicParamsValid_{false};
    
    //==========================================================================
    // OPTIMIZATION FLAGS
    //==========================================================================
    
    /** @brief Atomic cache validity flag for lock-free fast path optimization */
    mutable std::atomic<bool> cacheValidAtomic_{false};
    
    /** @brief True if this is the unit interval [0,1] for optimization */
    mutable bool isUnitInterval_{true};
    
    /** @brief True if this is the standard interval [-1,1] for optimization */
    mutable bool isStandardInterval_{false};
    
    /** @brief True if this is a symmetric interval around zero [-c,c] */
    mutable bool isSymmetricInterval_{false};
    
    /** @brief True if the width is very small (< tolerance) for numerical stability */
    mutable bool isNarrowInterval_{false};
    
    /** @brief True if the width is very large (> 1000) for numerical stability */
    mutable bool isWideInterval_{false};

    /**
     * Updates cached values when parameters change - assumes mutex is already held
     */
    void updateCacheUnsafe() const noexcept override {
        // Primary calculations - compute once, reuse multiple times
        width_ = b_ - a_;
        invWidth_ = constants::math::ONE / width_;
        midpoint_ = (a_ + b_) * constants::math::HALF;
        widthSquared_ = width_ * width_;
        variance_ = widthSquared_ * constants::math::ONE_TWELFTH;
        
        // Optimization flags
        isUnitInterval_ = (std::abs(a_ - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) &&
                         (std::abs(b_ - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        
        isStandardInterval_ = (std::abs(a_ + constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE) &&
                             (std::abs(b_ - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE);
        
        isSymmetricInterval_ = (std::abs(a_ + b_) <= constants::precision::DEFAULT_TOLERANCE);
        
        isNarrowInterval_ = (width_ < constants::precision::DEFAULT_TOLERANCE);
        isWideInterval_ = (width_ > constants::math::THOUSAND);
        
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
        
        // CRITICAL: Update atomic parameters for lock-free access
        atomicA_.store(a_, std::memory_order_release);
        atomicB_.store(b_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    
    /**
     * Validates parameters for the Uniform distribution
     * @param a Lower bound parameter
     * @param b Upper bound parameter (must be > a)
     * @throws std::invalid_argument if parameters are invalid
     */
    static void validateParameters(double a, double b) {
        if (std::isnan(a) || std::isinf(a) || std::isnan(b) || std::isinf(b)) {
            throw std::invalid_argument("Uniform distribution parameters must be finite numbers");
        }
        if (a >= b) {
            throw std::invalid_argument("Upper bound (b) must be strictly greater than lower bound (a)");
        }
    }

    friend std::istream& operator>>(std::istream& is,
            libstats::UniformDistribution& distribution);

public:
    //==========================================================================
    // CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================
    
    /**
     * Constructs a Uniform distribution with given bounds.
     * 
     * @param a Lower bound (default: 0.0)
     * @param b Upper bound (default: 1.0, must be > a)
     * @throws std::invalid_argument if parameters are invalid
     * 
     * Implementation in .cpp: Complex validation and cache initialization logic
     */
    explicit UniformDistribution(double a = constants::math::ZERO_DOUBLE, 
                                double b = constants::math::ONE);
    
    /**
     * Thread-safe copy constructor
     * 
     * Implementation in .cpp: Complex thread-safe copying with lock management,
     * parameter validation, and efficient cache value copying
     */
    UniformDistribution(const UniformDistribution& other);
    
    /**
     * Copy assignment operator
     * 
     * Implementation in .cpp: Complex thread-safe assignment with deadlock prevention,
     * atomic lock acquisition using std::lock, and parameter validation
     */
    UniformDistribution& operator=(const UniformDistribution& other);
    
    /**
     * Move constructor (DEFENSIVE THREAD SAFETY)
     * Implementation in .cpp: Thread-safe move with locking for legacy compatibility
     */
    UniformDistribution(UniformDistribution&& other) noexcept;
    
    /**
     * Move assignment operator (C++11 COMPLIANT)
     * Implementation in .cpp: Thread-safe move with atomic operations
     * @note noexcept compliant using atomic state management
     */
    UniformDistribution& operator=(UniformDistribution&& other) noexcept;

    /**
     * @brief Destructor - explicitly defaulted to satisfy Rule of Five
     * Implementation inline: Trivial destruction, kept for performance
     * 
     * Note: C++20 Best Practice - Rule of Five uses complexity-based placement:
     * - Simple operations (destructor) stay inline for performance
     * - Complex operations (copy/move) moved to .cpp for maintainability
     */
    ~UniformDistribution() override = default;
    
    //==========================================================================
    // SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================
    
    /**
     * @brief Safely create a Uniform distribution without throwing exceptions
     * 
     * This factory method provides exception-free construction to work around
     * ABI compatibility issues with Homebrew LLVM libc++ on macOS where
     * exceptions thrown from the library cause segfaults during unwinding.
     * 
     * @param a Lower bound parameter
     * @param b Upper bound parameter (must be > a)
     * @return Result containing either a valid UniformDistribution or error info
     * 
     * @par Usage Example:
     * @code
     * auto result = UniformDistribution::create(-2.0, 3.0);
     * if (result.isOk()) {
     *     auto distribution = std::move(result.value);
     *     // Use distribution safely...
     * } else {
     *     std::cout << "Error: " << result.message << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] static Result<UniformDistribution> create(double a = 0.0, double b = 1.0) noexcept {
        auto validation = validateUniformParameters(a, b);
        if (validation.isError()) {
            return Result<UniformDistribution>::makeError(validation.error_code, validation.message);
        }
        
        // Use private factory to bypass validation
        return Result<UniformDistribution>::ok(createUnchecked(a, b));
    }
    
    /**
     * @brief Safely try to set parameters without throwing exceptions
     * 
     * @param a New lower bound parameter
     * @param b New upper bound parameter (must be > a)
     * @return VoidResult indicating success or failure
     * 
     * Implementation in .cpp: Complex thread-safe parameter validation and atomic state management
     */
    [[nodiscard]] VoidResult trySetParameters(double a, double b) noexcept;
    
    /**
     * @brief Check if current parameters are valid
     * @return VoidResult indicating validity
     */
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return validateUniformParameters(a_, b_);
    }

    //==========================================================================
    // CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * Computes the probability density function for the Uniform distribution.
     * 
     * For uniform distribution: f(x) = 1/(b-a) for a ≤ x ≤ b, 0 otherwise
     * 
     * @param x The value at which to evaluate the PDF
     * @return Probability density (constant 1/(b-a) within support, 0 outside)
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * Computes the logarithm of the probability density function for numerical stability.
     * 
     * For uniform distribution: log(f(x)) = -log(b-a) for a ≤ x ≤ b, -∞ otherwise
     * 
     * @param x The value at which to evaluate the log-PDF
     * @return Natural logarithm of the probability density, or -∞ for values outside support
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * Evaluates the CDF at x using the standard uniform CDF formula.
     * 
     * For uniform distribution: F(x) = 0 for x < a, (x-a)/(b-a) for a ≤ x ≤ b, 1 for x > b
     * 
     * @param x The value at which to evaluate the CDF
     * @return Cumulative probability P(X ≤ x)
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;
    
    /**
     * @brief Computes the quantile function (inverse CDF)
     * 
     * For uniform distribution: F^(-1)(p) = a + p(b-a)
     * 
     * @param p Probability value in [0,1]
     * @return x such that P(X ≤ x) = p
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double p) const override;
    
    /**
     * @brief Generate single random sample from distribution
     * 
     * Uses linear transformation of uniform(0,1): X = a + (b-a) * U where U ~ Uniform(0,1)
     * 
     * @param rng Random number generator
     * @return Single random sample
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
    [[nodiscard]] double getLowerBound() const noexcept { 
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return a_; 
    }
    
    /**
     * Gets the upper bound parameter b.
     * Thread-safe: acquires shared lock to protect b_
     * 
     * @return Current upper bound value
     */
    [[nodiscard]] double getUpperBound() const noexcept { 
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return b_; 
    }
    
    /**
     * @brief Atomic lock-free getter for lower bound parameter a
     * 
     * High-performance lock-free access for multi-threaded applications.
     * Falls back to locked version if atomic copy is invalid.
     * 
     * @return Current lower bound value (lock-free)
     */
    [[nodiscard]] double getLowerBoundAtomic() const noexcept {
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            return atomicA_.load(std::memory_order_acquire);
        }
        // Fallback to locked version if atomic copy not valid
        return getLowerBound();
    }
    
    /**
     * @brief Atomic lock-free getter for upper bound parameter b
     * 
     * High-performance lock-free access for multi-threaded applications.
     * Falls back to locked version if atomic copy is invalid.
     * 
     * @return Current upper bound value (lock-free)
     */
    [[nodiscard]] double getUpperBoundAtomic() const noexcept {
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            return atomicB_.load(std::memory_order_acquire);
        }
        // Fallback to locked version if atomic copy not valid
        return getUpperBound();
    }
    
    /**
     * Sets the lower bound parameter a.
     * 
     * @param a New lower bound (must be < current upper bound)
     * @throws std::invalid_argument if a >= b or parameters are invalid
     */
    void setLowerBound(double a);
    
    /**
     * @brief Safely set the lower bound parameter a without throwing exceptions (Result-based API).
     * 
     * @param a New lower bound (must be < current upper bound)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetLowerBound(double a) noexcept;
    
    /**
     * Sets the upper bound parameter b.
     * 
     * @param b New upper bound (must be > current lower bound)
     * @throws std::invalid_argument if b <= a or parameters are invalid
     */
    void setUpperBound(double b);
    
    /**
     * @brief Safely set the upper bound parameter b without throwing exceptions (Result-based API).
     * 
     * @param b New upper bound (must be > current lower bound)
     * @return VoidResult indicating success or failure
     */
    [[nodiscard]] VoidResult trySetUpperBound(double b) noexcept;
    
    /**
     * Sets both bounds simultaneously.
     * 
     * @param a New lower bound
     * @param b New upper bound (must be > a)
     * @throws std::invalid_argument if parameters are invalid
     */
    void setBounds(double a, double b);
    
    /**
     * Gets the mean of the distribution.
     * For Uniform distribution, mean = (a + b)/2
     * Uses cached value to eliminate addition and division.
     * 
     * @return Mean value
     */
    [[nodiscard]] double getMean() const noexcept override;
    
    /**
     * Gets the variance of the distribution.
     * For Uniform distribution, variance = (b - a)²/12
     * Uses cached value to eliminate multiplications and divisions.
     * 
     * @return Variance value
     */
    [[nodiscard]] double getVariance() const noexcept override;
    
    /**
     * @brief Gets the skewness of the distribution.
     * For Uniform distribution, skewness = 0 (perfectly symmetric)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Skewness value (always 0)
     */
    [[nodiscard]] double getSkewness() const noexcept override {
        return 0.0;  // Uniform distribution is perfectly symmetric
    }
    
    /**
     * @brief Gets the kurtosis of the distribution.
     * For Uniform distribution, excess kurtosis = -1.2 (platykurtic)
     * Inline for performance - no thread safety needed for constant
     * 
     * @return Excess kurtosis value (always -1.2)
     */
    [[nodiscard]] double getKurtosis() const noexcept override {
        return -1.2;  // Uniform distribution is platykurtic (lighter tails)
    }
    
    /**
     * @brief Gets the number of parameters for this distribution.
     * For Uniform distribution, there are 2 parameters: a (lower) and b (upper)
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
        return "Uniform";
    }
    
    /**
     * @brief Checks if the distribution is discrete.
     * For Uniform distribution, it's continuous
     * Inline for performance - no thread safety needed for constant
     * 
     * @return false (always continuous)
     */
    [[nodiscard]] bool isDiscrete() const noexcept override {
        return false;
    }
    
    /**
     * @brief Gets the lower bound of the distribution support.
     * For Uniform distribution, support is [a, b]
     * 
     * @return Lower bound (parameter a)
     */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return a_;
    }
    
    /**
     * @brief Gets the upper bound of the distribution support.
     * For Uniform distribution, support is [a, b]
     * 
     * @return Upper bound (parameter b)
     */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return b_;
    }
    
    /**
     * Gets the width of the distribution (b - a).
     * This is also the range of the distribution.
     * Uses cached value to eliminate subtraction.
     * 
     * @return Width of the distribution
     */
    [[nodiscard]] double getWidth() const noexcept;
    
    /**
     * Gets the midpoint of the distribution (a + b)/2.
     * This is also the mean and median of the distribution.
     * Uses cached value to eliminate addition and division.
     * 
     * @return Midpoint of the distribution
     */
    [[nodiscard]] double getMidpoint() const noexcept;

    //==========================================================================
    // DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * Fits the distribution parameters to the given data.
     * For Uniform distribution, uses sample minimum and maximum as bounds.
     * 
     * @param values Vector of observed data
     * @throws std::invalid_argument if values is empty or contains invalid data
     */
    void fit(const std::vector<double>& values) override;

    /**
     * Resets the distribution to default parameters (a = 0.0, b = 1.0).
     * This corresponds to the standard uniform distribution.
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
    bool operator==(const UniformDistribution& other) const;
    
    /**
     * Inequality comparison operator with thread-safe locking
     * @param other Other distribution to compare with
     * @return true if parameters are not equal
     */
    bool operator!=(const UniformDistribution& other) const { return !(*this == other); }
    
    //==========================================================================
    // SMART AUTO-DISPATCH BATCH OPERATIONS (New Simplified API)
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
    void getProbability(std::span<const double> values, std::span<double> results,
                       const performance::PerformanceHint& hint = {}) const;
    
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
                          const performance::PerformanceHint& hint = {}) const;
    
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
                                 const performance::PerformanceHint& hint = {}) const;
    
    //==========================================================================
    // SIMD BATCH OPERATIONS (Legacy - prefer auto-dispatch methods above)
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
    // ADVANCED STATISTICAL METHODS
    //==========================================================================
    
    /**
     * @brief Kolmogorov-Smirnov goodness-of-fit test
     * 
     * Tests the null hypothesis that data follows the specified Uniform distribution.
     * 
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (KS_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> kolmogorovSmirnovTest(
        const std::vector<double>& data,
        const UniformDistribution& distribution,
        double alpha = 0.05);
    
    /**
     * @brief Anderson-Darling goodness-of-fit test
     * 
     * Tests the null hypothesis that data follows the specified Uniform distribution.
     * More sensitive to deviations in the tails than KS test.
     * 
     * @param data Sample data to test
     * @param distribution Theoretical distribution to test against
     * @param alpha Significance level (default: 0.05)
     * @return Tuple of (AD_statistic, p_value, reject_null)
     */
    static std::tuple<double, double, bool> andersonDarlingTest(
        const std::vector<double>& data,
        const UniformDistribution& distribution,
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
     * the distribution parameters (lower and upper bounds).
     * 
     * @param data Sample data for bootstrap resampling
     * @param confidence_level Confidence level (e.g., 0.95 for 95% CI)
     * @param n_bootstrap Number of bootstrap samples (default: 1000)
     * @param random_seed Seed for random sampling (default: 42)
     * @return Tuple of ((a_CI_lower, a_CI_upper), (b_CI_lower, b_CI_upper))
     */
    static std::tuple<std::pair<double, double>, std::pair<double, double>> bootstrapParameterConfidenceIntervals(
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
     * @param fitted_distribution The fitted Uniform distribution
     * @return Tuple of (AIC, BIC, AICc, log_likelihood)
     */
    static std::tuple<double, double, double, double> computeInformationCriteria(
        const std::vector<double>& data,
        const UniformDistribution& fitted_distribution);

private:
    //==========================================================================
    // PRIVATE FACTORY METHODS
    //==========================================================================
    
    /**
     * @brief Create a distribution without parameter validation (for internal use)
     * @param a Lower bound parameter (assumed valid)
     * @param b Upper bound parameter (assumed valid)
     * @return UniformDistribution with the given parameters
     */
    static UniformDistribution createUnchecked(double a, double b) noexcept {
        UniformDistribution dist(a, b, true); // bypass validation
        return dist;
    }
    
    /**
     * @brief Private constructor that bypasses validation (for internal use)
     * @param a Lower bound parameter (assumed valid)
     * @param b Upper bound parameter (assumed valid)
     * @param bypassValidation Internal flag to skip validation
     */
    UniformDistribution(double a, double b, bool /*bypassValidation*/) noexcept
        : DistributionBase(), a_(a), b_(b) {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    
    //==========================================================================
    // PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================
    
    /** @brief Internal implementation for batch PDF calculation */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double a, double b, double inv_width) const noexcept;
    
    /** @brief Internal implementation for batch log PDF calculation */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double a, double b, double log_inv_width) const noexcept;
    
    /** @brief Internal implementation for batch CDF calculation */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                 double a, double b, double inv_width) const noexcept;
};

/**
 * @brief Stream output operator
 * @param os Output stream
 * @param dist Distribution to output
 * @return Reference to the output stream
 */
std::ostream& operator<<(std::ostream& os, const UniformDistribution& dist);

} // namespace libstats

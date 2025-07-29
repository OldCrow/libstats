#ifndef LIBSTATS_SAFETY_H_
#define LIBSTATS_SAFETY_H_

#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <span>
#include "constants.h"
#include "../platform/simd.h"

/**
 * @file safety.h
 * @brief Memory safety, bounds checking, and numerical stability utilities for libstats
 * 
 * This header provides comprehensive safety utilities including bounds checking,
 * memory safety verification, numerical stability checks, and error recovery
 * mechanisms for statistical computations.
 * 
 * ## Design Pattern for Safety Functions
 * 
 * This module uses a dual-layer approach for mathematical safety functions:
 * 
 * 1. **Scalar Functions (inline)**: Small, frequently-used functions like safe_log(),
 *    safe_exp(), safe_sqrt() are implemented as inline functions in this header
 *    for maximum performance. These handle edge cases and provide numerically
 *    stable versions of standard math functions.
 * 
 * 2. **Vector Functions (compiled)**: Larger, more complex functions like
 *    vector_safe_log(), vector_safe_exp(), vector_safe_sqrt() are implemented
 *    in safety.cpp. These functions:
 *    - Use SIMD optimization for large arrays
 *    - Fall back to scalar implementations for small arrays
 *    - Call the inline scalar functions to ensure consistent behavior
 *    - Handle memory layout and chunking for optimal performance
 * 
 * This design ensures that:
 * - Single-value operations are fast (inlined)
 * - Array operations are optimized (vectorized when beneficial)
 * - Behavior is consistent between scalar and vector operations
 * - Code duplication is minimized
 */

namespace libstats {
namespace safety {

//==============================================================================
// MEMORY SAFETY AND BOUNDS CHECKING
//==============================================================================

/**
 * @brief Safe bounds checking for array/vector access
 * @param index The index being accessed
 * @param size The size/bound of the container
 * @param context Description for error messages
 * @throws std::out_of_range if index >= size
 */
inline void check_bounds(std::size_t index, std::size_t size, const char* context = "array access") {
    if (index >= size) {
        throw std::out_of_range(std::string("Index ") + std::to_string(index) + 
                               " out of bounds [0, " + std::to_string(size) + ") in " + context);
    }
}

/**
 * @brief Safe bounds checking for 2D matrix access
 * @param row Row index
 * @param col Column index
 * @param rows Total number of rows
 * @param cols Total number of columns
 * @param context Description for error messages
 * @throws std::out_of_range if indices are out of bounds
 */
inline void check_matrix_bounds(std::size_t row, std::size_t col, 
                               std::size_t rows, std::size_t cols, 
                               const char* context = "matrix access") {
    if (row >= rows || col >= cols) {
        throw std::out_of_range(std::string("Matrix index (") + std::to_string(row) + 
                               ", " + std::to_string(col) + ") out of bounds [0, " + 
                               std::to_string(rows) + ") x [0, " + std::to_string(cols) + 
                               ") in " + context);
    }
}

/**
 * @brief Safe linear index calculation for row-major 2D matrices
 * @param row Row index
 * @param col Column index
 * @param cols Total number of columns
 * @param rows Total number of rows (for bounds checking)
 * @param context Description for error messages
 * @return Linear index
 * @throws std::out_of_range if indices are out of bounds
 */
inline std::size_t safe_matrix_index(std::size_t row, std::size_t col, 
                                    std::size_t cols, std::size_t rows,
                                    const char* context = "matrix indexing") {
    check_matrix_bounds(row, col, rows, cols, context);
    return row * cols + col;
}

/**
 * @brief Safe pointer arithmetic with bounds checking
 * @param base_ptr Base pointer
 * @param offset Offset from base
 * @param max_offset Maximum allowed offset
 * @param context Description for error messages
 * @return Safe offset pointer
 * @throws std::out_of_range if offset is out of bounds
 */
template<typename T>
inline T* safe_pointer_offset(T* base_ptr, std::size_t offset, std::size_t max_offset,
                             const char* context = "pointer offset") {
    if (offset >= max_offset) {
        throw std::out_of_range(std::string("Pointer offset ") + std::to_string(offset) + 
                               " out of bounds [0, " + std::to_string(max_offset) + 
                               ") in " + context);
    }
    return base_ptr + offset;
}

/**
 * @brief Safe array size calculation with overflow checking
 * @param rows Number of rows
 * @param cols Number of columns
 * @param element_size Size of each element
 * @return Total size in bytes
 * @throws std::overflow_error if calculation would overflow
 */
inline std::size_t safe_array_size(std::size_t rows, std::size_t cols, std::size_t element_size) {
    // Check for multiplication overflow
    if (rows > 0 && cols > SIZE_MAX / rows) {
        throw std::overflow_error("Matrix size calculation overflow: rows * cols");
    }
    
    const std::size_t total_elements = rows * cols;
    if (total_elements > 0 && element_size > SIZE_MAX / total_elements) {
        throw std::overflow_error("Array size calculation overflow: elements * element_size");
    }
    
    return total_elements * element_size;
}

/**
 * @brief Verify SIMD alignment of a pointer
 * @param ptr Pointer to check
 * @param alignment Required alignment (must be power of 2)
 * @param context Description for error messages
 * @throws std::runtime_error if pointer is not properly aligned
 */
template<typename T>
inline void verify_simd_alignment(const T* ptr, std::size_t alignment, const char* context = "SIMD operation") {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & (alignment - 1)) != 0) {
        throw std::runtime_error(std::string("Pointer not aligned to ") + std::to_string(alignment) + 
                                " bytes for " + context);
    }
}

/**
 * @brief Debug-only bounds checking macros that compile to nothing in release builds
 */
#ifdef NDEBUG
    #define LIBSTATS_ASSERT_BOUNDS(index, size, context) do { } while(false)
    #define LIBSTATS_ASSERT_MATRIX_BOUNDS(row, col, rows, cols, context) do { } while(false)
    #define LIBSTATS_ASSERT_ALIGNMENT(ptr, alignment, context) do { } while(false)
#else
    #define LIBSTATS_ASSERT_BOUNDS(index, size, context) \
        assert((index) < (size) && "Bounds check failed in " context)
    
    #define LIBSTATS_ASSERT_MATRIX_BOUNDS(row, col, rows, cols, context) \
        assert((row) < (rows) && (col) < (cols) && "Matrix bounds check failed in " context)
    
    #define LIBSTATS_ASSERT_ALIGNMENT(ptr, alignment, context) \
        assert((reinterpret_cast<uintptr_t>(ptr) & ((alignment) - 1)) == 0 && \
               "Alignment check failed in " context)
#endif

//==============================================================================
// NUMERICAL SAFETY AND VALIDATION
//==============================================================================

/**
 * @brief Check if a value is finite and valid for computation
 * @param value Value to check
 * @param name Variable name for error messages
 * @throws std::runtime_error if value is not finite
 */
inline void check_finite(double value, const std::string& name = "value") {
    if (!std::isfinite(value)) {
        throw std::runtime_error("Value " + name + " is not finite: " + std::to_string(value));
    }
}

/**
 * @brief Check if a probability is in valid range [0, 1]
 * @param prob Probability value to check
 * @param name Variable name for error messages
 * @throws std::runtime_error if probability is invalid
 */
inline void check_probability(double prob, const std::string& name = "probability") {
    if (!std::isfinite(prob) || prob < 0.0 || prob > 1.0) {
        throw std::runtime_error("Invalid probability " + name + ": " + std::to_string(prob) + 
                                " (must be in [0, 1])");
    }
}

/**
 * @brief Check if a log probability is in valid range
 * @param log_prob Log probability value to check
 * @param name Variable name for error messages
 * @throws std::runtime_error if log probability is invalid
 */
inline void check_log_probability(double log_prob, const std::string& name = "log_probability") {
    if (!std::isfinite(log_prob) || log_prob > 0.0) {
        throw std::runtime_error("Invalid log probability " + name + ": " + std::to_string(log_prob) + 
                                " (must be <= 0)");
    }
}

/**
 * @brief Clamp probability to valid range [MIN_PROBABILITY, MAX_PROBABILITY]
 * @param prob Probability to clamp
 * @return Clamped probability
 */
inline double clamp_probability(double prob) noexcept {
    if (std::isnan(prob)) return constants::probability::MIN_PROBABILITY;
    if (prob <= 0.0) return constants::probability::MIN_PROBABILITY;
    if (prob >= 1.0) return constants::probability::MAX_PROBABILITY;
    return prob;
}

/**
 * @brief Clamp log probability to valid range
 * @param log_prob Log probability to clamp
 * @return Clamped log probability
 */
inline double clamp_log_probability(double log_prob) noexcept {
    if (std::isnan(log_prob)) return constants::probability::MIN_LOG_PROBABILITY;
    if (log_prob > 0.0) return constants::probability::MAX_LOG_PROBABILITY;
    if (log_prob < constants::probability::MIN_LOG_PROBABILITY) {
        return constants::probability::MIN_LOG_PROBABILITY;
    }
    return log_prob;
}

/**
 * @brief Safe logarithm that handles edge cases
 * @param value Value to take logarithm of
 * @return Safe logarithm, clamped to valid range
 */
inline double safe_log(double value) noexcept {
    if (value <= 0.0 || std::isnan(value)) {
        return constants::probability::MIN_LOG_PROBABILITY;
    }
    if (std::isinf(value)) {
        return std::numeric_limits<double>::max();
    }
    return std::log(value);
}

/**
 * @brief Safe exponential that handles edge cases
 * @param value Value to exponentiate
 * @return Safe exponential, clamped to valid range
 */
inline double safe_exp(double value) noexcept {
    if (std::isnan(value)) return 0.0;
    if (value < constants::probability::MIN_LOG_PROBABILITY) {
        return constants::probability::MIN_PROBABILITY;
    }
    if (value > constants::thresholds::LOG_EXP_OVERFLOW_THRESHOLD) {  // Prevent overflow
        return std::numeric_limits<double>::max();
    }
    
    double result = std::exp(value);
    // Handle underflow: if exp() returns 0 but value is finite, clamp to MIN_PROBABILITY
    if (result == 0.0 && std::isfinite(value)) {
        return constants::probability::MIN_PROBABILITY;
    }
    return result;
}

/**
 * @brief Safe square root that handles edge cases
 * @param value Value to take square root of
 * @return Safe square root, returns 0 for negative inputs
 */
inline double safe_sqrt(double value) noexcept {
    if (std::isnan(value) || value < 0.0) {
        return 0.0;
    }
    if (std::isinf(value)) {
        return std::numeric_limits<double>::max();
    }
    return std::sqrt(value);
}

//==============================================================================
// VECTORIZED SAFETY FUNCTIONS
//==============================================================================

/**
 * @brief Vectorized safe logarithm with SIMD optimization
 * @param input Input values
 * @param output Output array for safe_log(input[i])
 * @note Automatically selects optimal SIMD implementation based on CPU capabilities
 * @note For small arrays, falls back to scalar implementation to avoid overhead
 */
void vector_safe_log(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized safe exponential with SIMD optimization
 * @param input Input values
 * @param output Output array for safe_exp(input[i])
 * @note Automatically selects optimal SIMD implementation based on CPU capabilities
 * @note For small arrays, falls back to scalar implementation to avoid overhead
 */
void vector_safe_exp(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized safe square root with SIMD optimization
 * @param input Input values
 * @param output Output array for safe_sqrt(input[i])
 * @note Automatically selects optimal SIMD implementation based on CPU capabilities
 * @note For small arrays, falls back to scalar implementation to avoid overhead
 */
void vector_safe_sqrt(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized probability clamping with SIMD optimization
 * @param input Input values
 * @param output Output array for clamp_probability(input[i])
 * @note Automatically selects optimal SIMD implementation based on CPU capabilities
 */
void vector_clamp_probability(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized log probability clamping with SIMD optimization
 * @param input Input values
 * @param output Output array for clamp_log_probability(input[i])
 * @note Automatically selects optimal SIMD implementation based on CPU capabilities
 */
void vector_clamp_log_probability(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Check if vectorized safety operations should be used for given array size
 * @param size Number of elements to process
 * @return true if SIMD vectorization is beneficial for this size
 */
[[nodiscard]] bool should_use_vectorized_safety(std::size_t size) noexcept;

/**
 * @brief Get minimum array size threshold for vectorized safety operations
 * @return Minimum number of elements where vectorization becomes beneficial
 */
[[nodiscard]] std::size_t vectorized_safety_threshold() noexcept;

/**
 * @brief Check if a vector contains only finite values
 * @param data Vector to check
 * @param name Vector name for error messages
 * @throws std::runtime_error if vector contains non-finite values
 */
inline void check_vector_finite(const std::vector<double>& data, const std::string& name = "vector") {
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            throw std::runtime_error("Vector " + name + " contains non-finite value at index " + 
                                    std::to_string(i) + ": " + std::to_string(data[i]));
        }
    }
}

/**
 * @brief Normalize probabilities to sum to 1, handling edge cases
 * @param probs Probability vector to normalize (modified in place)
 * @return True if normalization was successful, false if all probabilities were zero
 */
inline bool normalize_probabilities(std::vector<double>& probs) noexcept {
    double sum = 0.0;
    
    // First pass: calculate sum and check for negative values
    for (double& prob : probs) {
        if (prob < 0.0 || std::isnan(prob)) {
            prob = 0.0;
        }
        sum += prob;
    }
    
    // If sum is too small, set to uniform distribution
    if (sum < constants::precision::ZERO) {
        const double uniform_prob = 1.0 / probs.size();
        for (double& prob : probs) {
            prob = uniform_prob;
        }
        return false;
    }
    
    // Normalize
    for (double& prob : probs) {
        prob /= sum;
    }
    
    return true;
}

/**
 * @brief Check if probabilities sum to approximately 1
 * @param probs Probability vector to check
 * @param tolerance Tolerance for sum check
 * @return True if probabilities are properly normalized
 */
inline bool is_probability_distribution(const std::vector<double>& probs, 
                                       double tolerance = constants::precision::DEFAULT_TOLERANCE) noexcept {
    double sum = 0.0;
    for (double prob : probs) {
        if (prob < 0.0 || prob > 1.0 || std::isnan(prob)) {
            return false;
        }
        sum += prob;
    }
    return std::abs(sum - 1.0) <= tolerance;
}

//==============================================================================
// CONVERGENCE DETECTION
//==============================================================================

/**
 * @brief Convergence detector for iterative algorithms
 */
class ConvergenceDetector {
private:
    std::vector<double> history_;
    double tolerance_;
    std::size_t max_iterations_;
    std::size_t window_size_;
    std::size_t current_iteration_;
    
public:
    /**
     * @brief Constructor with convergence parameters
     * @param tolerance Convergence tolerance
     * @param max_iterations Maximum iterations before forced termination
     * @param window_size Number of previous values to consider for convergence
     */
    explicit ConvergenceDetector(double tolerance = constants::precision::DEFAULT_TOLERANCE,
                                std::size_t max_iterations = 1000,
                                std::size_t window_size = 5)
        : tolerance_(tolerance), max_iterations_(max_iterations), 
          window_size_(window_size), current_iteration_(0) {
        history_.reserve(window_size_);
    }
    
    /**
     * @brief Add a new value and check convergence
     * @param value New value (e.g., log-likelihood)
     * @return True if algorithm has converged
     */
    bool add_value(double value) {
        history_.push_back(value);
        ++current_iteration_;
        
        // Keep only the last window_size_ values
        if (history_.size() > window_size_) {
            history_.erase(history_.begin());
        }
        
        // Need at least 2 values to check convergence
        if (history_.size() < 2) {
            return false;
        }
        
        // Check if recent values are within tolerance
        for (std::size_t i = 1; i < history_.size(); ++i) {
            if (std::abs(history_[i] - history_[i-1]) > tolerance_) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * @brief Check if maximum iterations reached
     * @return True if max iterations exceeded
     */
    bool is_max_iterations_reached() const noexcept {
        return current_iteration_ >= max_iterations_;
    }
    
    /**
     * @brief Get current iteration count
     * @return Current iteration number
     */
    std::size_t get_current_iteration() const noexcept {
        return current_iteration_;
    }
    
    /**
     * @brief Get convergence history
     * @return Vector of recent values
     */
    const std::vector<double>& get_history() const noexcept {
        return history_;
    }
    
    /**
     * @brief Reset the detector for a new run
     */
    void reset() noexcept {
        history_.clear();
        current_iteration_ = 0;
    }
    
    /**
     * @brief Check for oscillation (values bouncing back and forth)
     * @return True if oscillation detected
     */
    bool is_oscillating() const noexcept {
        if (history_.size() < 4) return false;
        
        // Simple oscillation check: alternating increases and decreases
        bool increasing = history_[1] > history_[0];
        for (std::size_t i = 2; i < history_.size(); ++i) {
            bool current_increasing = history_[i] > history_[i-1];
            if (current_increasing == increasing) {
                return false;  // Not oscillating
            }
            increasing = current_increasing;
        }
        return true;
    }
    
    /**
     * @brief Check for stagnation (values not changing significantly)
     * @return True if stagnation detected
     */
    bool is_stagnating() const noexcept {
        if (history_.size() < window_size_) return false;
        
        double max_change = 0.0;
        for (std::size_t i = 1; i < history_.size(); ++i) {
            max_change = std::max(max_change, std::abs(history_[i] - history_[i-1]));
        }
        
        return max_change < tolerance_ * 0.1;  // Very small changes
    }
};

//==============================================================================
// ERROR RECOVERY STRATEGIES
//==============================================================================

#ifdef STRICT
#undef STRICT
#endif
#ifdef GRACEFUL
#undef GRACEFUL
#endif
#ifdef ROBUST
#undef ROBUST
#endif
#ifdef ADAPTIVE
#undef ADAPTIVE
#endif
/**
 * @brief Error recovery strategies for numerical failures
 */
enum class RecoveryStrategy {
    StrictMode,      ///< Throw exception on any numerical issue
    GracefulMode,    ///< Try to recover with degraded precision
    RobustMode,      ///< Aggressively recover, may sacrifice some accuracy
    AdaptiveMode     ///< Choose strategy based on problem characteristics
};

inline bool recover_from_underflow(std::vector<double>& probs, RecoveryStrategy strategy = RecoveryStrategy::GracefulMode) {
    bool had_underflow = false;
    for (double& prob : probs) {
        if (prob < constants::probability::MIN_PROBABILITY) {
            had_underflow = true;
            switch (strategy) {
                case RecoveryStrategy::StrictMode:
                    throw std::runtime_error("Probability underflow detected in strict mode");
                case RecoveryStrategy::GracefulMode:
                case RecoveryStrategy::RobustMode:
                case RecoveryStrategy::AdaptiveMode:
                    prob = constants::probability::MIN_PROBABILITY;
                    break;
            }
        }
    }
    if (had_underflow) {
        normalize_probabilities(probs);
    }
    return had_underflow;
}

inline std::size_t handle_nan_values(std::vector<double>& values, RecoveryStrategy strategy = RecoveryStrategy::GracefulMode) {
    std::size_t nan_count = 0;
    for (double& value : values) {
        if (std::isnan(value)) {
            ++nan_count;
            switch (strategy) {
                case RecoveryStrategy::StrictMode:
                    throw std::runtime_error("NaN value detected in strict mode");
                case RecoveryStrategy::GracefulMode:
                case RecoveryStrategy::AdaptiveMode:
                    value = 0.0;
                    break;
                case RecoveryStrategy::RobustMode:
                    value = constants::precision::ZERO;
                    break;
            }
        }
    }
    return nan_count;
}

} // namespace safety
} // namespace libstats

#endif // LIBSTATS_SAFETY_H_

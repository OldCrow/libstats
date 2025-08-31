#pragma once

#include "../platform/simd.h"
#include "mathematical_constants.h"
#include "precision_constants.h"
#include "threshold_constants.h"

#include <array>
#include <cmath>
#include <limits>

namespace stats {

/**
 * @brief High-performance log-space arithmetic operations
 *
 * This class provides optimized implementations of log-space arithmetic
 * operations commonly used in statistical calculations. Key optimizations include:
 * - Precomputed lookup tables for frequently used values
 * - SIMD-vectorized operations
 * - Numerically stable log-sum-exp implementations
 * - Efficient handling of log(0) cases
 *
 * @note THREAD SAFETY:
 * - All operations are thread-safe after initialization
 * - Initialization is automatically thread-safe via std::once_flag
 * - Multiple threads can safely call all static methods concurrently
 * - The global initializer ensures tables are ready when library is loaded
 * - No additional synchronization is required for normal usage
 */
class LogSpaceOps {
   public:
    /// Log-space representation of zero (negative infinity)
    static constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

    /// Threshold below which exp() terms are considered negligible
    static constexpr double LOG_SUM_THRESHOLD = detail::LOG_SUM_EXP_THRESHOLD;

    /// Size of precomputed lookup tables
    static constexpr std::size_t LOOKUP_TABLE_SIZE = detail::LOG_SPACE_LOOKUP_TABLE_SIZE;

    /**
     * @brief Initialize precomputed lookup tables
     * Call this once at program startup for optimal performance
     *
     * @note THREAD SAFETY:
     * - This method is thread-safe and can be called from multiple threads
     * - Uses std::once_flag internally to ensure initialization happens exactly once
     * - Safe to call multiple times - subsequent calls are no-ops
     * - The global initializer calls this automatically when the library is loaded
     * - Manual calls are optional but can be used for explicit control
     */
    static void initialize();

    /**
     * @brief Numerically stable log-sum-exp: log(exp(a) + exp(b))
     *
     * Highly optimized version using lookup tables and avoiding
     * expensive exp/log operations when possible.
     *
     * @param logA First log value
     * @param logB Second log value
     * @return log(exp(logA) + exp(logB))
     */
    static double logSumExp(double logA, double logB) noexcept;

    /**
     * @brief Fast log-sum-exp for arrays using SIMD
     *
     * @param logValues Array of log values
     * @param size Number of values
     * @return log(sum(exp(logValues[i])))
     */
    static double logSumExpArray(const double* logValues, std::size_t size) noexcept;

    /**
     * @brief Precompute log values for probability matrix
     *
     * Converts probability matrix to log-space once and caches results.
     * Much faster than repeated log() calls during computation.
     *
     * @param probMatrix Input probability matrix
     * @param logMatrix Output log matrix (must be pre-allocated)
     * @param rows Number of rows
     * @param cols Number of columns
     */
    static void precomputeLogMatrix(const double* probMatrix, double* logMatrix, std::size_t rows,
                                    std::size_t cols) noexcept;

    /**
     * @brief SIMD-optimized log-space matrix-vector multiplication
     *
     * Performs: result[i] = logSumExp_j(logMatrix[i*cols + j] + logVector[j])
     *
     * @param logMatrix Log-space matrix (row-major)
     * @param logVector Log-space vector
     * @param result Output log-space vector
     * @param rows Number of matrix rows
     * @param cols Number of matrix columns
     */
    static void logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                        double* result, std::size_t rows,
                                        std::size_t cols) noexcept;

    /**
     * @brief SIMD-optimized transposed log-space matrix-vector multiplication
     *
     * Performs: result[j] = logSumExp_i(logMatrix[i*cols + j] + logVector[i])
     *
     * @param logMatrix Log-space matrix (row-major)
     * @param logVector Log-space vector
     * @param result Output log-space vector
     * @param rows Number of matrix rows
     * @param cols Number of matrix columns
     */
    static void logMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                                  double* result, std::size_t rows,
                                                  std::size_t cols) noexcept;

    /**
     * @brief Check if log value represents zero (is LOG_ZERO or NaN)
     */
    static bool isLogZero(double logValue) noexcept {
        return std::isnan(logValue) || logValue <= LOG_ZERO;
    }

    /**
     * @brief Safe conversion from probability to log-space
     */
    static double safeLog(double prob) noexcept { return (prob > 0.0) ? std::log(prob) : LOG_ZERO; }

   private:
    /// Precomputed lookup table for log(1 + exp(x)) for x in [-50, 0]
    static std::array<double, LOOKUP_TABLE_SIZE> logOnePlusExpTable_;
    static bool initialized_;

    /// Internal helper for lookup table access
    static double lookupLogOnePlusExp(double x) noexcept;

    /// SIMD implementations
    static double logSumExpArraySIMD(const double* logValues, std::size_t size) noexcept;
    static double logSumExpArrayScalar(const double* logValues, std::size_t size) noexcept;
};

/**
 * @brief RAII class to automatically initialize log-space operations
 *
 * Create one instance of this at program startup to ensure
 * lookup tables are properly initialized.
 */
class LogSpaceInitializer {
   public:
    LogSpaceInitializer() { LogSpaceOps::initialize(); }
};

/// Global initializer - ensures tables are ready when library is loaded
static LogSpaceInitializer globalLogSpaceInit;

}  // namespace stats

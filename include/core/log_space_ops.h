#pragma once

#include "libstats/platform/simd.h"
#include "math_constants.h"
#include "statistical_constants.h"

#include <cmath>
#include <limits>

namespace stats {

/**
 * @brief High-performance log-space arithmetic operations
 *
 * This class provides numerically stable log-space arithmetic operations
 * commonly used in statistical calculations:
 * - Numerically stable log-sum-exp (scalar and array)
 * - SIMD-vectorized array operations via the max-shift trick
 * - Efficient handling of log(0) via LOG_ZERO sentinel
 *
 * @note THREAD SAFETY:
 * - All operations are thread-safe without any initialization requirement
 * - @c initialize() is a no-op retained for API compatibility
 */
class LogSpaceOps {
   public:
    /// Log-space representation of zero (negative infinity)
    static constexpr double LOG_ZERO = -std::numeric_limits<double>::infinity();

    /// Threshold below which exp() terms are considered negligible
    static constexpr double LOG_SUM_THRESHOLD = detail::LOG_SUM_EXP_THRESHOLD;

    /**
     * @brief No-op retained for API compatibility.
     *
     * The lookup-table optimization previously populated here was removed
     * because 1024-point linear interpolation over [-50,0] only achieves
     * ~6e-5 accuracy — insufficient for 1e-9 tolerances. All operations
     * now use @c std::log1p / @c std::exp directly.
     */
    static void initialize();

    /**
     * @brief Numerically stable log-sum-exp: log(exp(a) + exp(b))
     *
     * Uses the max-shift trick to avoid catastrophic cancellation, then
     * computes the correction via std::log1p(std::exp(diff)) for full
     * machine-precision accuracy.
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

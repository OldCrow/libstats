#include "libstats/core/log_space_ops.h"

#include "libstats/core/math_constants.h"
#include "libstats/core/safety.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/platform/simd_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace stats {

// =============================================================================
// INITIALIZATION (no-op — lookup table removed)
// =============================================================================

void LogSpaceOps::initialize() {}

// =============================================================================
// CORE LOG-SPACE OPERATIONS
// =============================================================================

double LogSpaceOps::logSumExp(double logA, double logB) noexcept {
    // Handle special cases
    if (isLogZero(logA)) {
        return isLogZero(logB) ? LOG_ZERO : logB;
    }
    if (isLogZero(logB)) {
        return logA;
    }

    // Ensure logA >= logB for numerical stability
    if (logA < logB) {
        std::swap(logA, logB);
    }

    double diff = logB - logA;  // diff is in (-inf, 0]

    // If the difference is too large, the smaller term is negligible
    if (diff < LOG_SUM_THRESHOLD) {
        return logA;
    }

    // Always use the numerically-exact direct computation.
    // The lookup table (1024 points over [-50,0], step ~0.049) only achieves
    // ~6e-5 interpolation accuracy — insufficient for the 1e-9 tolerances
    // required by log-space arithmetic tests and downstream callers.
    return logA + std::log1p(std::exp(diff));
}

double LogSpaceOps::logSumExpArray(const double* logValues, std::size_t size) noexcept {
    if (size == 0) {
        return LOG_ZERO;
    }

    if (size == 1) {
        return logValues[0];
    }

    // Use SIMD implementation if available and size is large enough
    if constexpr (arch::simd::has_simd_support()) {
        if (arch::simd::SIMDPolicy::shouldUseSIMD(size)) {
            return logSumExpArraySIMD(logValues, size);
        }
    }

    return logSumExpArrayScalar(logValues, size);
}

void LogSpaceOps::precomputeLogMatrix(const double* probMatrix, double* logMatrix, std::size_t rows,
                                      std::size_t cols) noexcept {
    const std::size_t total_size = rows * cols;

    // Use SIMD for large matrices
    if constexpr (arch::simd::has_simd_support()) {
        if (arch::simd::SIMDPolicy::shouldUseSIMD(total_size)) {
            arch::simd::VectorOps::vector_log(probMatrix, logMatrix, total_size);
            return;
        }
    }

    // Scalar fallback
    for (std::size_t i = 0; i < total_size; ++i) {
        logMatrix[i] = safeLog(probMatrix[i]);
    }
}

void LogSpaceOps::logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                          double* result, std::size_t rows,
                                          std::size_t cols) noexcept {
    // Initialize result vector
    for (std::size_t i = 0; i < rows; ++i) {
        result[i] = LOG_ZERO;
    }

    // Perform matrix-vector multiplication in log space
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const std::size_t matrix_idx = i * cols + j;
            const double logProduct = logMatrix[matrix_idx] + logVector[j];

            if (!isLogZero(logProduct)) {
                result[i] = logSumExp(result[i], logProduct);
            }
        }
    }
}

void LogSpaceOps::logMatrixVectorMultiplyTransposed(const double* logMatrix,
                                                    const double* logVector, double* result,
                                                    std::size_t rows, std::size_t cols) noexcept {
    // Initialize result vector
    for (std::size_t j = 0; j < cols; ++j) {
        result[j] = LOG_ZERO;
    }

    // Perform transposed matrix-vector multiplication in log space
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const std::size_t matrix_idx = i * cols + j;
            const double logProduct = logMatrix[matrix_idx] + logVector[i];

            if (!isLogZero(logProduct)) {
                result[j] = logSumExp(result[j], logProduct);
            }
        }
    }
}

// =============================================================================
// SIMD IMPLEMENTATIONS
// =============================================================================

double LogSpaceOps::logSumExpArraySIMD(const double* logValues, std::size_t size) noexcept {
    // Find the maximum value for numerical stability
    double max_val = *std::max_element(logValues, logValues + size);

    // Handle the case where all values are -infinity
    if (std::isinf(max_val) && max_val < 0) {
        return LOG_ZERO;
    }

    // Use SIMD to compute sum of exp(logValues[i] - max_val)
    // This is a simplified version - in practice, you'd use specific SIMD intrinsics
    double sum_exp = detail::ZERO_DOUBLE;

    // Process SIMD blocks
    const std::size_t simd_width = arch::simd::double_vector_width();
    const std::size_t simd_blocks = size / simd_width;

    for (std::size_t block = 0; block < simd_blocks; ++block) {
        const std::size_t base_idx = block * simd_width;

        // In a real implementation, this would use SIMD intrinsics
        for (std::size_t i = 0; i < simd_width; ++i) {
            double val = logValues[base_idx + i];
            if (std::isfinite(val)) {
                sum_exp += std::exp(val - max_val);
            }
        }
    }

    // Handle remaining elements
    for (std::size_t i = simd_blocks * simd_width; i < size; ++i) {
        double val = logValues[i];
        if (std::isfinite(val)) {
            sum_exp += std::exp(val - max_val);
        }
    }

    return detail::safe_log(sum_exp) + max_val;
}

double LogSpaceOps::logSumExpArrayScalar(const double* logValues, std::size_t size) noexcept {
    // Find the maximum value for numerical stability
    double max_val = logValues[0];
    for (std::size_t i = 1; i < size; ++i) {
        if (logValues[i] > max_val) {
            max_val = logValues[i];
        }
    }

    // Handle the case where all values are -infinity
    if (std::isinf(max_val) && max_val < 0) {
        return LOG_ZERO;
    }

    // Compute sum of exp(logValues[i] - max_val)
    double sum_exp = detail::ZERO_DOUBLE;
    for (std::size_t i = 0; i < size; ++i) {
        double val = logValues[i];
        if (std::isfinite(val)) {
            sum_exp += std::exp(val - max_val);
        }
    }

    return detail::safe_log(sum_exp) + max_val;
}

// =============================================================================
// INTEGRATION WITH EXISTING SIMD INFRASTRUCTURE
// =============================================================================
//
// LogSpaceOps integrates with the existing arch::simd::VectorOps infrastructure
// rather than implementing its own platform-specific SIMD code.
// This ensures consistency and leverages the well-tested SIMD implementations.

}  // namespace stats

#include "libstats/core/log_space_ops.h"

#include "libstats/core/math_constants.h"
#include "libstats/core/safety.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/platform/simd.h"
#include "libstats/platform/simd_policy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

namespace stats {

// =============================================================================
// INITIALIZATION (no-op — lookup table removed)
// =============================================================================

void LogSpaceOps::initialize() {
    // call_once guard makes repeated calls idempotent and safe under concurrent
    // initialization from multiple TUs, each of which instantiates the static
    // globalLogSpaceInit from the header. Currently this function is a no-op,
    // but the guard prevents any future content from running multiple times.
    static std::once_flag initFlag;
    std::call_once(initFlag, []() {
        // No-op: lookup table removed (1024-point interpolation over [-50,0]
        // achieved only ~6e-5 accuracy, insufficient for 1e-9 tolerances).
        // Retaining the guard for future expansion.
    });
}

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

    // logSumExpArrayFallback is partial-SIMD: VectorOps::scalar_add +
    // VectorOps::vector_exp vectorise the expensive shift+exponentiate step;
    // the horizontal reduction is a scalar loop (no vector_hadd primitive yet).
    // Gate to promote to full-SIMD: implement vector_hadd across all five
    // SIMD backends (SSE2/AVX/AVX2/NEON/AVX-512) once logSumExpArray becomes
    // a measured bottleneck or a second caller needs a reduction primitive.
    if constexpr (arch::simd::has_simd_support()) {
        if (arch::simd::SIMDPolicy::shouldUseSIMD(size)) {
            return logSumExpArrayFallback(logValues, size);
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
            // Harmonise with safeLog semantics: vector_log returns NaN for
            // non-positive inputs; safeLog returns -inf. Replace NaN → -inf so
            // SIMD and scalar paths agree for zero-probability transitions.
            for (std::size_t i = 0; i < total_size; ++i)
                if (std::isnan(logMatrix[i]))
                    logMatrix[i] = -std::numeric_limits<double>::infinity();
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
// PARTIAL-SIMD IMPLEMENTATION (shift+exp via VectorOps; reduction scalar)
// =============================================================================

double LogSpaceOps::logSumExpArrayFallback(const double* logValues, std::size_t size) noexcept {
    double max_val = *std::max_element(logValues, logValues + size);

    if (std::isinf(max_val) && max_val < 0) {
        return LOG_ZERO;
    }

    // Shift by -max_val and exponentiate using SIMD, then reduce scalar.
    // Scratch buffer avoids modifying the caller's data.
    std::vector<double, arch::simd::aligned_allocator<double>> scratch(size);
    arch::simd::VectorOps::scalar_add(logValues, -max_val, scratch.data(), size);
    arch::simd::VectorOps::vector_exp(scratch.data(), scratch.data(), size);

    // Horizontal sum — scalar because no vector_hadd primitive exists yet.
    double sum_exp = detail::ZERO_DOUBLE;
    for (std::size_t i = 0; i < size; ++i) {
        sum_exp += scratch[i];
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

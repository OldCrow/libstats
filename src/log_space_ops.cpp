#include "../include/core/log_space_ops.h"
#include "../include/core/safety.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace libstats {

// =============================================================================
// STATIC MEMBER INITIALIZATION
// =============================================================================

std::array<double, LogSpaceOps::LOOKUP_TABLE_SIZE> LogSpaceOps::logOnePlusExpTable_{};
bool LogSpaceOps::initialized_ = false;

// =============================================================================
// INITIALIZATION
// =============================================================================

void LogSpaceOps::initialize() {
    if (initialized_) {
        return;
    }
    
    // Precompute log(1 + exp(x)) for x in [-50, 0]
    // This covers the range where numerical precision matters most
    constexpr double x_min = -50.0;
    constexpr double x_max = 0.0;
    constexpr double step = (x_max - x_min) / (LOOKUP_TABLE_SIZE - 1);
    
    for (std::size_t i = 0; i < LOOKUP_TABLE_SIZE; ++i) {
        double x = x_min + i * step;
        logOnePlusExpTable_[i] = std::log1p(std::exp(x));
    }
    
    initialized_ = true;
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
    
    double diff = logB - logA;
    
    // If the difference is too large, the smaller term is negligible
    if (diff < LOG_SUM_THRESHOLD) {
        return logA;
    }
    
    // Use lookup table for common range
    if (diff >= -50.0 && diff <= 0.0) {
        return logA + lookupLogOnePlusExp(diff);
    }
    
    // Fallback to direct computation
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
    if constexpr (simd::has_simd_support()) {
        if (size >= simd::tuned::min_states_for_simd()) {
            return logSumExpArraySIMD(logValues, size);
        }
    }
    
    return logSumExpArrayScalar(logValues, size);
}

void LogSpaceOps::precomputeLogMatrix(const double* probMatrix, double* logMatrix, 
                                     std::size_t rows, std::size_t cols) noexcept {
    const std::size_t total_size = rows * cols;
    
    // Use SIMD for large matrices
    if constexpr (simd::has_simd_support()) {
        if (total_size >= simd::tuned::min_states_for_simd()) {
            simd::VectorOps::vector_log(probMatrix, logMatrix, total_size);
            return;
        }
    }
    
    // Scalar fallback
    for (std::size_t i = 0; i < total_size; ++i) {
        logMatrix[i] = safeLog(probMatrix[i]);
    }
}

void LogSpaceOps::logMatrixVectorMultiply(const double* logMatrix, const double* logVector,
                                         double* result, std::size_t rows, std::size_t cols) noexcept {
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

void LogSpaceOps::logMatrixVectorMultiplyTransposed(const double* logMatrix, const double* logVector,
                                                   double* result, std::size_t rows, std::size_t cols) noexcept {
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
// INTERNAL HELPER FUNCTIONS
// =============================================================================

double LogSpaceOps::lookupLogOnePlusExp(double x) noexcept {
    // x should be in [-50, 0] for the lookup table
    if (x < -50.0) {
        return std::exp(x);  // log(1 + exp(x)) ≈ exp(x) for very small x
    }
    if (x > 0.0) {
        return x + std::log1p(std::exp(-x));  // Use alternative form for x > 0
    }
    
    // Linear interpolation in lookup table
    constexpr double x_min = -50.0;
    constexpr double x_max = 0.0;
    constexpr double step = (x_max - x_min) / (LOOKUP_TABLE_SIZE - 1);
    
    const double index_real = (x - x_min) / step;
    const std::size_t index_low = static_cast<std::size_t>(std::floor(index_real));
    const std::size_t index_high = std::min(index_low + 1, LOOKUP_TABLE_SIZE - 1);
    
    if (index_low >= LOOKUP_TABLE_SIZE - 1) {
        return logOnePlusExpTable_[LOOKUP_TABLE_SIZE - 1];
    }
    
    // Linear interpolation
    const double frac = index_real - index_low;
    const double low_val = logOnePlusExpTable_[index_low];
    const double high_val = logOnePlusExpTable_[index_high];
    
    return low_val + frac * (high_val - low_val);
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
    double sum_exp = 0.0;
    
    // Process SIMD blocks
    const std::size_t simd_width = simd::double_vector_width();
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
    
    return safety::safe_log(sum_exp) + max_val;
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
    double sum_exp = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        double val = logValues[i];
        if (std::isfinite(val)) {
            sum_exp += std::exp(val - max_val);
        }
    }
    
    return safety::safe_log(sum_exp) + max_val;
}

// =============================================================================
// INTEGRATION WITH EXISTING SIMD INFRASTRUCTURE
// =============================================================================
// 
// LogSpaceOps integrates with the existing simd::VectorOps infrastructure
// rather than implementing its own platform-specific SIMD code.
// This ensures consistency and leverages the well-tested SIMD implementations.

} // namespace libstats

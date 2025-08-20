// Scalar fallback implementations - no SIMD instructions
// These implementations work on any CPU and serve as the baseline

#include "../include/core/constants.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/platform_constants.h"
#include "../include/platform/simd.h"
#include "../include/platform/simd_policy.h"

#include <algorithm>

namespace libstats {
namespace simd {

//========== Fallback Implementations (Scalar) ==========

double VectorOps::dot_product_fallback(const double* a, const double* b,
                                       std::size_t size) noexcept {
    double sum = 0.0;
    for (std::size_t i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void VectorOps::vector_add_fallback(const double* a, const double* b, double* result,
                                    std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_fallback(const double* a, const double* b, double* result,
                                         std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_fallback(const double* a, const double* b, double* result,
                                         std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_fallback(const double* a, double scalar, double* result,
                                         std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_fallback(const double* a, double scalar, double* result,
                                    std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void VectorOps::vector_exp_fallback(const double* values, double* results,
                                    std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}

void VectorOps::vector_log_fallback(const double* values, double* results,
                                    std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}

void VectorOps::vector_pow_fallback(const double* base, double exponent, double* results,
                                    std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_erf_fallback(const double* values, double* results,
                                    std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}

//========== Helper Functions with Platform-Aware Optimization ==========

bool VectorOps::should_use_simd(std::size_t size) noexcept {
    // Delegate to the centralized SIMD policy
    return SIMDPolicy::shouldUseSIMD(size);
}

std::size_t VectorOps::min_simd_size() noexcept {
    // Return platform-adaptive minimum size
    return constants::platform::get_min_simd_size();
}

bool VectorOps::supports_vectorization() noexcept {
// Check if any form of vectorization is available
// This includes SIMD or platform-specific optimizations
#if defined(LIBSTATS_HAS_AVX512) || defined(LIBSTATS_HAS_AVX2) || defined(LIBSTATS_HAS_AVX) ||     \
    defined(LIBSTATS_HAS_SSE2) || defined(LIBSTATS_HAS_NEON)
    return true;
#else
    // Even without SIMD, we can provide vectorization through
    // optimized scalar implementations and compiler auto-vectorization
    return true;
#endif
}

std::size_t VectorOps::double_vector_width() noexcept {
    // Return platform-optimal SIMD register width for double precision
    return constants::platform::get_optimal_simd_block_size();
}

}  // namespace simd
}  // namespace libstats

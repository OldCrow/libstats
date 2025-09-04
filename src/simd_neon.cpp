// ARM NEON-specific SIMD implementations
// This file is compiled ONLY with NEON flags to ensure safety

#if defined(__GNUC__) && !defined(__clang__)
    #if defined(__aarch64__) || defined(_M_ARM64)
        // NEON is mandatory in AArch64, no explicit target needed
        #pragma GCC target("neon")
    #elif defined(__arm__) || defined(_M_ARM)
        #pragma GCC target("neon")
    #endif
#elif defined(__clang__)
    #if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        // Clang uses different target attribute syntax
        #pragma clang attribute push(__attribute__((target("neon"))), apply_to = function)
    #endif
#endif

#include "../include/common/simd_implementation_common.h"

// Only include NEON intrinsics on ARM platforms
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #include <arm_neon.h>
#endif

#include <cmath>

namespace stats {
namespace simd {
namespace ops {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)

// All NEON functions use double-precision (64-bit) values
// NEON processes 2 doubles per 128-bit register

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if NEON not supported
    if (!stats::arch::supports_neon()) {
        return dot_product_fallback(a, b, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    // Apple Silicon optimization: Use multiple accumulators to exploit
    // superscalar execution and out-of-order capabilities
    #if defined(LIBSTATS_APPLE_SILICON)
    if (size >= stats::arch::simd::OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD * 2) {
        float64x2_t sum1 = vdupq_n_f64(detail::ZERO_DOUBLE);
        float64x2_t sum2 = vdupq_n_f64(detail::ZERO_DOUBLE);

        const std::size_t unroll_end =
            (size / (stats::arch::simd::NEON_UNROLL * 2)) * (stats::arch::simd::NEON_UNROLL * 2);

        // Process 4 doubles per iteration (2 NEON registers)
        for (std::size_t i = 0; i < unroll_end; i += stats::arch::simd::NEON_UNROLL * 2) {
            // Load data
            float64x2_t va1 = vld1q_f64(&a[i]);
            float64x2_t vb1 = vld1q_f64(&b[i]);
            float64x2_t va2 = vld1q_f64(&a[i + 2]);
            float64x2_t vb2 = vld1q_f64(&b[i + 2]);

            // Multiply and accumulate with independent accumulators
            sum1 = vfmaq_f64(sum1, va1, vb1);
            sum2 = vfmaq_f64(sum2, va2, vb2);
        }

        // Combine accumulators
        float64x2_t sum = vaddq_f64(sum1, sum2);

        // Handle remaining SIMD-width elements
        for (std::size_t i = unroll_end; i < simd_end; i += NEON_DOUBLE_WIDTH) {
            float64x2_t va = vld1q_f64(&a[i]);
            float64x2_t vb = vld1q_f64(&b[i]);
            sum = vfmaq_f64(sum, va, vb);
        }

        // Extract horizontal sum
        double final_sum = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

        // Handle remaining scalar elements
        for (std::size_t i = simd_end; i < size; ++i) {
            final_sum += a[i] * b[i];
        }

        return final_sum;
    }
    #endif

    // Standard NEON implementation for smaller sizes or non-Apple Silicon
    float64x2_t sum = vdupq_n_f64(detail::ZERO_DOUBLE);

    // Process pairs of doubles
    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);

        // Multiply and accumulate: sum += va * vb
        sum = vfmaq_f64(sum, va, vb);
    }

    // Extract horizontal sum
    double final_sum = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_neon(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    // Apple Silicon optimization: Loop unrolling for better throughput
    #if defined(LIBSTATS_APPLE_SILICON)
    if (size >= stats::arch::simd::OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD * 2) {
        const std::size_t unroll_end =
            (size / (stats::arch::simd::NEON_UNROLL * 2)) * (stats::arch::simd::NEON_UNROLL * 2);

        // Process 4 doubles per iteration
        for (std::size_t i = 0; i < unroll_end; i += stats::arch::simd::NEON_UNROLL * 2) {
            // Load and process 2 NEON registers worth of data
            float64x2_t va1 = vld1q_f64(&a[i]);
            float64x2_t vb1 = vld1q_f64(&b[i]);
            float64x2_t va2 = vld1q_f64(&a[i + 2]);
            float64x2_t vb2 = vld1q_f64(&b[i + 2]);

            // Compute results
            float64x2_t vresult1 = vaddq_f64(va1, vb1);
            float64x2_t vresult2 = vaddq_f64(va2, vb2);

            // Store results
            vst1q_f64(&result[i], vresult1);
            vst1q_f64(&result[i + detail::TWO_INT], vresult2);
        }

        // Handle remaining SIMD-width elements
        for (std::size_t i = unroll_end; i < simd_end; i += NEON_DOUBLE_WIDTH) {
            float64x2_t va = vld1q_f64(&a[i]);
            float64x2_t vb = vld1q_f64(&b[i]);
            float64x2_t vresult = vaddq_f64(va, vb);
            vst1q_f64(&result[i], vresult);
        }
    } else
    #endif
    {
        // Standard NEON implementation
        for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
            float64x2_t va = vld1q_f64(&a[i]);
            float64x2_t vb = vld1q_f64(&b[i]);
            float64x2_t vresult = vaddq_f64(va, vb);
            vst1q_f64(&result[i], vresult);
        }
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vsubq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vmulq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    float64x2_t vscalar = vdupq_n_f64(scalar);
    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vmulq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    float64x2_t vscalar = vdupq_n_f64(scalar);
    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vaddq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// Transcendental functions - NEON doesn't have native support, use scalar fallback
void VectorOps::vector_exp_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_exp_fallback(a, result, size);
    }
    // NEON doesn't have native exponential instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::exp(a[i]);
    }
}

void VectorOps::vector_log_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_log_fallback(a, result, size);
    }
    // NEON doesn't have native logarithm instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::log(a[i]);
    }
}

void VectorOps::vector_pow_neon(const double* base, double exponent, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_pow_fallback(base, exponent, result, size);
    }
    // NEON doesn't have native power instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_pow_elementwise_neon(const double* base, const double* exponent,
                                            double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            result[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }
    // NEON doesn't have native power instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::pow(base[i], exponent[i]);
    }
}

void VectorOps::vector_erf_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_erf_fallback(a, result, size);
    }
    // NEON doesn't have native error function instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::erf(a[i]);
    }
}

#else

// Fallback implementations for non-ARM platforms
// These will never be called, but we need them for linking

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    return dot_product_fallback(a, b, size);
}

void VectorOps::vector_add_neon(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    vector_add_fallback(a, b, result, size);
}

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    vector_subtract_fallback(a, b, result, size);
}

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    vector_multiply_fallback(a, b, result, size);
}

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    scalar_multiply_fallback(a, scalar, result, size);
}

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    scalar_add_fallback(a, scalar, result, size);
}

void VectorOps::vector_exp_neon(const double* a, double* result, std::size_t size) noexcept {
    vector_exp_fallback(a, result, size);
}

void VectorOps::vector_log_neon(const double* a, double* result, std::size_t size) noexcept {
    vector_log_fallback(a, result, size);
}

void VectorOps::vector_pow_neon(const double* base, double exponent, double* result,
                                std::size_t size) noexcept {
    vector_pow_fallback(base, exponent, result, size);
}

void VectorOps::vector_pow_elementwise_neon(const double* base, const double* exponent,
                                            double* result, std::size_t size) noexcept {
    // Fallback to scalar implementation for non-ARM platforms
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::pow(base[i], exponent[i]);
    }
}

void VectorOps::vector_erf_neon(const double* a, double* result, std::size_t size) noexcept {
    vector_erf_fallback(a, result, size);
}

#endif  // ARM platform check

}  // namespace ops
}  // namespace simd
}  // namespace stats

#ifdef __clang__
    #if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        #pragma clang attribute pop
    #endif
#endif

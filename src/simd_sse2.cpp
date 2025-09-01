// SSE2-specific SIMD implementations
// This file is compiled ONLY with SSE2 flags to ensure safety

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC target("sse2")
    #pragma GCC target("no-avx512f,no-avx2,no-avx")
#elif defined(__clang__)
    // Clang uses different target attribute syntax
    #pragma clang attribute push(__attribute__((target("sse2"))), apply_to = function)
#elif defined(_MSC_VER)
// MSVC doesn't need target pragmas - uses /arch flags in CMake
// and has different intrinsic handling
#endif

#include "../include/common/cpu_detection_fwd.h"       // Use lightweight forward declarations
#include "../include/common/platform_constants_fwd.h"  // Use lightweight forward declarations
#include "../include/core/mathematical_constants.h"
#include "../include/core/threshold_constants.h"
#include "../include/platform/simd.h"

#include <cmath>
#include <emmintrin.h>  // SSE2 intrinsics

namespace stats {
namespace simd {
namespace ops {

// All SSE2 functions use double-precision (64-bit) values
// SSE2 processes 2 doubles per 128-bit register

double VectorOps::dot_product_sse2(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if SSE2 not supported
    if (!stats::arch::supports_sse2()) {
        return dot_product_fallback(a, b, size);
    }

    __m128d sum = _mm_setzero_pd();
    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    // Process pairs of doubles
    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d prod = _mm_mul_pd(va, vb);
        sum = _mm_add_pd(sum, prod);
    }

    // Extract horizontal sum
    double result[2];
    _mm_storeu_pd(result, sum);
    double final_sum = result[0] + result[1];

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_sse2(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_add_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_sse2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_sub_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_sse2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_mul_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_sse2(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m128d vscalar = _mm_set1_pd(scalar);
    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vresult = _mm_mul_pd(va, vscalar);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_sse2(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m128d vscalar = _mm_set1_pd(scalar);
    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vresult = _mm_add_pd(va, vscalar);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats

#ifdef __clang__
    #pragma clang attribute pop
#endif

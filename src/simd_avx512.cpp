// AVX-512-specific SIMD implementations
// This file is compiled ONLY with AVX-512 flags and includes runtime safety checks

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC target("avx512f")
#elif defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx512f"))), apply_to = function)
#elif defined(_MSC_VER)
// MSVC doesn't need target pragmas - uses /arch flags in CMake
// and has different intrinsic handling
#endif

#include "../include/common/simd_implementation_common.h"

#include <cmath>
#include <immintrin.h>  // AVX-512 intrinsics

namespace stats {
namespace simd {
namespace ops {

// All AVX-512 functions use double-precision (64-bit) values
// AVX-512 processes 8 doubles per 512-bit register

double VectorOps::dot_product_avx512(const double* a, const double* b, std::size_t size) noexcept {
    // CRITICAL: Runtime safety check - bail out if AVX-512 not supported
    // This prevents illegal instruction crashes on CPUs without AVX-512
    if (!stats::arch::supports_avx512()) {
        return dot_product_fallback(a, b, size);
    }

    __m512d sum = _mm512_setzero_pd();
    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    // Process octets of doubles
    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        // Use FMA instruction for efficiency: sum = sum + (va * vb)
        sum = _mm512_fmadd_pd(va, vb, sum);
    }

    // Extract horizontal sum
    double result[8];
    _mm512_storeu_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] +
                       result[6] + result[7];

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_avx512(const double* a, const double* b, double* result,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_avx512(const double* a, const double* b, double* result,
                                       std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx512(const double* a, const double* b, double* result,
                                       std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx512(const double* a, double scalar, double* result,
                                       std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m512d vscalar = _mm512_set1_pd(scalar);
    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vresult = _mm512_mul_pd(va, vscalar);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx512(const double* a, double scalar, double* result,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m512d vscalar = _mm512_set1_pd(scalar);
    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vresult = _mm512_add_pd(va, vscalar);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// AVX512 transcendental functions - for now, delegate to AVX implementations
// Future: could use SVML (Intel Short Vector Math Library) for better performance

void VectorOps::vector_exp_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_exp_fallback(values, results, size);
    }
    // For now, delegate to AVX implementation
    // Future: use _mm512_exp_pd from SVML if available
    return vector_exp_avx(values, results, size);
}

void VectorOps::vector_log_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_log_fallback(values, results, size);
    }
    // For now, delegate to AVX implementation
    // Future: use _mm512_log_pd from SVML if available
    return vector_log_avx(values, results, size);
}

void VectorOps::vector_pow_avx512(const double* base, double exponent, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_pow_fallback(base, exponent, results, size);
    }
    // For now, delegate to AVX implementation
    // Future: use _mm512_pow_pd from SVML if available
    return vector_pow_avx(base, exponent, results, size);
}

void VectorOps::vector_pow_elementwise_avx512(const double* base, const double* exponent,
                                              double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            results[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }
    // For now, delegate to AVX implementation
    return vector_pow_elementwise_avx(base, exponent, results, size);
}

void VectorOps::vector_erf_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_erf_fallback(values, results, size);
    }
    // For now, delegate to AVX implementation
    // Future: use _mm512_erf_pd from SVML if available
    return vector_erf_avx(values, results, size);
}

}  // namespace ops
}  // namespace simd
}  // namespace stats

#ifdef __clang__
    #pragma clang attribute pop
#endif

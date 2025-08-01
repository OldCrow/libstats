// AVX-512-specific SIMD implementations
// This file is compiled ONLY with AVX-512 flags and includes runtime safety checks

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC target("avx512f")
#elif defined(__clang__)
    #pragma clang attribute push (__attribute__((target("avx512f"))), apply_to=function)
#elif defined(_MSC_VER)
    // MSVC doesn't need target pragmas - uses /arch flags in CMake
    // and has different intrinsic handling
#endif

#include "../include/platform/simd.h"
#include "../include/platform/cpu_detection.h"
#include "../include/core/constants.h"
#include "../include/platform/platform_constants.h"
#include <immintrin.h> // AVX-512 intrinsics
#include <cmath>

namespace libstats {
namespace simd {

// All AVX-512 functions use double-precision (64-bit) values
// AVX-512 processes 8 doubles per 512-bit register

double VectorOps::dot_product_avx512(const double* a, const double* b, std::size_t size) noexcept {
    // CRITICAL: Runtime safety check - bail out if AVX-512 not supported
    // This prevents illegal instruction crashes on CPUs without AVX-512
    if (!cpu::supports_avx512()) {
        return dot_product_fallback(a, b, size);
    }
    
    __m512d sum = _mm512_setzero_pd();
    constexpr std::size_t AVX512_DOUBLE_WIDTH = constants::simd::registers::AVX512_DOUBLES;
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
    double final_sum = result[0] + result[1] + result[2] + result[3] + 
                       result[4] + result[5] + result[6] + result[7];
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }
    
    return final_sum;
}

void VectorOps::vector_add_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx512()) {
        return vector_add_fallback(a, b, result, size);
    }
    
    constexpr std::size_t AVX512_DOUBLE_WIDTH = constants::simd::registers::AVX512_DOUBLES;
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

void VectorOps::vector_subtract_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx512()) {
        return vector_subtract_fallback(a, b, result, size);
    }
    
    constexpr std::size_t AVX512_DOUBLE_WIDTH = constants::simd::registers::AVX512_DOUBLES;
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

void VectorOps::vector_multiply_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx512()) {
        return vector_multiply_fallback(a, b, result, size);
    }
    
    constexpr std::size_t AVX512_DOUBLE_WIDTH = constants::simd::registers::AVX512_DOUBLES;
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

void VectorOps::scalar_multiply_avx512(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx512()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }
    
    __m512d vscalar = _mm512_set1_pd(scalar);
    constexpr std::size_t AVX512_DOUBLE_WIDTH = constants::simd::registers::AVX512_DOUBLES;
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

void VectorOps::scalar_add_avx512(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx512()) {
        return scalar_add_fallback(a, scalar, result, size);
    }
    
    __m512d vscalar = _mm512_set1_pd(scalar);
    constexpr std::size_t AVX512_DOUBLE_WIDTH = constants::simd::registers::AVX512_DOUBLES;
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

} // namespace simd
} // namespace libstats

#ifdef __clang__
    #pragma clang attribute pop
#endif

// AVX2-specific SIMD implementations
// This file is compiled ONLY with AVX2 flags to ensure safety

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC target("avx2")
    #pragma GCC target("no-avx512f")
#elif defined(__clang__)
    // Clang uses different target attribute syntax
    #pragma clang attribute push (__attribute__((target("avx2"))), apply_to=function)
#endif

#include "../include/simd.h"
#include "../include/cpu_detection.h"
#include "../include/constants.h"
#include <immintrin.h> // AVX2 intrinsics
#include <cmath>

namespace libstats {
namespace simd {

// All AVX2 functions use double-precision (64-bit) values
// AVX2 processes 4 doubles per 256-bit register (same as AVX but with better integer support)

double VectorOps::dot_product_avx2(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if AVX2 not supported
    if (!cpu::supports_avx2()) {
        return dot_product_fallback(a, b, size);
    }
    
    __m256d sum = _mm256_setzero_pd();
    constexpr std::size_t AVX2_DOUBLE_WIDTH = constants::simd::registers::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;
    
    // Process quartets of doubles with FMA if available
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        
        // Use FMA instruction if available for better performance and accuracy
        #ifdef __FMA__
        sum = _mm256_fmadd_pd(va, vb, sum);
        #else
        __m256d prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
        #endif
    }
    
    // Extract horizontal sum
    double result[4];
    _mm256_storeu_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3];
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }
    
    return final_sum;
}

void VectorOps::vector_add_avx2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx2()) {
        return vector_add_fallback(a, b, result, size);
    }
    
    constexpr std::size_t AVX2_DOUBLE_WIDTH = constants::simd::registers::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;
    
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_avx2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx2()) {
        return vector_subtract_fallback(a, b, result, size);
    }
    
    constexpr std::size_t AVX2_DOUBLE_WIDTH = constants::simd::registers::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;
    
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx2()) {
        return vector_multiply_fallback(a, b, result, size);
    }
    
    constexpr std::size_t AVX2_DOUBLE_WIDTH = constants::simd::registers::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;
    
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx2(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx2()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }
    
    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX2_DOUBLE_WIDTH = constants::simd::registers::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;
    
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_mul_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx2(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!cpu::supports_avx2()) {
        return scalar_add_fallback(a, scalar, result, size);
    }
    
    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX2_DOUBLE_WIDTH = constants::simd::registers::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;
    
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_add_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
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

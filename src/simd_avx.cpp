// AVX-specific SIMD implementations
// This file is compiled ONLY with AVX flags to ensure safety

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC target("avx")
    #pragma GCC target("no-avx512f,no-avx2")
#elif defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx"))), apply_to = function)
#elif defined(_MSC_VER)
// MSVC doesn't need target pragmas - uses /arch flags in CMake
// and has different intrinsic handling
#endif

#include "../include/core/constants.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/platform_constants.h"
#include "../include/platform/simd.h"

#include <cmath>
#include <immintrin.h>  // AVX intrinsics

namespace libstats {
namespace simd {

// All AVX functions use double-precision (64-bit) values
// AVX processes 4 doubles per 256-bit register

double VectorOps::dot_product_avx(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if AVX not supported
    if (!cpu::supports_avx()) {
        return dot_product_fallback(a, b, size);
    }

    __m256d sum = _mm256_setzero_pd();
    constexpr std::size_t AVX_DOUBLE_WIDTH = constants::simd::registers::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    // Process quartets of doubles
    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
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

void VectorOps::vector_add_avx(const double* a, const double* b, double* result,
                               std::size_t size) noexcept {
    if (!cpu::supports_avx()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = constants::simd::registers::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
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

void VectorOps::vector_subtract_avx(const double* a, const double* b, double* result,
                                    std::size_t size) noexcept {
    if (!cpu::supports_avx()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = constants::simd::registers::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx(const double* a, const double* b, double* result,
                                    std::size_t size) noexcept {
    if (!cpu::supports_avx()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = constants::simd::registers::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx(const double* a, double scalar, double* result,
                                    std::size_t size) noexcept {
    if (!cpu::supports_avx()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX_DOUBLE_WIDTH = constants::simd::registers::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_mul_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx(const double* a, double scalar, double* result,
                               std::size_t size) noexcept {
    if (!cpu::supports_avx()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX_DOUBLE_WIDTH = constants::simd::registers::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_add_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

}  // namespace simd
}  // namespace libstats

#ifdef __clang__
    #pragma clang attribute pop
#endif

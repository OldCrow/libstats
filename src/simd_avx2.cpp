// AVX2-specific SIMD implementations
// This file is compiled ONLY with AVX2 flags to ensure safety

#include "libstats/common/simd_implementation_common.h"

#include <cmath>
#include <immintrin.h>  // AVX2 intrinsics

namespace stats {
namespace simd {
namespace ops {

// All AVX2 functions use double-precision (64-bit) values
// AVX2 processes 4 doubles per 256-bit register (same as AVX but with better integer support)

double VectorOps::dot_product_avx2(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if AVX2 not supported
    if (!stats::arch::supports_avx2()) {
        return dot_product_fallback(a, b, size);
    }

    // 4-accumulator unrolled loop breaks the single-accumulator latency chain.
    // Each accumulator is independent so 4 FMA units (Kaby Lake, Zen 4) work in parallel.
    // NEON backend already uses 2-accumulator pattern; this extends to 4 for x86 AVX2.
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    constexpr std::size_t W = arch::simd::AVX2_DOUBLES;  // 4 doubles
    constexpr std::size_t U = 4;                         // unroll factor
    constexpr std::size_t UW = U * W;                    // 16 doubles per iteration
    const std::size_t unroll_end = (size / UW) * UW;
    const std::size_t simd_end = (size / W) * W;

#ifdef __FMA__
    for (std::size_t i = 0; i < unroll_end; i += UW) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 0]), _mm256_loadu_pd(&b[i + 0]), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 4]), _mm256_loadu_pd(&b[i + 4]), acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 8]), _mm256_loadu_pd(&b[i + 8]), acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i + 12]), _mm256_loadu_pd(&b[i + 12]), acc3);
    }
    // Drain the tail (< 16 elements) one W at a time into acc0
    for (std::size_t i = unroll_end; i < simd_end; i += W) {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i]), acc0);
    }
#else
    for (std::size_t i = 0; i < unroll_end; i += UW) {
        acc0 = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(&a[i + 0]), _mm256_loadu_pd(&b[i + 0])),
                             acc0);
        acc1 = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(&a[i + 4]), _mm256_loadu_pd(&b[i + 4])),
                             acc1);
        acc2 = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(&a[i + 8]), _mm256_loadu_pd(&b[i + 8])),
                             acc2);
        acc3 = _mm256_add_pd(
            _mm256_mul_pd(_mm256_loadu_pd(&a[i + 12]), _mm256_loadu_pd(&b[i + 12])), acc3);
    }
    for (std::size_t i = unroll_end; i < simd_end; i += W) {
        acc0 = _mm256_add_pd(_mm256_mul_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])), acc0);
    }
#endif

    // Combine accumulators and extract horizontal sum
    __m256d sum = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    __m128d lo = _mm256_castpd256_pd128(sum);
    __m128d hi = _mm256_extractf128_pd(sum, 1);
    __m128d s = _mm_add_pd(lo, hi);
    __m128d s2 = _mm_unpackhi_pd(s, s);
    double final_sum = _mm_cvtsd_f64(_mm_add_pd(s, s2));

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_avx2(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
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

void VectorOps::vector_subtract_avx2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
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

void VectorOps::vector_multiply_avx2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
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

void VectorOps::scalar_multiply_avx2(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
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

void VectorOps::scalar_add_avx2(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
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

}  // namespace ops
}  // namespace simd
}  // namespace stats

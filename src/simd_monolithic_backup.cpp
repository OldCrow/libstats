#include "../include/simd.h"
#include "../include/constants.h"
#include "../include/cpu_detection.h"
#include <cstring>
#include <cmath>
#include <climits>
#include <algorithm>

namespace libstats {
namespace simd {

//========== Public Interface Implementations ==========

double VectorOps::dot_product(const double* a, const double* b, std::size_t size) noexcept {
#ifdef LIBSTATS_HAS_AVX512
    if (supports_vectorization() && cpu::supports_avx512() && size >= double_vector_width()) {
        return dot_product_avx512(a, b, size);
    }
#endif
#ifdef LIBSTATS_HAS_AVX
    if (supports_vectorization() && cpu::supports_avx() && size >= DOUBLE_SIMD_WIDTH) {
        return dot_product_avx(a, b, size);
    }
#endif
#ifdef LIBSTATS_HAS_SSE2
    if (supports_vectorization() && cpu::supports_sse2() && size >= DOUBLE_SIMD_WIDTH) {
        return dot_product_sse2(a, b, size);
    }
#endif
#ifdef LIBSTATS_HAS_NEON
    if (supports_vectorization() && cpu::supports_neon() && size >= DOUBLE_SIMD_WIDTH) {
        return dot_product_neon(a, b, size);
    }
#endif
    return dot_product_fallback(a, b, size);
}

void VectorOps::vector_add(const double* a, const double* b, double* result, std::size_t size) noexcept {
#ifdef LIBSTATS_HAS_AVX512
    if (supports_vectorization() && cpu::supports_avx512() && size >= double_vector_width()) {
        vector_add_avx512(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_AVX
    if (supports_vectorization() && cpu::supports_avx() && size >= DOUBLE_SIMD_WIDTH) {
        vector_add_avx(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_SSE2
    if (supports_vectorization() && cpu::supports_sse2() && size >= DOUBLE_SIMD_WIDTH) {
        vector_add_sse2(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_NEON
    if (supports_vectorization() && cpu::supports_neon() && size >= DOUBLE_SIMD_WIDTH) {
        vector_add_neon(a, b, result, size);
        return;
    }
#endif
    vector_add_fallback(a, b, result, size);
}

void VectorOps::vector_subtract(const double* a, const double* b, double* result, std::size_t size) noexcept {
#ifdef LIBSTATS_HAS_AVX512
    if (supports_vectorization() && cpu::supports_avx512() && size >= double_vector_width()) {
        vector_subtract_avx512(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_AVX
    if (supports_vectorization() && cpu::supports_avx() && size >= DOUBLE_SIMD_WIDTH) {
        vector_subtract_avx(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_SSE2
    if (supports_vectorization() && cpu::supports_sse2() && size >= DOUBLE_SIMD_WIDTH) {
        vector_subtract_sse2(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_NEON
    if (supports_vectorization() && cpu::supports_neon() && size >= DOUBLE_SIMD_WIDTH) {
        vector_subtract_neon(a, b, result, size);
        return;
    }
#endif
    vector_subtract_fallback(a, b, result, size);
}

void VectorOps::vector_multiply(const double* a, const double* b, double* result, std::size_t size) noexcept {
#ifdef LIBSTATS_HAS_AVX512
    if (supports_vectorization() && cpu::supports_avx512() && size >= double_vector_width()) {
        vector_multiply_avx512(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_AVX
    if (supports_vectorization() && cpu::supports_avx() && size >= DOUBLE_SIMD_WIDTH) {
        vector_multiply_avx(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_SSE2
    if (supports_vectorization() && cpu::supports_sse2() && size >= DOUBLE_SIMD_WIDTH) {
        vector_multiply_sse2(a, b, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_NEON
    if (supports_vectorization() && cpu::supports_neon() && size >= DOUBLE_SIMD_WIDTH) {
        vector_multiply_neon(a, b, result, size);
        return;
    }
#endif
    vector_multiply_fallback(a, b, result, size);
}

void VectorOps::scalar_multiply(const double* a, double scalar, double* result, std::size_t size) noexcept {
#ifdef LIBSTATS_HAS_AVX512
    if (supports_vectorization() && cpu::supports_avx512() && size >= double_vector_width()) {
        scalar_multiply_avx512(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_AVX
    if (supports_vectorization() && cpu::supports_avx() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_multiply_avx(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_SSE2
    if (supports_vectorization() && cpu::supports_sse2() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_multiply_sse2(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_NEON
    if (supports_vectorization() && cpu::supports_neon() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_multiply_neon(a, scalar, result, size);
        return;
    }
#endif
    scalar_multiply_fallback(a, scalar, result, size);
}

void VectorOps::scalar_add(const double* a, double scalar, double* result, std::size_t size) noexcept {
#ifdef LIBSTATS_HAS_AVX512
    if (supports_vectorization() && cpu::supports_avx512() && size >= double_vector_width()) {
        scalar_add_avx512(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_AVX
    if (supports_vectorization() && cpu::supports_avx() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_add_avx(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_SSE2
    if (supports_vectorization() && cpu::supports_sse2() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_add_sse2(a, scalar, result, size);
        return;
    }
#endif
#ifdef LIBSTATS_HAS_NEON
    if (supports_vectorization() && cpu::supports_neon() && size >= DOUBLE_SIMD_WIDTH) {
        scalar_add_neon(a, scalar, result, size);
        return;
    }
#endif
    scalar_add_fallback(a, scalar, result, size);
}

void VectorOps::vector_exp(const double* values, double* results, std::size_t size) noexcept {
    if (should_use_simd(size)) {
#ifdef LIBSTATS_HAS_AVX
        vector_exp_avx(values, results, size);
        return;
#endif
#ifdef LIBSTATS_HAS_SSE2
        vector_exp_sse2(values, results, size);
        return;
#endif
#ifdef LIBSTATS_HAS_NEON
        vector_exp_neon(values, results, size);
        return;
#endif
    }
    vector_exp_fallback(values, results, size);
}

void VectorOps::vector_log(const double* values, double* results, std::size_t size) noexcept {
    // For now, use optimized fallback implementation
    // Full SIMD implementations of log() are complex and require careful implementation
    // of range reduction, polynomial approximation, and special case handling
    vector_log_fallback(values, results, size);
}

void VectorOps::vector_pow(const double* base, double exponent, double* results, std::size_t size) noexcept {
    // For now, use fallback implementation
    vector_pow_fallback(base, exponent, results, size);
}

void VectorOps::vector_erf(const double* values, double* results, std::size_t size) noexcept {
    // For now, use fallback implementation
    // SIMD implementations of erf() are complex and can be added later
    vector_erf_fallback(values, results, size);
}

bool VectorOps::should_use_simd(std::size_t size) noexcept {
    // Runtime integration with CPU detection
    const auto& features = cpu::get_features();
    
    // Check if SIMD is available at runtime
    if (!features.sse2 && !features.avx && !features.avx512f && !features.neon) {
        return false;
    }
    
    // Check if size is sufficient to benefit from SIMD
    return size >= min_simd_size();
}

std::size_t VectorOps::min_simd_size() noexcept {
    // Runtime integration with CPU detection
    const auto& features = cpu::get_features();
    
    if (features.avx512f) {
        return tuned::min_states_for_simd(); // 16 for AVX-512
    } else if (features.avx || features.avx2) {
        return tuned::min_states_for_simd(); // 8 for AVX
    } else if (features.sse2) {
        return tuned::min_states_for_simd(); // 4 for SSE2
    } else if (features.neon) {
        return tuned::min_states_for_simd(); // 4 for NEON
    } else {
        return SIZE_MAX; // No SIMD available
    }
}


//========== Fallback Implementations ==========

double VectorOps::dot_product_fallback(const double* a, const double* b, std::size_t size) noexcept {
    double result = constants::math::ZERO_DOUBLE;
    
    // Loop unrolling for better performance
    std::size_t unroll_limit = (size / 4) * 4;
    std::size_t i = 0;
    
    // Process 4 elements at a time
    for (; i < unroll_limit; i += 4) {
        result += a[i] * b[i] + a[i+1] * b[i+1] + a[i+2] * b[i+2] + a[i+3] * b[i+3];
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void VectorOps::vector_add_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept {
    // Loop unrolling for better performance
    std::size_t unroll_limit = (size / 4) * 4;
    std::size_t i = 0;
    
    // Process 4 elements at a time
    for (; i < unroll_limit; i += 4) {
        result[i] = a[i] + b[i];
        result[i+1] = a[i+1] + b[i+1];
        result[i+2] = a[i+2] + b[i+2];
        result[i+3] = a[i+3] + b[i+3];
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept {
    // Loop unrolling for better performance
    std::size_t unroll_limit = (size / 4) * 4;
    std::size_t i = 0;
    
    // Process 4 elements at a time
    for (; i < unroll_limit; i += 4) {
        result[i] = a[i] - b[i];
        result[i+1] = a[i+1] - b[i+1];
        result[i+2] = a[i+2] - b[i+2];
        result[i+3] = a[i+3] - b[i+3];
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept {
    // Loop unrolling for better performance
    std::size_t unroll_limit = (size / 4) * 4;
    std::size_t i = 0;
    
    // Process 4 elements at a time
    for (; i < unroll_limit; i += 4) {
        result[i] = a[i] * b[i];
        result[i+1] = a[i+1] * b[i+1];
        result[i+2] = a[i+2] * b[i+2];
        result[i+3] = a[i+3] * b[i+3];
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_fallback(const double* a, double scalar, double* result, std::size_t size) noexcept {
    // Loop unrolling for better performance
    std::size_t unroll_limit = (size / 4) * 4;
    std::size_t i = 0;
    
    // Process 4 elements at a time
    for (; i < unroll_limit; i += 4) {
        result[i] = a[i] * scalar;
        result[i+1] = a[i+1] * scalar;
        result[i+2] = a[i+2] * scalar;
        result[i+3] = a[i+3] * scalar;
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_fallback(const double* a, double scalar, double* result, std::size_t size) noexcept {
    // Loop unrolling for better performance
    std::size_t unroll_limit = (size / 4) * 4;
    std::size_t i = 0;
    
    // Process 4 elements at a time
    for (; i < unroll_limit; i += 4) {
        result[i] = a[i] + scalar;
        result[i+1] = a[i+1] + scalar;
        result[i+2] = a[i+2] + scalar;
        result[i+3] = a[i+3] + scalar;
    }
    
    // Process remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void VectorOps::vector_exp_fallback(const double* values, double* results, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}

void VectorOps::vector_log_fallback(const double* values, double* results, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}

void VectorOps::vector_pow_fallback(const double* base, double exponent, double* results, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_erf_fallback(const double* values, double* results, std::size_t size) noexcept {
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}


//========== AVX Implementations ==========

#ifdef LIBSTATS_HAS_AVX

double VectorOps::dot_product_avx(const double* a, const double* b, std::size_t size) noexcept {
    __m256d sum = _mm256_setzero_pd();
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks with unaligned loads for robustness
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        
#if defined(LIBSTATS_HAS_AVX2) && defined(__FMA__)
        // Use FMA if available for better precision and performance
        sum = _mm256_fmadd_pd(va, vb, sum);
#else
        __m256d prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
#endif
    }
    
    // Extract horizontal sum
    __m128d sum_low = _mm256_castpd256_pd128(sum);
    __m128d sum_high = _mm256_extractf128_pd(sum, 1);
    __m128d sum_combined = _mm_add_pd(sum_low, sum_high);
    __m128d sum_final = _mm_hadd_pd(sum_combined, sum_combined);
    
    double result;
    _mm_store_sd(&result, sum_final);
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void VectorOps::vector_add_avx(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
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

void VectorOps::scalar_multiply_avx(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m256d vscalar = _mm256_set1_pd(scalar);
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_mul_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::vector_subtract_avx(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
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

void VectorOps::vector_multiply_avx(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_add_avx(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m256d vscalar = _mm256_set1_pd(scalar);
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_add_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void VectorOps::vector_exp_avx(const double* values, double* results, std::size_t size) noexcept {
    // Check for runtime CPU support
    const auto& features = cpu::get_features();
    if (!features.avx) {
        return vector_exp_fallback(values, results, size);
    }
    
    // If size is too small, use fallback
    if (size < tuned::min_states_for_simd()) {
        return vector_exp_fallback(values, results, size);
    }
    
    // Currently using fallback implementation
    // TODO: Implement optimized AVX exponential using polynomial approximation
    // This would require careful handling of range reduction and special cases
    // For now, process in chunks to maintain cache efficiency
    
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;
    
    // Process in SIMD-sized chunks but use scalar exp for accuracy
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        // Prefetch next cache line
        prefetch_read(values + i + DOUBLE_SIMD_WIDTH);
        prefetch_write(results + i + DOUBLE_SIMD_WIDTH);
        
        // Process 4 elements
        for (std::size_t j = 0; j < DOUBLE_SIMD_WIDTH; ++j) {
            results[i + j] = std::exp(values[i + j]);
        }
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}

void VectorOps::vector_log_avx(const double* values, double* results, std::size_t size) noexcept {
    // Check for runtime CPU support
    const auto& features = cpu::get_features();
    if (!features.avx) {
        return vector_log_fallback(values, results, size);
    }

    // If size is too small, use fallback
    if (size < tuned::min_states_for_simd()) {
        return vector_log_fallback(values, results, size);
    }

    // For now, process in chunks to maintain cache efficiency
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;

    // Process in SIMD-sized chunks but use scalar log for accuracy
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        prefetch_read(values + i + DOUBLE_SIMD_WIDTH);
        prefetch_write(results + i + DOUBLE_SIMD_WIDTH);

        // Process 4 elements
        for (std::size_t j = 0; j < DOUBLE_SIMD_WIDTH; ++j) {
            results[i + j] = std::log(values[i + j]);
        }
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}

void VectorOps::vector_pow_avx(const double* base, double exponent, double* results, std::size_t size) noexcept {
    // Check for runtime CPU support
    const auto& features = cpu::get_features();
    if (!features.avx) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    // If size is too small, use fallback
    if (size < tuned::min_states_for_simd()) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    // For now, process in chunks to maintain cache efficiency
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;

    // Process in SIMD-sized chunks but use scalar pow for accuracy
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        prefetch_read(base + i + DOUBLE_SIMD_WIDTH);
        prefetch_write(results + i + DOUBLE_SIMD_WIDTH);

        // Process 4 elements
        for (std::size_t j = 0; j < DOUBLE_SIMD_WIDTH; ++j) {
            results[i + j] = std::pow(base[i + j], exponent);
        }
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_erf_avx(const double* values, double* results, std::size_t size) noexcept {
    // Check for runtime CPU support
    const auto& features = cpu::get_features();
    if (!features.avx) {
        return vector_erf_fallback(values, results, size);
    }

    // If size is too small, use fallback
    if (size < tuned::min_states_for_simd()) {
        return vector_erf_fallback(values, results, size);
    }

    // For now, process in chunks to maintain cache efficiency
    std::size_t simd_end = (size / DOUBLE_SIMD_WIDTH) * DOUBLE_SIMD_WIDTH;

    // Process in SIMD-sized chunks but use scalar erf for accuracy
    for (std::size_t i = 0; i < simd_end; i += DOUBLE_SIMD_WIDTH) {
        prefetch_read(values + i + DOUBLE_SIMD_WIDTH);
        prefetch_write(results + i + DOUBLE_SIMD_WIDTH);

        // Process 4 elements
        for (std::size_t j = 0; j < DOUBLE_SIMD_WIDTH; ++j) {
            results[i + j] = std::erf(values[i + j]);
        }
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}

#endif // LIBSTATS_HAS_AVX

//========== AVX512 Implementations ==========

#ifdef LIBSTATS_HAS_AVX512

double VectorOps::dot_product_avx512(const double* a, const double* b, std::size_t size) noexcept {
    __m512d sum = _mm512_setzero_pd();
    std::size_t simd_end = (size / 8) * 8;
    
    // Process SIMD blocks with unaligned loads for robustness
    for (std::size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        sum = _mm512_fmadd_pd(va, vb, sum);
    }
    
    // Extract horizontal sum
    double result[8];
    _mm512_storeu_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }
    
    return final_sum;
}

void VectorOps::vector_add_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 8) * 8;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += 8) {
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
    std::size_t simd_end = (size / 8) * 8;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 8) * 8;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx512(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m512d vscalar = _mm512_set1_pd(scalar);
    std::size_t simd_end = (size / 8) * 8;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vresult = _mm512_mul_pd(va, vscalar);
        _mm512_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx512(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m512d vscalar = _mm512_set1_pd(scalar);
    std::size_t simd_end = (size / 8) * 8;
    
    // Process SIMD blocks with unaligned operations for robustness
    for (std::size_t i = 0; i < simd_end; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vresult = _mm512_add_pd(va, vscalar);
        _mm512_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void VectorOps::vector_exp_avx512(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.avx512f) {
        return vector_exp_fallback(values, results, size);
    }
    
    if (size < tuned::min_states_for_simd()) {
        return vector_exp_fallback(values, results, size);
    }
    
    std::size_t simd_end = (size / 8) * 8;
    
    for (std::size_t i = 0; i < simd_end; i += 8) {
        prefetch_read(values + i + 8);
        prefetch_write(results + i + 8);
        
        // Fallback to scalar computation within vectorized loop
        for (std::size_t j = 0; j < 8; ++j) {
            results[i + j] = std::exp(values[i + j]);
        }
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}

void VectorOps::vector_log_avx512(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.avx512f) {
        return vector_log_fallback(values, results, size);
    }
    
    if (size < tuned::min_states_for_simd()) {
        return vector_log_fallback(values, results, size);
    }
    
    std::size_t simd_end = (size / 8) * 8;
    
    for (std::size_t i = 0; i < simd_end; i += 8) {
        prefetch_read(values + i + 8);
        prefetch_write(results + i + 8);
        
        // Fallback to scalar computation within vectorized loop
        for (std::size_t j = 0; j < 8; ++j) {
            results[i + j] = std::log(values[i + j]);
        }
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}

void VectorOps::vector_pow_avx512(const double* base, double exponent, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.avx512f) {
        return vector_pow_fallback(base, exponent, results, size);
    }
    
    if (size < tuned::min_states_for_simd()) {
        return vector_pow_fallback(base, exponent, results, size);
    }
    
    std::size_t simd_end = (size / 8) * 8;
    
    for (std::size_t i = 0; i < simd_end; i += 8) {
        prefetch_read(base + i + 8);
        prefetch_write(results + i + 8);
        
        // Fallback to scalar computation within vectorized loop
        for (std::size_t j = 0; j < 8; ++j) {
            results[i + j] = std::pow(base[i + j], exponent);
        }
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_erf_avx512(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.avx512f) {
        return vector_erf_fallback(values, results, size);
    }
    
    if (size < tuned::min_states_for_simd()) {
        return vector_erf_fallback(values, results, size);
    }
    
    std::size_t simd_end = (size / 8) * 8;
    
    for (std::size_t i = 0; i < simd_end; i += 8) {
        prefetch_read(values + i + 8);
        prefetch_write(results + i + 8);
        
        // Fallback to scalar computation within vectorized loop
        for (std::size_t j = 0; j < 8; ++j) {
            results[i + j] = std::erf(values[i + j]);
        }
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}

#endif // LIBSTATS_HAS_AVX512

//========== SSE2 Implementations ==========

#ifdef LIBSTATS_HAS_SSE2

double VectorOps::dot_product_sse2(const double* a, const double* b, std::size_t size) noexcept {
    __m128d sum = _mm_setzero_pd();
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks (2 doubles at a time) with unaligned loads
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d prod = _mm_mul_pd(va, vb);
        sum = _mm_add_pd(sum, prod);
    }
    
    // Extract horizontal sum
    __m128d sum_shuf = _mm_shuffle_pd(sum, sum, 1);
    sum = _mm_add_sd(sum, sum_shuf);
    
    double result;
    _mm_store_sd(&result, sum);
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void VectorOps::vector_add_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks with unaligned operations
    for (std::size_t i = 0; i < simd_end; i += 2) {
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

void VectorOps::scalar_multiply_sse2(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m128d vscalar = _mm_set1_pd(scalar);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks with unaligned operations
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vresult = _mm_mul_pd(va, vscalar);
        _mm_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::vector_subtract_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks with unaligned operations
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_sub_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks with unaligned operations
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_mul_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_add_sse2(const double* a, double scalar, double* result, std::size_t size) noexcept {
    __m128d vscalar = _mm_set1_pd(scalar);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process SIMD blocks with unaligned operations
    for (std::size_t i = 0; i < simd_end; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vresult = _mm_add_pd(va, vscalar);
        _mm_storeu_pd(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void VectorOps::vector_exp_sse2(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.sse2) {
        return vector_exp_fallback(values, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_exp_fallback(values, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(values + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::exp(values[i + j]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}

void VectorOps::vector_log_sse2(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.sse2) {
        return vector_log_fallback(values, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_log_fallback(values, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(values + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::log(values[i + j]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}

void VectorOps::vector_pow_sse2(const double* base, double exponent, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.sse2) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(base + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::pow(base[i + j], exponent);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_erf_sse2(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.sse2) {
        return vector_erf_fallback(values, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_erf_fallback(values, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(values + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::erf(values[i + j]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}

#endif // LIBSTATS_HAS_SSE2

//========== ARM NEON Implementations ==========

#ifdef LIBSTATS_HAS_NEON

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    float64x2_t sum = vdupq_n_f64(0.0);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks (2 doubles at a time)
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t prod = vmulq_f64(va, vb);
        sum = vaddq_f64(sum, prod);
    }
    
    // Extract horizontal sum
    double result = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void VectorOps::vector_add_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vaddq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    float64x2_t vscalar = vdupq_n_f64(scalar);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vmulq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vsubq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vmulq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    float64x2_t vscalar = vdupq_n_f64(scalar);
    std::size_t simd_end = (size / 2) * 2;
    
    // Process NEON blocks
    for (std::size_t i = 0; i < simd_end; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vaddq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }
    
    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

void VectorOps::vector_log_neon(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.neon) {
        return vector_log_fallback(values, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_log_fallback(values, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(values + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::log(values[i + j]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}

void VectorOps::vector_pow_neon(const double* base, double exponent, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.neon) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(base + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::pow(base[i + j], exponent);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_erf_neon(const double* values, double* results, std::size_t size) noexcept {
    const auto& features = cpu::get_features();
    if (!features.neon) {
        return vector_erf_fallback(values, results, size);
    }

    if (size < tuned::min_states_for_simd()) {
        return vector_erf_fallback(values, results, size);
    }

    std::size_t simd_end = (size / 2) * 2;

    for (std::size_t i = 0; i < simd_end; i += 2) {
        prefetch_read(values + i + 2);
        prefetch_write(results + i + 2);

        for (std::size_t j = 0; j < 2; ++j) {
            results[i + j] = std::erf(values[i + j]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}

#endif // LIBSTATS_HAS_NEON

} // namespace simd
} // namespace libstats

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
        #pragma clang attribute push (__attribute__((target("neon"))), apply_to=function)
    #endif
#endif

#include "../include/simd.h"
#include "../include/cpu_detection.h"
#include "../include/constants.h"

// Only include NEON intrinsics on ARM platforms
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #include <arm_neon.h>
#endif

#include <cmath>

namespace libstats {
namespace simd {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)

// All NEON functions use double-precision (64-bit) values
// NEON processes 2 doubles per 128-bit register

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if NEON not supported
    if (!cpu::supports_neon()) {
        return dot_product_fallback(a, b, size);
    }
    
    constexpr std::size_t NEON_DOUBLE_WIDTH = constants::simd::registers::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;
    
    // Apple Silicon optimization: Use multiple accumulators to exploit
    // superscalar execution and out-of-order capabilities
    #if defined(LIBSTATS_APPLE_SILICON)
    if (size >= constants::simd::tuning::apple_silicon::NEON_UNROLL_MIN_SIZE) {
        float64x2_t sum1 = vdupq_n_f64(0.0);
        float64x2_t sum2 = vdupq_n_f64(0.0);
        float64x2_t sum3 = vdupq_n_f64(0.0);
        float64x2_t sum4 = vdupq_n_f64(0.0);
        
        const std::size_t unroll_end = (size / constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR) * constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR;
        
        // Process 8 doubles per iteration (4 NEON registers)
        for (std::size_t i = 0; i < unroll_end; i += constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR) {
            // Load data with prefetch hints for Apple Silicon
            float64x2_t va1 = vld1q_f64(&a[i]);
            float64x2_t vb1 = vld1q_f64(&b[i]);
            float64x2_t va2 = vld1q_f64(&a[i + 2]);
            float64x2_t vb2 = vld1q_f64(&b[i + 2]);
            float64x2_t va3 = vld1q_f64(&a[i + 4]);
            float64x2_t vb3 = vld1q_f64(&b[i + 4]);
            float64x2_t va4 = vld1q_f64(&a[i + 6]);
            float64x2_t vb4 = vld1q_f64(&b[i + 6]);
            
            // Multiply and accumulate with independent accumulators
            sum1 = vfmaq_f64(sum1, va1, vb1);
            sum2 = vfmaq_f64(sum2, va2, vb2);
            sum3 = vfmaq_f64(sum3, va3, vb3);
            sum4 = vfmaq_f64(sum4, va4, vb4);
        }
        
        // Combine accumulators
        float64x2_t sum = vaddq_f64(vaddq_f64(sum1, sum2), vaddq_f64(sum3, sum4));
        
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
    float64x2_t sum = vdupq_n_f64(0.0);
    
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

void VectorOps::vector_add_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_neon()) {
        return vector_add_fallback(a, b, result, size);
    }
    
    constexpr std::size_t NEON_DOUBLE_WIDTH = constants::simd::registers::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;
    
    // Apple Silicon optimization: Loop unrolling for better throughput
    #if defined(LIBSTATS_APPLE_SILICON)
    if (size >= tuned::cache_friendly_step()) {
        const std::size_t unroll_end = (size / constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR) * constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR;
        
        // Process 8 doubles per iteration with prefetching
        for (std::size_t i = 0; i < unroll_end; i += constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR) {
            // Prefetch next cache line on Apple Silicon
            prefetch_read(&a[i + tuned::prefetch_distance() * constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR]);
            prefetch_read(&b[i + tuned::prefetch_distance() * constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR]);
            prefetch_write(&result[i + tuned::prefetch_distance() * constants::simd::tuning::apple_silicon::NEON_UNROLL_FACTOR]);
            
            // Load and process 4 NEON registers worth of data
            float64x2_t va1 = vld1q_f64(&a[i]);
            float64x2_t vb1 = vld1q_f64(&b[i]);
            float64x2_t va2 = vld1q_f64(&a[i + 2]);
            float64x2_t vb2 = vld1q_f64(&b[i + 2]);
            float64x2_t va3 = vld1q_f64(&a[i + 4]);
            float64x2_t vb3 = vld1q_f64(&b[i + 4]);
            float64x2_t va4 = vld1q_f64(&a[i + 6]);
            float64x2_t vb4 = vld1q_f64(&b[i + 6]);
            
            // Compute results
            float64x2_t vresult1 = vaddq_f64(va1, vb1);
            float64x2_t vresult2 = vaddq_f64(va2, vb2);
            float64x2_t vresult3 = vaddq_f64(va3, vb3);
            float64x2_t vresult4 = vaddq_f64(va4, vb4);
            
            // Store results
            vst1q_f64(&result[i], vresult1);
            vst1q_f64(&result[i + 2], vresult2);
            vst1q_f64(&result[i + 4], vresult3);
            vst1q_f64(&result[i + 6], vresult4);
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

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_neon()) {
        return vector_subtract_fallback(a, b, result, size);
    }
    
    constexpr std::size_t NEON_DOUBLE_WIDTH = constants::simd::registers::NEON_DOUBLES;
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

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!cpu::supports_neon()) {
        return vector_multiply_fallback(a, b, result, size);
    }
    
    constexpr std::size_t NEON_DOUBLE_WIDTH = constants::simd::registers::NEON_DOUBLES;
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

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!cpu::supports_neon()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }
    
    float64x2_t vscalar = vdupq_n_f64(scalar);
    constexpr std::size_t NEON_DOUBLE_WIDTH = constants::simd::registers::NEON_DOUBLES;
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

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!cpu::supports_neon()) {
        return scalar_add_fallback(a, scalar, result, size);
    }
    
    float64x2_t vscalar = vdupq_n_f64(scalar);
    constexpr std::size_t NEON_DOUBLE_WIDTH = constants::simd::registers::NEON_DOUBLES;
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

#else

// Fallback implementations for non-ARM platforms
// These will never be called, but we need them for linking

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    return dot_product_fallback(a, b, size);
}

void VectorOps::vector_add_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    vector_add_fallback(a, b, result, size);
}

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    vector_subtract_fallback(a, b, result, size);
}

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result, std::size_t size) noexcept {
    vector_multiply_fallback(a, b, result, size);
}

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    scalar_multiply_fallback(a, scalar, result, size);
}

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result, std::size_t size) noexcept {
    scalar_add_fallback(a, scalar, result, size);
}

#endif // ARM platform check

} // namespace simd
} // namespace libstats

#ifdef __clang__
    #if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
        #pragma clang attribute pop
    #endif
#endif

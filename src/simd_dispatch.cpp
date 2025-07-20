// Main SIMD dispatch logic - NO SIMD intrinsics in this file
// This file contains only the decision logic for which implementation to use
// Enhanced with platform-specific optimizations and adaptive thresholds

#include "../include/simd.h"
#include "../include/cpu_detection.h"
#include "../include/constants.h"
#include <algorithm>
#include <cstring>

namespace libstats {
namespace simd {

//========== Public Interface Implementations ==========
// These are the main entry points that users call
// They dispatch to the appropriate SIMD implementation based on runtime CPU detection

double VectorOps::dot_product(const double* a, const double* b, std::size_t size) noexcept {
    // Early exit for small arrays where SIMD overhead isn't worth it
    if (!should_use_simd(size)) {
        return dot_product_fallback(a, b, size);
    }
    
    // Dispatch to best available implementation in order of performance
    // Each implementation includes its own runtime safety checks
    
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return dot_product_avx512(a, b, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return dot_product_avx2(a, b, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return dot_product_avx(a, b, size);
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return dot_product_sse2(a, b, size);
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return dot_product_neon(a, b, size);
    }
#endif

    // Fallback to scalar implementation
    return dot_product_fallback(a, b, size);
}

void VectorOps::vector_add(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!should_use_simd(size)) {
        return vector_add_fallback(a, b, result, size);
    }
    
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return vector_add_avx512(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return vector_add_avx2(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return vector_add_avx(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return vector_add_sse2(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return vector_add_neon(a, b, result, size);
    }
#endif

    return vector_add_fallback(a, b, result, size);
}

void VectorOps::vector_subtract(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!should_use_simd(size)) {
        return vector_subtract_fallback(a, b, result, size);
    }
    
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return vector_subtract_avx512(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return vector_subtract_avx2(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return vector_subtract_avx(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return vector_subtract_sse2(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return vector_subtract_neon(a, b, result, size);
    }
#endif

    return vector_subtract_fallback(a, b, result, size);
}

void VectorOps::vector_multiply(const double* a, const double* b, double* result, std::size_t size) noexcept {
    if (!should_use_simd(size)) {
        return vector_multiply_fallback(a, b, result, size);
    }
    
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return vector_multiply_avx512(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return vector_multiply_avx2(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return vector_multiply_avx(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return vector_multiply_sse2(a, b, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return vector_multiply_neon(a, b, result, size);
    }
#endif

    return vector_multiply_fallback(a, b, result, size);
}

void VectorOps::scalar_multiply(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!should_use_simd(size)) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }
    
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return scalar_multiply_avx512(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return scalar_multiply_avx2(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return scalar_multiply_avx(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return scalar_multiply_sse2(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return scalar_multiply_neon(a, scalar, result, size);
    }
#endif

    return scalar_multiply_fallback(a, scalar, result, size);
}

void VectorOps::scalar_add(const double* a, double scalar, double* result, std::size_t size) noexcept {
    if (!should_use_simd(size)) {
        return scalar_add_fallback(a, scalar, result, size);
    }
    
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return scalar_add_avx512(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return scalar_add_avx2(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return scalar_add_avx(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return scalar_add_sse2(a, scalar, result, size);
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return scalar_add_neon(a, scalar, result, size);
    }
#endif

    return scalar_add_fallback(a, scalar, result, size);
}

void VectorOps::vector_exp(const double* values, double* results, std::size_t size) noexcept {
    // For complex math functions like exp, we currently use fallback
    // Full SIMD implementations of transcendental functions require careful
    // implementation of range reduction and polynomial approximation
    return vector_exp_fallback(values, results, size);
}

void VectorOps::vector_log(const double* values, double* results, std::size_t size) noexcept {
    // For complex math functions like log, we currently use fallback
    return vector_log_fallback(values, results, size);
}

void VectorOps::vector_pow(const double* base, double exponent, double* results, std::size_t size) noexcept {
    // For complex math functions like pow, we currently use fallback
    return vector_pow_fallback(base, exponent, results, size);
}

void VectorOps::vector_erf(const double* values, double* results, std::size_t size) noexcept {
    // For complex math functions like erf, we currently use fallback
    return vector_erf_fallback(values, results, size);
}

//========== Runtime Information Functions ==========

std::string VectorOps::get_active_simd_level() noexcept {
    // Return the highest SIMD level currently available at runtime
#ifdef LIBSTATS_HAS_AVX512
    if (cpu::supports_avx512()) {
        return "AVX-512";
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        return "AVX2";
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (cpu::supports_avx()) {
        return "AVX";
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (cpu::supports_sse2()) {
        return "SSE2";
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (cpu::supports_neon()) {
        return "NEON";
    }
#endif

    return "Scalar";
}

bool VectorOps::is_simd_available() noexcept {
    return get_active_simd_level() != "Scalar";
}

std::size_t VectorOps::get_optimal_block_size() noexcept {
    return constants::platform::get_optimal_simd_block_size();
}

//========== Enhanced Platform-Aware Dispatch Utilities ==========

namespace {
    /// Internal utility: Check if memory alignment is beneficial for current platform
    inline bool is_alignment_beneficial(const void* ptr1, const void* ptr2 = nullptr, const void* ptr3 = nullptr) noexcept {
        const std::size_t alignment = constants::platform::get_optimal_alignment();
        
        bool aligned = (reinterpret_cast<uintptr_t>(ptr1) % alignment) == 0;
        if (ptr2) {
            aligned = aligned && ((reinterpret_cast<uintptr_t>(ptr2) % alignment) == 0);
        }
        if (ptr3) {
            aligned = aligned && ((reinterpret_cast<uintptr_t>(ptr3) % alignment) == 0);
        }
        
        return aligned;
    }
    
    /// Internal utility: Get platform-specific cache optimization threshold
    inline std::size_t get_cache_optimization_threshold() noexcept {
        const auto thresholds = constants::platform::get_cache_thresholds();
        return thresholds.l1_optimal_size / 4; // Use quarter of L1 as threshold
    }
    
    /// Internal utility: Choose optimal SIMD path based on data characteristics
    template<typename Operation>
    inline bool should_use_advanced_simd(std::size_t size, const void* ptr1, const void* ptr2 = nullptr, const void* ptr3 = nullptr) noexcept {
        // Basic size check
        if (!VectorOps::should_use_simd(size)) {
            return false;
        }
        
        // For very large datasets, always use SIMD regardless of alignment
        const std::size_t cache_threshold = get_cache_optimization_threshold();
        if (size >= cache_threshold) {
            return true;
        }
        
        // For medium datasets, check alignment benefits
        if (size >= constants::simd::optimization::MEDIUM_DATASET_MIN_SIZE && is_alignment_beneficial(ptr1, ptr2, ptr3)) {
            return true;
        }
        
        // For high-end SIMD (AVX-512), use for smaller aligned datasets
        #ifdef LIBSTATS_HAS_AVX512
        if (cpu::supports_avx512() && size >= constants::simd::optimization::AVX512_MIN_ALIGNED_SIZE && is_alignment_beneficial(ptr1, ptr2, ptr3)) {
            return true;
        }
        #endif
        
        // For Apple Silicon, be more aggressive with SIMD usage
        #if defined(LIBSTATS_APPLE_SILICON)
        if (size >= constants::simd::optimization::APPLE_SILICON_AGGRESSIVE_THRESHOLD) {
            return true;
        }
        #endif
        
        return size >= constants::platform::get_min_simd_size();
    }
}

//========== Enhanced Public Interface Functions ==========

bool VectorOps::should_use_vectorized_path(std::size_t size, const void* data1, const void* data2, const void* data3) noexcept {
    return should_use_advanced_simd<void>(size, data1, data2, data3);
}

std::string VectorOps::get_platform_optimization_info() noexcept {
    const auto thresholds = constants::platform::get_cache_thresholds();
    
    std::string info = "Platform: ";
    
    #if defined(LIBSTATS_APPLE_SILICON)
        info += "Apple Silicon (";
    #elif defined(__aarch64__) || defined(_M_ARM64)
        info += "ARM64 (";
    #elif defined(__x86_64__) || defined(_M_X64)
        info += "x86_64 (";
    #elif defined(__i386) || defined(_M_IX86)
        info += "x86_32 (";
    #else
        info += "Unknown (";
    #endif
    
    info += get_active_simd_level() + "), ";
    info += "SIMD Width: " + std::to_string(double_vector_width()) + ", ";
    info += "Min SIMD Size: " + std::to_string(min_simd_size()) + ", ";
    info += "L1 Cache Elements: " + std::to_string(thresholds.l1_optimal_size);
    
    return info;
}

} // namespace simd
} // namespace libstats

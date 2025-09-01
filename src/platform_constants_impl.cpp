#include "../include/common/platform_constants_fwd.h"
#include "../include/core/mathematical_constants.h"
#include "../include/platform/platform_constants.h"
#include "../include/platform/simd_policy.h"
#include "platform/cpu_detection.h"
#include "platform/cpu_vendor_constants.h"

// Heavy STL includes are now contained in this implementation file
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>  // for operator==

/**
 * @file platform_constants_impl.cpp
 * @brief Implementation of platform constants - Phase 2 PIMPL optimization
 *
 * This file contains all the heavy implementation details that were previously
 * in the header file. By moving them here, we eliminate ~85% of compilation
 * overhead for files that only need to access constant values.
 */

namespace stats {
namespace arch {

/// Runtime adaptive parallel optimization functions
/// These provide CPU feature-based optimization for parallel processing

// Adaptive functions (runtime platform optimization)
std::size_t get_min_elements_for_parallel() {
    const auto& features = stats::arch::get_features();

    if (features.avx512f) {
        return parallel::avx512::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.avx2) {
        return parallel::avx2::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (stats::arch::is_sandy_ivy_bridge()) {
        return parallel::avx::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.avx) {
        return parallel::avx::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.sse2) {
        return parallel::sse::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.neon) {
        return parallel::neon::MIN_ELEMENTS_FOR_PARALLEL;
    } else {
        return parallel::fallback::MIN_ELEMENTS_FOR_PARALLEL;
    }
}

std::size_t get_min_elements_for_distribution_parallel() {
    const auto& features = stats::arch::get_features();

    if (features.avx512f) {
        return parallel::avx512::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.avx2) {
        return parallel::avx2::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (stats::arch::is_sandy_ivy_bridge()) {
        return parallel::avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.avx) {
        return parallel::avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.sse2) {
        return parallel::sse::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.neon) {
        return parallel::neon::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else {
        return parallel::fallback::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    }
}

std::size_t get_min_elements_for_simple_distribution_parallel() {
    const auto& features = stats::arch::get_features();

    if (features.avx512f) {
        return parallel::avx512::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.avx2) {
        return parallel::avx2::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (stats::arch::is_sandy_ivy_bridge()) {
        return parallel::avx::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.avx) {
        return parallel::avx::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.sse2) {
        return parallel::sse::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.neon) {
        return parallel::neon::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else {
        return parallel::fallback::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    }
}

std::size_t get_default_grain_size() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Use vendor-specific grain sizes when available
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon - use vendor-specific optimizations
    return stats::arch::cpu::apple_silicon::DEFAULT_GRAIN_SIZE;
#elif defined(__x86_64__) || defined(_M_X64)
    // x86-64 - consider both vendor and SIMD features
    if (features.vendor == "GenuineIntel") {
        if (stats::arch::is_sandy_ivy_bridge()) {
            return stats::arch::cpu::intel::legacy::MAX_GRAIN_SIZE;  // Use larger grain size for
                                                                     // legacy Intel
        } else {
            return stats::arch::cpu::intel::modern::DEFAULT_GRAIN_SIZE;
        }
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::ryzen::DEFAULT_GRAIN_SIZE;
    } else {
        // Unknown x86 vendor - fallback to SIMD-based selection
        if (features.avx512f) {
            return parallel::avx512::DEFAULT_GRAIN_SIZE;
        } else if (features.avx2) {
            return parallel::avx2::DEFAULT_GRAIN_SIZE;
        } else if (features.avx) {
            return parallel::avx::DEFAULT_GRAIN_SIZE;
        } else if (features.sse2) {
            return parallel::sse::DEFAULT_GRAIN_SIZE;
        }
    }
#else
    // ARM or other architectures
    if (features.neon) {
        return stats::arch::cpu::arm::DEFAULT_GRAIN_SIZE;
    }
#endif
    return parallel::fallback::DEFAULT_GRAIN_SIZE;
}

std::size_t get_simple_operation_grain_size() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Use vendor-specific grain sizes when available
#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::SIMPLE_OPERATION_GRAIN_SIZE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::modern::SIMPLE_OPERATION_GRAIN_SIZE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::ryzen::SIMPLE_OPERATION_GRAIN_SIZE;
    } else {
        // Fallback to SIMD-based selection
        if (features.avx512f) {
            return parallel::avx512::SIMPLE_OPERATION_GRAIN_SIZE;
        } else if (features.avx2) {
            return parallel::avx2::SIMPLE_OPERATION_GRAIN_SIZE;
        } else if (features.avx) {
            return parallel::avx::SIMPLE_OPERATION_GRAIN_SIZE;
        } else if (features.sse2) {
            return parallel::sse::SIMPLE_OPERATION_GRAIN_SIZE;
        }
    }
#else
    if (features.neon) {
        return stats::arch::cpu::arm::SIMPLE_OPERATION_GRAIN_SIZE;
    }
#endif
    return parallel::fallback::SIMPLE_OPERATION_GRAIN_SIZE;
}

std::size_t get_complex_operation_grain_size() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Use vendor-specific grain sizes when available
#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::COMPLEX_OPERATION_GRAIN_SIZE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::modern::COMPLEX_OPERATION_GRAIN_SIZE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::ryzen::COMPLEX_OPERATION_GRAIN_SIZE;
    } else {
        // Fallback to SIMD-based selection
        if (features.avx512f) {
            return parallel::avx512::COMPLEX_OPERATION_GRAIN_SIZE;
        } else if (features.avx2) {
            return parallel::avx2::COMPLEX_OPERATION_GRAIN_SIZE;
        } else if (features.avx) {
            return parallel::avx::COMPLEX_OPERATION_GRAIN_SIZE;
        } else if (features.sse2) {
            return parallel::sse::COMPLEX_OPERATION_GRAIN_SIZE;
        }
    }
#else
    if (features.neon) {
        return stats::arch::cpu::arm::COMPLEX_OPERATION_GRAIN_SIZE;
    }
#endif
    return parallel::fallback::COMPLEX_OPERATION_GRAIN_SIZE;
}

std::size_t get_monte_carlo_grain_size() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Use vendor-specific grain sizes when available
#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::MONTE_CARLO_GRAIN_SIZE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::modern::MONTE_CARLO_GRAIN_SIZE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::ryzen::MONTE_CARLO_GRAIN_SIZE;
    } else {
        // Fallback to SIMD-based selection
        if (features.avx512f) {
            return parallel::avx512::MONTE_CARLO_GRAIN_SIZE;
        } else if (features.avx2) {
            return parallel::avx2::MONTE_CARLO_GRAIN_SIZE;
        } else if (features.avx) {
            return parallel::avx::MONTE_CARLO_GRAIN_SIZE;
        } else if (features.sse2) {
            return parallel::sse::MONTE_CARLO_GRAIN_SIZE;
        }
    }
#else
    if (features.neon) {
        return stats::arch::cpu::arm::MONTE_CARLO_GRAIN_SIZE;
    }
#endif
    return parallel::fallback::MONTE_CARLO_GRAIN_SIZE;
}

std::size_t get_max_grain_size() {
    const auto& features = stats::arch::get_features();

    if (features.avx512f) {
        return parallel::avx512::MAX_GRAIN_SIZE;
    } else if (features.avx2) {
        return parallel::avx2::MAX_GRAIN_SIZE;
    } else if (features.avx) {
        return parallel::avx::MAX_GRAIN_SIZE;
    } else if (features.sse2) {
        return parallel::sse::MAX_GRAIN_SIZE;
    } else if (features.neon) {
        return parallel::neon::MAX_GRAIN_SIZE;
    } else {
        return parallel::fallback::MAX_GRAIN_SIZE;
    }
}

/// Memory access and prefetching optimization constants (implementation)
namespace memory {
namespace prefetch {
// Now using vendor-specific constants from cpu_vendor_constants.h
// No more duplication - just reference the proper namespace

std::size_t get_sequential_prefetch_distance() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

// Platform-specific prefetch distances based on architecture
#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::SEQUENTIAL_PREFETCH_DISTANCE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::SEQUENTIAL_PREFETCH_DISTANCE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::SEQUENTIAL_PREFETCH_DISTANCE;
    } else {
        return stats::arch::cpu::intel::SEQUENTIAL_PREFETCH_DISTANCE;  // Default to Intel
    }
#else
    return stats::arch::cpu::arm::SEQUENTIAL_PREFETCH_DISTANCE;
#endif
}

std::size_t get_random_prefetch_distance() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::RANDOM_PREFETCH_DISTANCE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::RANDOM_PREFETCH_DISTANCE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::RANDOM_PREFETCH_DISTANCE;
    } else {
        return stats::arch::cpu::intel::RANDOM_PREFETCH_DISTANCE;
    }
#else
    return stats::arch::cpu::arm::RANDOM_PREFETCH_DISTANCE;
#endif
}

std::size_t get_matrix_prefetch_distance() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::MATRIX_PREFETCH_DISTANCE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::MATRIX_PREFETCH_DISTANCE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::MATRIX_PREFETCH_DISTANCE;
    } else {
        return stats::arch::cpu::intel::MATRIX_PREFETCH_DISTANCE;
    }
#else
    return stats::arch::cpu::arm::MATRIX_PREFETCH_DISTANCE;
#endif
}

std::size_t get_prefetch_stride() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::PREFETCH_STRIDE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::PREFETCH_STRIDE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::PREFETCH_STRIDE;
    } else {
        return stats::arch::cpu::intel::PREFETCH_STRIDE;
    }
#else
    return stats::arch::cpu::arm::PREFETCH_STRIDE;
#endif
}
}  // namespace prefetch

// No duplicate constants needed - all are defined in header as constexpr
}  // namespace memory

/// Platform-specific tuning functions (implementation)
std::size_t get_optimal_simd_block_size() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Use vendor-specific SIMD block sizes when available, fallback to SIMD features
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon - use vendor-specific SIMD optimization
    return stats::arch::cpu::apple_silicon::OPTIMAL_SIMD_BLOCK;
#elif defined(__x86_64__) || defined(_M_X64)
    // x86-64 - consider both vendor and SIMD capabilities
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::OPTIMAL_SIMD_BLOCK;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::OPTIMAL_SIMD_BLOCK;
    } else {
        // Unknown vendor - use SIMD features
        if (features.avx512f) {
            return 8;  // AVX-512: 8 doubles per register
        } else if (features.avx || features.avx2) {
            return 4;  // AVX/AVX2: 4 doubles per register
        } else if (features.sse2) {
            return 2;  // SSE2: 2 doubles per register
        }
    }
#else
    // ARM or other architectures
    if (features.neon) {
        return stats::arch::cpu::arm::OPTIMAL_SIMD_BLOCK;
    }
#endif
    return 1;  // No SIMD support
}

std::size_t get_optimal_alignment() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Use vendor-specific cache line sizes when available
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon has 128-byte cache lines
    return stats::arch::cpu::apple_silicon::CACHE_LINE_SIZE;
#elif defined(__x86_64__) || defined(_M_X64)
    // x86-64 vendors - prefer SIMD alignment for vectorized operations,
    // fall back to vendor-specific cache line size for general alignment
    if (features.avx512f) {
        return stats::arch::cpu::intel::CACHE_LINE_SIZE;  // AVX-512: Use full cache line alignment
    } else if (features.avx || features.avx2) {
        // For AVX/AVX2, use 32-byte SIMD alignment but consider cache lines
        if (features.vendor == "GenuineIntel") {
            return std::max<std::size_t>(32, stats::arch::cpu::intel::CACHE_LINE_SIZE / 2);
        } else if (features.vendor == "AuthenticAMD") {
            return std::max<std::size_t>(32, stats::arch::cpu::amd::CACHE_LINE_SIZE / 2);
        } else {
            return 32;  // Default AVX alignment
        }
    } else if (features.sse2) {
        // For SSE2, use 16-byte SIMD alignment
        return 16;
    } else {
        // No SIMD: Use vendor-specific cache line size for general alignment
        if (features.vendor == "GenuineIntel") {
            return stats::arch::cpu::intel::CACHE_LINE_SIZE;
        } else if (features.vendor == "AuthenticAMD") {
            return stats::arch::cpu::amd::CACHE_LINE_SIZE;
        } else {
            return stats::arch::cpu::intel::CACHE_LINE_SIZE;  // Default to Intel cache line size
        }
    }
#else
    // ARM or other architectures
    if (features.neon) {
        // For NEON, prefer cache line alignment over SIMD register alignment
        return stats::arch::cpu::arm::CACHE_LINE_SIZE;
    } else {
        return stats::arch::cpu::arm::CACHE_LINE_SIZE;  // Use ARM cache line size
    }
#endif
}

std::size_t get_cache_line_size() {
    [[maybe_unused]] const auto& features = stats::arch::get_features();

    // Return vendor-specific cache line size based on runtime detection
#if defined(__APPLE__) && defined(__aarch64__)
    return stats::arch::cpu::apple_silicon::CACHE_LINE_SIZE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return stats::arch::cpu::intel::CACHE_LINE_SIZE;
    } else if (features.vendor == "AuthenticAMD") {
        return stats::arch::cpu::amd::CACHE_LINE_SIZE;
    } else {
        return stats::arch::cpu::intel::CACHE_LINE_SIZE;  // Default to Intel cache line size
    }
#else
    // ARM or other architectures
    return stats::arch::cpu::arm::CACHE_LINE_SIZE;
#endif
}

std::size_t get_min_simd_size() {
    // Delegate to the centralized SIMDPolicy for consistent thresholds
    // This ensures VectorOps::min_simd_size() and VectorOps::should_use_simd()
    // use the same threshold values
    return arch::simd::SIMDPolicy::getMinThreshold();
}

std::size_t get_min_parallel_elements() {
    const auto& features = stats::arch::get_features();

    // More powerful SIMD allows for lower parallel thresholds
    if (features.avx512f) {
        return 256;
    } else if (features.avx2 || features.fma) {
        return 384;
    } else if (features.avx) {
        return 512;
    } else if (features.sse4_2) {
        return 768;
    } else if (features.sse2 || features.neon) {
        return 1024;
    } else {
        return 2048;  // Higher threshold for scalar operations
    }
}

std::size_t get_optimal_grain_size() noexcept {
    const auto& features = stats::arch::get_features();
    const std::size_t optimal_block = get_optimal_simd_block_size();

    // Grain size should be a multiple of SIMD block size
    // and account for cache line efficiency
    const std::size_t cache_line_elements = features.cache_line_size / sizeof(double);
    const std::size_t base_grain = std::max(optimal_block * 8, cache_line_elements);

    // Adjust based on CPU capabilities
    if (features.avx512f) {
        return base_grain * 2;  // Can handle larger chunks efficiently
    } else if (features.avx2 || features.fma) {
        return static_cast<std::size_t>(std::round(static_cast<double>(base_grain) * 1.5));
    } else {
        return base_grain;
    }
}

bool supports_fast_transcendental() {
    const auto& features = stats::arch::get_features();
    // FMA typically indicates more modern CPU with better transcendental support
    return features.fma || features.avx2 || features.avx512f;
}

stats::arch::CacheThresholds get_cache_thresholds() {
    const auto& features = stats::arch::get_features();
    stats::arch::CacheThresholds thresholds{};

    // Convert cache sizes from bytes to number of doubles
    thresholds.l1_optimal_size = features.l1_cache_size > 0
                                     ? (features.l1_cache_size / sizeof(double)) / detail::TWO_INT
                                     : 4096;  // Use half of L1

    thresholds.l2_optimal_size = features.l2_cache_size > 0
                                     ? (features.l2_cache_size / sizeof(double)) / detail::TWO_INT
                                     : 32768;

    thresholds.l3_optimal_size = features.l3_cache_size > 0
                                     ? (features.l3_cache_size / sizeof(double)) / detail::FOUR_INT
                                     : 262144;

// Special handling for platforms where L3 cache might not be detected (e.g., Apple Silicon)
// Ensure L3 optimal size is at least as large as L2 optimal size
#ifdef __APPLE__
    if (thresholds.l3_optimal_size < thresholds.l2_optimal_size) {
        thresholds.l3_optimal_size = thresholds.l2_optimal_size;
    }
#endif

    // Blocking size for cache tiling (typically sqrt of L1 size)
    thresholds.blocking_size =
        static_cast<std::size_t>(std::sqrt(static_cast<double>(thresholds.l1_optimal_size)));

    return thresholds;
}
}  // namespace arch

// Note: Bridge functions declared in constants_bridge.h are implemented
// by the functions above in this file via the forward declaration system.

}  // namespace stats

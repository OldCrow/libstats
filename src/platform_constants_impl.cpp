#include "../include/common/platform_constants_fwd.h"
#include "platform/cpu_detection.h"

// Heavy STL includes are now contained in this implementation file
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

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

/// SIMD optimization parameters and architectural constants (implementation)
namespace simd {
// Static constant definitions
static constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;
static constexpr std::size_t MIN_SIMD_SIZE = 4;
static constexpr std::size_t MAX_BLOCK_SIZE = 64;
static constexpr std::size_t SIMD_ALIGNMENT = 32;

std::size_t get_default_block_size() {
    return DEFAULT_BLOCK_SIZE;
}
std::size_t get_min_simd_size() {
    return MIN_SIMD_SIZE;
}
std::size_t get_max_block_size() {
    return MAX_BLOCK_SIZE;
}
std::size_t get_simd_alignment() {
    return SIMD_ALIGNMENT;
}

namespace alignment {
[[maybe_unused]] static constexpr std::size_t AVX512_ALIGNMENT = 64;
[[maybe_unused]] static constexpr std::size_t AVX_ALIGNMENT = 32;
[[maybe_unused]] static constexpr std::size_t SSE_ALIGNMENT = 16;
[[maybe_unused]] static constexpr std::size_t NEON_ALIGNMENT = 16;
[[maybe_unused]] static constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;
[[maybe_unused]] static constexpr std::size_t MIN_SAFE_ALIGNMENT = 8;

std::size_t get_avx512_alignment() {
    return AVX512_ALIGNMENT;
}
std::size_t get_avx_alignment() {
    return AVX_ALIGNMENT;
}
std::size_t get_sse_alignment() {
    return SSE_ALIGNMENT;
}
std::size_t get_neon_alignment() {
    return NEON_ALIGNMENT;
}
std::size_t get_cache_line_alignment() {
    return CACHE_LINE_ALIGNMENT;
}
std::size_t get_min_safe_alignment() {
    return MIN_SAFE_ALIGNMENT;
}
}  // namespace alignment

namespace registers {
[[maybe_unused]] static constexpr std::size_t AVX512_DOUBLES = 8;
[[maybe_unused]] static constexpr std::size_t AVX_DOUBLES = 4;
[[maybe_unused]] static constexpr std::size_t AVX2_DOUBLES = 4;
[[maybe_unused]] static constexpr std::size_t SSE_DOUBLES = 2;
[[maybe_unused]] static constexpr std::size_t NEON_DOUBLES = 2;

std::size_t get_avx512_doubles() {
    return AVX512_DOUBLES;
}
std::size_t get_avx_doubles() {
    return AVX_DOUBLES;
}
std::size_t get_sse_doubles() {
    return SSE_DOUBLES;
}
std::size_t get_neon_doubles() {
    return NEON_DOUBLES;
}
}  // namespace registers
}  // namespace simd

/// Parallel processing optimization constants (implementation)
namespace parallel {
// Legacy constants for backward compatibility
[[maybe_unused]] const std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
[[maybe_unused]] const std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
[[maybe_unused]] const std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
[[maybe_unused]] const std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;
[[maybe_unused]] const std::size_t DEFAULT_GRAIN_SIZE = 256;

// Architecture-specific constants
namespace sse {
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 2048;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
[[maybe_unused]] static constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
[[maybe_unused]] static constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
[[maybe_unused]] static constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
[[maybe_unused]] static constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
[[maybe_unused]] static constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace sse

namespace avx {
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
[[maybe_unused]] static constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;
[[maybe_unused]] static constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;
[[maybe_unused]] static constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 512;
[[maybe_unused]] static constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;
[[maybe_unused]] static constexpr std::size_t MAX_GRAIN_SIZE = 4096;
// Legacy Intel constants (now flattened)
[[maybe_unused]] static constexpr std::size_t LEGACY_INTEL_MAX_GRAIN_SIZE = 32768;
}  // namespace avx

namespace avx2 {
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1536;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
[[maybe_unused]] static constexpr std::size_t DEFAULT_GRAIN_SIZE = 512;
[[maybe_unused]] static constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 256;
[[maybe_unused]] static constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 1024;
[[maybe_unused]] static constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 128;
[[maybe_unused]] static constexpr std::size_t MAX_GRAIN_SIZE = 8192;
}  // namespace avx2

namespace avx512 {
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 8192;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 65536;
[[maybe_unused]] static constexpr std::size_t DEFAULT_GRAIN_SIZE = 1024;
[[maybe_unused]] static constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 512;
[[maybe_unused]] static constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 2048;
[[maybe_unused]] static constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 256;
[[maybe_unused]] static constexpr std::size_t MAX_GRAIN_SIZE = 16384;
}  // namespace avx512

namespace neon {
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 1536;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
[[maybe_unused]] static constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
[[maybe_unused]] static constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
[[maybe_unused]] static constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
[[maybe_unused]] static constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 48;
[[maybe_unused]] static constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace neon

namespace fallback {
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 2048;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
[[maybe_unused]] static constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
[[maybe_unused]] static constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
[[maybe_unused]] static constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
[[maybe_unused]] static constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
[[maybe_unused]] static constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
[[maybe_unused]] static constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace fallback

// Adaptive functions (runtime platform optimization)
std::size_t get_min_elements_for_parallel() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.avx2) {
        return avx2::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (is_sandy_ivy_bridge()) {
        return avx::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.avx) {
        return avx::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.sse2) {
        return sse::MIN_ELEMENTS_FOR_PARALLEL;
    } else if (features.neon) {
        return neon::MIN_ELEMENTS_FOR_PARALLEL;
    } else {
        return fallback::MIN_ELEMENTS_FOR_PARALLEL;
    }
}

std::size_t get_min_elements_for_distribution_parallel() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.avx2) {
        return avx2::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (is_sandy_ivy_bridge()) {
        return avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.avx) {
        return avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.sse2) {
        return sse::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else if (features.neon) {
        return neon::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    } else {
        return fallback::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    }
}

std::size_t get_min_elements_for_simple_distribution_parallel() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.avx2) {
        return avx2::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (is_sandy_ivy_bridge()) {
        return avx::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.avx) {
        return avx::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.sse2) {
        return sse::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else if (features.neon) {
        return neon::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    } else {
        return fallback::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
    }
}

std::size_t get_default_grain_size() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::DEFAULT_GRAIN_SIZE;
    } else if (features.avx2) {
        return avx2::DEFAULT_GRAIN_SIZE;
    } else if (is_sandy_ivy_bridge()) {
        return avx::DEFAULT_GRAIN_SIZE;
    } else if (features.avx) {
        return avx::DEFAULT_GRAIN_SIZE;
    } else if (features.sse2) {
        return sse::DEFAULT_GRAIN_SIZE;
    } else if (features.neon) {
        return neon::DEFAULT_GRAIN_SIZE;
    } else {
        return fallback::DEFAULT_GRAIN_SIZE;
    }
}

std::size_t get_simple_operation_grain_size() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::SIMPLE_OPERATION_GRAIN_SIZE;
    } else if (features.avx2) {
        return avx2::SIMPLE_OPERATION_GRAIN_SIZE;
    } else if (features.avx) {
        return avx::SIMPLE_OPERATION_GRAIN_SIZE;
    } else if (features.sse2) {
        return sse::SIMPLE_OPERATION_GRAIN_SIZE;
    } else if (features.neon) {
        return neon::SIMPLE_OPERATION_GRAIN_SIZE;
    } else {
        return fallback::SIMPLE_OPERATION_GRAIN_SIZE;
    }
}

std::size_t get_complex_operation_grain_size() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::COMPLEX_OPERATION_GRAIN_SIZE;
    } else if (features.avx2) {
        return avx2::COMPLEX_OPERATION_GRAIN_SIZE;
    } else if (features.avx) {
        return avx::COMPLEX_OPERATION_GRAIN_SIZE;
    } else if (features.sse2) {
        return sse::COMPLEX_OPERATION_GRAIN_SIZE;
    } else if (features.neon) {
        return neon::COMPLEX_OPERATION_GRAIN_SIZE;
    } else {
        return fallback::COMPLEX_OPERATION_GRAIN_SIZE;
    }
}

std::size_t get_monte_carlo_grain_size() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::MONTE_CARLO_GRAIN_SIZE;
    } else if (features.avx2) {
        return avx2::MONTE_CARLO_GRAIN_SIZE;
    } else if (features.avx) {
        return avx::MONTE_CARLO_GRAIN_SIZE;
    } else if (features.sse2) {
        return sse::MONTE_CARLO_GRAIN_SIZE;
    } else if (features.neon) {
        return neon::MONTE_CARLO_GRAIN_SIZE;
    } else {
        return fallback::MONTE_CARLO_GRAIN_SIZE;
    }
}

std::size_t get_max_grain_size() {
    const auto& features = get_features();

    if (features.avx512f) {
        return avx512::MAX_GRAIN_SIZE;
    } else if (features.avx2) {
        return avx2::MAX_GRAIN_SIZE;
    } else if (features.avx) {
        return avx::MAX_GRAIN_SIZE;
    } else if (features.sse2) {
        return sse::MAX_GRAIN_SIZE;
    } else if (features.neon) {
        return neon::MAX_GRAIN_SIZE;
    } else {
        return fallback::MAX_GRAIN_SIZE;
    }
}
}  // namespace parallel

/// Memory access and prefetching optimization constants (implementation)
namespace memory {
namespace prefetch {
namespace apple_silicon {
[[maybe_unused]] static constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 256;
[[maybe_unused]] static constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 64;
[[maybe_unused]] static constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 128;
[[maybe_unused]] static constexpr std::size_t PREFETCH_STRIDE = 8;
}  // namespace apple_silicon

namespace intel {
[[maybe_unused]] static constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 192;
[[maybe_unused]] static constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 48;
[[maybe_unused]] static constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 96;
[[maybe_unused]] static constexpr std::size_t PREFETCH_STRIDE = 4;
}  // namespace intel

namespace amd {
[[maybe_unused]] static constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 128;
[[maybe_unused]] static constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 32;
[[maybe_unused]] static constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 64;
[[maybe_unused]] static constexpr std::size_t PREFETCH_STRIDE = 4;
}  // namespace amd

namespace arm {
[[maybe_unused]] static constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 64;
[[maybe_unused]] static constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 16;
[[maybe_unused]] static constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 32;
[[maybe_unused]] static constexpr std::size_t PREFETCH_STRIDE = 2;
}  // namespace arm

std::size_t get_sequential_prefetch_distance() {
    [[maybe_unused]] const auto& features = get_features();

// Platform-specific prefetch distances based on architecture
#if defined(__APPLE__) && defined(__aarch64__)
    return apple_silicon::SEQUENTIAL_PREFETCH_DISTANCE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return intel::SEQUENTIAL_PREFETCH_DISTANCE;
    } else if (features.vendor == "AuthenticAMD") {
        return amd::SEQUENTIAL_PREFETCH_DISTANCE;
    } else {
        return intel::SEQUENTIAL_PREFETCH_DISTANCE;  // Default to Intel
    }
#else
    return arm::SEQUENTIAL_PREFETCH_DISTANCE;
#endif
}

std::size_t get_random_prefetch_distance() {
    [[maybe_unused]] const auto& features = get_features();

#if defined(__APPLE__) && defined(__aarch64__)
    return apple_silicon::RANDOM_PREFETCH_DISTANCE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return intel::RANDOM_PREFETCH_DISTANCE;
    } else if (features.vendor == "AuthenticAMD") {
        return amd::RANDOM_PREFETCH_DISTANCE;
    } else {
        return intel::RANDOM_PREFETCH_DISTANCE;
    }
#else
    return arm::RANDOM_PREFETCH_DISTANCE;
#endif
}

std::size_t get_matrix_prefetch_distance() {
    [[maybe_unused]] const auto& features = get_features();

#if defined(__APPLE__) && defined(__aarch64__)
    return apple_silicon::MATRIX_PREFETCH_DISTANCE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return intel::MATRIX_PREFETCH_DISTANCE;
    } else if (features.vendor == "AuthenticAMD") {
        return amd::MATRIX_PREFETCH_DISTANCE;
    } else {
        return intel::MATRIX_PREFETCH_DISTANCE;
    }
#else
    return arm::MATRIX_PREFETCH_DISTANCE;
#endif
}

std::size_t get_prefetch_stride() {
    [[maybe_unused]] const auto& features = get_features();

#if defined(__APPLE__) && defined(__aarch64__)
    return apple_silicon::PREFETCH_STRIDE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (features.vendor == "GenuineIntel") {
        return intel::PREFETCH_STRIDE;
    } else if (features.vendor == "AuthenticAMD") {
        return amd::PREFETCH_STRIDE;
    } else {
        return intel::PREFETCH_STRIDE;
    }
#else
    return arm::PREFETCH_STRIDE;
#endif
}
}  // namespace prefetch

namespace access {
[[maybe_unused]] const std::size_t CACHE_LINE_SIZE_BYTES = 64;
[[maybe_unused]] const std::size_t DOUBLES_PER_CACHE_LINE = 8;
[[maybe_unused]] const std::size_t CACHE_LINE_ALIGNMENT = 64;
}  // namespace access
}  // namespace memory

/// Platform-specific tuning functions (implementation)
std::size_t get_optimal_simd_block_size() {
    const auto& features = get_features();

    // AVX-512: 8 doubles per register
    if (features.avx512f) {
        return 8;
    }
    // AVX/AVX2: 4 doubles per register
    else if (features.avx || features.avx2) {
        return 4;
    }
    // SSE2: 2 doubles per register
    else if (features.sse2) {
        return 2;
    }
    // ARM NEON: 2 doubles per register
    else if (features.neon) {
        return 2;
    }
    // No SIMD support
    else {
        return 1;
    }
}

std::size_t get_optimal_alignment() {
    const auto& features = get_features();

    // AVX-512: 64-byte alignment
    if (features.avx512f) {
        return 64;
    }
    // AVX/AVX2: 32-byte alignment
    else if (features.avx || features.avx2) {
        return 32;
    }
    // SSE2: 16-byte alignment
    else if (features.sse2) {
        return 16;
    }
    // ARM NEON: 16-byte alignment
    else if (features.neon) {
        return 16;
    }
    // Default cache line alignment
    else {
        return features.cache_line_size > 0 ? features.cache_line_size : 64;
    }
}

std::size_t get_min_simd_size() {
    const auto& features = get_features();

    // Higher-end SIMD can handle smaller datasets efficiently
    if (features.avx512f) {
        return 4;
    } else if (features.avx2 || features.fma) {
        return 6;
    } else if (features.avx || features.sse4_2) {
        return 8;
    } else if (features.sse2 || features.neon) {
        return 12;
    } else {
        return 32;  // No SIMD benefit until larger sizes
    }
}

std::size_t get_min_parallel_elements() {
    const auto& features = get_features();

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

std::size_t get_optimal_grain_size() {
    const auto& features = get_features();
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
    const auto& features = get_features();
    // FMA typically indicates more modern CPU with better transcendental support
    return features.fma || features.avx2 || features.avx512f;
}

CacheThresholds get_cache_thresholds() {
    const auto& features = get_features();
    CacheThresholds thresholds{};

    // Convert cache sizes from bytes to number of doubles
    thresholds.l1_optimal_size = features.l1_cache_size > 0
                                     ? (features.l1_cache_size / sizeof(double)) / 2
                                     : 4096;  // Use half of L1

    thresholds.l2_optimal_size =
        features.l2_cache_size > 0 ? (features.l2_cache_size / sizeof(double)) / 2 : 32768;

    thresholds.l3_optimal_size =
        features.l3_cache_size > 0 ? (features.l3_cache_size / sizeof(double)) / 4 : 262144;

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

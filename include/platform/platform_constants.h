#pragma once

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

// Forward declaration for platform-specific tuning
#include "cpu_detection.h"
#include "cpu_vendor_constants.h"

/**
 * @file platform/platform_constants.h
 * @brief Platform-dependent optimization constants and runtime tuning functions
 *
 * This header contains all platform-specific constants, SIMD optimization parameters,
 * parallel processing thresholds, and memory optimization constants that depend on
 * the target hardware architecture.
 */

namespace stats {
namespace arch {
/// SIMD optimization parameters and architectural constants
namespace simd {
/// Default SIMD block size for vectorized operations
inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;

/// Minimum problem size to benefit from SIMD
inline constexpr std::size_t MIN_SIMD_SIZE = 4;

/// Maximum block size for cache optimization
inline constexpr std::size_t MAX_BLOCK_SIZE = 64;

/// Platform-specific SIMD register widths (in number of doubles)
/// AVX-512: 8 doubles per register
inline constexpr std::size_t AVX512_DOUBLES = 8;

/// AVX/AVX2: 4 doubles per register
inline constexpr std::size_t AVX_DOUBLES = 4;
inline constexpr std::size_t AVX2_DOUBLES = 4;

/// SSE2: 2 doubles per register
inline constexpr std::size_t SSE_DOUBLES = 2;

/// ARM NEON: 2 doubles per register
inline constexpr std::size_t NEON_DOUBLES = 2;

/// ARM NEON unroll factor
inline constexpr std::size_t NEON_UNROLL = 2;

/// CPU detection and runtime constants
/// Maximum backoff time during CPU feature detection (nanoseconds)
inline constexpr uint64_t CPU_MAX_BACKOFF_NANOSECONDS = 1000;

/// Default cache line size fallback (bytes)
inline constexpr uint32_t CPU_DEFAULT_CACHE_LINE_SIZE = 64;

/// Default TSC frequency measurement duration (milliseconds)
inline constexpr uint32_t CPU_DEFAULT_TSC_SAMPLE_MS = 10;

/// Conversion factor from nanoseconds to Hertz
inline constexpr double CPU_NANOSECONDS_TO_HZ = 1e9;

/// SIMD optimization thresholds and platform-specific constants
/// Medium dataset minimum size for alignment benefits
inline constexpr std::size_t OPT_MEDIUM_DATASET_MIN_SIZE = 32;

/// Minimum threshold for alignment benefit checks
inline constexpr std::size_t OPT_ALIGNMENT_BENEFIT_THRESHOLD = 32;

/// Minimum size for AVX-512 aligned datasets
inline constexpr std::size_t OPT_AVX512_MIN_ALIGNED_SIZE = 8;

/// Aggressive SIMD threshold for Apple Silicon
inline constexpr std::size_t OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD = 6;

/// Minimum size threshold for AVX-512 small benefit
inline constexpr std::size_t OPT_AVX512_SMALL_BENEFIT_THRESHOLD = 4;
}  // namespace simd

/// Parallel processing optimization constants.
///
/// Each sub-namespace (sse, avx, avx2, avx512, neon, fallback) holds
/// compile-time constants for one SIMD tier. The `get_*` runtime functions
/// below this namespace are the intended public API: they dispatch to the
/// correct sub-namespace at runtime based on detected CPU features.
/// Callers outside platform_constants_impl.cpp should use `get_*` functions,
/// not the sub-namespace constants directly.
namespace parallel {

namespace sse {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace sse

namespace avx {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 512;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;
inline constexpr std::size_t MAX_GRAIN_SIZE = 4096;
}  // namespace avx

namespace avx2 {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1536;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 512;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 1024;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 128;
inline constexpr std::size_t MAX_GRAIN_SIZE = 8192;
}  // namespace avx2

namespace avx512 {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 8192;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 65536;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 1024;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 512;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 2048;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 256;
inline constexpr std::size_t MAX_GRAIN_SIZE = 16384;
}  // namespace avx512

namespace neon {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 1536;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 48;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace neon

namespace fallback {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace fallback

/// Monte Carlo simulations benefit from parallelization above this threshold.
inline constexpr std::size_t MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL = 10000;

}  // namespace parallel

/// Runtime adaptive parallel optimization functions (forward declarations)
/// These provide CPU feature-based optimization for parallel processing
/// Implementation is in platform_constants_impl.cpp for PIMPL optimization

/// Get platform-optimized minimum elements for parallel processing
std::size_t get_min_elements_for_parallel();

/// Get platform-optimized minimum elements for distribution parallel processing
std::size_t get_min_elements_for_distribution_parallel();

/// Get platform-optimized minimum elements for simple distribution parallel processing
std::size_t get_min_elements_for_simple_distribution_parallel();

/// Get platform-optimized grain size
std::size_t get_default_grain_size();

/// Get platform-optimized simple operation grain size
std::size_t get_simple_operation_grain_size();

/// Get platform-optimized complex operation grain size
std::size_t get_complex_operation_grain_size();

/// Get platform-optimized Monte Carlo grain size
std::size_t get_monte_carlo_grain_size();

/// Get platform-optimized maximum grain size
std::size_t get_max_grain_size();

/// Platform-specific tuning functions
// Consolidated into detail namespace (was: namespace platform)
/**
 * @brief Get optimized SIMD block size based on detected CPU features
 * @return Optimal block size for SIMD operations
 */
std::size_t get_optimal_simd_block_size();

/**
 * @brief Get optimized memory alignment based on detected CPU features
 * @return Optimal memory alignment in bytes
 */
std::size_t get_optimal_alignment();

/**
 * @brief Get cache line size based on detected CPU vendor
 * @return Platform-specific cache line size in bytes
 */
std::size_t get_cache_line_size();

/**
 * @brief Get optimized minimum size for SIMD operations
 * @return Minimum size threshold for SIMD benefit
 */
std::size_t get_min_simd_size();

/**
 * @brief Get optimized parallel processing thresholds based on CPU features
 * @return Optimal minimum elements for parallel processing
 */
std::size_t get_min_parallel_elements();

/**
 * @brief Get platform-optimized grain size for parallel operations
 * @return Optimal grain size for work distribution
 */
std::size_t get_optimal_grain_size() noexcept;

/**
 * @brief Check if platform supports efficient transcendental functions
 * @return True if CPU has hardware support for fast transcendental operations
 */
bool supports_fast_transcendental();

/**
 * @brief Get cache-optimized thresholds for algorithms
 * @return Structure with GPU-accelerated thresholds
 */
struct CacheThresholds {
    std::size_t l1_optimal_size;  // Optimal size for L1 cache
    std::size_t l2_optimal_size;  // Optimal size for L2 cache
    std::size_t l3_optimal_size;  // Optimal size for L3 cache
    std::size_t blocking_size;    // Optimal blocking size for cache tiling
};

CacheThresholds get_cache_thresholds();
// End of consolidated platform constants

}  // namespace arch
}  // namespace stats

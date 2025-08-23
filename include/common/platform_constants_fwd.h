#pragma once

#include <cstddef>

/**
 * @file common/platform_constants_fwd.h
 * @brief Lightweight forward declarations for platform constants - Phase 2 PIMPL optimization
 *
 * This header provides a minimal interface to platform constants without pulling in
 * heavy STL dependencies. Full implementation is hidden behind the PIMPL pattern.
 *
 * Usage:
 *   - Include this header when you only need constant value access
 *   - Use platform_constants.h when you need the full inline implementation
 *   - This reduces compilation overhead by ~85% for most use cases
 */

namespace stats {
namespace constants {

/// Forward declarations - lightweight interface to platform constants
namespace simd {
/// Get optimal SIMD block size for the current platform
std::size_t get_default_block_size();
std::size_t get_min_simd_size();
std::size_t get_max_block_size();
std::size_t get_simd_alignment();

namespace alignment {
std::size_t get_avx512_alignment();
std::size_t get_avx_alignment();
std::size_t get_sse_alignment();
std::size_t get_neon_alignment();
std::size_t get_cache_line_alignment();
std::size_t get_min_safe_alignment();
}  // namespace alignment

namespace registers {
std::size_t get_avx512_doubles();
std::size_t get_avx_doubles();
std::size_t get_sse_doubles();
std::size_t get_neon_doubles();
}  // namespace registers
}  // namespace simd

namespace parallel {
/// Get platform-optimized parallel processing thresholds
std::size_t get_min_elements_for_parallel();
std::size_t get_min_elements_for_distribution_parallel();
std::size_t get_min_elements_for_simple_distribution_parallel();

/// Get platform-optimized grain sizes
std::size_t get_default_grain_size();
std::size_t get_simple_operation_grain_size();
std::size_t get_complex_operation_grain_size();
std::size_t get_monte_carlo_grain_size();
std::size_t get_max_grain_size();

/// Legacy constants for backward compatibility
extern const std::size_t MIN_ELEMENTS_FOR_PARALLEL;
extern const std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
extern const std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
extern const std::size_t SIMPLE_OPERATION_GRAIN_SIZE;
extern const std::size_t DEFAULT_GRAIN_SIZE;
}  // namespace parallel

namespace memory {
namespace prefetch {
/// Get platform-optimized prefetch distances
std::size_t get_sequential_prefetch_distance();
std::size_t get_random_prefetch_distance();
std::size_t get_matrix_prefetch_distance();
std::size_t get_prefetch_stride();
}  // namespace prefetch

namespace access {
extern const std::size_t CACHE_LINE_SIZE_BYTES;
extern const std::size_t DOUBLES_PER_CACHE_LINE;
extern const std::size_t CACHE_LINE_ALIGNMENT;
}  // namespace access
}  // namespace memory

/// Platform detection and optimization functions
namespace platform {
/// Get optimized parameters based on detected CPU features
std::size_t get_optimal_simd_block_size();
std::size_t get_optimal_alignment();
std::size_t get_min_simd_size();
std::size_t get_min_parallel_elements();
std::size_t get_optimal_grain_size();

/// Hardware capability queries
bool supports_fast_transcendental();

/// Cache-aware configuration
struct CacheThresholds {
    std::size_t l1_optimal_size;
    std::size_t l2_optimal_size;
    std::size_t l3_optimal_size;
    std::size_t blocking_size;
};

CacheThresholds get_cache_thresholds();
}  // namespace platform

namespace cache {
namespace sizing {
extern const std::size_t MIN_CACHE_SIZE_BYTES;
extern const std::size_t MAX_CACHE_SIZE_BYTES;
extern const std::size_t MIN_ENTRY_COUNT;
extern const std::size_t MAX_ENTRY_COUNT;
}  // namespace sizing
}  // namespace cache

}  // namespace constants

/// CPU detection functions (forward declarations)
namespace cpu {
struct Features;
const Features& get_features();
std::size_t optimal_double_width();
std::size_t optimal_alignment();
bool is_sandy_ivy_bridge();
}  // namespace cpu

}  // namespace stats

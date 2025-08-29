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
namespace arch {

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

/// Get optimized parameters based on detected CPU features
std::size_t get_optimal_simd_block_size();
std::size_t get_optimal_alignment();
std::size_t get_cache_line_size();
std::size_t get_min_simd_size();
std::size_t get_min_parallel_elements();
std::size_t get_optimal_grain_size() noexcept;

/// Hardware capability queries
bool supports_fast_transcendental();

/// Cache-aware configuration
struct CacheThresholds;  // Forward declaration - full definition in platform_constants.h

CacheThresholds get_cache_thresholds();

/// CPU detection functions
struct Features;
const Features& get_features();
std::size_t optimal_double_width();
std::size_t optimal_alignment();
bool is_sandy_ivy_bridge();

}  // namespace arch
}  // namespace stats

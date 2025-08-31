#pragma once

#include <cstddef>
#include <string>

/**
 * @file common/simd_policy_fwd.h
 * @brief Lightweight forward declarations for SIMD policy - Phase 2 PIMPL optimization
 *
 * This header provides a minimal interface to SIMD policy capabilities without
 * pulling in heavy platform-specific dependencies and CPU detection overhead.
 *
 * Benefits:
 *   - Eliminates ~95% of compilation overhead for basic SIMD queries
 *   - Removes platform-specific header dependencies (immintrin.h, arm_neon.h, etc.)
 *   - Hides complex CPU detection logic behind implementation
 *   - Provides clean API for SIMD decision making
 */

namespace stats {
namespace arch {
namespace simd {

/// SIMD instruction set levels (forward declaration)
enum class SIMDLevel {
    None,   ///< No SIMD support - use scalar implementation
    NEON,   ///< ARM NEON - 128-bit vectors
    SSE2,   ///< SSE2 - 128-bit vectors, 2 doubles
    AVX,    ///< AVX - 256-bit vectors, 4 doubles
    AVX2,   ///< AVX2 + FMA - enhanced 256-bit vectors
    AVX512  ///< AVX-512 - 512-bit vectors, 8 doubles
};

/// Core SIMD decision functions (implementation hidden)
bool should_use_simd(std::size_t count) noexcept;
SIMDLevel get_best_simd_level() noexcept;
std::size_t get_simd_min_threshold() noexcept;
std::size_t get_simd_optimal_block_size() noexcept;
std::size_t get_simd_optimal_alignment() noexcept;

/// SIMD capability queries (implementation hidden)
std::string get_simd_level_string() noexcept;
std::string get_simd_capability_string() noexcept;
void refresh_simd_cache() noexcept;

/// Specific SIMD support queries (implementation hidden)
bool supports_simd_sse2() noexcept;
bool supports_simd_avx() noexcept;
bool supports_simd_avx2() noexcept;
bool supports_simd_avx512() noexcept;
bool supports_simd_neon() noexcept;

}  // namespace simd
}  // namespace arch
}  // namespace stats

// Simplified SIMD policy macros (platform-independent)
#define LIBSTATS_SIMD_IF_AVAILABLE(size) (stats::arch::simd::should_use_simd(size))
#define LIBSTATS_SIMD_BLOCK_SIZE() (stats::arch::simd::get_simd_optimal_block_size())
#define LIBSTATS_SIMD_ALIGNMENT() (stats::arch::simd::get_simd_optimal_alignment())

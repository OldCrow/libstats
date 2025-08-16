#pragma once

#include "../platform/platform_constants_fwd.h"

namespace libstats {
namespace constants {

// Bridge parallel constants to match expected namespace structure
// Now using lightweight PIMPL interface (Phase 2 optimization)
namespace parallel {
    // Bridge to PIMPL functions for runtime optimization
    inline std::size_t get_simple_operation_grain_size() {
        return libstats::constants::parallel::get_simple_operation_grain_size();
    }
    
    inline std::size_t get_min_elements_for_distribution_parallel() {
        return libstats::constants::parallel::get_min_elements_for_distribution_parallel();
    }
    
    // Legacy constants for backward compatibility (static fallbacks)
    // NOTE: SIMPLE_OPERATION_GRAIN_SIZE is defined in platform_constants_impl.cpp to avoid ODR violations
    inline constexpr std::size_t MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL = 10000;
    inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
    
    // Adaptive functions - bridge to PIMPL implementation
    namespace adaptive {
        inline std::size_t grain_size() {
            return libstats::constants::parallel::get_default_grain_size();
        }
    }
}

// SIMD constants bridge - now using PIMPL
namespace simd {
    inline std::size_t get_default_block_size() {
        return libstats::constants::simd::get_default_block_size();
    }
    
    // Legacy constant for backward compatibility
    inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;
    
    // Additional SIMD optimization namespace for backward compatibility
    namespace optimization {
        inline constexpr std::size_t MEDIUM_DATASET_MIN_SIZE = 32;
        inline constexpr std::size_t ALIGNMENT_BENEFIT_THRESHOLD = 32;
        inline constexpr std::size_t AVX512_MIN_ALIGNED_SIZE = 8;
        inline constexpr std::size_t APPLE_SILICON_AGGRESSIVE_THRESHOLD = 6;
        inline constexpr std::size_t AVX512_SMALL_BENEFIT_THRESHOLD = 4;
    }
}

// Platform functions bridge - lightweight access to PIMPL functions
namespace platform {
    inline std::size_t get_optimal_simd_block_size() {
        return libstats::constants::platform::get_optimal_simd_block_size();
    }
    
    inline std::size_t get_optimal_alignment() {
        return libstats::constants::platform::get_optimal_alignment();
    }
    
    inline std::size_t get_min_simd_size() {
        return libstats::constants::platform::get_min_simd_size();
    }
    
    inline auto get_cache_thresholds() {
        return libstats::constants::platform::get_cache_thresholds();
    }
    
    inline bool supports_fast_transcendental() {
        return libstats::constants::platform::supports_fast_transcendental();
    }
}

} // namespace constants
} // namespace libstats

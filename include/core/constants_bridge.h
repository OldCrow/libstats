#pragma once

#include "../platform/platform_constants_fwd.h"

namespace libstats {
namespace constants {

// Bridge parallel constants to match expected namespace structure
// Now using lightweight PIMPL interface (Phase 2 optimization)
namespace parallel {
    // Bridge to PIMPL functions for runtime optimization
    std::size_t get_simple_operation_grain_size();
    
    std::size_t get_min_elements_for_distribution_parallel();
    
    // Legacy constants for backward compatibility (static fallbacks)
    // NOTE: SIMPLE_OPERATION_GRAIN_SIZE is defined in platform_constants_impl.cpp to avoid ODR violations
    inline constexpr std::size_t MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL = 10000;
    inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
    
    // Adaptive functions - bridge to PIMPL implementation
    namespace adaptive {
        std::size_t grain_size();
    }
}

// SIMD constants bridge - now using PIMPL
namespace simd {
    std::size_t get_default_block_size();
    
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
    std::size_t get_optimal_simd_block_size();
    
    std::size_t get_optimal_alignment();
    
    std::size_t get_min_simd_size();
    
    libstats::constants::platform::CacheThresholds get_cache_thresholds();
    
    bool supports_fast_transcendental();
}

} // namespace constants
} // namespace libstats

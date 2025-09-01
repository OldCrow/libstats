// simd_implementation_common.h
// Common includes and declarations for SIMD implementation files
// This header reduces compilation time by consolidating frequently used includes

#pragma once

#ifndef LIBSTATS_SIMD_IMPLEMENTATION_COMMON_H
    #define LIBSTATS_SIMD_IMPLEMENTATION_COMMON_H

    // Core lightweight headers - always needed
    #include "../core/mathematical_constants.h"
    #include "../core/threshold_constants.h"

    // Forward declarations for heavy headers
    #include "cpu_detection_fwd.h"
    #include "platform_constants_fwd.h"

    // The main SIMD interface is still needed
    #include "../platform/simd.h"

    // Standard library essentials for SIMD operations
    #include <cmath>
    #include <cstddef>
    #include <cstdint>

// Platform-specific SIMD intrinsics are included conditionally in each implementation file
// This avoids pulling in unnecessary intrinsics headers

namespace stats {
namespace simd {
namespace ops {

// Common SIMD helper functions and constants can be declared here
// These are implementation details shared across SIMD variants

// Helper to check alignment (useful across all SIMD implementations)
inline bool is_aligned(const void* ptr, std::size_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// Common SIMD width constants are already in platform_constants_fwd.h
// Additional implementation-specific helpers can be added here as needed

}  // namespace ops
}  // namespace simd
}  // namespace stats

#endif  // LIBSTATS_SIMD_IMPLEMENTATION_COMMON_H

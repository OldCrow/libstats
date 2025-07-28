#ifndef LIBSTATS_CONSTANTS_BRIDGE_H_
#define LIBSTATS_CONSTANTS_BRIDGE_H_

#include "../platform/platform_constants.h"

namespace libstats {
namespace constants {

// Bridge parallel constants to match expected namespace structure
namespace parallel {
    // Use the platform constants for backward compatibility
    inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = platform::SIMPLE_OPERATION_GRAIN_SIZE;
    inline constexpr std::size_t MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL = platform::MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL;
    inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = platform::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
    
    // Adaptive functions
    namespace adaptive {
        inline std::size_t grain_size() {
            return platform::adaptive::grain_size();
        }
    }
}

// SIMD constants bridge
namespace simd {
    inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 32;
}

} // namespace constants
} // namespace libstats

#endif // LIBSTATS_CONSTANTS_BRIDGE_H_

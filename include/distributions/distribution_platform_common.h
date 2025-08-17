#pragma once

/**
 * @file distributions/distribution_platform_common.h
 * @brief Common platform headers for statistical distributions
 * 
 * This header consolidates the platform dependencies that are commonly needed
 * by most distribution implementations. This follows our balanced consolidation
 * approach: reduce redundancy while preserving Single Responsibility Principle.
 * 
 * Used by most distribution headers, but individual distributions may still
 * include additional platform headers specific to their computational needs.
 * 
 * Design Principle: Consolidates only the platform headers used by 4+ distributions,
 * while respecting the architectural distinctiveness of each distribution type.
 */

// Core platform optimization headers used by most distributions
#include "../platform/simd.h"                   // SIMD operations (used by all distributions)
#include "../platform/parallel_execution.h"     // Parallel execution policies (used by all)
#include "../platform/work_stealing_pool.h"     // Work-stealing parallelism (used by most)

// Thread pool integration - used by distributions with heavy batch operations
#include "../platform/thread_pool.h"            // Traditional thread pool (used by most)

namespace libstats {
namespace distributions {

/**
 * @brief Forward declarations for common distribution platform types
 * These reduce compile-time dependencies across distribution headers
 */
namespace platform_support {
    // Common platform integration types used across distributions
    using SIMDVectorWidth = std::size_t;
    using ParallelThreshold = std::size_t;
    // Note: CacheStrategy moved to platform/cache_platform.h for components that need caching
}

} // namespace distributions
} // namespace libstats

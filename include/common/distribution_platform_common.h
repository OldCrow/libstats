#pragma once

/**
 * @file common/distribution_platform_common.h
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

// AQ-7 (v2.0.0): The heavy platform headers (simd.h, parallel_execution.h,
// work_stealing_pool.h, thread_pool.h) have been moved to
// distribution_impl_common.h, which is included only in distribution .cpp files.
// Distribution *headers* should not need SIMD intrinsic types or threading
// infrastructure in their public class definitions.

namespace stats {
namespace distributions {

/**
 * @brief Forward declarations for common distribution platform types
 * These reduce compile-time dependencies across distribution headers
 */
namespace platform_support {
// Common platform integration types used across distributions
using SIMDVectorWidth = std::size_t;
using ParallelThreshold = std::size_t;
}  // namespace platform_support

}  // namespace distributions
}  // namespace stats

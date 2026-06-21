#pragma once

/**
 * @file common/distribution_impl_common.h
 * @brief Heavy platform includes for distribution .cpp implementation files.
 *
 * AQ-7 (v2.0.0): Split the platform-infrastructure includes out of
 * distribution_platform_common.h (which is included by all distribution
 * *headers*) into this file, which is included only in distribution *source*
 * files that actually use SIMD, parallel, or threading machinery.
 *
 * This reduces the transitive include depth of every distribution header from
 * ~1,287 to ~1,050 by avoiding pulling simd.h, work_stealing_pool.h,
 * thread_pool.h, and parallel_execution.h into public-facing headers.
 *
 * **Only include this file in .cpp files, never in .h files.**
 */

#include "libstats/platform/parallel_execution.h"  // C++20 parallel execution wrappers
#include "libstats/platform/simd.h"                // SIMD intrinsics and VectorOps
#include "libstats/platform/thread_pool.h"         // Traditional thread pool
#include "libstats/platform/work_stealing_pool.h"  // Work-stealing parallel pool

#pragma once

/**
 * @file platform/cache_platform.h
 * @brief Platform header for cache-specific functionality
 * 
 * This header provides cache-related platform dependencies for components
 * that actually need cache functionality. Most distribution implementations
 * don't need direct cache access - only specialized components like
 * dispatch utilities and cache-aware batch operations require this.
 * 
 * Design Rationale:
 * - Separates cache concerns from general distribution platform needs
 * - Reduces compilation dependencies for most distribution code
 * - Provides clear indication of which components use caching
 * - Follows Single Responsibility Principle for platform headers
 * 
 * Components that should include this header:
 * - Dispatch utilities with cache-aware strategies
 * - Performance dispatchers with caching
 * - Distribution cache management systems
 * - Batch operation implementations with cache-aware lambdas
 */

// Core cache infrastructure
#include "../cache/adaptive_cache.h"         // Adaptive cache implementation

// Cache integration platform support
#include "../platform/parallel_execution.h"  // For cache-aware parallel operations
#include "../core/performance_dispatcher.h"  // For cache performance integration

namespace libstats {
namespace cache {
namespace platform {

/**
 * @brief Platform-specific cache configuration and integration
 * 
 * These types and constants define how caching integrates with
 * platform-specific optimizations and parallel execution strategies.
 */

// Cache integration with platform parallelism
using CacheAwareParallelThreshold = std::size_t;

// Platform cache configuration constants
constexpr std::size_t DEFAULT_CACHE_SIZE = 1024;
constexpr std::size_t CACHE_LINE_SIZE = 64;  // Typical CPU cache line size

/**
 * @brief Cache strategy selection based on platform capabilities
 * 
 * Different platforms may benefit from different cache strategies
 * based on memory hierarchy, NUMA topology, etc.
 */
enum class PlatformCacheStrategy {
    ADAPTIVE_LRU,      // Standard adaptive LRU caching
    NUMA_AWARE,        // NUMA-topology aware caching (future)
    GPU_UNIFIED,       // Unified CPU-GPU cache (future)
    STREAMING          // Streaming-optimized cache (future)
};

/**
 * @brief Determine optimal cache strategy for current platform
 * @return Recommended cache strategy for this platform
 */
inline PlatformCacheStrategy getOptimalCacheStrategy() noexcept {
    // For now, always use adaptive LRU
    // Future versions may detect platform capabilities
    return PlatformCacheStrategy::ADAPTIVE_LRU;
}

/**
 * @brief Check if cache-aware operations are beneficial for given workload size
 * @param workload_size Size of the computational workload
 * @return True if cache-aware strategies should be used
 */
inline bool shouldUseCacheAwareStrategy(std::size_t workload_size) noexcept {
    // Cache-aware strategies have overhead - only beneficial for larger workloads
    // and when cache hit rates are expected to be reasonable
    return workload_size >= 100;  // Empirically determined threshold
}

} // namespace platform
} // namespace cache
} // namespace libstats

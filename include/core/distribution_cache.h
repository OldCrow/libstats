#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace stats {

// =============================================================================
// THREAD-SAFE CACHE MANAGEMENT INTERFACE
// =============================================================================

/**
 * @brief Thread-safe cache management base class for distributions
 *
 * Provides the infrastructure for thread-safe caching in distribution classes
 * using the distribution cache adapter pattern.
 *
 * ### Dual-flag pattern (cache_valid_ + cacheValidAtomic_)
 * Two flags represent the same state deliberately:
 * - cacheValidAtomic_ enables a contention-reduced fast path in
 *   getCachedValue(): the atomic load (acquire) avoids a lock entirely when
 *   the cache is stale (proceeding directly to the unique-lock slow path);
 *   when valid it still acquires shared_lock before reading values.  This is
 *   NOT a fully lock-free read — it is a contention-reduced path that avoids
 *   the exclusive (write) lock on the hot path.
 * - cache_valid_ is the plain bool read under shared_lock; it serves as the
 *   ground-truth flag for code paths that already hold cache_mutex_.
 *
 * ### Per-parameter atomic fast paths (derived classes)
 * Some derived classes (e.g. GaussianDistribution) expose individual
 * parameter atomics (atomicMean_, atomicStdDev_) for lock-free
 * getMean()/getStdDev() calls.  Each value is individually consistent, but
 * reading two atomics across separate calls does not form an atomic pair:
 * a concurrent setParameters() between the two reads yields a torn snapshot.
 * Callers that need a consistent pair of parameters should use the snapshot
 * pattern (shared_lock, read both under the same lock).
 *
 * v2.0.0 removed the old derived-class cacheValidAtomic_ shadows; this base
 * member is now the single atomic cache-validity flag for all distributions.
 * The plain bool remains for code paths that already hold cache_mutex_.
 */
class ThreadSafeCacheManager {
   protected:
    /**
     * @brief Thread-safe cache management infrastructure
     */
    mutable std::shared_mutex cache_mutex_;
    mutable bool cache_valid_{false};

    /**
     * @brief Atomic cache validity flag for lock-free fast paths
     * @see Class-level documentation for the dual-flag pattern rationale.
     */
    mutable std::atomic<bool> cacheValidAtomic_{false};

   public:
    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     */
    virtual ~ThreadSafeCacheManager() = default;

    /**
     * @brief Update cached statistical properties (must be overridden)
     * @note Called under unique lock; implementation should set cache_valid_ = true
     */
    virtual void updateCacheUnsafe() const = 0;

    /**
     * @brief Invalidate cache when parameters change
     * @note Thread-safe; call whenever parameters are modified
     */
    void invalidateCache() noexcept {
        std::unique_lock lock(cache_mutex_);
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }

    /**
     * @brief Thread-safe cached value access with double-checked locking
     * @param accessor Function to access cached value
     * @return Cached value
     * @note Implementation in src/distribution_cache.cpp with explicit instantiations
     */
    template <typename Func>
    auto getCachedValue(Func&& accessor) const -> decltype(accessor());
};

}  // namespace stats

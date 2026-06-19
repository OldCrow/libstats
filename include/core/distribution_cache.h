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
 * - cacheValidAtomic_ enables a lock-free fast path in getCachedValue(). An
 *   acquire load synchronizes-with the release store in updateCacheUnsafe(),
 *   ensuring cache_valid_ and the cached values are visible before the atomic
 *   is seen as true.
 * - cache_valid_ is the plain bool read under shared_lock in the slow path.
 * Consolidating to a single atomic<bool> is a v2.0.0 task; it is a
 * prerequisite for noexcept move constructors (which cannot hold a mutex).
 * Do NOT collapse the two flags here without completing that larger migration.
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

// =============================================================================
// CACHED PROPERTY TEMPLATE
// =============================================================================

/**
 * @brief Template helper for cached statistical properties
 * @tparam PropertyType Type of cached property
 *
 * @warning **Not thread-safe.** This class has no synchronisation of its own.
 * Use it only within a scope that already holds an appropriate lock from
 * ThreadSafeCacheManager (e.g. under cache_mutex_ unique_lock). Concurrent
 * access from multiple threads without external locking is a data race.
 * Full per-property atomic protection is planned for v2.0.0 as part of the
 * noexcept move-constructor migration.
 */
template <typename PropertyType>
class CachedProperty {
   private:
    mutable PropertyType value_;
    mutable bool valid_{false};

   public:
    template <typename ComputeFunc>
    PropertyType get(ComputeFunc&& compute_func) const {
        if (!valid_) {
            value_ = compute_func();
            valid_ = true;
        }
        return value_;
    }

    void invalidate() noexcept { valid_ = false; }

    bool isValid() const noexcept { return valid_; }
};

}  // namespace stats

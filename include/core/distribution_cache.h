#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace libstats {

// =============================================================================
// THREAD-SAFE CACHE MANAGEMENT INTERFACE
// =============================================================================

/**
 * @brief Thread-safe cache management base class for distributions
 *
 * Provides the infrastructure for thread-safe caching in distribution classes
 * using the distribution cache adapter pattern.
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

}  // namespace libstats

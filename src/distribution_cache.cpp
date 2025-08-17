#include "cache/distribution_cache.h"
#include <functional>

namespace libstats {

template<typename Func>
auto ThreadSafeCacheManager::getCachedValue(Func&& accessor) const -> decltype(accessor()) {
    // Fast path: check cache validity atomically without lock
    if (cacheValidAtomic_.load(std::memory_order_acquire)) {
        std::shared_lock lock(cache_mutex_);
        if (cache_valid_) {
            return accessor();
        }
    }

    // Slow path: acquire unique lock and update cache
    std::unique_lock lock(cache_mutex_);
    if (!cache_valid_) {
        updateCacheUnsafe();
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);
    }

    return accessor();
}

// =============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// =============================================================================

// Instantiate for common accessor types to reduce compilation overhead.
// Add more instantiations as needed for other return types or function signatures.

template double ThreadSafeCacheManager::getCachedValue<std::function<double()>>(std::function<double()>&&) const;
template bool ThreadSafeCacheManager::getCachedValue<std::function<bool()>>(std::function<bool()>&&) const;
template int ThreadSafeCacheManager::getCachedValue<std::function<int()>>(std::function<int()>&&) const;
template size_t ThreadSafeCacheManager::getCachedValue<std::function<size_t()>>(std::function<size_t()>&&) const;

}


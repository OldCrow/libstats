#ifndef LIBSTATS_DISTRIBUTION_CACHE_ADAPTER_H_
#define LIBSTATS_DISTRIBUTION_CACHE_ADAPTER_H_

#include "adaptive_cache.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>

namespace libstats {

// =============================================================================
// DISTRIBUTION-SPECIFIC CACHE METRICS
// =============================================================================

/**
 * @brief Distribution-specific cache performance metrics (internal)
 * @note Contains atomic counters for thread-safe updates
 */
struct DistributionCacheMetrics {
    std::atomic<size_t> statistical_hits{0};       // Cache hits for statistical properties
    std::atomic<size_t> parameter_hits{0};         // Cache hits for parameters
    std::atomic<size_t> computation_hits{0};       // Cache hits for expensive computations
    std::atomic<size_t> invalidations{0};          // Cache invalidations due to parameter changes
    std::atomic<size_t> adaptive_resizes{0};       // Automatic cache resizing events
    
    void resetMetrics() noexcept {
        statistical_hits.store(0);
        parameter_hits.store(0);
        computation_hits.store(0);
        invalidations.store(0);
        adaptive_resizes.store(0);
    }
};

/**
 * @brief Distribution cache metrics snapshot for external access
 * @note Copyable snapshot of atomic metrics
 */
struct DistributionCacheMetricsSnapshot {
    size_t statistical_hits;
    size_t parameter_hits;
    size_t computation_hits;
    size_t invalidations;
    size_t adaptive_resizes;
    
    double statisticalHitRate() const noexcept {
        size_t total = statistical_hits + parameter_hits + computation_hits;
        return total > 0 ? static_cast<double>(statistical_hits) / total : 0.0;
    }
    
    double totalHitRate() const noexcept {
        size_t total_hits = statistical_hits + parameter_hits + computation_hits;
        return total_hits > 0 ? static_cast<double>(total_hits) / (total_hits + 1) : 0.0;
    }
};

// =============================================================================
// DISTRIBUTION CACHE ADAPTER
// =============================================================================

/**
 * @brief Distribution cache adapter that wraps platform::AdaptiveCache
 * 
 * This adapter provides distribution-specific caching patterns while leveraging
 * the platform-optimized adaptive cache for actual storage and performance.
 * 
 * @tparam Key Cache key type (typically std::string for property names)
 * @tparam Value Cache value type (typically double for statistical values)
 */
template<typename Key, typename Value>
class DistributionCacheAdapter {
private:
    mutable cache::AdaptiveCache<Key, Value> platform_cache_;
    mutable DistributionCacheMetrics distribution_metrics_;
    mutable std::shared_mutex adapter_mutex_;
    
public:
    /**
     * @brief Construct adapter with platform cache configuration
     * @param platform_config Configuration for underlying platform cache
     */
    explicit DistributionCacheAdapter(const cache::AdaptiveCacheConfig& platform_config = {})
        : platform_cache_(platform_config) {}
    
    /**
     * @brief Get cached statistical property with distribution-aware metrics
     * @param key Property identifier (e.g., "mean", "variance", "skewness")
     * @return Cached value if available
     */
    std::optional<Value> getStatisticalProperty(const Key& key) const {
        auto result = platform_cache_.get(key);
        if (result.has_value()) {
            distribution_metrics_.statistical_hits.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    
    /**
     * @brief Cache statistical property with appropriate TTL
     * @param key Property identifier
     * @param value Property value
     */
    void putStatisticalProperty(const Key& key, const Value& value) {
        platform_cache_.put(key, value);
    }
    
    /**
     * @brief Get cached parameter with parameter-aware metrics
     * @param key Parameter identifier (e.g., "mean_param", "stddev_param")
     * @return Cached value if available
     */
    std::optional<Value> getParameter(const Key& key) const {
        auto result = platform_cache_.get(key);
        if (result.has_value()) {
            distribution_metrics_.parameter_hits.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    
    /**
     * @brief Cache parameter value
     * @param key Parameter identifier
     * @param value Parameter value
     */
    void putParameter(const Key& key, const Value& value) {
        platform_cache_.put(key, value);
    }
    
    /**
     * @brief Get cached expensive computation result
     * @param key Computation identifier (e.g., "pdf_normalization", "cdf_integration")
     * @return Cached result if available
     */
    std::optional<Value> getComputationResult(const Key& key) const {
        auto result = platform_cache_.get(key);
        if (result.has_value()) {
            distribution_metrics_.computation_hits.fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }
    
    /**
     * @brief Cache expensive computation result
     * @param key Computation identifier
     * @param value Computation result
     */
    void putComputationResult(const Key& key, const Value& value) {
        platform_cache_.put(key, value);
    }
    
    /**
     * @brief Invalidate all distribution caches (e.g., when parameters change)
     */
    void invalidateAll() {
        std::unique_lock lock(adapter_mutex_);
        platform_cache_.clear();
        distribution_metrics_.invalidations.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief Get distribution-specific cache metrics snapshot
     * @return Distribution cache metrics snapshot
     */
    DistributionCacheMetricsSnapshot getDistributionMetrics() const noexcept {
        DistributionCacheMetricsSnapshot snapshot;
        snapshot.statistical_hits = distribution_metrics_.statistical_hits.load();
        snapshot.parameter_hits = distribution_metrics_.parameter_hits.load();
        snapshot.computation_hits = distribution_metrics_.computation_hits.load();
        snapshot.invalidations = distribution_metrics_.invalidations.load();
        snapshot.adaptive_resizes = distribution_metrics_.adaptive_resizes.load();
        return snapshot;
    }
    
    /**
     * @brief Get underlying platform cache metrics
     * @return Platform cache metrics
     */
    auto getPlatformMetrics() const {
        return platform_cache_.getMetrics();
    }
    
    /**
     * @brief Get cache size
     * @return Number of cached entries
     */
    size_t size() const {
        return platform_cache_.size();
    }
    
    /**
     * @brief Check if cache is empty
     * @return true if cache is empty
     */
    bool empty() const {
        return platform_cache_.empty();
    }
    
    /**
     * @brief Reset all metrics
     */
    void resetAllMetrics() {
        distribution_metrics_.resetMetrics();
        platform_cache_.resetMetrics();
    }
};

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
    template<typename Func>
    auto getCachedValue(Func&& accessor) const -> decltype(accessor());
};

// =============================================================================
// CACHED PROPERTY TEMPLATE
// =============================================================================

/**
 * @brief Template helper for cached statistical properties
 * @tparam PropertyType Type of cached property
 */
template<typename PropertyType>
class CachedProperty {
private:
    mutable PropertyType value_;
    mutable bool valid_{false};
    
public:
    template<typename ComputeFunc>
    PropertyType get(ComputeFunc&& compute_func) const {
        if (!valid_) {
            value_ = compute_func();
            valid_ = true;
        }
        return value_;
    }
    
    void invalidate() noexcept {
        valid_ = false;
    }
    
    bool isValid() const noexcept {
        return valid_;
    }
};

} // namespace libstats

#endif // LIBSTATS_DISTRIBUTION_CACHE_ADAPTER_H_

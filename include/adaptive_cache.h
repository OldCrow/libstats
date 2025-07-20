#ifndef LIBSTATS_ADAPTIVE_CACHE_H_
#define LIBSTATS_ADAPTIVE_CACHE_H_

#include <unordered_map>
#include <unordered_set>
#include <list>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <optional>
#include <functional>
#include <vector>
#include <algorithm>
#include <memory>
#include <thread>

namespace libstats {
namespace cache {

/**
 * @brief Eviction policy types
 */
enum class EvictionPolicy {
    LRU,        ///< Least Recently Used
    LFU,        ///< Least Frequently Used
    TTL,        ///< Time To Live based
    ADAPTIVE    ///< Adaptive hybrid policy
};

/**
 * @brief Cache performance metrics with detailed statistics
 */
struct CacheMetrics {
    std::atomic<size_t> hits{0};
    std::atomic<size_t> misses{0};
    std::atomic<size_t> evictions{0};
    std::atomic<size_t> memory_usage{0};
    std::atomic<size_t> cache_size{0};
    std::atomic<double> average_access_time{0.0};
    std::atomic<double> hit_rate{0.0};
    
    // Advanced metrics
    std::atomic<size_t> prefetch_hits{0};
    std::atomic<size_t> prefetch_misses{0};
    std::atomic<size_t> memory_pressure_events{0};
    std::atomic<size_t> adaptive_resizes{0};
    
    // Make CacheMetrics copyable
    CacheMetrics() = default;
    
    CacheMetrics(const CacheMetrics& other)
        : hits(other.hits.load())
        , misses(other.misses.load())
        , evictions(other.evictions.load())
        , memory_usage(other.memory_usage.load())
        , cache_size(other.cache_size.load())
        , average_access_time(other.average_access_time.load())
        , hit_rate(other.hit_rate.load())
        , prefetch_hits(other.prefetch_hits.load())
        , prefetch_misses(other.prefetch_misses.load())
        , memory_pressure_events(other.memory_pressure_events.load())
        , adaptive_resizes(other.adaptive_resizes.load())
    {}
    
    CacheMetrics& operator=(const CacheMetrics& other) {
        if (this != &other) {
            hits.store(other.hits.load());
            misses.store(other.misses.load());
            evictions.store(other.evictions.load());
            memory_usage.store(other.memory_usage.load());
            cache_size.store(other.cache_size.load());
            average_access_time.store(other.average_access_time.load());
            hit_rate.store(other.hit_rate.load());
            prefetch_hits.store(other.prefetch_hits.load());
            prefetch_misses.store(other.prefetch_misses.load());
            memory_pressure_events.store(other.memory_pressure_events.load());
            adaptive_resizes.store(other.adaptive_resizes.load());
        }
        return *this;
    }
    
    void updateHitRate() noexcept {
        size_t total = hits.load() + misses.load();
        if (total > 0) {
            hit_rate.store(static_cast<double>(hits.load()) / total, std::memory_order_relaxed);
        }
    }
    
    double getPrefetchEffectiveness() const noexcept {
        size_t total_prefetch = prefetch_hits.load() + prefetch_misses.load();
        return total_prefetch > 0 ? static_cast<double>(prefetch_hits.load()) / total_prefetch : 0.0;
    }
};

/**
 * @brief Advanced cache configuration with adaptive parameters
 */
struct AdaptiveCacheConfig {
    // Basic configuration
    size_t max_memory_bytes = 1024 * 1024;    // 1MB default
    size_t min_cache_size = 64;               // Minimum entries
    size_t max_cache_size = 4096;             // Maximum entries
    std::chrono::milliseconds ttl{10000};     // 10 second TTL
    
    // Eviction configuration
    EvictionPolicy eviction_policy = EvictionPolicy::ADAPTIVE;
    double eviction_threshold = 0.85;         // Start eviction at 85% capacity
    size_t batch_eviction_size = 10;          // Evict multiple entries at once
    
    // Adaptive behavior
    bool enable_adaptive_sizing = true;       // Enable dynamic resizing
    bool enable_prefetching = true;           // Enable predictive prefetching
    bool memory_pressure_aware = true;        // React to memory pressure
    
    // Performance tuning
    std::chrono::milliseconds metrics_update_interval{1000};  // 1 second
    double hit_rate_target = 0.85;            // Target hit rate for sizing
    double memory_efficiency_target = 0.7;    // Target memory efficiency
    
    // Advanced features
    bool enable_background_optimization = true; // Enable background thread
    size_t prefetch_queue_size = 32;          // Maximum prefetch queue size
    double access_pattern_sensitivity = 0.1;  // Sensitivity to access patterns
};

/**
 * @brief Cache entry with comprehensive metadata
 */
template<typename T>
struct CacheEntry {
    T value;
    std::chrono::steady_clock::time_point creation_time;
    mutable std::chrono::steady_clock::time_point last_access_time;
    mutable std::atomic<size_t> access_count{0};
    size_t memory_size;
    double access_frequency = 0.0;  // Calculated access frequency
    
    CacheEntry() = default;
    
    explicit CacheEntry(T val) 
        : value(std::move(val))
        , creation_time(std::chrono::steady_clock::now())
        , last_access_time(creation_time)
        , memory_size(sizeof(T)) {}
    
    // Make it movable by providing move operations
    CacheEntry(const CacheEntry&) = delete;
    CacheEntry& operator=(const CacheEntry&) = delete;
    
    CacheEntry(CacheEntry&& other) noexcept
        : value(std::move(other.value))
        , creation_time(other.creation_time)
        , last_access_time(other.last_access_time)
        , access_count(other.access_count.load())
        , memory_size(other.memory_size)
        , access_frequency(other.access_frequency) {}
    
    CacheEntry& operator=(CacheEntry&& other) noexcept {
        if (this != &other) {
            value = std::move(other.value);
            creation_time = other.creation_time;
            last_access_time = other.last_access_time;
            access_count.store(other.access_count.load());
            memory_size = other.memory_size;
            access_frequency = other.access_frequency;
        }
        return *this;
    }
    
    bool isExpired(std::chrono::milliseconds ttl) const noexcept {
        auto now = std::chrono::steady_clock::now();
        return (now - creation_time) > ttl;
    }
    
    void updateAccess() const noexcept {
        access_count.fetch_add(1, std::memory_order_relaxed);
        last_access_time = std::chrono::steady_clock::now();
    }
    
    double getRecency() const noexcept {
        auto now = std::chrono::steady_clock::now();
        auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_access_time).count();
        return 1.0 / (1.0 + age_ms / 1000.0);  // Exponential decay
    }
    
    double getFrequency() const noexcept {
        auto now = std::chrono::steady_clock::now();
        auto lifetime_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - creation_time).count();
        return lifetime_ms > 0 ? static_cast<double>(access_count.load()) / (lifetime_ms / 1000.0) : 0.0;
    }
    
    double getPriority() const noexcept {
        return 0.6 * getRecency() + 0.4 * getFrequency();
    }
};

/**
 * @brief Advanced adaptive cache with predictive capabilities
 */
template<typename Key, typename Value>
class AdaptiveCache {
public:
    using KeyType = Key;
    using ValueType = Value;
    using EntryType = CacheEntry<Value>;
    
private:
    // Core cache storage
    mutable std::unordered_map<Key, EntryType> cache_;
    mutable std::shared_mutex cache_mutex_;
    
    // Configuration and metrics
    AdaptiveCacheConfig config_;
    mutable CacheMetrics metrics_;
    
    // Access pattern tracking
    mutable std::list<Key> access_history_;
    mutable std::mutex access_pattern_mutex_;  // Separate mutex for access pattern tracking
    mutable std::unordered_set<Key> prefetch_queue_;
    mutable std::shared_mutex prefetch_mutex_;
    
    // Background optimization
    std::unique_ptr<std::thread> background_thread_;
    std::atomic<bool> shutdown_requested_{false};
    mutable std::condition_variable optimization_cv_;
    mutable std::mutex cv_mutex_;
    
    // Timing for metrics
    mutable std::chrono::steady_clock::time_point last_metrics_update_;
    
public:
    explicit AdaptiveCache(const AdaptiveCacheConfig& config = AdaptiveCacheConfig{})
        : config_(config)
        , last_metrics_update_(std::chrono::steady_clock::now()) {
        
        if (config_.enable_background_optimization) {
            startBackgroundOptimization();
        }
    }
    
    ~AdaptiveCache() {
        if (background_thread_) {
            shutdown_requested_.store(true);
            optimization_cv_.notify_all();
            if (background_thread_->joinable()) {
                background_thread_->join();
            }
        }
    }
    
    // Disable copy operations (move-only)
    AdaptiveCache(const AdaptiveCache&) = delete;
    AdaptiveCache& operator=(const AdaptiveCache&) = delete;
    
    AdaptiveCache(AdaptiveCache&& other) noexcept 
        : cache_(std::move(other.cache_))
        , config_(std::move(other.config_))
        , metrics_(std::move(other.metrics_))
        , access_history_(std::move(other.access_history_))
        , prefetch_queue_(std::move(other.prefetch_queue_))
        , shutdown_requested_(false)  // Start fresh
        , last_metrics_update_(other.last_metrics_update_) {
        
        // THREAD SAFETY FIX: Properly shutdown other's thread before moving
        if (other.background_thread_) {
            other.shutdown_requested_.store(true);
            other.optimization_cv_.notify_all();
            if (other.background_thread_->joinable()) {
                other.background_thread_->join();
            }
        }
        
        // Start our own background thread if enabled
        if (config_.enable_background_optimization) {
            startBackgroundOptimization();
        }
        
        // Reset other object to safe state
        other.background_thread_.reset();
    }
    
    AdaptiveCache& operator=(AdaptiveCache&& other) noexcept {
        if (this != &other) {
            // Clean up current background thread
            if (background_thread_) {
                shutdown_requested_.store(true);
                optimization_cv_.notify_all();
                if (background_thread_->joinable()) {
                    background_thread_->join();
                }
            }
            
            // Move data from other
            cache_ = std::move(other.cache_);
            config_ = std::move(other.config_);
            metrics_ = std::move(other.metrics_);
            access_history_ = std::move(other.access_history_);
            prefetch_queue_ = std::move(other.prefetch_queue_);
            background_thread_ = std::move(other.background_thread_);
            shutdown_requested_.store(other.shutdown_requested_.load());
            last_metrics_update_ = other.last_metrics_update_;
            
            // Reset other object to safe state
            other.shutdown_requested_.store(true);
        }
        return *this;
    }
    
    /**
     * @brief Get value from cache with access pattern tracking
     */
    std::optional<Value> get(const Key& key) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        {
            std::shared_lock lock(cache_mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                if (!it->second.isExpired(config_.ttl)) {
                    it->second.updateAccess();
                    updateAccessPattern(key);
                    
                    auto end_time = std::chrono::high_resolution_clock::now();
                    updateAccessTime(start_time, end_time);
                    
                    metrics_.hits.fetch_add(1, std::memory_order_relaxed);
                    updateMetrics();  // Update metrics after hit
                    return it->second.value;
                } else {
                    // Entry expired - will be removed in put() or background cleanup
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        updateAccessTime(start_time, end_time);
        
        metrics_.misses.fetch_add(1, std::memory_order_relaxed);
        updateMetrics();
        
        return std::nullopt;
    }
    
    /**
     * @brief Put value in cache with adaptive management
     */
    void put(const Key& key, const Value& value) {
        std::unique_lock lock(cache_mutex_);
        
        // Clean expired entries first
        cleanExpiredEntries();
        
        // Check if this is a new key (will increase cache size)
        bool is_new_key = cache_.find(key) == cache_.end();
        
        // If adding a new key would exceed capacity, evict first
        if (is_new_key && cache_.size() >= config_.max_cache_size) {
            performEviction();
        }
        
        // Insert or update entry
        EntryType entry(value);
        auto [it, inserted] = cache_.try_emplace(key, std::move(entry));
        if (!inserted) {
            // Update the existing entry's value directly rather than trying to move-assign
            it->second.value = value;
            it->second.last_access_time = std::chrono::steady_clock::now();
        }
        
        updateAccessPattern(key);
        updateMemoryUsage();
        
        // Final check - if somehow we're still over limit, force eviction
        if (cache_.size() > config_.max_cache_size) {
            performEviction();
        }
        
        updateMetrics();
    }
    
    /**
     * @brief Prefetch likely-to-be-accessed values
     */
    template<typename Generator>
    void prefetch(const std::vector<Key>& keys, Generator&& value_generator) {
        if (!config_.enable_prefetching) return;
        
        std::unique_lock prefetch_lock(prefetch_mutex_);
        
        for (const auto& key : keys) {
            if (prefetch_queue_.size() >= config_.prefetch_queue_size) break;
            
            // Only prefetch if not already in cache
            {
                std::shared_lock cache_lock(cache_mutex_);
                if (cache_.find(key) != cache_.end()) continue;
            }
            
            if (prefetch_queue_.insert(key).second) {
                try {
                    auto value = value_generator(key);
                    put(key, value);
                    metrics_.prefetch_hits.fetch_add(1, std::memory_order_relaxed);
                } catch (...) {
                    metrics_.prefetch_misses.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        
        // Cleanup prefetch queue
        for (auto it = prefetch_queue_.begin(); it != prefetch_queue_.end();) {
            std::shared_lock cache_lock(cache_mutex_);
            if (cache_.find(*it) != cache_.end()) {
                it = prefetch_queue_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    /**
     * @brief Clear entire cache
     */
    void clear() {
        std::unique_lock cache_lock(cache_mutex_);
        std::lock_guard<std::mutex> pattern_lock(access_pattern_mutex_);
        cache_.clear();
        access_history_.clear();
        metrics_.memory_usage.store(0, std::memory_order_relaxed);
        metrics_.cache_size.store(0, std::memory_order_relaxed);
    }
    
    /**
     * @brief Get current cache metrics
     */
    CacheMetrics getMetrics() const {
        updateMetrics(true);  // Force update to ensure current metrics
        return metrics_;
    }
    
    /**
     * @brief Get cache statistics summary
     */
    struct CacheStats {
        size_t size;
        size_t memory_usage;
        double hit_rate;
        double memory_efficiency;
        double prefetch_effectiveness;
        size_t evictions;
        double average_access_time;
    };
    
    CacheStats getStats() const {
        std::shared_lock lock(cache_mutex_);
        auto metrics = getMetrics();
        
        CacheStats stats;
        stats.size = cache_.size();
        stats.memory_usage = metrics.memory_usage.load();
        stats.hit_rate = metrics.hit_rate.load();
        stats.memory_efficiency = stats.memory_usage > 0 ? 
            static_cast<double>(metrics.hits.load()) / stats.memory_usage : 0.0;
        stats.prefetch_effectiveness = metrics.getPrefetchEffectiveness();
        stats.evictions = metrics.evictions.load();
        stats.average_access_time = metrics.average_access_time.load();
        
        return stats;
    }
    
    /**
     * @brief Update cache configuration
     */
    void updateConfig(const AdaptiveCacheConfig& new_config) {
        std::unique_lock lock(cache_mutex_);
        config_ = new_config;
        
        // Apply immediate changes
        if (cache_.size() > config_.max_cache_size) {
            performEviction();
        }
    }
    
    /**
     * @brief Get cached computation parameters for a specific operation
     * Used for predictive cache warming and optimization
     * @param cache_key Unique identifier for the computation type
     * @return Optional cached parameters (placeholder implementation)
     */
    std::optional<std::pair<size_t, double>> getCachedComputationParams(const std::string& cache_key) const {
        // Placeholder implementation - returns adaptive grain size suggestion
        // In a full implementation, this would maintain performance history per operation type
        std::shared_lock lock(cache_mutex_);
        
        // Extract size hint from cache key if present
        size_t pos = cache_key.find("_batch_");
        if (pos != std::string::npos) {
            try {
                size_t batch_size = std::stoull(cache_key.substr(pos + 7));
                // Return adaptive grain size based on cache performance
                double hit_rate = metrics_.hit_rate.load();
                size_t grain_size = std::max(size_t(8), batch_size / (config_.enable_background_optimization ? 16 : 8));
                return std::make_pair(grain_size, hit_rate);
            } catch (const std::exception&) {
                // Fallback for invalid key format
            }
        }
        
        return std::nullopt;
    }
    
    /**
     * @brief Get optimal grain size for parallel operations based on cache behavior
     * @param data_size Size of the data to be processed
     * @param operation_type Type of operation ("gaussian_pdf", "gaussian_cdf", etc.)
     * @return Recommended grain size for parallel processing
     */
    size_t getOptimalGrainSize(size_t data_size, const std::string& operation_type) const {
        std::shared_lock lock(cache_mutex_);
        
        // Base grain size calculation
        size_t base_grain = std::max(size_t(8), data_size / 16);  // Target ~16 chunks
        
        // Adjust based on cache performance metrics
        double hit_rate = metrics_.hit_rate.load();
        double memory_pressure = static_cast<double>(metrics_.memory_usage.load()) / config_.max_memory_bytes;
        
        // If hit rate is low, use smaller grains to improve locality
        if (hit_rate < 0.7) {
            base_grain = std::max(size_t(4), base_grain / 2);
        }
        
        // If memory pressure is high, use larger grains to reduce overhead
        if (memory_pressure > 0.8) {
            base_grain = std::min(data_size / 4, base_grain * 2);
        }
        
        // Operation-specific tuning
        if (operation_type.find("pdf") != std::string::npos) {
            // PDF operations are typically compute-intensive, can use larger grains
            base_grain = std::min(data_size / 8, base_grain * 3 / 2);
        } else if (operation_type.find("cdf") != std::string::npos) {
            // CDF operations may have more irregular access patterns
            base_grain = std::max(size_t(16), base_grain);
        }
        
        return std::clamp(base_grain, size_t(1), data_size);
    }
    
    /**
     * @brief Record performance metrics for batch operations
     * Used to improve future cache and parallelization decisions
     * @param cache_key Unique identifier for the operation
     * @param data_size Size of the processed data
     * @param grain_size Grain size used for the operation
     */
    void recordBatchPerformance([[maybe_unused]] const std::string& cache_key, 
                               [[maybe_unused]] size_t data_size, 
                               [[maybe_unused]] size_t grain_size) {
        // Update metrics for adaptive behavior
        // In a full implementation, this would maintain detailed performance history
        
        std::unique_lock lock(cache_mutex_);
        
        // Update basic metrics
        metrics_.cache_size.store(cache_.size(), std::memory_order_relaxed);
        updateMemoryUsage();
        updateMetrics();
        
        // Log performance for future optimization (placeholder)
        // Real implementation would store operation_type -> performance mapping
        // and use machine learning or heuristics to optimize grain sizes
        
        // Trigger background optimization if enabled
        if (config_.enable_background_optimization && background_thread_) {
            optimization_cv_.notify_one();
        }
    }
    
private:
    void updateAccessPattern(const Key& key) const {
        // Track access history for pattern detection using separate mutex
        std::lock_guard<std::mutex> pattern_lock(access_pattern_mutex_);
        access_history_.push_back(key);
        if (access_history_.size() > 100) {  // Keep last 100 accesses
            access_history_.pop_front();
        }
    }
    
    void updateAccessTime(const std::chrono::high_resolution_clock::time_point& start,
                         const std::chrono::high_resolution_clock::time_point& end) const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double current_avg = metrics_.average_access_time.load();
        double new_time = duration.count();
        
        // Exponential moving average
        double alpha = 0.1;
        double new_avg = alpha * new_time + (1.0 - alpha) * current_avg;
        metrics_.average_access_time.store(new_avg, std::memory_order_relaxed);
    }
    
    void updateMemoryUsage() const {
        size_t total_memory = 0;
        for (const auto& [key, entry] : cache_) {
            total_memory += entry.memory_size + sizeof(key);
        }
        metrics_.memory_usage.store(total_memory, std::memory_order_relaxed);
        metrics_.cache_size.store(cache_.size(), std::memory_order_relaxed);
    }
    
    void updateMetrics(bool force = false) const {
        auto now = std::chrono::steady_clock::now();
        if (force || now - last_metrics_update_ >= config_.metrics_update_interval) {
            metrics_.updateHitRate();
            updateMemoryUsage();
            last_metrics_update_ = now;
        }
    }
    
    bool shouldEvict() const {
        // Check memory-based eviction threshold
        if (config_.memory_pressure_aware) {
            size_t current_memory = metrics_.memory_usage.load();
            if (current_memory >= config_.max_memory_bytes * config_.eviction_threshold) {
                return true;
            }
        }
        
        // Check if we're getting close to the size limit (soft threshold)
        return cache_.size() >= static_cast<size_t>(config_.max_cache_size * config_.eviction_threshold);
    }
    
    void performEviction() {
        if (cache_.empty()) return;
        
        size_t current_size = cache_.size();
        size_t max_size = config_.max_cache_size;
        
        // Calculate how many entries to evict
        size_t eviction_count = 1; // Default to evicting at least 1
        if (current_size > max_size) {
            // We're over the limit - evict enough to get back under
            eviction_count = current_size - max_size + 1;
        } else if (current_size >= static_cast<size_t>(max_size * 0.9)) {
            // We're close to the limit - evict a few to make room
            eviction_count = std::max(size_t(1), static_cast<size_t>(current_size * 0.1));
        } else {
            return; // No eviction needed
        }
        
        // Limit eviction count to reasonable batch size
        eviction_count = std::min(eviction_count, config_.batch_eviction_size);
        eviction_count = std::min(eviction_count, current_size);
        
        // Simple eviction: just remove the first N entries
        // This is more reliable than complex sorting
        size_t evicted = 0;
        auto it = cache_.begin();
        while (it != cache_.end() && evicted < eviction_count) {
            it = cache_.erase(it);
            ++evicted;
        }
        
        metrics_.evictions.fetch_add(evicted, std::memory_order_relaxed);
        updateMemoryUsage();
    }
    
    void cleanExpiredEntries() {
        // Note: now variable is useful for TTL comparison in the loop below
        size_t cleaned = 0;
        
        for (auto it = cache_.begin(); it != cache_.end();) {
            if (it->second.isExpired(config_.ttl)) {
                it = cache_.erase(it);
                ++cleaned;
            } else {
                ++it;
            }
        }
        
        if (cleaned > 0) {
            metrics_.evictions.fetch_add(cleaned, std::memory_order_relaxed);
            updateMemoryUsage();
        }
    }
    
    void considerResizing() {
        double hit_rate = metrics_.hit_rate.load();
        size_t cache_size = cache_.size();
        
        if (hit_rate < config_.hit_rate_target && cache_size < config_.max_cache_size) {
            // Increase cache size if hit rate is low
            config_.max_cache_size = std::min(config_.max_cache_size * 1.2, 
                                            static_cast<double>(config_.max_cache_size * 2));
            metrics_.adaptive_resizes.fetch_add(1, std::memory_order_relaxed);
        } else if (hit_rate > 0.95 && cache_size > config_.min_cache_size * 2) {
            // Decrease cache size if hit rate is very high (over-provisioned)
            config_.max_cache_size = std::max(config_.max_cache_size * 0.9,
                                            static_cast<double>(config_.min_cache_size));
            metrics_.adaptive_resizes.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    void startBackgroundOptimization() {
        background_thread_ = std::make_unique<std::thread>([this]() {
            while (!shutdown_requested_.load()) {
                std::unique_lock<std::mutex> lock(cv_mutex_);
                optimization_cv_.wait_for(lock, std::chrono::seconds(10), 
                                        [this]() { return shutdown_requested_.load(); });
                
                if (shutdown_requested_.load()) break;
                
                // Perform optimization with cache lock
                {
                    std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
                    
                    // Periodic maintenance
                    cleanExpiredEntries();
                    updateMetrics();
                    
                    // Adaptive tuning based on performance
                    auto stats = getStats();
                    if (stats.hit_rate < 0.7) {
                        // Poor hit rate - try to optimize
                        if (config_.enable_adaptive_sizing) {
                            considerResizing();
                        }
                    }
                    
                    if (stats.memory_efficiency < config_.memory_efficiency_target) {
                        // Poor memory efficiency - more aggressive eviction
                        config_.eviction_threshold = std::max(0.6, config_.eviction_threshold * 0.9);
                        performEviction();
                    }
                }
            }
        });
    }
};

/**
 * @brief Factory function for creating optimized caches
 */
template<typename Key, typename Value>
std::unique_ptr<AdaptiveCache<Key, Value>> createOptimizedCache(
    size_t max_memory_mb = 1,
    std::chrono::milliseconds ttl = std::chrono::milliseconds(10000),
    bool enable_prefetching = true) {
    
    AdaptiveCacheConfig config;
    config.max_memory_bytes = max_memory_mb * 1024 * 1024;
    config.ttl = ttl;
    config.enable_prefetching = enable_prefetching;
    config.enable_adaptive_sizing = true;
    config.enable_background_optimization = true;
    
    return std::make_unique<AdaptiveCache<Key, Value>>(config);
}

} // namespace cache
} // namespace libstats

#endif // LIBSTATS_ADAPTIVE_CACHE_H_

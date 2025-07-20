#include "../include/adaptive_cache.h"
#include "../include/constants.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <deque>
#include <unordered_set>

// Conditional CPU detection support
#ifdef LIBSTATS_ENABLE_CPU_DETECTION
#include "../include/cpu_detection.h"
namespace cpu_impl = libstats::cpu;
#else
// Fallback CPU detection stubs
namespace cpu_impl {
    struct CacheInfo {
        uint32_t size = 0;
    };
    
    struct Features {
        bool neon = false;
        bool avx512f = false;
        bool avx2 = false;
        bool avx = false;
        bool sse2 = false;
        std::string vendor = "Unknown";
    };
    
    inline std::optional<CacheInfo> get_l2_cache() { return std::nullopt; }
    inline std::optional<CacheInfo> get_l3_cache() { return std::nullopt; }
    inline bool supports_avx2() { return false; }
    inline std::optional<uint64_t> get_tsc_frequency() { return std::nullopt; }
    inline const Features& get_features() {
        static Features features;
        return features;
    }
}
#endif

namespace libstats {
namespace cache {

/**
 * @brief Cache monitoring and diagnostic utilities
 */
class CacheMonitor {
private:
    std::vector<CacheMetrics> history_;
    mutable std::mutex history_mutex_;
    std::chrono::steady_clock::time_point start_time_;
    
public:
    CacheMonitor() : start_time_(std::chrono::steady_clock::now()) {}
    
    void recordMetrics(const CacheMetrics& metrics) {
        std::lock_guard<std::mutex> lock(history_mutex_);
        history_.push_back(metrics);
        
        // Keep only last 1000 entries
        if (history_.size() > 1000) {
            history_.erase(history_.begin(), history_.begin() + 500);
        }
    }
    
    struct PerformanceTrend {
        double hit_rate_trend;          // Positive = improving
        double memory_efficiency_trend;  // Positive = improving
        double access_time_trend;       // Negative = improving
        size_t sample_count;
        std::chrono::duration<double> observation_period;
    };
    
    PerformanceTrend analyzeTrends(std::chrono::seconds /* window */ = std::chrono::seconds(300)) const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        
        PerformanceTrend trend{};
        if (history_.size() < 2) return trend;
        
        auto now = std::chrono::steady_clock::now();
        // auto cutoff_time = now - window;  // Future use for time-based filtering
        
        std::vector<double> hit_rates;
        std::vector<double> memory_effs;
        std::vector<double> access_times;
        
        // Collect recent metrics
        for (const auto& metrics : history_) {
            // Simple linear approximation for time-based filtering
            hit_rates.push_back(metrics.hit_rate.load());
            
            double mem_eff = metrics.memory_usage.load() > 0 ? 
                static_cast<double>(metrics.hits.load()) / metrics.memory_usage.load() : 0.0;
            memory_effs.push_back(mem_eff);
            
            access_times.push_back(metrics.average_access_time.load());
        }
        
        trend.sample_count = hit_rates.size();
        trend.observation_period = std::chrono::duration_cast<std::chrono::duration<double>>(
            now - start_time_);
        
        if (hit_rates.size() > 1) {
            // Calculate simple linear trends
            trend.hit_rate_trend = calculateTrend(hit_rates);
            trend.memory_efficiency_trend = calculateTrend(memory_effs);
            trend.access_time_trend = calculateTrend(access_times);
        }
        
        return trend;
    }
    
    std::string generateReport(const CacheMetrics& current_metrics) const {
        std::ostringstream report;
        report << std::fixed << std::setprecision(2);
        
        report << "=== Cache Performance Report ===\n";
        report << "Current Metrics:\n";
        report << "  Hit Rate: " << current_metrics.hit_rate.load() * 100 << "%\n";
        report << "  Total Accesses: " << (current_metrics.hits.load() + current_metrics.misses.load()) << "\n";
        report << "  Memory Usage: " << formatBytes(current_metrics.memory_usage.load()) << "\n";
        report << "  Cache Size: " << current_metrics.cache_size.load() << " entries\n";
        report << "  Average Access Time: " << current_metrics.average_access_time.load() << " μs\n";
        report << "  Evictions: " << current_metrics.evictions.load() << "\n";
        
        if (current_metrics.prefetch_hits.load() + current_metrics.prefetch_misses.load() > 0) {
            report << "  Prefetch Effectiveness: " 
                   << current_metrics.getPrefetchEffectiveness() * 100 << "%\n";
        }
        
        auto trend = analyzeTrends();
        if (trend.sample_count > 1) {
            report << "\nPerformance Trends (last " << trend.observation_period.count() << "s):\n";
            report << "  Hit Rate: " << formatTrend(trend.hit_rate_trend) << "\n";
            report << "  Memory Efficiency: " << formatTrend(trend.memory_efficiency_trend) << "\n";
            report << "  Access Time: " << formatTrend(-trend.access_time_trend) << " (lower is better)\n";
        }
        
        return report.str();
    }
    
private:
    double calculateTrend(const std::vector<double>& values) const {
        if (values.size() < 2) return 0.0;
        
        // Simple linear regression slope
        double n = values.size();
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        
        for (size_t i = 0; i < values.size(); ++i) {
            double x = i;
            double y = values[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        
        double denominator = n * sum_x2 - sum_x * sum_x;
        if (std::abs(denominator) < 1e-10) return 0.0;
        
        return (n * sum_xy - sum_x * sum_y) / denominator;
    }
    
    std::string formatBytes(size_t bytes) const {
        const char* units[] = {"B", "KB", "MB", "GB"};
        int unit_index = 0;
        double size = bytes;
        
        while (size >= 1024.0 && unit_index < 3) {
            size /= 1024.0;
            ++unit_index;
        }
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1) << size << " " << units[unit_index];
        return oss.str();
    }
    
    std::string formatTrend(double trend) const {
        if (std::abs(trend) < 0.001) return "stable";
        else if (trend > 0) return "improving +" + std::to_string(trend * 100) + "%";
        else return "declining " + std::to_string(trend * 100) + "%";
    }
};

/**
 * @brief Memory pressure detector using CPU cache information
 */
class MemoryPressureDetector {
private:
    mutable std::mutex state_mutex_;
    mutable std::chrono::steady_clock::time_point last_check_;
    mutable double current_pressure_level_ = 0.0;
    
public:
    struct MemoryPressureInfo {
        double pressure_level;      // 0.0 to 1.0
        size_t available_cache_mb;  // Estimated available cache memory
        bool high_pressure;         // True if pressure > 0.8
        std::string recommendation;
    };
    
    MemoryPressureInfo detectPressure() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        
        auto now = std::chrono::steady_clock::now();
        if (now - last_check_ > std::chrono::seconds(10)) {
            updatePressureLevel();
            last_check_ = now;
        }
        
        MemoryPressureInfo info;
        info.pressure_level = current_pressure_level_;
        info.high_pressure = current_pressure_level_ > 0.8;
        
        // Use CPU cache information to estimate available cache memory
        // Provide a fallback without dependencies
        auto l3_cache = cpu_impl::get_l3_cache();
        auto l2_cache = cpu_impl::get_l2_cache();
        if (l3_cache) {
            // Conservatively estimate 10% of L3 cache available for application caching
            size_t cache_portion = l3_cache->size / (1024 * 1024 * 10);
            info.available_cache_mb = std::max(size_t(1), cache_portion);
        } else if (l2_cache) {
            // Use 5% of L2 cache if no L3
            size_t cache_portion = l2_cache->size / (1024 * 1024 * 20);
            info.available_cache_mb = std::max(size_t(1), cache_portion);
        } else {
            // Conservative fallback
            info.available_cache_mb = 1;
        }
        
        // Generate recommendation
        if (info.high_pressure) {
            info.recommendation = "Reduce cache size or increase eviction aggressiveness";
        } else if (info.pressure_level < 0.3) {
            info.recommendation = "Cache can be expanded for better performance";
        } else {
            info.recommendation = "Cache pressure is optimal";
        }
        
        return info;
    }
    
private:
    void updatePressureLevel() const {
        // Simple heuristic based on available system information
        // In a real implementation, this could query system memory, swap usage, etc.
        
        // For now, use CPU cache sizes as a proxy
        auto l3_cache = cpu_impl::get_l3_cache();
        auto l2_cache = cpu_impl::get_l2_cache();
        
        if (l3_cache && l3_cache->size > 8 * 1024 * 1024) {  // > 8MB L3
            current_pressure_level_ = 0.2;  // Low pressure on high-end CPUs
        } else if (l2_cache && l2_cache->size > 1024 * 1024) {  // > 1MB L2
            current_pressure_level_ = 0.5;  // Medium pressure
        } else {
            current_pressure_level_ = 0.8;  // High pressure on low-end systems
        }
        
        // Add some randomness to simulate dynamic conditions
        current_pressure_level_ += (std::rand() % 20 - 10) / 100.0;
        current_pressure_level_ = std::max(0.0, std::min(1.0, current_pressure_level_));
    }
};

/**
 * @brief Cache advisor for optimization recommendations
 */
class CacheAdvisor {
public:
    struct OptimizationRecommendation {
        enum class Action {
            INCREASE_SIZE,
            DECREASE_SIZE,
            ADJUST_TTL,
            ENABLE_PREFETCHING,
            DISABLE_PREFETCHING,
            CHANGE_EVICTION_POLICY,
            NO_ACTION
        };
        
        Action action;
        std::string description;
        double expected_improvement;  // Expected performance improvement (0-1)
        int priority;                 // 1-10, higher is more important
    };
    
    std::vector<OptimizationRecommendation> analyzeAndRecommend(
        const CacheMetrics& metrics,
        const AdaptiveCacheConfig& config,
        const MemoryPressureDetector::MemoryPressureInfo& memory_info) const {
        
        std::vector<OptimizationRecommendation> recommendations;
        
        double hit_rate = metrics.hit_rate.load();
        double memory_usage_mb = metrics.memory_usage.load() / (1024.0 * 1024.0);
        double prefetch_effectiveness = metrics.getPrefetchEffectiveness();
        
        // Analyze hit rate
        if (hit_rate < 0.7 && !memory_info.high_pressure) {
            OptimizationRecommendation rec;
            rec.action = OptimizationRecommendation::Action::INCREASE_SIZE;
            rec.description = "Low hit rate detected. Increasing cache size could improve performance.";
            rec.expected_improvement = 0.3;
            rec.priority = 8;
            recommendations.push_back(rec);
        } else if (hit_rate > 0.95 && memory_usage_mb > memory_info.available_cache_mb) {
            OptimizationRecommendation rec;
            rec.action = OptimizationRecommendation::Action::DECREASE_SIZE;
            rec.description = "Very high hit rate with high memory usage. Cache can be reduced.";
            rec.expected_improvement = 0.1;
            rec.priority = 4;
            recommendations.push_back(rec);
        }
        
        // Analyze prefetching effectiveness
        if (config.enable_prefetching && prefetch_effectiveness < 0.3) {
            OptimizationRecommendation rec;
            rec.action = OptimizationRecommendation::Action::DISABLE_PREFETCHING;
            rec.description = "Prefetching is ineffective and may be wasting resources.";
            rec.expected_improvement = 0.15;
            rec.priority = 6;
            recommendations.push_back(rec);
        } else if (!config.enable_prefetching && hit_rate < 0.8) {
            OptimizationRecommendation rec;
            rec.action = OptimizationRecommendation::Action::ENABLE_PREFETCHING;
            rec.description = "Enabling prefetching might improve hit rate.";
            rec.expected_improvement = 0.2;
            rec.priority = 5;
            recommendations.push_back(rec);
        }
        
        // Analyze TTL
        size_t evictions = metrics.evictions.load();
        size_t total_accesses = metrics.hits.load() + metrics.misses.load();
        double eviction_rate = total_accesses > 0 ? 
            static_cast<double>(evictions) / total_accesses : 0.0;
        
        if (eviction_rate > 0.1) {  // High eviction rate
            OptimizationRecommendation rec;
            rec.action = OptimizationRecommendation::Action::ADJUST_TTL;
            rec.description = "High eviction rate suggests TTL might be too short.";
            rec.expected_improvement = 0.25;
            rec.priority = 7;
            recommendations.push_back(rec);
        }
        
        // Memory pressure response
        if (memory_info.high_pressure && memory_usage_mb > memory_info.available_cache_mb * 0.8) {
            OptimizationRecommendation rec;
            rec.action = OptimizationRecommendation::Action::CHANGE_EVICTION_POLICY;
            rec.description = "High memory pressure. Consider more aggressive eviction policy.";
            rec.expected_improvement = 0.2;
            rec.priority = 9;
            recommendations.push_back(rec);
        }
        
        // Sort by priority (highest first)
        std::sort(recommendations.begin(), recommendations.end(),
                 [](const auto& a, const auto& b) { return a.priority > b.priority; });
        
        return recommendations;
    }
};

/**
 * @brief Global cache management utilities
 */
namespace utils {

/**
 * @brief Detect platform architecture for cache optimization
 */
enum class PlatformArchitecture {
    APPLE_SILICON,
    INTEL,
    AMD,
    ARM_GENERIC,
    UNKNOWN
};

/**
 * @brief Detect the current platform architecture
 */
PlatformArchitecture detectPlatformArchitecture() {
#ifdef LIBSTATS_ENABLE_CPU_DETECTION
    const auto& features = cpu_impl::get_features();
    
    // Check for Apple Silicon (M1/M2/M3)
    if (features.neon && features.vendor.find("Apple") != std::string::npos) {
        return PlatformArchitecture::APPLE_SILICON;
    }
    
    // Check for Intel
    if (features.vendor.find("Intel") != std::string::npos ||
        features.vendor.find("GenuineIntel") != std::string::npos) {
        return PlatformArchitecture::INTEL;
    }
    
    // Check for AMD
    if (features.vendor.find("AMD") != std::string::npos ||
        features.vendor.find("AuthenticAMD") != std::string::npos) {
        return PlatformArchitecture::AMD;
    }
    
    // Check for ARM (other than Apple)
    if (features.neon) {
        return PlatformArchitecture::ARM_GENERIC;
    }
#endif
    
    return PlatformArchitecture::UNKNOWN;
}

/**
 * @brief Create cache configuration optimized for current platform
 */
AdaptiveCacheConfig createOptimalConfig() {
    AdaptiveCacheConfig config;
    PlatformArchitecture arch = detectPlatformArchitecture();
    
    // Apply platform-specific base configuration
    switch (arch) {
        case PlatformArchitecture::APPLE_SILICON:
            config.max_memory_bytes = constants::cache::platform::apple_silicon::DEFAULT_MAX_MEMORY_MB * 1024 * 1024;
            config.max_cache_size = constants::cache::platform::apple_silicon::DEFAULT_MAX_ENTRIES;
            config.prefetch_queue_size = constants::cache::platform::apple_silicon::PREFETCH_QUEUE_SIZE;
            config.eviction_threshold = constants::cache::platform::apple_silicon::EVICTION_THRESHOLD;
            config.batch_eviction_size = constants::cache::platform::apple_silicon::BATCH_EVICTION_SIZE;
            config.ttl = constants::cache::platform::apple_silicon::DEFAULT_TTL;
            config.hit_rate_target = constants::cache::platform::apple_silicon::HIT_RATE_TARGET;
            config.memory_efficiency_target = constants::cache::platform::apple_silicon::MEMORY_EFFICIENCY_TARGET;
            break;
            
        case PlatformArchitecture::INTEL:
            config.max_memory_bytes = constants::cache::platform::intel::DEFAULT_MAX_MEMORY_MB * 1024 * 1024;
            config.max_cache_size = constants::cache::platform::intel::DEFAULT_MAX_ENTRIES;
            config.prefetch_queue_size = constants::cache::platform::intel::PREFETCH_QUEUE_SIZE;
            config.eviction_threshold = constants::cache::platform::intel::EVICTION_THRESHOLD;
            config.batch_eviction_size = constants::cache::platform::intel::BATCH_EVICTION_SIZE;
            config.ttl = constants::cache::platform::intel::DEFAULT_TTL;
            config.hit_rate_target = constants::cache::platform::intel::HIT_RATE_TARGET;
            config.memory_efficiency_target = constants::cache::platform::intel::MEMORY_EFFICIENCY_TARGET;
            break;
            
        case PlatformArchitecture::AMD:
            config.max_memory_bytes = constants::cache::platform::amd::DEFAULT_MAX_MEMORY_MB * 1024 * 1024;
            config.max_cache_size = constants::cache::platform::amd::DEFAULT_MAX_ENTRIES;
            config.prefetch_queue_size = constants::cache::platform::amd::PREFETCH_QUEUE_SIZE;
            config.eviction_threshold = constants::cache::platform::amd::EVICTION_THRESHOLD;
            config.batch_eviction_size = constants::cache::platform::amd::BATCH_EVICTION_SIZE;
            config.ttl = constants::cache::platform::amd::DEFAULT_TTL;
            config.hit_rate_target = constants::cache::platform::amd::HIT_RATE_TARGET;
            config.memory_efficiency_target = constants::cache::platform::amd::MEMORY_EFFICIENCY_TARGET;
            break;
            
        case PlatformArchitecture::ARM_GENERIC:
            config.max_memory_bytes = constants::cache::platform::arm::DEFAULT_MAX_MEMORY_MB * 1024 * 1024;
            config.max_cache_size = constants::cache::platform::arm::DEFAULT_MAX_ENTRIES;
            config.prefetch_queue_size = constants::cache::platform::arm::PREFETCH_QUEUE_SIZE;
            config.eviction_threshold = constants::cache::platform::arm::EVICTION_THRESHOLD;
            config.batch_eviction_size = constants::cache::platform::arm::BATCH_EVICTION_SIZE;
            config.ttl = constants::cache::platform::arm::DEFAULT_TTL;
            config.hit_rate_target = constants::cache::platform::arm::HIT_RATE_TARGET;
            config.memory_efficiency_target = constants::cache::platform::arm::MEMORY_EFFICIENCY_TARGET;
            break;
            
        default: // UNKNOWN
            // Conservative defaults
            config.max_memory_bytes = 2 * 1024 * 1024;  // 2MB
            config.max_cache_size = 512;
            config.prefetch_queue_size = 16;
            config.eviction_threshold = 0.80;
            config.batch_eviction_size = 8;
            config.ttl = std::chrono::milliseconds(8000);
            config.hit_rate_target = 0.80;
            config.memory_efficiency_target = 0.65;
    }
    
    // Fine-tune based on actual CPU cache hierarchy
#ifdef LIBSTATS_ENABLE_CPU_DETECTION
    if (auto l3_cache = cpu_impl::get_l3_cache()) {
        // Adjust memory limit based on L3 cache size
        size_t l3_based_limit = static_cast<size_t>(l3_cache->size * constants::cache::sizing::L3_CACHE_FRACTION);
        l3_based_limit = std::clamp(l3_based_limit, 
                                   constants::cache::sizing::MIN_CACHE_SIZE_BYTES,
                                   constants::cache::sizing::MAX_CACHE_SIZE_BYTES);
        config.max_memory_bytes = std::min(config.max_memory_bytes, l3_based_limit);
    } else if (auto l2_cache = cpu_impl::get_l2_cache()) {
        // Adjust memory limit based on L2 cache size
        size_t l2_based_limit = static_cast<size_t>(l2_cache->size * constants::cache::sizing::L2_CACHE_FRACTION);
        l2_based_limit = std::clamp(l2_based_limit,
                                   constants::cache::sizing::MIN_CACHE_SIZE_BYTES,
                                   constants::cache::sizing::MAX_CACHE_SIZE_BYTES);
        config.max_memory_bytes = std::min(config.max_memory_bytes, l2_based_limit);
    }
    
    // Adjust entry count based on memory limit
    config.max_cache_size = std::clamp(
        config.max_memory_bytes / constants::cache::sizing::BYTES_PER_ENTRY_ESTIMATE,
        constants::cache::sizing::MIN_ENTRY_COUNT,
        constants::cache::sizing::MAX_ENTRY_COUNT
    );
    
    // Fine-tune TTL based on CPU frequency
    if (auto cpu_freq = cpu_impl::get_tsc_frequency()) {
        if (*cpu_freq >= constants::cache::tuning::ULTRA_HIGH_FREQ_THRESHOLD_HZ) {
            config.ttl = constants::cache::tuning::ULTRA_HIGH_FREQ_TTL;
        } else if (*cpu_freq >= constants::cache::tuning::HIGH_FREQ_THRESHOLD_HZ) {
            config.ttl = constants::cache::tuning::HIGH_FREQ_TTL;
        } else {
            config.ttl = constants::cache::tuning::BASE_TTL;
        }
    }
    
    // Adjust prefetch behavior based on SIMD capabilities
    const auto& features = cpu_impl::get_features();
    if (features.avx512f) {
        config.prefetch_queue_size *= constants::cache::tuning::AVX512_PREFETCH_MULTIPLIER;
        config.enable_prefetching = true;
        config.enable_background_optimization = true;
    } else if (features.avx2 || features.avx) {
        config.prefetch_queue_size *= constants::cache::tuning::AVX2_PREFETCH_MULTIPLIER;
        config.enable_prefetching = true;
        config.enable_background_optimization = true;
    } else if (features.sse2 || features.neon) {
        config.prefetch_queue_size *= constants::cache::tuning::SSE_PREFETCH_MULTIPLIER;
        config.enable_prefetching = arch != PlatformArchitecture::ARM_GENERIC; // Conservative on generic ARM
        config.enable_background_optimization = true;
    } else {
        config.enable_prefetching = false;
        config.enable_background_optimization = false;
    }
#endif
    
    // Ensure configuration bounds
    config.min_cache_size = std::min(config.min_cache_size, config.max_cache_size / 4);
    config.prefetch_queue_size = std::min(config.prefetch_queue_size, config.max_cache_size / 4);
    
    return config;
}

/**
 * @brief Access pattern analyzer for cache optimization
 */
class AccessPatternAnalyzer {
public:
    enum class PatternType {
        SEQUENTIAL,
        RANDOM,
        MIXED,
        UNKNOWN
    };
    
    struct PatternInfo {
        PatternType type;
        double sequential_ratio;  // 0.0 = completely random, 1.0 = completely sequential
        double locality_score;    // 0.0 = no locality, 1.0 = perfect locality
        size_t unique_keys_accessed;
        std::string description;
    };
    
private:
    std::deque<uint64_t> access_history_;
    std::unordered_set<uint64_t> unique_accesses_;
    mutable std::mutex pattern_mutex_;
    
public:
    template<typename Key>
    void recordAccess(const Key& key) {
        std::lock_guard<std::mutex> lock(pattern_mutex_);
        
        // Simple hash for pattern analysis
        uint64_t hash_key = std::hash<Key>{}(key);
        
        access_history_.push_back(hash_key);
        unique_accesses_.insert(hash_key);
        
        // Keep history bounded
        if (access_history_.size() > constants::cache::patterns::MAX_PATTERN_HISTORY) {
            auto old_key = access_history_.front();
            access_history_.pop_front();
            
            // Remove from unique set if no longer in history
            if (std::find(access_history_.begin(), access_history_.end(), old_key) == access_history_.end()) {
                unique_accesses_.erase(old_key);
            }
        }
    }
    
    PatternInfo analyzePattern() const {
        std::lock_guard<std::mutex> lock(pattern_mutex_);
        
        PatternInfo info;
        info.unique_keys_accessed = unique_accesses_.size();
        
        if (access_history_.size() < 10) {
            info.type = PatternType::UNKNOWN;
            info.sequential_ratio = 0.0;
            info.locality_score = 0.0;
            info.description = "Insufficient data for pattern analysis";
            return info;
        }
        
        // Calculate sequential ratio
        size_t sequential_pairs = 0;
        size_t total_pairs = access_history_.size() - 1;
        
        for (size_t i = 1; i < access_history_.size(); ++i) {
            if (std::abs(static_cast<int64_t>(access_history_[i]) - static_cast<int64_t>(access_history_[i-1])) <= 1) {
                ++sequential_pairs;
            }
        }
        
        info.sequential_ratio = static_cast<double>(sequential_pairs) / total_pairs;
        
        // Calculate locality score (how often recent keys are re-accessed)
        size_t recent_reaccess = 0;
        size_t window_size = std::min(access_history_.size(), size_t(32));
        std::unordered_set<uint64_t> recent_keys(access_history_.end() - window_size, access_history_.end());
        
        for (size_t i = access_history_.size() - window_size; i > 0 && i > access_history_.size() - 2 * window_size; --i) {
            if (recent_keys.count(access_history_[i-1]) > 0) {
                ++recent_reaccess;
            }
        }
        
        info.locality_score = window_size > 0 ? static_cast<double>(recent_reaccess) / window_size : 0.0;
        
        // Determine pattern type
        if (info.sequential_ratio >= constants::cache::patterns::SEQUENTIAL_PATTERN_THRESHOLD) {
            info.type = PatternType::SEQUENTIAL;
            info.description = "Sequential access pattern detected";
        } else if (info.sequential_ratio <= constants::cache::patterns::RANDOM_PATTERN_THRESHOLD) {
            info.type = PatternType::RANDOM;
            info.description = "Random access pattern detected";
        } else {
            info.type = PatternType::MIXED;
            info.description = "Mixed access pattern detected";
        }
        
        return info;
    }
};

/**
 * @brief Create cache configuration with access pattern awareness
 */
AdaptiveCacheConfig createPatternAwareConfig(const AccessPatternAnalyzer::PatternInfo& pattern_info = {}) {
    // Start with platform-optimized configuration
    AdaptiveCacheConfig config = createOptimalConfig();
    
    // Adjust based on access pattern
    switch (pattern_info.type) {
        case AccessPatternAnalyzer::PatternType::SEQUENTIAL:
            // Sequential patterns benefit from larger caches and longer TTL
            config.max_cache_size = static_cast<size_t>(config.max_cache_size * constants::cache::patterns::SEQUENTIAL_SIZE_MULTIPLIER);
            config.max_memory_bytes = static_cast<size_t>(config.max_memory_bytes * constants::cache::patterns::SEQUENTIAL_SIZE_MULTIPLIER);
            config.ttl = std::chrono::milliseconds(static_cast<long>(config.ttl.count() * 1.5));
            config.enable_prefetching = true;
            config.prefetch_queue_size *= 2;  // More aggressive prefetching
            break;
            
        case AccessPatternAnalyzer::PatternType::RANDOM:
            // Random patterns benefit from smaller, faster caches
            config.max_cache_size = static_cast<size_t>(config.max_cache_size * constants::cache::patterns::RANDOM_SIZE_MULTIPLIER);
            config.max_memory_bytes = static_cast<size_t>(config.max_memory_bytes * constants::cache::patterns::RANDOM_SIZE_MULTIPLIER);
            config.eviction_threshold *= 0.9;  // More aggressive eviction
            config.enable_prefetching = false; // Prefetching less effective for random access
            break;
            
        case AccessPatternAnalyzer::PatternType::MIXED:
            // Mixed patterns use default configuration with some tuning
            config.max_cache_size = static_cast<size_t>(config.max_cache_size * constants::cache::patterns::MIXED_SIZE_MULTIPLIER);
            config.enable_prefetching = pattern_info.locality_score > 0.5; // Enable if good locality
            break;
            
        case AccessPatternAnalyzer::PatternType::UNKNOWN:
        default:
            // Use conservative defaults
            break;
    }
    
    // Adjust based on locality score
    if (pattern_info.locality_score > 0.8) {
        // High locality - extend TTL and enable background optimization
        config.ttl = std::chrono::milliseconds(static_cast<long>(config.ttl.count() * 1.3));
        config.enable_background_optimization = true;
    } else if (pattern_info.locality_score < 0.3) {
        // Low locality - shorter TTL and more aggressive eviction
        config.ttl = std::chrono::milliseconds(static_cast<long>(config.ttl.count() * 0.7));
        config.eviction_threshold *= 0.85;
    }
    
    // Ensure bounds are still respected
    config.max_cache_size = std::clamp(config.max_cache_size,
                                      constants::cache::sizing::MIN_ENTRY_COUNT,
                                      constants::cache::sizing::MAX_ENTRY_COUNT);
    config.max_memory_bytes = std::clamp(config.max_memory_bytes,
                                        constants::cache::sizing::MIN_CACHE_SIZE_BYTES,
                                        constants::cache::sizing::MAX_CACHE_SIZE_BYTES);
    config.prefetch_queue_size = std::min(config.prefetch_queue_size, config.max_cache_size / 4);
    
    return config;
}

/**
 * @brief Performance benchmarking for cache configurations
 */
template<typename Key, typename Value>
struct BenchmarkResult {
    double hit_rate;
    double average_access_time_us;
    double memory_efficiency;
    size_t operations_per_second;
    std::string config_description;
};

template<typename Key, typename Value>
BenchmarkResult<Key, Value> benchmarkCache(
    AdaptiveCache<Key, Value>& cache,
    const std::vector<std::pair<Key, Value>>& test_data,
    size_t num_operations = 10000) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Populate cache
    for (const auto& [key, value] : test_data) {
        cache.put(key, value);
    }
    
    // Perform random access test
    size_t hits = 0;
    for (size_t i = 0; i < num_operations; ++i) {
        const auto& [key, expected_value] = test_data[i % test_data.size()];
        if (cache.get(key)) {
            ++hits;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    auto stats = cache.getStats();
    
    BenchmarkResult<Key, Value> result;
    result.hit_rate = static_cast<double>(hits) / num_operations;
    result.average_access_time_us = stats.average_access_time;
    result.memory_efficiency = stats.memory_efficiency;
    result.operations_per_second = static_cast<size_t>(num_operations * 1000000.0 / duration.count());
    result.config_description = "Adaptive Cache Benchmark";
    
    return result;
}

} // namespace utils

} // namespace cache
} // namespace libstats

#include "../include/platform/adaptive_cache.h"
#include "../include/core/constants.h"
#include "../include/platform/platform_constants.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <deque>
#include <unordered_set>
#include <cstdlib>
#include <chrono>

// Conditional CPU detection support
#ifdef LIBSTATS_ENABLE_CPU_DETECTION
#include "../include/platform/cpu_detection.h"
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

// Implementation of CacheMonitor class
CacheMonitor::CacheMonitor() : start_time_(std::chrono::steady_clock::now()) {}

void CacheMonitor::recordMetrics(const CacheMetrics& metrics) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    history_.push_back(metrics);
    
    // Keep only last entries
    if (history_.size() > constants::cache::patterns::MAX_PATTERN_HISTORY) {
        history_.erase(history_.begin(), history_.begin() + static_cast<size_t>(constants::cache::patterns::MAX_PATTERN_HISTORY * constants::math::HALF));
    }
}

CacheMonitor::PerformanceTrend CacheMonitor::analyzeTrends(std::chrono::seconds /* window */) const {
    std::lock_guard<std::mutex> lock(history_mutex_);
    
    PerformanceTrend trend{};
    if (history_.size() < constants::math::TWO_INT) return trend;
    
    auto now = std::chrono::steady_clock::now();
    // auto cutoff_time = now - window;  // Future use for time-based filtering
    
    std::vector<double> hit_rates;
    std::vector<double> memory_effs;
    std::vector<double> access_times;
    
    // Collect recent metrics
    for (const auto& metrics : history_) {
        // Simple linear approximation for time-based filtering
        hit_rates.push_back(metrics.hit_rate.load());
        
        double mem_eff = metrics.memory_usage.load() > constants::math::ZERO_INT ? 
            static_cast<double>(metrics.hits.load()) / static_cast<double>(metrics.memory_usage.load()) : constants::math::ZERO_DOUBLE;
        memory_effs.push_back(mem_eff);
        
        access_times.push_back(metrics.average_access_time.load());
    }
    
    trend.sample_count = hit_rates.size();
    trend.observation_period = std::chrono::duration_cast<std::chrono::duration<double>>(
        now - start_time_);
    
    if (hit_rates.size() > constants::math::ONE_INT) {
        // Calculate simple linear trends
        trend.hit_rate_trend = calculateTrend(hit_rates);
        trend.memory_efficiency_trend = calculateTrend(memory_effs);
        trend.access_time_trend = calculateTrend(access_times);
    }
    
    return trend;
}

std::string CacheMonitor::generateReport(const CacheMetrics& current_metrics) const {
    std::ostringstream report;
    report << std::fixed << std::setprecision(2);
    
    report << "=== Cache Performance Report ===\n";
    report << "Current Metrics:\n";
    report << "  Hit Rate: " << current_metrics.hit_rate.load() * constants::math::HUNDRED << "%\n";
    report << "  Total Accesses: " << (current_metrics.hits.load() + current_metrics.misses.load()) << "\n";
    report << "  Memory Usage: " << formatBytes(current_metrics.memory_usage.load()) << "\n";
    report << "  Cache Size: " << current_metrics.cache_size.load() << " entries\n";
    report << "  Average Access Time: " << current_metrics.average_access_time.load() << " μs\n";
    report << "  Evictions: " << current_metrics.evictions.load() << "\n";
    
    if (current_metrics.prefetch_hits.load() + current_metrics.prefetch_misses.load() > constants::math::ZERO_INT) {
        report << "  Prefetch Effectiveness: " 
               << current_metrics.getPrefetchEffectiveness() * constants::math::HUNDRED << "%\n";
    }
    
    auto trend = analyzeTrends();
    if (trend.sample_count > constants::math::ONE_INT) {
        report << "\nPerformance Trends (last " << trend.observation_period.count() << "s):\n";
        report << "  Hit Rate: " << formatTrend(trend.hit_rate_trend) << "\n";
        report << "  Memory Efficiency: " << formatTrend(trend.memory_efficiency_trend) << "\n";
        report << "  Access Time: " << formatTrend(-trend.access_time_trend) << " (lower is better)\n";
    }
    
    return report.str();
}

double CacheMonitor::calculateTrend(const std::vector<double>& values) const {
    if (values.size() < constants::math::TWO_INT) return constants::math::ZERO_DOUBLE;
    
    // Simple linear regression slope
    double n = static_cast<double>(values.size());
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
    
    for (size_t i = 0; i < values.size(); ++i) {
        double x = static_cast<double>(i);
        double y = values[i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }
    
    double denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE) return constants::math::ZERO_DOUBLE;
    
    return (n * sum_xy - sum_x * sum_y) / denominator;
}

std::string CacheMonitor::formatBytes(size_t bytes) const {
    const char* units[] = {"B", "KB", "MB", "GB"};
    int unit_index = 0;
    double size = static_cast<double>(bytes);
    
    const double KILOBYTE = 1024.0;
    while (size >= KILOBYTE && unit_index < constants::math::THREE_INT) {
        size /= KILOBYTE;
        ++unit_index;
    }
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << size << " " << units[unit_index];
    return oss.str();
}

std::string CacheMonitor::formatTrend(double trend) const {
    if (std::abs(trend) < constants::math::THOUSANDTH) return "stable";
    else if (trend > constants::math::ZERO_DOUBLE) return "improving +" + std::to_string(trend * constants::math::HUNDRED) + "%";
    else return "declining " + std::to_string(trend * constants::math::HUNDRED) + "%";
}

// Implementation of MemoryPressureDetector class
MemoryPressureDetector::MemoryPressureDetector() 
    : last_check_(std::chrono::steady_clock::now() - std::chrono::seconds(11)) {}

MemoryPressureDetector::MemoryPressureInfo MemoryPressureDetector::detectPressure() const {
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

void MemoryPressureDetector::updatePressureLevel() const {
    // Simple heuristic based on available system information
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

// Implementation of CacheAdvisor class
std::vector<CacheAdvisor::OptimizationRecommendation> CacheAdvisor::analyzeAndRecommend(
    const CacheMetrics& metrics,
    const AdaptiveCacheConfig& config,
    const MemoryPressureDetector::MemoryPressureInfo& memory_info) const {
    
    std::vector<OptimizationRecommendation> recommendations;
    
    double hit_rate = metrics.hit_rate.load();
    double memory_usage_mb = static_cast<double>(metrics.memory_usage.load()) / (1024.0 * 1024.0);
    double prefetch_effectiveness = metrics.getPrefetchEffectiveness();
    
    // Analyze hit rate
    if (hit_rate < 0.7 && !memory_info.high_pressure) {
        OptimizationRecommendation rec;
        rec.action = OptimizationRecommendation::Action::INCREASE_SIZE;
        rec.description = "Low hit rate detected. Increasing cache size could improve performance.";
        rec.expected_improvement = 0.3;
        rec.priority = 8;
        recommendations.push_back(rec);
    } else if (hit_rate > 0.95 && memory_usage_mb > static_cast<double>(memory_info.available_cache_mb)) {
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
        static_cast<double>(evictions) / static_cast<double>(total_accesses) : 0.0;
    
    if (eviction_rate > 0.1) {  // High eviction rate
        OptimizationRecommendation rec;
        rec.action = OptimizationRecommendation::Action::ADJUST_TTL;
        rec.description = "High eviction rate suggests TTL might be too short.";
        rec.expected_improvement = 0.25;
        rec.priority = 7;
        recommendations.push_back(rec);
    }
    
    // Memory pressure response
    if (memory_info.high_pressure && memory_usage_mb > static_cast<double>(memory_info.available_cache_mb) * 0.8) {
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

// Implementation of utils namespace functions
namespace utils {

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
            config.max_memory_bytes = static_cast<size_t>(2 * 1024 * 1024);  // 2MB
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
#endif
    
    // Ensure configuration bounds
    config.min_cache_size = std::min(config.min_cache_size, config.max_cache_size / 4);
    config.prefetch_queue_size = std::min(config.prefetch_queue_size, config.max_cache_size / 4);
    
    return config;
}

AdaptiveCacheConfig createPatternAwareConfig(const AccessPatternAnalyzer::PatternInfo& pattern_info) {
    // Start with platform-optimized configuration
    AdaptiveCacheConfig config = createOptimalConfig();
    
    // Adjust based on access pattern
    switch (pattern_info.type) {
        case AccessPatternAnalyzer::PatternType::SEQUENTIAL:
            // Sequential patterns benefit from larger caches and longer TTL
            config.max_cache_size = static_cast<size_t>(static_cast<double>(config.max_cache_size) * constants::cache::patterns::SEQUENTIAL_SIZE_MULTIPLIER);
            config.max_memory_bytes = static_cast<size_t>(static_cast<double>(config.max_memory_bytes) * constants::cache::patterns::SEQUENTIAL_SIZE_MULTIPLIER);
            config.ttl = std::chrono::milliseconds(static_cast<long>(static_cast<double>(config.ttl.count()) * 1.5));
            config.enable_prefetching = true;
            config.prefetch_queue_size *= 2;  // More aggressive prefetching
            break;
            
        case AccessPatternAnalyzer::PatternType::RANDOM:
            // Random patterns benefit from smaller, faster caches
            config.max_cache_size = static_cast<size_t>(static_cast<double>(config.max_cache_size) * constants::cache::patterns::RANDOM_SIZE_MULTIPLIER);
            config.max_memory_bytes = static_cast<size_t>(static_cast<double>(config.max_memory_bytes) * constants::cache::patterns::RANDOM_SIZE_MULTIPLIER);
            config.eviction_threshold *= 0.9;  // More aggressive eviction
            config.enable_prefetching = false; // Prefetching less effective for random access
            break;
            
        case AccessPatternAnalyzer::PatternType::MIXED:
            // Mixed patterns use default configuration with some tuning
            config.max_cache_size = static_cast<size_t>(static_cast<double>(config.max_cache_size) * constants::cache::patterns::MIXED_SIZE_MULTIPLIER);
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
        config.ttl = std::chrono::milliseconds(static_cast<long>(static_cast<double>(config.ttl.count()) * 1.3));
        config.enable_background_optimization = true;
    } else if (pattern_info.locality_score < 0.3) {
        // Low locality - shorter TTL and more aggressive eviction
        config.ttl = std::chrono::milliseconds(static_cast<long>(static_cast<double>(config.ttl.count()) * 0.7));
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

// Implementation of AccessPatternAnalyzer::analyzePattern
AccessPatternAnalyzer::PatternInfo AccessPatternAnalyzer::analyzePattern() const {
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
    
    info.sequential_ratio = static_cast<double>(sequential_pairs) / static_cast<double>(total_pairs);
    
    // Calculate locality score (how often recent keys are re-accessed)
    size_t recent_reaccess = 0;
    size_t window_size = std::min(access_history_.size(), size_t(32));
    std::unordered_set<uint64_t> recent_keys(access_history_.end() - window_size, access_history_.end());
    
    for (size_t i = access_history_.size() - window_size; i > 0 && i > access_history_.size() - 2 * window_size; --i) {
        if (recent_keys.count(access_history_[i-1]) > 0) {
            ++recent_reaccess;
        }
    }
    
    info.locality_score = window_size > 0 ? static_cast<double>(recent_reaccess) / static_cast<double>(window_size) : 0.0;
    
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

} // namespace utils

} // namespace cache
} // namespace libstats

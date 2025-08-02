#include "../include/core/performance_history.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace libstats {
namespace performance {

void PerformanceHistory::recordPerformance(
    Strategy strategy,
    DistributionType distribution_type,
    std::size_t batch_size,
    std::uint64_t execution_time_ns
) noexcept {
    std::string key = generateKey(strategy, distribution_type, categorizeBatchSize(batch_size));
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    auto& stats = performance_data_[key];
    lock.unlock();
    stats.total_time_ns.fetch_add(execution_time_ns, std::memory_order_release);
    stats.execution_count.fetch_add(1, std::memory_order_release);
    stats.min_time_ns.store(std::min(stats.min_time_ns.load(std::memory_order_acquire), execution_time_ns), std::memory_order_release);
    stats.max_time_ns.store(std::max(stats.max_time_ns.load(std::memory_order_acquire), execution_time_ns), std::memory_order_release);
    total_executions_.fetch_add(1, std::memory_order_release);
}

std::optional<PerformanceSnapshot> PerformanceHistory::getPerformanceStats(
    Strategy strategy,
    DistributionType distribution_type
) const noexcept {
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    
    // Aggregate performance data across all batch size categories
    PerformanceSnapshot aggregated;
    bool found_data = false;
    
    for (const auto& [key, stats] : performance_data_) {
        // Check if this key matches our strategy and distribution
        std::string expected_prefix = std::string(strategyToString(strategy)) + "_" + distributionTypeToString(distribution_type) + "_";
        if (key.starts_with(expected_prefix)) {
            found_data = true;
            aggregated.total_time_ns += stats.total_time_ns.load(std::memory_order_acquire);
            aggregated.execution_count += stats.execution_count.load(std::memory_order_acquire);
            
            auto min_time = stats.min_time_ns.load(std::memory_order_acquire);
            auto max_time = stats.max_time_ns.load(std::memory_order_acquire);
            
            if (min_time < aggregated.min_time_ns) {
                aggregated.min_time_ns = min_time;
            }
            if (max_time > aggregated.max_time_ns) {
                aggregated.max_time_ns = max_time;
            }
        }
    }
    
    return found_data ? std::make_optional(aggregated) : std::nullopt;
}

PerformanceHistory::StrategyRecommendation PerformanceHistory::getBestStrategy(
    DistributionType distribution_type,
    std::size_t batch_size
) const noexcept {
    const std::size_t batch_category = categorizeBatchSize(batch_size);
    
    // Collect performance data for all strategies for this distribution and batch size category
    std::vector<std::pair<Strategy, std::uint64_t>> strategy_performance;
    
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    
    // Check all available strategies
    for (auto strategy : {Strategy::SCALAR, Strategy::SIMD_BATCH, Strategy::PARALLEL_SIMD, 
                         Strategy::WORK_STEALING, Strategy::CACHE_AWARE}) {
        std::string key = generateKey(strategy, distribution_type, batch_category);
        auto it = performance_data_.find(key);
        
        if (it != performance_data_.end() && it->second.hasReliableData()) {
            std::uint64_t avg_time = it->second.getAverageTimeNs();
            strategy_performance.emplace_back(strategy, avg_time);
        }
    }
    
    lock.unlock();
    
    // If we have no historical data, return a conservative default
    if (strategy_performance.empty()) {
        return {Strategy::SCALAR, 0.0, 0, false};
    }
    
    // Find the strategy with the best (lowest) average time
    auto best_it = std::min_element(strategy_performance.begin(), strategy_performance.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    // Calculate confidence score based on the number of strategies we have data for
    // and the performance difference between best and worst
    double confidence_score = 0.0;
    if (strategy_performance.size() >= 2) {
        auto worst_it = std::max_element(strategy_performance.begin(), strategy_performance.end(),
            [](const auto& a, const auto& b) {
                return a.second < b.second;
            });
        
        // Confidence increases with performance difference and number of data points
        double performance_ratio = static_cast<double>(worst_it->second) / best_it->second;
        double data_confidence = std::min(1.0, strategy_performance.size() / 5.0);
        confidence_score = std::min(1.0, (performance_ratio - 1.0) * data_confidence);
    } else {
        confidence_score = 0.5; // Medium confidence with only one data point
    }
    
    return {
        best_it->first,
        confidence_score,
        best_it->second,
        true
    };
}

std::optional<std::pair<std::size_t, std::size_t>> PerformanceHistory::learnOptimalThresholds(
    DistributionType distribution_type
) const noexcept {
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    
    // Collect performance data for different strategies across batch sizes
    std::map<std::size_t, std::map<Strategy, std::uint64_t>> batch_performance;
    
    for (const auto& [key, stats] : performance_data_) {
        std::string dist_str = distributionTypeToString(distribution_type);
        if (key.find(dist_str) != std::string::npos && stats.hasReliableData()) {
            // Parse the key to extract strategy and batch category
            auto last_underscore = key.find_last_of('_');
            if (last_underscore != std::string::npos) {
                std::size_t batch_category = std::stoull(key.substr(last_underscore + 1));
                
                // Determine strategy from key
                Strategy strategy = Strategy::SCALAR;
                if (key.find("SCALAR") != std::string::npos) strategy = Strategy::SCALAR;
                else if (key.find("SIMD_BATCH") != std::string::npos) strategy = Strategy::SIMD_BATCH;
                else if (key.find("PARALLEL_SIMD") != std::string::npos) strategy = Strategy::PARALLEL_SIMD;
                
                batch_performance[batch_category][strategy] = stats.getAverageTimeNs();
            }
        }
    }
    
    // Find crossover points where one strategy becomes better than another
    std::size_t simd_threshold = 100;  // Default fallback
    std::size_t parallel_threshold = 5000;  // Default fallback
    
    // Look for the batch size where SIMD becomes better than SCALAR
    for (const auto& [batch_cat, strategies] : batch_performance) {
        auto scalar_it = strategies.find(Strategy::SCALAR);
        auto simd_it = strategies.find(Strategy::SIMD_BATCH);
        
        if (scalar_it != strategies.end() && simd_it != strategies.end()) {
            if (simd_it->second < scalar_it->second) {
                simd_threshold = std::max(simd_threshold, batch_cat * 1000); // Convert back from category
                break;
            }
        }
    }
    
    // Look for the batch size where PARALLEL becomes better than SIMD
    for (const auto& [batch_cat, strategies] : batch_performance) {
        auto simd_it = strategies.find(Strategy::SIMD_BATCH);
        auto parallel_it = strategies.find(Strategy::PARALLEL_SIMD);
        
        if (simd_it != strategies.end() && parallel_it != strategies.end()) {
            if (parallel_it->second < simd_it->second) {
                parallel_threshold = std::max(parallel_threshold, batch_cat * 1000); // Convert back from category
                break;
            }
        }
    }
    
    // Only return thresholds if we have sufficient data
    if (batch_performance.size() >= 3) {
        return std::make_pair(simd_threshold, parallel_threshold);
    }
    
    return std::nullopt;
}

void PerformanceHistory::clearHistory() noexcept {
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    performance_data_.clear();
    total_executions_.store(0, std::memory_order_release);
}

bool PerformanceHistory::hasSufficientLearningData([[maybe_unused]] DistributionType distribution_type) const noexcept {
    return total_executions_.load(std::memory_order_acquire) >= min_learning_threshold;
}

std::string PerformanceHistory::generateKey(
    Strategy strategy,
    DistributionType distribution_type,
    std::size_t batch_size_category
) noexcept {
    return std::string(strategyToString(strategy)) + "_" + distributionTypeToString(distribution_type) + "_" + std::to_string(batch_size_category);
}

std::size_t PerformanceHistory::categorizeBatchSize(std::size_t batch_size) noexcept {
    return batch_size / 1000;
}

const char* PerformanceHistory::strategyToString(Strategy strategy) noexcept {
    switch(strategy) {
        case Strategy::SCALAR: return "SCALAR";
        case Strategy::SIMD_BATCH: return "SIMD_BATCH";
        case Strategy::PARALLEL_SIMD: return "PARALLEL_SIMD";
        case Strategy::WORK_STEALING: return "WORK_STEALING";
        case Strategy::CACHE_AWARE: return "CACHE_AWARE";
        default: return "UNKNOWN";
    }
}

const char* PerformanceHistory::distributionTypeToString(DistributionType dist_type) noexcept {
    switch(dist_type) {
        case DistributionType::UNIFORM: return "UNIFORM";
        case DistributionType::GAUSSIAN: return "GAUSSIAN";
        case DistributionType::EXPONENTIAL: return "EXPONENTIAL";
        case DistributionType::DISCRETE: return "DISCRETE";
        case DistributionType::POISSON: return "POISSON";
        case DistributionType::GAMMA: return "GAMMA";
        default: return "UNKNOWN";
    }
}

} // namespace performance
} // namespace libstats


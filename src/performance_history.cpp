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
                         Strategy::WORK_STEALING, Strategy::GPU_ACCELERATED}) {
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
        double performance_ratio = static_cast<double>(worst_it->second) / static_cast<double>(best_it->second);
        double data_confidence = std::min(1.0, static_cast<double>(strategy_performance.size()) / 5.0);
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
                
                // Determine strategy from key - improved parsing
                Strategy strategy = Strategy::SCALAR;
                if (key.find("WORK_STEALING") != std::string::npos) strategy = Strategy::WORK_STEALING;
                else if (key.find("GPU_ACCELERATED") != std::string::npos) strategy = Strategy::GPU_ACCELERATED;
                else if (key.find("PARALLEL_SIMD") != std::string::npos) strategy = Strategy::PARALLEL_SIMD;
                else if (key.find("SIMD_BATCH") != std::string::npos) strategy = Strategy::SIMD_BATCH;
                else if (key.find("SCALAR") != std::string::npos) strategy = Strategy::SCALAR;
                
                batch_performance[batch_category][strategy] = stats.getAverageTimeNs();
            }
        }
    }
    
    // Need at least 3 batch sizes with data for meaningful learning
    if (batch_performance.size() < 3) {
        return std::nullopt;
    }
    
    // Advanced threshold detection with stability analysis
    std::size_t simd_threshold = findOptimalThreshold(batch_performance, Strategy::SCALAR, Strategy::SIMD_BATCH);
    std::size_t parallel_threshold = findOptimalThreshold(batch_performance, Strategy::SIMD_BATCH, Strategy::PARALLEL_SIMD);
    
    return std::make_pair(simd_threshold, parallel_threshold);
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
    // Enhanced categorization with better granularity around threshold boundaries
    // Focus on preserving critical crossover points
    if (batch_size <= 8) return 8;
    else if (batch_size <= 10) return 10;
    else if (batch_size <= 16) return 16;
    else if (batch_size <= 20) return 20;
    else if (batch_size <= 25) return 25;
    else if (batch_size <= 32) return 32;
    else if (batch_size <= 40) return 40;
    else if (batch_size <= 50) return 50;
    else if (batch_size <= 64) return 64;
    else if (batch_size <= 80) return 80;
    else if (batch_size <= 100) return 100;
    else if (batch_size <= 128) return 128;
    else if (batch_size <= 160) return 160;
    else if (batch_size <= 200) return 200;
    else if (batch_size <= 250) return 250;
    else if (batch_size <= 320) return 320;
    else if (batch_size <= 400) return 400;
    else if (batch_size <= 500) return 500;
    else if (batch_size <= 640) return 640;
    else if (batch_size <= 800) return 800;
    else if (batch_size <= 1000) return 1000;
    else if (batch_size <= 1280) return 1280;
    else if (batch_size <= 1600) return 1600;
    else if (batch_size <= 2000) return 2000;
    else if (batch_size <= 2500) return 2500;
    else if (batch_size <= 3200) return 3200;
    else if (batch_size <= 4000) return 4000;
    else if (batch_size <= 5000) return 5000;
    else if (batch_size <= 6400) return 6400;
    else if (batch_size <= 8000) return 8000;
    else if (batch_size <= 10000) return 10000;
    else if (batch_size <= 12800) return 12800;
    else if (batch_size <= 16000) return 16000;
    else if (batch_size <= 20000) return 20000;
    else if (batch_size <= 25000) return 25000;
    else if (batch_size <= 32000) return 32000;
    else if (batch_size <= 40000) return 40000;
    else if (batch_size <= 50000) return 50000;
    else if (batch_size <= 64000) return 64000;
    else if (batch_size <= 80000) return 80000;
    else return 100000;
}

const char* PerformanceHistory::strategyToString(Strategy strategy) noexcept {
    switch(strategy) {
        case Strategy::SCALAR: return "SCALAR";
        case Strategy::SIMD_BATCH: return "SIMD_BATCH";
        case Strategy::PARALLEL_SIMD: return "PARALLEL_SIMD";
        case Strategy::WORK_STEALING: return "WORK_STEALING";
        case Strategy::GPU_ACCELERATED: return "GPU_ACCELERATED";
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

std::size_t PerformanceHistory::findOptimalThreshold(
    const std::map<std::size_t, std::map<Strategy, std::uint64_t>>& batch_performance,
    Strategy baseline_strategy,
    Strategy target_strategy
) noexcept {
    // Fallback thresholds based on strategy type
    std::size_t fallback_threshold = 100;  // Default for SIMD
    if (target_strategy == Strategy::PARALLEL_SIMD) fallback_threshold = 5000;
    else if (target_strategy == Strategy::WORK_STEALING) fallback_threshold = 10000;
    else if (target_strategy == Strategy::GPU_ACCELERATED) fallback_threshold = 50000;
    
    // Collect crossover candidates with performance ratios
    std::vector<std::pair<std::size_t, double>> crossover_candidates;
    
    for (const auto& [batch_size, strategies] : batch_performance) {
        auto baseline_it = strategies.find(baseline_strategy);
        auto target_it = strategies.find(target_strategy);
        
        if (baseline_it != strategies.end() && target_it != strategies.end()) {
            uint64_t baseline_time = baseline_it->second;
            uint64_t target_time = target_it->second;
            
            // Only consider if target is actually better than baseline
            if (target_time < baseline_time && baseline_time > 0) {
                // Calculate performance improvement ratio
                double improvement_ratio = static_cast<double>(baseline_time) / static_cast<double>(target_time);
                // Only consider significant improvements (at least 5%)
                if (improvement_ratio > 1.05) {
                    crossover_candidates.emplace_back(batch_size, improvement_ratio);
                }
            }
        }
    }
    
    // If no crossover points found, return fallback
    if (crossover_candidates.empty()) {
        return fallback_threshold;
    }
    
    // Find the earliest batch size with stable performance advantage
    // Sort by batch size
    std::sort(crossover_candidates.begin(), crossover_candidates.end());
    
    // Advanced stability analysis: look for consistent performance advantage
    // across multiple consecutive batch sizes
    constexpr double MIN_IMPROVEMENT_RATIO = 1.1;  // Require 10% improvement
    constexpr size_t MIN_STABILITY_WINDOW = 2;     // Require stability across 2+ sizes
    
    for (size_t i = 0; i < crossover_candidates.size(); ++i) {
        const auto& [candidate_size, improvement_ratio] = crossover_candidates[i];
        
        // Check if this candidate has sufficient improvement
        if (improvement_ratio < MIN_IMPROVEMENT_RATIO) {
            continue;
        }
        
        // Check stability: count consecutive candidates with good performance
        size_t stability_count = 1;
        for (size_t j = i + 1; j < crossover_candidates.size() && 
             j < i + 3; ++j) {  // Look at most 3 ahead
            if (crossover_candidates[j].second >= MIN_IMPROVEMENT_RATIO) {
                stability_count++;
            } else {
                break;
            }
        }
        
        // If we have sufficient stability, this is our threshold
        if (stability_count >= MIN_STABILITY_WINDOW) {
            return candidate_size;
        }
    }
    
    // Fallback: find the candidate with the highest improvement ratio
    auto best_candidate = std::max_element(
        crossover_candidates.begin(), crossover_candidates.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;  // Compare improvement ratios
        }
    );
    
    if (best_candidate != crossover_candidates.end() && 
        best_candidate->second >= MIN_IMPROVEMENT_RATIO) {
        return best_candidate->first;
    }
    
    // Ultimate fallback
    return fallback_threshold;
}

} // namespace performance
} // namespace libstats


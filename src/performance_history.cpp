#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/distribution_meta.h"  // distributionEnumName()
#include "libstats/core/performance_history.h"

#include "libstats/core/math_constants.h"
#include "libstats/core/performance_constants.h"
#include "libstats/core/statistical_constants.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace stats {
namespace detail {  // Performance utilities

void PerformanceHistory::recordPerformance(Strategy strategy, DistributionType distribution_type,
                                           std::size_t batch_size,
                                           std::uint64_t execution_time_ns) noexcept {
    std::string key = generateKey(strategy, distribution_type, categorizeBatchSize(batch_size));
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    auto& stats = performance_data_[key];
    // Hold lock through all updates — prevents clearHistory() from calling
    // performance_data_.clear() and destroying the node that `stats` references.
    stats.total_time_ns.fetch_add(execution_time_ns, std::memory_order_release);
    stats.execution_count.fetch_add(1, std::memory_order_release);

    // CAS loop for min: separate load and store are not atomic together, so two
    // racing threads can each load the old value and one silently overwrites the
    // other's result. compare_exchange_weak retries until the update lands.
    {
        auto cur = stats.min_time_ns.load(std::memory_order_acquire);
        while (execution_time_ns < cur &&
               !stats.min_time_ns.compare_exchange_weak(cur, execution_time_ns,
                                                        std::memory_order_release,
                                                        std::memory_order_acquire)) {}
    }
    {
        auto cur = stats.max_time_ns.load(std::memory_order_acquire);
        while (execution_time_ns > cur &&
               !stats.max_time_ns.compare_exchange_weak(cur, execution_time_ns,
                                                        std::memory_order_release,
                                                        std::memory_order_acquire)) {}
    }

    total_executions_.fetch_add(1, std::memory_order_release);
}

std::optional<PerformanceSnapshot> PerformanceHistory::getPerformanceStats(
    Strategy strategy, DistributionType distribution_type) const noexcept {
    std::unique_lock<std::timed_mutex> lock(data_mutex_);

    // Aggregate performance data across all batch size categories
    PerformanceSnapshot aggregated;
    bool found_data = false;

    for (const auto& [key, stats] : performance_data_) {
        // Check if this key matches our strategy and distribution
        std::string expected_prefix = std::string(strategyToString(strategy)) + "_" +
                                      distributionTypeToString(distribution_type) + "_";
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
    DistributionType distribution_type, std::size_t batch_size) const noexcept {
    const std::size_t batch_category = categorizeBatchSize(batch_size);

    // Collect performance data for all strategies for this distribution and batch size category
    std::vector<std::pair<Strategy, std::uint64_t>> strategy_performance;

    std::unique_lock<std::timed_mutex> lock(data_mutex_);

    // Check all available strategies
    for (auto strategy :
         {Strategy::SCALAR, Strategy::VECTORIZED, Strategy::PARALLEL, Strategy::WORK_STEALING}) {
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
        return {Strategy::SCALAR, detail::ZERO_DOUBLE, 0, false};
    }

    // Find the strategy with the best (lowest) average time
    auto best_it =
        std::min_element(strategy_performance.begin(), strategy_performance.end(),
                         [](const auto& a, const auto& b) { return a.second < b.second; });

    // Calculate confidence score based on the number of strategies we have data for
    // and the performance difference between best and worst
    double confidence_score = detail::ZERO_DOUBLE;
    if (strategy_performance.size() >= 2) {
        auto worst_it =
            std::max_element(strategy_performance.begin(), strategy_performance.end(),
                             [](const auto& a, const auto& b) { return a.second < b.second; });

        // Confidence increases with performance difference and number of data points
        double performance_ratio =
            static_cast<double>(worst_it->second) / static_cast<double>(best_it->second);
        double data_confidence =
            std::min(detail::ONE, static_cast<double>(strategy_performance.size()) / detail::FIVE);
        confidence_score =
            std::min(detail::ONE, (performance_ratio - detail::ONE) * data_confidence);
    } else {
        confidence_score = detail::SINGLE_SAMPLE_CONFIDENCE;  // Medium confidence with only one data point
    }

    return {best_it->first, confidence_score, best_it->second, true};
}

std::optional<std::pair<std::size_t, std::size_t>> PerformanceHistory::learnOptimalThresholds(
    DistributionType distribution_type) const noexcept {
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
                if (key.find("WORK_STEALING") != std::string::npos)
                    strategy = Strategy::WORK_STEALING;
                else if (key.find("PARALLEL") != std::string::npos)
                    strategy = Strategy::PARALLEL;
                else if (key.find("VECTORIZED") != std::string::npos)
                    strategy = Strategy::VECTORIZED;
                else if (key.find("SCALAR") != std::string::npos)
                    strategy = Strategy::SCALAR;

                batch_performance[batch_category][strategy] = stats.getAverageTimeNs();
            }
        }
    }

    // Need at least 3 batch sizes with data for meaningful learning
    if (batch_performance.size() < 3) {
        return std::nullopt;
    }

    // Advanced threshold detection with stability analysis
    std::size_t simd_threshold =
        findOptimalThreshold(batch_performance, Strategy::SCALAR, Strategy::VECTORIZED);
    std::size_t parallel_threshold =
        findOptimalThreshold(batch_performance, Strategy::VECTORIZED, Strategy::PARALLEL);

    return std::make_pair(simd_threshold, parallel_threshold);
}

void PerformanceHistory::clearHistory() noexcept {
    std::unique_lock<std::timed_mutex> lock(data_mutex_);
    performance_data_.clear();
    total_executions_.store(0, std::memory_order_release);
}

bool PerformanceHistory::hasSufficientLearningData(
    [[maybe_unused]] DistributionType distribution_type) const noexcept {
    return total_executions_.load(std::memory_order_acquire) >= min_learning_threshold;
}

std::string PerformanceHistory::generateKey(Strategy strategy, DistributionType distribution_type,
                                            std::size_t batch_size_category) noexcept {
    return std::string(strategyToString(strategy)) + "_" +
           distributionTypeToString(distribution_type) + "_" + std::to_string(batch_size_category);
}

std::size_t PerformanceHistory::categorizeBatchSize(std::size_t batch_size) noexcept {
    // Bucket boundaries in ascending order. Each value is both the threshold
    // and the category label returned for all batch sizes up to that boundary.
    // std::lower_bound selects the smallest bucket >= batch_size in O(log N).
    static constexpr std::array<std::size_t, 41> kBuckets = {
        8,     10,    16,    20,    25,    32,    40,    50,    64,    80,    100,
        128,   160,   200,   250,   320,   400,   500,   640,   800,   1000,  1280,
        1600,  2000,  2500,  3200,  4000,  5000,  6400,  8000,  10000, 12800, 16000,
        20000, 25000, 32000, 40000, 50000, 64000, 80000, 100000};
    auto it = std::lower_bound(kBuckets.begin(), kBuckets.end(), batch_size);
    return (it != kBuckets.end()) ? *it : kBuckets.back();
}

const char* PerformanceHistory::strategyToString(Strategy strategy) noexcept {
    switch (strategy) {
        case Strategy::SCALAR:
            return "SCALAR";
        case Strategy::VECTORIZED:
            return "VECTORIZED";
        case Strategy::PARALLEL:
            return "PARALLEL";
        case Strategy::WORK_STEALING:
            return "WORK_STEALING";
        default:
            return "UNKNOWN";
    }
}

const char* PerformanceHistory::distributionTypeToString(DistributionType dist_type) noexcept {
    return distributionEnumName(dist_type).data();
}

std::size_t PerformanceHistory::findOptimalThreshold(
    const std::map<std::size_t, std::map<Strategy, std::uint64_t>>& batch_performance,
    Strategy baseline_strategy, Strategy target_strategy) noexcept {
    // Fallback thresholds based on strategy type
    std::size_t fallback_threshold = detail::MAX_NEWTON_ITERATIONS;  // Default for SIMD
    if (target_strategy == Strategy::PARALLEL)
        fallback_threshold = detail::MAX_DATA_POINTS_FOR_SW_TEST;
    else if (target_strategy == Strategy::WORK_STEALING)
        fallback_threshold = detail::WORK_STEALING_FALLBACK_THRESHOLD;

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
                double improvement_ratio =
                    static_cast<double>(baseline_time) / static_cast<double>(target_time);
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
        for (size_t j = i + 1; j < crossover_candidates.size() && j < i + 3;
             ++j) {  // Look at most 3 ahead
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
        crossover_candidates.begin(), crossover_candidates.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;  // Compare improvement ratios
        });

    if (best_candidate != crossover_candidates.end() &&
        best_candidate->second >= MIN_IMPROVEMENT_RATIO) {
        return best_candidate->first;
    }

    // Ultimate fallback
    return fallback_threshold;
}

}  // namespace detail
}  // namespace stats

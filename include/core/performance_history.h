#pragma once

#include "performance_dispatcher.h"

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

/**
 * @file performance_history.h
 * @brief Performance learning system for adaptive optimization
 *
 * This system tracks execution performance across different strategies and workloads,
 * enabling the PerformanceDispatcher to make increasingly intelligent decisions
 * based on historical performance data.
 */

namespace libstats {
namespace performance {

/**
 * @brief Learning system that tracks performance metrics for adaptive optimization
 *
 * The PerformanceHistory class maintains a record of execution times for different
 * strategies across various workload characteristics. This enables the system to
 * learn optimal thresholds and make better dispatching decisions over time.
 *
 * @par Thread Safety:
 * - All methods are thread-safe using atomic operations and mutexes
 * - Recording performance is lock-free in the common case
 * - Learning updates use reader-writer locks for optimal concurrent access
 *
 * @par Key Features:
 * - Adaptive threshold learning based on performance feedback
 * - Distribution-specific performance tracking
 * - Batch size sensitivity analysis
 * - Statistical confidence in recommendations
 */
/**
 * @brief Simple snapshot of performance statistics (copyable, no atomics)
 */
struct PerformanceSnapshot {
    std::uint64_t total_time_ns = 0;         ///< Cumulative execution time
    std::uint32_t execution_count = 0;       ///< Number of executions
    std::uint64_t min_time_ns = UINT64_MAX;  ///< Fastest execution
    std::uint64_t max_time_ns = 0;           ///< Slowest execution

    /**
     * @brief Get average execution time in nanoseconds
     * @return Average execution time, or 0 if no executions recorded
     */
    [[nodiscard]] std::uint64_t getAverageTimeNs() const noexcept {
        if (execution_count == 0) {
            return 0;
        }
        return total_time_ns / execution_count;
    }

    /**
     * @brief Check if we have sufficient data for reliable statistics
     * @return True if we have at least minimum required samples
     */
    [[nodiscard]] bool hasReliableData() const noexcept { return execution_count >= 5; }
};

class PerformanceHistory {
   public:
    /**
     * @brief Performance statistics for a specific strategy and workload (atomic version)
     */
    struct PerformanceStats {
        std::atomic<std::uint64_t> total_time_ns{0};         ///< Cumulative execution time
        std::atomic<std::uint32_t> execution_count{0};       ///< Number of executions
        std::atomic<std::uint64_t> min_time_ns{UINT64_MAX};  ///< Fastest execution
        std::atomic<std::uint64_t> max_time_ns{0};           ///< Slowest execution

        /**
         * @brief Get average execution time in nanoseconds
         * @return Average execution time, or 0 if no executions recorded
         */
        [[nodiscard]] std::uint64_t getAverageTimeNs() const noexcept {
            const auto count = execution_count.load(std::memory_order_acquire);
            if (count == 0) {
                return 0;
            }
            return total_time_ns.load(std::memory_order_acquire) / count;
        }

        /**
         * @brief Check if we have sufficient data for reliable statistics
         * @return True if we have at least minimum required samples
         */
        [[nodiscard]] bool hasReliableData() const noexcept {
            return execution_count.load(std::memory_order_acquire) >= min_samples_for_reliability;
        }

       private:
        static constexpr std::uint32_t min_samples_for_reliability = 5;
    };

    /**
     * @brief Strategy performance recommendation based on historical data
     */
    struct StrategyRecommendation {
        Strategy recommended_strategy;   ///< Best performing strategy
        double confidence_score;         ///< Confidence in recommendation (0.0-1.0)
        std::uint64_t expected_time_ns;  ///< Expected execution time
        bool has_sufficient_data;        ///< True if recommendation is based on adequate data
    };

    /**
     * @brief Record performance data for a specific execution
     *
     * @param strategy The execution strategy used
     * @param distribution_type Type of distribution processed
     * @param batch_size Number of elements processed
     * @param execution_time_ns Actual execution time in nanoseconds
     */
    void recordPerformance(Strategy strategy, DistributionType distribution_type,
                           std::size_t batch_size, std::uint64_t execution_time_ns) noexcept;

    /**
     * @brief Get performance statistics for a specific strategy and distribution
     *
     * @param strategy The execution strategy to query
     * @param distribution_type The distribution type to query
     * @return Performance statistics, or nullopt if no data available
     */
    [[nodiscard]] std::optional<PerformanceSnapshot> getPerformanceStats(
        Strategy strategy, DistributionType distribution_type) const noexcept;

    /**
     * @brief Get the best strategy recommendation for given parameters
     *
     * @param distribution_type Type of distribution to process
     * @param batch_size Number of elements to process
     * @return Strategy recommendation with confidence metrics
     */
    [[nodiscard]] StrategyRecommendation getBestStrategy(DistributionType distribution_type,
                                                         std::size_t batch_size) const noexcept;

    /**
     * @brief Learn optimal thresholds based on performance history
     *
     * @param distribution_type Distribution to optimize thresholds for
     * @return Suggested threshold values, or nullopt if insufficient data
     */
    [[nodiscard]] std::optional<std::pair<std::size_t, std::size_t>> learnOptimalThresholds(
        DistributionType distribution_type) const noexcept;

    /**
     * @brief Clear all performance history (useful for testing or reset)
     */
    void clearHistory() noexcept;

    /**
     * @brief Get total number of recorded executions across all strategies
     * @return Total execution count
     */
    [[nodiscard]] std::uint64_t getTotalExecutions() const noexcept {
        return total_executions_.load(std::memory_order_acquire);
    }

    /**
     * @brief Check if the learning system has sufficient data for reliable recommendations
     * @param distribution_type Distribution type to check
     * @return True if we have enough data for this distribution type
     */
    [[nodiscard]] bool hasSufficientLearningData(DistributionType distribution_type) const noexcept;

   private:
    /**
     * @brief Generate unique key for performance tracking
     */
    [[nodiscard]] static std::string generateKey(Strategy strategy,
                                                 DistributionType distribution_type,
                                                 std::size_t batch_size_category) noexcept;

    /**
     * @brief Categorize batch size for statistical grouping
     */
    [[nodiscard]] static std::size_t categorizeBatchSize(std::size_t batch_size) noexcept;

    /**
     * @brief Convert strategy enum to string for key generation
     */
    [[nodiscard]] static const char* strategyToString(Strategy strategy) noexcept;

    /**
     * @brief Convert distribution type enum to string for key generation
     */
    [[nodiscard]] static const char* distributionTypeToString(DistributionType dist_type) noexcept;

    /**
     * @brief Find optimal threshold between two strategies using advanced analysis
     *
     * @param batch_performance Performance data across batch sizes
     * @param baseline_strategy The baseline strategy (e.g., SCALAR)
     * @param target_strategy The target strategy (e.g., SIMD_BATCH)
     * @return Optimal threshold size, or fallback value if no clear crossover
     */
    [[nodiscard]] static std::size_t findOptimalThreshold(
        const std::map<std::size_t, std::map<Strategy, std::uint64_t>>& batch_performance,
        Strategy baseline_strategy, Strategy target_strategy) noexcept;

    /// Performance data storage - thread-safe map with atomic statistics
    mutable std::unordered_map<std::string, PerformanceStats> performance_data_;

    /// Mutex for protecting map modifications (not needed for individual PerformanceStats updates)
    // Replace shared_mutex with a suitable alternative for compatibility
    mutable std::timed_mutex data_mutex_;

    /// Total execution counter across all strategies
    std::atomic<std::uint64_t> total_executions_{0};

    /// Minimum executions needed before making learning-based recommendations
    static constexpr std::uint32_t min_learning_threshold = 10;
};

}  // namespace performance
}  // namespace libstats

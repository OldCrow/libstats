#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

#include "core/performance_dispatcher.h"
#include "core/performance_history.h"

#include <chrono>
#include <gtest/gtest.h>
#include <random>
#include <thread>

using namespace libstats::performance;

class PerformanceHistoryTest : public ::testing::Test {
   protected:
    void SetUp() override {
        history.clearHistory();  // Start with clean slate
    }

    PerformanceHistory history;
};

TEST_F(PerformanceHistoryTest, BasicRecordingAndRetrieval) {
    // Record some performance data
    history.recordPerformance(Strategy::SCALAR, DistributionType::GAUSSIAN, 100, 1000);
    history.recordPerformance(Strategy::SCALAR, DistributionType::GAUSSIAN, 200, 2000);
    history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN, 100, 800);

    // Retrieve statistics
    auto scalar_stats = history.getPerformanceStats(Strategy::SCALAR, DistributionType::GAUSSIAN);
    ASSERT_TRUE(scalar_stats.has_value());
    EXPECT_EQ(scalar_stats->execution_count, 2);
    EXPECT_EQ(scalar_stats->getAverageTimeNs(), 1500);  // (1000 + 2000) / 2
    EXPECT_EQ(scalar_stats->min_time_ns, 1000);
    EXPECT_EQ(scalar_stats->max_time_ns, 2000);

    auto simd_stats = history.getPerformanceStats(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN);
    ASSERT_TRUE(simd_stats.has_value());
    EXPECT_EQ(simd_stats->execution_count, 1);
    EXPECT_EQ(simd_stats->getAverageTimeNs(), 800);
}

TEST_F(PerformanceHistoryTest, ThreadSafetyTest) {
    constexpr int num_threads = 4;
    constexpr int records_per_thread = 100;

    std::vector<std::thread> threads;

    // Launch multiple threads recording performance data
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([this, t]() {
            std::mt19937 rng(static_cast<unsigned int>(t));  // Seed with thread number
            std::uniform_int_distribution<uint64_t> time_dist(100, 10000);

            for (int i = 0; i < records_per_thread; ++i) {
                Strategy strategy = (i % 2 == 0) ? Strategy::SCALAR : Strategy::SIMD_BATCH;
                DistributionType dist_type =
                    (i % 3 == 0) ? DistributionType::GAUSSIAN : DistributionType::EXPONENTIAL;
                size_t batch_size = 100 + static_cast<size_t>(i % 900);  // 100-999
                uint64_t exec_time = time_dist(rng);

                history.recordPerformance(strategy, dist_type, batch_size, exec_time);
            }
        });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify total executions
    EXPECT_EQ(history.getTotalExecutions(), static_cast<size_t>(num_threads * records_per_thread));

    // Verify we can retrieve data for different strategies
    auto scalar_gaussian =
        history.getPerformanceStats(Strategy::SCALAR, DistributionType::GAUSSIAN);
    auto simd_gaussian =
        history.getPerformanceStats(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN);

    EXPECT_TRUE(scalar_gaussian.has_value());
    EXPECT_TRUE(simd_gaussian.has_value());
    EXPECT_GT(scalar_gaussian->execution_count, 0);
    EXPECT_GT(simd_gaussian->execution_count, 0);
}

TEST_F(PerformanceHistoryTest, StrategyRecommendation) {
    // Record data showing SIMD is faster than scalar for medium batches
    for (int i = 0; i < 10; ++i) {
        history.recordPerformance(Strategy::SCALAR, DistributionType::GAUSSIAN, 1000,
                                  static_cast<uint64_t>(5000 + i * 100));
        history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN, 1000,
                                  static_cast<uint64_t>(2000 + i * 50));
    }

    // Get recommendation
    auto recommendation = history.getBestStrategy(DistributionType::GAUSSIAN, 1000);

    EXPECT_EQ(recommendation.recommended_strategy, Strategy::SIMD_BATCH);
    EXPECT_GT(recommendation.confidence_score, 0.5);
    EXPECT_TRUE(recommendation.has_sufficient_data);
    EXPECT_LT(recommendation.expected_time_ns, 3000);  // Should be close to SIMD average
}

TEST_F(PerformanceHistoryTest, InsufficientDataHandling) {
    // Record only a few data points
    history.recordPerformance(Strategy::SCALAR, DistributionType::GAMMA, 500, 1000);
    history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::GAMMA, 500, 800);

    auto recommendation = history.getBestStrategy(DistributionType::GAMMA, 500);

    // Should indicate insufficient data
    EXPECT_FALSE(recommendation.has_sufficient_data);
    EXPECT_LT(recommendation.confidence_score, 0.5);
}

TEST_F(PerformanceHistoryTest, OptimalThresholdLearning) {
    // Record performance data across different batch sizes to learn thresholds
    // Small batches: scalar is better
    for (int i = 0; i < 10; ++i) {
        history.recordPerformance(Strategy::SCALAR, DistributionType::EXPONENTIAL, 10,
                                  static_cast<uint64_t>(100 + i));
        history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::EXPONENTIAL, 10,
                                  static_cast<uint64_t>(200 + i));
    }

    // Medium batches: SIMD is better
    for (int i = 0; i < 10; ++i) {
        history.recordPerformance(Strategy::SCALAR, DistributionType::EXPONENTIAL, 1000,
                                  static_cast<uint64_t>(5000 + i * 100));
        history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::EXPONENTIAL, 1000,
                                  static_cast<uint64_t>(2000 + i * 50));
    }

    // Large batches: parallel is best
    for (int i = 0; i < 10; ++i) {
        history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::EXPONENTIAL, 10000,
                                  static_cast<uint64_t>(15000 + i * 200));
        history.recordPerformance(Strategy::PARALLEL_SIMD, DistributionType::EXPONENTIAL, 10000,
                                  static_cast<uint64_t>(8000 + i * 100));
    }

    auto thresholds = history.learnOptimalThresholds(DistributionType::EXPONENTIAL);
    ASSERT_TRUE(thresholds.has_value());

    // Should learn reasonable threshold values based on batch categorization
    // The algorithm uses batch_size/1000 categorization, so:
    // - batch_size 10 -> category 0
    // - batch_size 1000 -> category 1
    // - batch_size 10000 -> category 10
    // Algorithm finds crossover points at category boundaries
    EXPECT_GE(thresholds->first, 10);      // SIMD threshold should be at least 10
    EXPECT_LE(thresholds->first, 1000);    // Could be exactly 1000 (category 1 * 1000)
    EXPECT_GE(thresholds->second, 1000);   // Parallel threshold should be at least 1000
    EXPECT_LE(thresholds->second, 10000);  // Could be exactly 10000 (category 10 * 1000)

    // Parallel threshold should be higher than or equal to SIMD threshold
    EXPECT_GE(thresholds->second, thresholds->first);
}

TEST_F(PerformanceHistoryTest, ClearHistory) {
    // Record some data
    history.recordPerformance(Strategy::SCALAR, DistributionType::UNIFORM, 100, 1000);
    history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::UNIFORM, 200, 800);

    EXPECT_EQ(history.getTotalExecutions(), 2);

    // Clear history
    history.clearHistory();

    EXPECT_EQ(history.getTotalExecutions(), 0);

    // Should return no data
    auto stats = history.getPerformanceStats(Strategy::SCALAR, DistributionType::UNIFORM);
    EXPECT_FALSE(stats.has_value());
}

TEST_F(PerformanceHistoryTest, SufficientLearningDataCheck) {
    // Initially no data
    EXPECT_FALSE(history.hasSufficientLearningData(DistributionType::POISSON));

    // Add some data but not enough
    for (int i = 0; i < 3; ++i) {
        history.recordPerformance(Strategy::SCALAR, DistributionType::POISSON, 100, 1000);
    }
    EXPECT_FALSE(history.hasSufficientLearningData(DistributionType::POISSON));

    // Add enough data
    for (int i = 0; i < 10; ++i) {
        history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::POISSON, 100, 800);
    }
    EXPECT_TRUE(history.hasSufficientLearningData(DistributionType::POISSON));
}

TEST_F(PerformanceHistoryTest, PerformanceSnapshotCopyable) {
    // Test that PerformanceSnapshot can be copied (no atomics)
    history.recordPerformance(Strategy::SCALAR, DistributionType::DISCRETE, 500, 1500);

    auto stats_opt = history.getPerformanceStats(Strategy::SCALAR, DistributionType::DISCRETE);
    ASSERT_TRUE(stats_opt.has_value());

    // Should be copyable
    PerformanceSnapshot copy = *stats_opt;
    EXPECT_EQ(copy.execution_count, 1);
    EXPECT_EQ(copy.total_time_ns, 1500);
    EXPECT_TRUE(copy.hasReliableData() == false);  // Need at least 5 samples

    // Test assignment
    PerformanceSnapshot assigned;
    assigned = copy;
    EXPECT_EQ(assigned.getAverageTimeNs(), 1500);
}

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

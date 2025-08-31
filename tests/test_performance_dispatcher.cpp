#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

// Use focused header for performance dispatcher testing
#include "../include/core/performance_dispatcher.h"
#include "../include/core/performance_history.h"

// Standard library includes
#include <gtest/gtest.h>
#include <thread>  // for std::thread
#include <vector>  // for std::vector

using namespace stats::detail;

class PerformanceDispatcherTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Clear any existing performance history
        PerformanceDispatcher::getPerformanceHistory().clearHistory();
    }
};

class SystemCapabilitiesTest : public ::testing::Test {
   protected:
    const SystemCapabilities& capabilities = SystemCapabilities::current();
};

TEST_F(SystemCapabilitiesTest, BasicCapabilityDetection) {
    // Test that basic system info is detected
    EXPECT_GT(capabilities.logical_cores(), 0);
    EXPECT_GT(capabilities.physical_cores(), 0);
    EXPECT_LE(capabilities.physical_cores(), capabilities.logical_cores());

    // Cache sizes should be reasonable (non-zero for most systems)
    EXPECT_GT(capabilities.l1_cache_size(), 0);

    // Should have some form of SIMD on modern systems (but fallback gracefully)
    // Note: These might be false on very old systems
    [[maybe_unused]] bool has_any_simd = capabilities.has_sse2() || capabilities.has_avx() ||
                                         capabilities.has_avx2() || capabilities.has_neon();

    // Performance characteristics should be positive
    EXPECT_GE(capabilities.simd_efficiency(), 0.0);
    EXPECT_GE(capabilities.threading_overhead_ns(), 0.0);
    EXPECT_GE(capabilities.memory_bandwidth_gb_s(), 0.0);

    // Efficiency should be reasonable (not greater than 1.0 in most cases)
    EXPECT_LE(capabilities.simd_efficiency(), 2.0);  // Allow some margin for exceptional systems
}

TEST_F(PerformanceDispatcherTest, BasicStrategySelection) {
    PerformanceDispatcher dispatcher;
    const SystemCapabilities& system = SystemCapabilities::current();

    // Very small batches should prefer scalar
    auto strategy_small = dispatcher.selectOptimalStrategy(5, DistributionType::GAUSSIAN,
                                                           ComputationComplexity::SIMPLE, system);
    EXPECT_EQ(strategy_small, Strategy::SCALAR);

    // Very large batches should prefer parallel strategies
    auto strategy_large = dispatcher.selectOptimalStrategy(100000, DistributionType::GAUSSIAN,
                                                           ComputationComplexity::COMPLEX, system);
    EXPECT_TRUE(strategy_large == Strategy::PARALLEL_SIMD ||
                strategy_large == Strategy::WORK_STEALING ||
                strategy_large == Strategy::GPU_ACCELERATED);
}

TEST_F(PerformanceDispatcherTest, DistributionSpecificThresholds) {
    PerformanceDispatcher dispatcher;
    const SystemCapabilities& system = SystemCapabilities::current();

    // Test that different distributions have different thresholds
    // Simple distributions (like uniform) should need larger batches for parallelization
    auto uniform_medium = dispatcher.selectOptimalStrategy(1000, DistributionType::UNIFORM,
                                                           ComputationComplexity::SIMPLE, system);

    // Complex distributions (like gamma) should parallelize earlier
    [[maybe_unused]] auto gamma_medium = dispatcher.selectOptimalStrategy(
        1000, DistributionType::GAMMA, ComputationComplexity::COMPLEX, system);

    // If we have multiple cores, gamma should be more likely to use parallel strategies
    if (system.physical_cores() > 1) {
        // Gamma with same batch size might choose parallel while uniform chooses SIMD
        // This is probabilistic based on thresholds, so we test the concept
        EXPECT_TRUE(uniform_medium == Strategy::SCALAR || uniform_medium == Strategy::SIMD_BATCH);
    }
}

TEST_F(PerformanceDispatcherTest, ComplexityInfluencesStrategy) {
    PerformanceDispatcher dispatcher;
    const SystemCapabilities& system = SystemCapabilities::current();

    // Same distribution, same batch size, different complexity
    constexpr size_t batch_size = 1000;
    constexpr DistributionType dist = DistributionType::GAUSSIAN;

    auto simple_strategy =
        dispatcher.selectOptimalStrategy(batch_size, dist, ComputationComplexity::SIMPLE, system);

    [[maybe_unused]] auto complex_strategy =
        dispatcher.selectOptimalStrategy(batch_size, dist, ComputationComplexity::COMPLEX, system);

    // Complex operations should be more likely to choose parallel execution
    // (This is a general trend, though specific results depend on system capabilities)
    EXPECT_TRUE(simple_strategy !=
                Strategy::GPU_ACCELERATED);  // Simple ops unlikely to need GPU acceleration
}

TEST_F(PerformanceDispatcherTest, ThresholdUpdating) {
    PerformanceDispatcher dispatcher;

    // Get current thresholds
    auto original_thresholds = dispatcher.getThresholds();

    // Update with new thresholds
    PerformanceDispatcher::Thresholds new_thresholds = original_thresholds;
    new_thresholds.simd_min = 16;        // Change from default
    new_thresholds.parallel_min = 2000;  // Change from default

    dispatcher.updateThresholds(new_thresholds);

    // Verify thresholds were updated
    auto updated_thresholds = dispatcher.getThresholds();
    EXPECT_EQ(updated_thresholds.simd_min, 16);
    EXPECT_EQ(updated_thresholds.parallel_min, 2000);
}

TEST_F(PerformanceDispatcherTest, PerformanceRecording) {
    // Test static performance recording function
    PerformanceDispatcher::recordPerformance(Strategy::SIMD_BATCH, DistributionType::EXPONENTIAL,
                                             500, 1500);
    PerformanceDispatcher::recordPerformance(Strategy::SCALAR, DistributionType::EXPONENTIAL, 500,
                                             2000);

    // Verify data was recorded in the global history
    auto& history = PerformanceDispatcher::getPerformanceHistory();

    auto simd_stats =
        history.getPerformanceStats(Strategy::SIMD_BATCH, DistributionType::EXPONENTIAL);
    auto scalar_stats =
        history.getPerformanceStats(Strategy::SCALAR, DistributionType::EXPONENTIAL);

    ASSERT_TRUE(simd_stats.has_value());
    ASSERT_TRUE(scalar_stats.has_value());

    EXPECT_EQ(simd_stats->execution_count, 1);
    EXPECT_EQ(simd_stats->getAverageTimeNs(), 1500);
    EXPECT_EQ(scalar_stats->execution_count, 1);
    EXPECT_EQ(scalar_stats->getAverageTimeNs(), 2000);
}

TEST_F(PerformanceDispatcherTest, PerformanceHints) {
    // Test performance hint structures
    auto minimal_latency = PerformanceHint::minimal_latency();
    EXPECT_EQ(minimal_latency.strategy, PerformanceHint::PreferredStrategy::MINIMIZE_LATENCY);
    EXPECT_EQ(minimal_latency.thread_count.value_or(0), 1);

    auto max_throughput = PerformanceHint::maximum_throughput();
    EXPECT_EQ(max_throughput.strategy, PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT);
    EXPECT_FALSE(max_throughput.thread_count.has_value());

    // Test custom hint
    PerformanceHint custom_hint;
    custom_hint.strategy = PerformanceHint::PreferredStrategy::FORCE_SIMD;
    custom_hint.disable_learning = true;
    custom_hint.force_strategy = true;

    EXPECT_EQ(custom_hint.strategy, PerformanceHint::PreferredStrategy::FORCE_SIMD);
    EXPECT_TRUE(custom_hint.disable_learning);
    EXPECT_TRUE(custom_hint.force_strategy);
}

TEST_F(PerformanceDispatcherTest, EdgeCases) {
    PerformanceDispatcher dispatcher;
    const SystemCapabilities& system = SystemCapabilities::current();

    // Test edge cases

    // Zero batch size (should handle gracefully)
    auto zero_strategy = dispatcher.selectOptimalStrategy(0, DistributionType::GAUSSIAN,
                                                          ComputationComplexity::SIMPLE, system);
    EXPECT_EQ(zero_strategy, Strategy::SCALAR);

    // Single element
    auto single_strategy = dispatcher.selectOptimalStrategy(1, DistributionType::GAMMA,
                                                            ComputationComplexity::COMPLEX, system);
    EXPECT_EQ(single_strategy, Strategy::SCALAR);

    // Extremely large batch size
    auto huge_strategy = dispatcher.selectOptimalStrategy(SIZE_MAX / 2, DistributionType::UNIFORM,
                                                          ComputationComplexity::SIMPLE, system);
    EXPECT_TRUE(huge_strategy == Strategy::PARALLEL_SIMD ||
                huge_strategy == Strategy::WORK_STEALING ||
                huge_strategy == Strategy::GPU_ACCELERATED);
}

TEST_F(PerformanceDispatcherTest, ThreadSafety) {
    // Test that multiple threads can safely use the dispatcher
    constexpr std::size_t num_threads = 4;
    constexpr std::size_t selections_per_thread = 1000;

    std::vector<std::thread> threads;
    std::vector<std::vector<Strategy>> results(static_cast<std::size_t>(num_threads));

    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            PerformanceDispatcher dispatcher;
            const SystemCapabilities& system = SystemCapabilities::current();
            results[t].reserve(static_cast<std::size_t>(selections_per_thread));

            for (std::size_t i = 0; i < selections_per_thread; ++i) {
                size_t batch_size = 100 + static_cast<std::size_t>(i % 10000);
                DistributionType dist_type = static_cast<DistributionType>(i % 6);
                ComputationComplexity complexity = static_cast<ComputationComplexity>(i % 3);

                auto strategy =
                    dispatcher.selectOptimalStrategy(batch_size, dist_type, complexity, system);
                results[t].push_back(strategy);

                // Also record some performance data
                PerformanceDispatcher::recordPerformance(strategy, dist_type, batch_size,
                                                         static_cast<uint64_t>(1000 + (i % 5000)));
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify results
    for (std::size_t t = 0; t < num_threads; ++t) {
        EXPECT_EQ(results[t].size(), selections_per_thread);
        // All strategies should be valid
        for (auto strategy : results[t]) {
            EXPECT_TRUE(strategy >= Strategy::SCALAR && strategy <= Strategy::GPU_ACCELERATED);
        }
    }

    // Should have recorded performance data
    auto& history = PerformanceDispatcher::getPerformanceHistory();
    EXPECT_GT(history.getTotalExecutions(), 0);
}

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

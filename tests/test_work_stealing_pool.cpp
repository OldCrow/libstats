/**
 * @file test_work_stealing_pool.cpp
 * @brief GTest suite for WorkStealingPool with Level 0-2 integration
 */
#include "libstats/platform/work_stealing_pool.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <vector>

using namespace stats;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

TEST(WorkStealingPool, BasicTaskSubmission) {
    WorkStealingPool pool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i) {
        pool.submit([&counter, i]() {
            counter.fetch_add(1);
            std::cout << "  Task " << i << " executed\n";
        });
    }

    pool.waitForAll();
    EXPECT_EQ(counter.load(), 10);
    std::cout << "  All 10 tasks executed successfully\n";
}

TEST(WorkStealingPool, CppRangesWithWorkStealing) {
    WorkStealingPool pool(4);

    std::vector<int> data;
    auto range = std::views::iota(0, 1000) | std::views::transform([](int i) { return i * 2; });
    std::ranges::copy(range, std::back_inserter(data));

    std::atomic<long long> sum{0};
    const size_t chunkSize = 100;
    for (size_t start = 0; start < data.size(); start += chunkSize) {
        size_t end = std::min(start + chunkSize, data.size());
        pool.submit([&data, &sum, start, end]() {
            long long local = 0;
            for (size_t i = start; i < end; ++i)
                local += data[i];
            sum.fetch_add(local);
        });
    }

    pool.waitForAll();

    long long expected = 0;
    for (auto v : data)
        expected += v;
    EXPECT_EQ(sum.load(), expected);
    std::cout << "  Parallel sum: " << sum.load() << " (expected: " << expected << ")\n";
}

TEST(WorkStealingPool, ParallelForWithConcepts) {
    WorkStealingPool pool(4);

    auto process_numeric = [&pool]<Numeric T>(std::vector<T>& vec, T multiplier) {
        pool.parallelFor(0, vec.size(), [&vec, multiplier](size_t i) { vec[i] *= multiplier; });
    };

    std::vector<int> int_data(1000);
    std::iota(int_data.begin(), int_data.end(), 1);
    process_numeric(int_data, 3);

    EXPECT_EQ(int_data[0], 3);
    EXPECT_EQ(int_data[1], 6);
    EXPECT_EQ(int_data[999], 3000);
    std::cout << "  Concept-based parallelFor processed 1000 elements\n";
}

TEST(WorkStealingPool, PerformanceAndStatistics) {
    unsigned int thread_count = std::max(2u, std::thread::hardware_concurrency());
    WorkStealingPool pool(thread_count);
    const int numTasks = 1000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numTasks; ++i) {
        pool.submit([i]() {
            auto work = std::views::iota(0, (i % 10 + 1) * 100) |
                        std::views::transform([](int x) { return x * x; }) | std::views::take(50);
            volatile long long s = 0;
            for (auto v : work)
                s = s + v;
        });
    }
    pool.waitForAll();
    auto end = std::chrono::high_resolution_clock::now();

    auto stats = pool.getStatistics();
    std::cout << "  Execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms, steals: " << stats.workSteals << "\n";

    EXPECT_EQ(static_cast<int>(stats.tasksExecuted), numTasks);
}

TEST(WorkStealingPool, GlobalUtilities) {
    std::atomic<int> counter{0};

    std::vector<int> indices;
    std::ranges::copy(std::views::iota(0, 50), std::back_inserter(indices));

    for ([[maybe_unused]] auto i : indices) {
        stats::workStealingSubmit([&counter]() { counter.fetch_add(1); });
    }

    stats::workStealingWaitForAll();
    EXPECT_EQ(counter.load(), 50);
    std::cout << "  Global utilities executed " << counter.load() << " tasks\n";
}

TEST(WorkStealingPool, Level0ConstantsIntegration) {
    auto parallelThreshold = arch::get_min_elements_for_parallel();
    auto distributionThreshold = arch::get_min_elements_for_distribution_parallel();
    auto defaultGrainSize = arch::get_default_grain_size();
    auto simdBlockSize = arch::get_optimal_simd_block_size();
    auto memoryAlignment = arch::get_optimal_alignment();

    EXPECT_GT(parallelThreshold, 0u);
    EXPECT_GT(distributionThreshold, 0u);
    EXPECT_GT(defaultGrainSize, 0u);
    EXPECT_GT(simdBlockSize, 0u);
    EXPECT_GT(memoryAlignment, 0u);

    std::cout << "  Min parallel: " << parallelThreshold << ", grain: " << defaultGrainSize
              << ", alignment: " << memoryAlignment << " bytes\n";
}

TEST(WorkStealingPool, CPUDetectionIntegration) {
    const auto& features = arch::get_features();
    auto optimalThreads = WorkStealingPool::getOptimalThreadCount();

    EXPECT_GT(optimalThreads, 0u);

    if (features.topology.hyperthreading && features.topology.logical_cores > 0) {
        EXPECT_LE(optimalThreads, features.topology.logical_cores);
    }

    std::cout << "  Physical: " << features.topology.physical_cores
              << ", Logical: " << features.topology.logical_cores << ", Optimal: " << optimalThreads
              << "\n";
}

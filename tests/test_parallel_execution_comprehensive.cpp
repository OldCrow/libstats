#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Test integration with main header
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <gtest/gtest.h>

TEST(ParallelExecutionComprehensive, GCDAlgorithms) {
    std::vector<int> count_data = {1, 2, 3, 2, 4, 2, 5, 2};

    auto count_result = stats::arch::safe_count(count_data.begin(), count_data.end(), 2);
    EXPECT_EQ(count_result, 4);
    std::cout << "  - safe_count (GCD): PASSED" << std::endl;

    auto count_if_result = stats::arch::safe_count_if(
        count_data.begin(), count_data.end(), [](int x) { return x > 2; });
    EXPECT_EQ(count_if_result, 3);
    std::cout << "  - safe_count_if (GCD): PASSED" << std::endl;

    std::vector<double> reduce_data(1000, 2.0);
    auto sum_result = stats::arch::safe_reduce(reduce_data.begin(), reduce_data.end(), 0.0);
    EXPECT_NEAR(sum_result, 2000.0, 1e-10);
    std::cout << "  - safe_reduce (GCD): PASSED" << std::endl;
}

TEST(ParallelExecutionComprehensive, EdgeCasesAndBoundaryConditions) {
    std::vector<int> single = {42};
    stats::arch::safe_fill(single.begin(), single.end(), 100);
    EXPECT_EQ(single[0], 100);
    std::cout << "  - Single element fill: PASSED" << std::endl;

    std::vector<double> large_data(100000);
    std::iota(large_data.begin(), large_data.end(), 1.0);

    std::vector<double> large_output(100000);
    stats::arch::safe_transform(large_data.begin(), large_data.end(), large_output.begin(),
                                [](double x) { return x * x; });
    EXPECT_EQ(large_output[0], 1.0);
    EXPECT_EQ(large_output[999], 1000000.0);
    std::cout << "  - Large dataset transform: PASSED" << std::endl;

    auto large_sum = stats::arch::safe_reduce(large_data.begin(), large_data.end(), 0.0);
    double expected_sum = (100000.0 * 100001.0) / 2.0;
    EXPECT_NEAR(large_sum, expected_sum, 1.0);
    std::cout << "  - Large dataset reduce: PASSED" << std::endl;
}

TEST(ParallelExecutionComprehensive, AlgorithmCorrectnessVerification) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    std::vector<double> test_data(5000);
    std::generate(test_data.begin(), test_data.end(), [&]() { return dis(gen); });

    std::vector<double> serial_result(5000);
    std::transform(test_data.begin(), test_data.end(), serial_result.begin(),
                   [](double x) { return std::sqrt(x); });

    std::vector<double> parallel_result(5000);
    stats::arch::safe_transform(test_data.begin(), test_data.end(), parallel_result.begin(),
                                [](double x) { return std::sqrt(x); });

    bool transforms_equal =
        std::equal(serial_result.begin(), serial_result.end(), parallel_result.begin(),
                   [](double a, double b) { return std::abs(a - b) < 1e-10; });
    EXPECT_TRUE(transforms_equal);
    std::cout << "  - Parallel vs serial transform equivalence: PASSED" << std::endl;
}

TEST(ParallelExecutionComprehensive, PerformanceCharacteristics) {
    std::vector<double> perf_data(50000);
    std::iota(perf_data.begin(), perf_data.end(), 1.0);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto perf_sum = stats::arch::safe_reduce(perf_data.begin(), perf_data.end(), 0.0);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "  - Parallel reduce of 50k elements took: "
              << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
              << " us" << std::endl;

    double expected = (50000.0 * 50001.0) / 2.0;
    EXPECT_NEAR(perf_sum, expected, 1.0);
    std::cout << "  - Performance test correctness: PASSED" << std::endl;
}

TEST(ParallelExecutionComprehensive, MemoryAndExceptionSafety) {
    std::vector<int> empty;
    auto empty_count = stats::arch::safe_count(empty.begin(), empty.end(), 5);
    EXPECT_EQ(empty_count, 0);
    std::cout << "  - Empty container count: PASSED" << std::endl;

    std::vector<int*> ptr_vec(100, nullptr);
    auto null_count = stats::arch::safe_count(ptr_vec.begin(), ptr_vec.end(), nullptr);
    EXPECT_EQ(null_count, 100);
    std::cout << "  - Null pointer handling: PASSED" << std::endl;
}

TEST(ParallelExecutionComprehensive, PlatformAdaptiveFeatures) {
    auto threshold = stats::arch::get_min_elements_for_distribution_parallel();
    EXPECT_GT(threshold, 0u);
    EXPECT_LT(threshold, 100000u);

    auto default_grain  = stats::arch::get_default_grain_size();
    auto optimal_grain  = stats::arch::get_optimal_grain_size();
    auto simple_grain   = stats::arch::get_simple_operation_grain_size();
    EXPECT_GT(default_grain, 0u);  EXPECT_LT(default_grain, 50000u);
    EXPECT_GT(optimal_grain, 0u);  EXPECT_LT(optimal_grain, 50000u);
    EXPECT_GT(simple_grain,  0u);  EXPECT_LT(simple_grain,  50000u);

    auto memory_grain  = stats::arch::get_adaptive_grain_size(0, 10000);
    auto compute_grain = stats::arch::get_adaptive_grain_size(1, 10000);
    auto mixed_grain   = stats::arch::get_adaptive_grain_size(2, 10000);
    EXPECT_GE(memory_grain,  simple_grain);
    EXPECT_GE(compute_grain, simple_grain);
    EXPECT_GE(mixed_grain,   simple_grain);
    EXPECT_GE(compute_grain, memory_grain);

    auto threads_small = stats::arch::get_optimal_thread_count(1000);
    auto threads_large = stats::arch::get_optimal_thread_count(100000);
    EXPECT_GE(threads_small, 1u);
    EXPECT_GE(threads_large, 1u);
    EXPECT_GE(threads_large, threads_small);

    std::cout << "  - Platform-aware adaptive features: PASSED" << std::endl;
}

TEST(ParallelExecutionComprehensive, ExecutionPolicyDetection) {
    [[maybe_unused]] bool has_std_exec = stats::arch::has_execution_policies();
    std::cout << "  - Standard execution policies: " << (has_std_exec ? "YES" : "NO") << std::endl;
    std::cout << "  - Execution support: " << stats::arch::execution_support_string() << std::endl;
}

TEST(ParallelExecutionComprehensive, PlatformSpecificOptimizations) {
    for (auto size : {1000u, 10000u, 100000u}) {
        auto grain = stats::arch::get_adaptive_grain_size(0, size);
        EXPECT_GE(grain, 32u);
        std::cout << "  - Data size " << size << ": grain = " << grain << std::endl;
    }
    EXPECT_GE(stats::arch::get_adaptive_grain_size(0, 1000000), 64u);

    bool should_parallel_large = stats::arch::should_use_distribution_parallel(100000);
    if (stats::arch::has_execution_policies()) {
        EXPECT_TRUE(should_parallel_large);
    }
    std::cout << "  - Platform-specific optimization validation: PASSED" << std::endl;
}

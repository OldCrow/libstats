#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

// Test integration with main header
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <gtest/gtest.h>

TEST(ParallelExecutionIntegration, HeaderAndThresholds) {
    std::cout << "Header inclusion - "
              << (stats::arch::has_execution_policies() ? "C++20 parallel execution available"
                                                        : "Fallback to serial execution")
              << std::endl;

    std::size_t optimal_threshold = stats::arch::get_min_elements_for_distribution_parallel();
    std::size_t optimal_grain = stats::arch::get_optimal_grain_size();
    std::cout << "Threshold: " << optimal_threshold << ", Grain: " << optimal_grain << std::endl;
    EXPECT_GT(optimal_threshold, 0u);
    EXPECT_GT(optimal_grain, 0u);
}

TEST(ParallelExecutionIntegration, BasicAlgorithms) {
    // safe_fill
    std::vector<double> data(1000);
    stats::arch::safe_fill(data.begin(), data.end(), 42.0);
    EXPECT_TRUE(std::all_of(data.begin(), data.end(), [](double x) { return x == 42.0; }));
    std::cout << "  - safe_fill: PASSED" << std::endl;

    // safe_transform
    std::vector<double> input(1000);
    std::vector<double> output(1000);
    std::iota(input.begin(), input.end(), 1.0);
    stats::arch::safe_transform(input.begin(), input.end(), output.begin(),
                                [](double x) { return x * 2.0; });
    EXPECT_EQ(output[0], 2.0);
    EXPECT_EQ(output[999], 2000.0);
    std::cout << "  - safe_transform: PASSED" << std::endl;

    // safe_reduce
    std::vector<double> values(100);
    std::fill(values.begin(), values.end(), 1.0);
    double sum = stats::arch::safe_reduce(values.begin(), values.end(), 0.0);
    EXPECT_EQ(sum, 100.0);
    std::cout << "  - safe_reduce: PASSED" << std::endl;

    // safe_for_each
    std::vector<int> counters(10, 0);
    stats::arch::safe_for_each(counters.begin(), counters.end(), [](int& x) { x = 1; });
    EXPECT_TRUE(std::all_of(counters.begin(), counters.end(), [](int x) { return x == 1; }));
    std::cout << "  - safe_for_each: PASSED" << std::endl;

    // safe_sort
    std::vector<double> unsorted = {5.0, 2.0, 8.0, 1.0, 9.0, 3.0};
    stats::arch::safe_sort(unsorted.begin(), unsorted.end());
    EXPECT_TRUE(std::is_sorted(unsorted.begin(), unsorted.end()));
    std::cout << "  - safe_sort: PASSED" << std::endl;
}

TEST(ParallelExecutionIntegration, ThresholdDecisions) {
    [[maybe_unused]] bool should_use_small = stats::arch::should_use_parallel(10);
    [[maybe_unused]] bool should_use_large = stats::arch::should_use_parallel(10000);
    [[maybe_unused]] bool should_use_dist  = stats::arch::should_use_distribution_parallel(1000);
    std::cout << "  - Small (10): "     << (should_use_small ? "parallel" : "serial") << std::endl;
    std::cout << "  - Large (10000): "  << (should_use_large ? "parallel" : "serial") << std::endl;
    std::cout << "  - Dist (1000): "    << (should_use_dist  ? "parallel" : "serial") << std::endl;
    // All queries must complete without crash; no strict direction requirement.
}

TEST(ParallelExecutionIntegration, EmptyContainerSafety) {
    EXPECT_NO_THROW({
        std::vector<double> empty_data;
        stats::arch::safe_fill(empty_data.begin(), empty_data.end(), 1.0);
    });
    std::cout << "  - Empty container handling: PASSED" << std::endl;
}

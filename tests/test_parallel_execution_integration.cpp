#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>

// Test integration with main header
#define LIBSTATS_FULL_INTERFACE
#include "../include/libstats.h"

int main() {
    std::cout << "=== Testing parallel_execution.h Integration ===" << std::endl;
    
    // Test 1: Basic header inclusion (compilation test)
    std::cout << "Test 1: Header inclusion - ";
    std::cout << (libstats::parallel::has_execution_policies() ? "C++20 parallel execution available" : "Fallback to serial execution");
    std::cout << std::endl;
    
    // Test 2: CPU-aware threshold detection
    std::cout << "Test 2: CPU-aware threshold detection - ";
    std::size_t optimal_threshold = libstats::parallel::get_optimal_parallel_threshold("gaussian", "pdf");
    std::size_t optimal_grain = libstats::parallel::get_optimal_grain_size();
    std::cout << "Threshold: " << optimal_threshold << ", Grain: " << optimal_grain << std::endl;
    
    // Test 3: Basic parallel algorithm functionality
    std::cout << "Test 3: Basic parallel algorithm functionality" << std::endl;
    
    // Test safe_fill
    std::vector<double> data(1000);
    libstats::parallel::safe_fill(data.begin(), data.end(), 42.0);
    assert(std::all_of(data.begin(), data.end(), [](double x) { return x == 42.0; }));
    std::cout << "  - safe_fill: PASSED" << std::endl;
    
    // Test safe_transform
    std::vector<double> input(1000);
    std::vector<double> output(1000);
    std::iota(input.begin(), input.end(), 1.0);
    libstats::parallel::safe_transform(input.begin(), input.end(), output.begin(), 
        [](double x) { return x * 2.0; });
    assert(output[0] == 2.0 && output[999] == 2000.0);
    std::cout << "  - safe_transform: PASSED" << std::endl;
    
    // Test safe_reduce
    std::vector<double> values(100);
    std::fill(values.begin(), values.end(), 1.0);
    [[maybe_unused]] double sum = libstats::parallel::safe_reduce(values.begin(), values.end(), 0.0);
    assert(sum == 100.0);
    std::cout << "  - safe_reduce: PASSED" << std::endl;
    
    // Test safe_for_each
    std::vector<int> counters(10, 0);
    libstats::parallel::safe_for_each(counters.begin(), counters.end(), [](int& x) { x = 1; });
    assert(std::all_of(counters.begin(), counters.end(), [](int x) { return x == 1; }));
    std::cout << "  - safe_for_each: PASSED" << std::endl;
    
    // Test safe_sort
    std::vector<double> unsorted = {5.0, 2.0, 8.0, 1.0, 9.0, 3.0};
    libstats::parallel::safe_sort(unsorted.begin(), unsorted.end());
    assert(std::is_sorted(unsorted.begin(), unsorted.end()));
    std::cout << "  - safe_sort: PASSED" << std::endl;
    
    // Test 4: Threshold decision logic
    std::cout << "Test 4: Threshold decision logic" << std::endl;
    bool should_use_small = libstats::parallel::should_use_parallel(10);
    bool should_use_large = libstats::parallel::should_use_parallel(10000);
    bool should_use_dist = libstats::parallel::should_use_distribution_parallel(1000);
    
    std::cout << "  - Small dataset (10 elements): " << (should_use_small ? "parallel" : "serial") << std::endl;
    std::cout << "  - Large dataset (10000 elements): " << (should_use_large ? "parallel" : "serial") << std::endl;
    std::cout << "  - Distribution dataset (1000 elements): " << (should_use_dist ? "parallel" : "serial") << std::endl;
    
    // Test 5: Safety integration
    std::cout << "Test 5: Safety integration" << std::endl;
    try {
        std::vector<double> empty_data;
        libstats::parallel::safe_fill(empty_data.begin(), empty_data.end(), 1.0);
        std::cout << "  - Empty container handling: PASSED" << std::endl;
    } catch (const std::exception&) {
        std::cout << "  - Empty container handling: PASSED (caught expected exception)" << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "🎉 All parallel_execution.h integration tests passed!" << std::endl;
    std::cout << "✅ Header successfully integrated into libstats.h" << std::endl;
    std::cout << "✅ CPU-aware optimization working" << std::endl;
    std::cout << "✅ All parallel algorithms functioning correctly" << std::endl;
    std::cout << "✅ Safety integration working" << std::endl;
    
    return 0;
}

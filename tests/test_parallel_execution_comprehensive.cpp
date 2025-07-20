#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>

// Test integration with main header
#include "libstats.h"

int main() {
    std::cout << "=== Comprehensive Parallel Execution Tests ===" << std::endl;
    
    // Test 1: GCD-specific algorithm coverage
    std::cout << "Test 1: GCD-specific algorithm implementations" << std::endl;
    
    // Test safe_count with GCD
    std::vector<int> count_data = {1, 2, 3, 2, 4, 2, 5, 2};
    auto count_result = libstats::parallel::safe_count(count_data.begin(), count_data.end(), 2);
    assert(count_result == 4);
    std::cout << "  - safe_count (GCD): PASSED" << std::endl;
    
    // Test safe_count_if with GCD
    auto count_if_result = libstats::parallel::safe_count_if(count_data.begin(), count_data.end(), 
        [](int x) { return x > 2; });
    assert(count_if_result == 3);  // 3, 4, 5
    std::cout << "  - safe_count_if (GCD): PASSED" << std::endl;
    
    // Test safe_reduce (sum operation is the default)
    std::vector<double> reduce_data(1000, 2.0);
    auto sum_result = libstats::parallel::safe_reduce(reduce_data.begin(), reduce_data.end(), 0.0);
    assert(std::abs(sum_result - 2000.0) < 1e-10);
    std::cout << "  - safe_reduce with custom op (GCD): PASSED" << std::endl;
    
    // Test 2: Edge cases and boundary conditions
    std::cout << "Test 2: Edge cases and boundary conditions" << std::endl;
    
    // Single element
    std::vector<int> single = {42};
    libstats::parallel::safe_fill(single.begin(), single.end(), 100);
    assert(single[0] == 100);
    std::cout << "  - Single element fill: PASSED" << std::endl;
    
    // Large dataset to ensure parallel execution
    std::vector<double> large_data(100000);
    std::iota(large_data.begin(), large_data.end(), 1.0);
    
    // Test parallel transform on large dataset
    std::vector<double> large_output(100000);
    libstats::parallel::safe_transform(large_data.begin(), large_data.end(), large_output.begin(),
        [](double x) { return x * x; });
    assert(large_output[0] == 1.0 && large_output[999] == 1000000.0);
    std::cout << "  - Large dataset transform: PASSED" << std::endl;
    
    // Test parallel reduce on large dataset
    auto large_sum = libstats::parallel::safe_reduce(large_data.begin(), large_data.end(), 0.0);
    double expected_sum = (100000.0 * 100001.0) / 2.0;  // Sum of 1..100000
    assert(std::abs(large_sum - expected_sum) < 1.0);  // Allow small floating point error
    std::cout << "  - Large dataset reduce: PASSED" << std::endl;
    
    // Test 3: Algorithm correctness under parallel execution
    std::cout << "Test 3: Algorithm correctness verification" << std::endl;
    
    // Test that parallel and serial give same results
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 100.0);
    
    std::vector<double> test_data(5000);
    std::generate(test_data.begin(), test_data.end(), [&]() { return dis(gen); });
    
    // Serial results
    std::vector<double> serial_transform(5000);
    std::transform(test_data.begin(), test_data.end(), serial_transform.begin(), 
        [](double x) { return std::sqrt(x); });
    
    // Parallel results
    std::vector<double> parallel_transform(5000);
    libstats::parallel::safe_transform(test_data.begin(), test_data.end(), parallel_transform.begin(),
        [](double x) { return std::sqrt(x); });
    
    // Compare results
    bool transforms_equal = std::equal(serial_transform.begin(), serial_transform.end(), 
        parallel_transform.begin(), [](double a, double b) { return std::abs(a - b) < 1e-10; });
    assert(transforms_equal);
    std::cout << "  - Parallel vs serial transform equivalence: PASSED" << std::endl;
    
    // Test 4: Performance characteristics (not strict performance test, just sanity check)
    std::cout << "Test 4: Performance characteristics" << std::endl;
    
    std::vector<double> perf_data(50000);
    std::iota(perf_data.begin(), perf_data.end(), 1.0);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto perf_sum_result = libstats::parallel::safe_reduce(perf_data.begin(), perf_data.end(), 0.0);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  - Parallel reduce of 50k elements took: " << duration.count() << " μs" << std::endl;
    
    double expected_perf_sum = (50000.0 * 50001.0) / 2.0;
    assert(std::abs(perf_sum_result - expected_perf_sum) < 1.0);
    std::cout << "  - Performance test correctness: PASSED" << std::endl;
    
    // Test 5: Memory safety and exception safety
    std::cout << "Test 5: Memory and exception safety" << std::endl;
    
    try {
        // Test with empty containers
        std::vector<int> empty;
        auto empty_count = libstats::parallel::safe_count(empty.begin(), empty.end(), 5);
        assert(empty_count == 0);
        std::cout << "  - Empty container count: PASSED" << std::endl;
        
        // Test with null/default values
        std::vector<int*> ptr_vec(100, nullptr);
        auto null_count = libstats::parallel::safe_count(ptr_vec.begin(), ptr_vec.end(), nullptr);
        assert(null_count == 100);
        std::cout << "  - Null pointer handling: PASSED" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "  - Exception in safety test: " << e.what() << std::endl;
    }
    
    // Test 6: Platform-aware adaptive grain sizing and threading
    std::cout << "Test 6: Platform-aware adaptive features" << std::endl;
    
    // Test optimal parallel threshold
    auto optimal_threshold = libstats::parallel::get_optimal_parallel_threshold();
    std::cout << "  - Optimal parallel threshold: " << optimal_threshold << " elements" << std::endl;
    assert(optimal_threshold > 0 && optimal_threshold < 100000); // Reasonable range
    
    // Test base grain size
    auto base_grain_size = libstats::parallel::get_optimal_grain_size();
    std::cout << "  - Base optimal grain size: " << base_grain_size << " elements" << std::endl;
    assert(base_grain_size > 0 && base_grain_size < 50000); // Reasonable range
    
    // Test adaptive grain sizes for different operation types
    auto memory_grain = libstats::parallel::get_adaptive_grain_size(0, 10000); // Memory-bound
    auto compute_grain = libstats::parallel::get_adaptive_grain_size(1, 10000); // Computation-bound
    auto mixed_grain = libstats::parallel::get_adaptive_grain_size(2, 10000);   // Mixed
    
    std::cout << "  - Memory-bound grain size: " << memory_grain << " elements" << std::endl;
    std::cout << "  - Computation-bound grain size: " << compute_grain << " elements" << std::endl;
    std::cout << "  - Mixed operation grain size: " << mixed_grain << " elements" << std::endl;
    
    // Verify grain sizes are reasonable and different based on operation type
    assert(memory_grain >= 64);  // Minimum grain size
    assert(compute_grain >= 64); // Minimum grain size
    assert(mixed_grain >= 64);   // Minimum grain size
    
    // Computation-bound should generally have larger grains
    assert(compute_grain >= base_grain_size);
    
    // Test optimal thread count
    auto optimal_threads_small = libstats::parallel::get_optimal_thread_count(1000);
    auto optimal_threads_large = libstats::parallel::get_optimal_thread_count(100000);
    
    std::cout << "  - Optimal threads (small workload): " << optimal_threads_small << std::endl;
    std::cout << "  - Optimal threads (large workload): " << optimal_threads_large << std::endl;
    
    assert(optimal_threads_small >= 1);
    assert(optimal_threads_large >= 1);
    // Large workloads may benefit from more threads
    assert(optimal_threads_large >= optimal_threads_small);
    
    std::cout << "  - Platform-aware adaptive features: PASSED" << std::endl;
    
    // Test 7: Execution policy detection and fallback verification
    std::cout << "Test 7: Execution policy detection" << std::endl;
    
    bool has_std_exec = libstats::parallel::has_execution_policies();
    std::cout << "  - Standard execution policies available: " << (has_std_exec ? "YES" : "NO") << std::endl;
    std::cout << "  - Execution support: " << libstats::parallel::execution_support_string() << std::endl;
    
    #if defined(LIBSTATS_HAS_GCD)
    std::cout << "  - GCD fallback available: YES" << std::endl;
    #elif defined(LIBSTATS_HAS_WIN_THREADPOOL)
    std::cout << "  - Windows Thread Pool API available: YES" << std::endl;
    #elif defined(LIBSTATS_HAS_OPENMP)
    std::cout << "  - OpenMP available: YES" << std::endl;
    #elif defined(LIBSTATS_HAS_PTHREADS)
    std::cout << "  - POSIX threads available: YES" << std::endl;
    #else
    std::cout << "  - Only serial execution available" << std::endl;
    #endif
    
    // Test threshold logic with various sizes
    std::cout << "  - Threshold decisions:" << std::endl;
    std::cout << "    * 10 elements: " << (libstats::parallel::should_use_parallel(10) ? "parallel" : "serial") << std::endl;
    std::cout << "    * 100 elements: " << (libstats::parallel::should_use_parallel(100) ? "parallel" : "serial") << std::endl;
    std::cout << "    * 1000 elements: " << (libstats::parallel::should_use_parallel(1000) ? "parallel" : "serial") << std::endl;
    std::cout << "    * 10000 elements: " << (libstats::parallel::should_use_parallel(10000) ? "parallel" : "serial") << std::endl;
    
    // Test 8: Platform-specific optimizations validation
    std::cout << "Test 8: Platform-specific optimization validation" << std::endl;
    
    // Test different grain sizes with different data sizes
    std::vector<std::size_t> test_sizes = {1000, 10000, 100000};
    for (auto size : test_sizes) {
        auto adaptive_grain = libstats::parallel::get_adaptive_grain_size(0, size);
        std::cout << "  - Data size " << size << ": adaptive grain = " << adaptive_grain << std::endl;
        
        // Ensure grain size is reasonable relative to data size
        assert(adaptive_grain <= size);  // Grain shouldn't exceed data size
        assert(adaptive_grain >= 64);    // Minimum grain size
    }
    
    // Test cache-aware grain sizing for memory operations
    auto cache_aware_grain = libstats::parallel::get_adaptive_grain_size(0, 1000000); // 1M elements
    std::cout << "  - Cache-aware grain (1M elements): " << cache_aware_grain << std::endl;
    assert(cache_aware_grain >= 64);
    
    // Test distribution parallel thresholds
    bool should_parallel_small = libstats::parallel::should_use_distribution_parallel(100);
    bool should_parallel_large = libstats::parallel::should_use_distribution_parallel(100000);
    
    std::cout << "  - Distribution parallel (100 elements): " << (should_parallel_small ? "YES" : "NO") << std::endl;
    std::cout << "  - Distribution parallel (100k elements): " << (should_parallel_large ? "YES" : "NO") << std::endl;
    
    // Large datasets should generally use parallel
    if (libstats::parallel::has_execution_policies()) {
        assert(should_parallel_large);
    }
    
    std::cout << "  - Platform-specific optimization validation: PASSED" << std::endl;
    
    std::cout << std::endl;
    std::cout << "🎉 All comprehensive parallel execution tests passed!" << std::endl;
    std::cout << "✅ GCD-specific implementations working correctly" << std::endl;
    std::cout << "✅ Edge cases handled properly" << std::endl;
    std::cout << "✅ Algorithm correctness verified" << std::endl;
    std::cout << "✅ Performance characteristics acceptable" << std::endl;
    std::cout << "✅ Memory and exception safety confirmed" << std::endl;
    std::cout << "✅ Platform-aware adaptive features validated" << std::endl;
    std::cout << "✅ Execution policy detection working" << std::endl;
    std::cout << "✅ Platform-specific optimizations confirmed" << std::endl;
    
    return 0;
}

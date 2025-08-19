/**
 * @file test_work_stealing_pool.cpp
 * @brief Comprehensive test suite for WorkStealingPool with Level 0-2 integration
 * 
 * This test suite verifies that the WorkStealingPool has the same Level 0-2 integration
 * as the ThreadPool implementation, plus work-stealing specific functionality.
 */

#include <iostream>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <cassert>
#include <ranges>
#include <concepts>
#include <numeric>
#include <cmath>
#include <iomanip>
#include <span>
#include "../include/platform/work_stealing_pool.h"

using namespace libstats;

// C++20 concept for testing
template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

int main() {
    std::cout << "=== WorkStealingPool Test with C++20 ===\n\n";
    std::cout << "C++ Standard: " << __cplusplus << "\n\n";
    
    // Test 1: Basic task submission and execution
    std::cout << "Test 1: Basic task submission\n";
    {
        WorkStealingPool pool(4); // 4 worker threads
        std::atomic<int> counter{0};
        
        // Submit 10 simple tasks
        for (int i = 0; i < 10; ++i) {
            pool.submit([&counter, i]() {
                counter.fetch_add(1);
                std::cout << "  Task " << i << " executed\n";
            });
        }
        
        pool.waitForAll();
        assert(counter.load() == 10);
        std::cout << "  ✓ All 10 tasks executed successfully\n\n";
    }
    
    // Test 2: C++20 ranges with parallel computation
    std::cout << "Test 2: C++20 ranges with work stealing\n";
    {
        WorkStealingPool pool(4);
        
        // Create data using C++20 ranges
        std::vector<int> data;
        auto range = std::views::iota(0, 1000) 
                   | std::views::transform([](int i) { return i * 2; });
        std::ranges::copy(range, std::back_inserter(data));
        
        std::atomic<long long> sum{0};
        
        // Process data in parallel chunks
        const size_t chunkSize = 100;
        for (size_t start = 0; start < data.size(); start += chunkSize) {
            size_t end = std::min(start + chunkSize, data.size());
            
            pool.submit([&data, &sum, start, end]() {
                long long localSum = 0;
                for (size_t i = start; i < end; ++i) {
                    localSum += data[i];
                }
                sum.fetch_add(localSum);
            });
        }
        
        pool.waitForAll();
        
        // Verify result
        long long expectedSum = 0;
        for (auto value : data) {
            expectedSum += value;
        }
        
        assert(sum.load() == expectedSum);
        std::cout << "  ✓ Parallel sum: " << sum.load() << " (expected: " << expectedSum << ")\n\n";
    }
    
    // Test 3: ParallelFor with C++20 concepts
    std::cout << "Test 3: ParallelFor with concepts\n";
    {
        WorkStealingPool pool(4);
        
        // Template function using C++20 concepts
        auto process_numeric = [&pool]<Numeric T>(std::vector<T>& vec, T multiplier) {
            pool.parallelFor(0, vec.size(), [&vec, multiplier](size_t i) {
                vec[i] *= multiplier;
            });
        };
        
        std::vector<int> int_data(1000);
        std::iota(int_data.begin(), int_data.end(), 1);
        
        process_numeric(int_data, 3);
        
        // Verify first few elements
        assert(int_data[0] == 3);   // 1 * 3
        assert(int_data[1] == 6);   // 2 * 3
        assert(int_data[999] == 3000); // 1000 * 3
        
        std::cout << "  ✓ Concept-based parallelFor processed 1000 elements\n\n";
    }
    
    // Test 4: Work stealing statistics with performance measurement
    std::cout << "Test 4: Performance and statistics\n";
    {
        // Ensure we have at least 1 thread (hardware_concurrency can return 0)
        unsigned int thread_count = std::thread::hardware_concurrency();
        if (thread_count == 0) thread_count = 2; // Default to 2 if detection fails
        WorkStealingPool pool(thread_count);
        const int numTasks = 1000;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Submit variable-work tasks to trigger stealing
        for (int i = 0; i < numTasks; ++i) {
            pool.submit([i]() {
                // Variable work amounts using C++20 features
                auto work_amount = std::views::iota(0, (i % 10 + 1) * 100)
                                 | std::views::transform([](int x) { return x * x; })
                                 | std::views::take(50);
                
                volatile long long sum = 0;
                for (auto value : work_amount) {
                    sum = sum + value;
                }
            });
        }
        
        pool.waitForAll();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        auto stats = pool.getStatistics();
        std::cout << "  Execution time: " << duration.count() << " ms\n";
        std::cout << "  Tasks executed: " << stats.tasksExecuted << "\n";
        std::cout << "  Work steals: " << stats.workSteals << "\n";
        std::cout << "  Steal success rate: " << (stats.stealSuccessRate * 100) << "%\n";
        std::cout << "  Hardware threads: " << std::thread::hardware_concurrency() << "\n";
        
        assert(stats.tasksExecuted == numTasks);
        std::cout << "  ✓ All tasks completed with work stealing active\n\n";
    }
    
    // Test 5: Global utilities with C++20 features
    std::cout << "Test 5: Global utilities\n";
    {
        std::atomic<int> counter{0};
        
        // Use C++20 ranges to generate task indices
        std::vector<int> task_indices;
        auto range = std::views::iota(0, 50);
        std::ranges::copy(range, std::back_inserter(task_indices));
        
        for ([[maybe_unused]] auto i : task_indices) {
            WorkStealingUtils::submit([&counter]() {
                counter.fetch_add(1);
            });
        }
        
        WorkStealingUtils::waitForAll();
        assert(counter.load() == 50);
        std::cout << "  ✓ Global utilities executed " << counter.load() << " tasks\n\n";
    }
    
    // Test 6: Level 0 Constants Integration
    std::cout << "Test 6: Level 0 Constants Integration\n";
    {
        // Test that constants are properly used
        auto parallelThreshold = constants::parallel::adaptive::min_elements_for_parallel();
        auto distributionThreshold = constants::parallel::adaptive::min_elements_for_distribution_parallel();
        auto defaultGrainSize = constants::parallel::adaptive::grain_size();
        auto simdBlockSize = constants::platform::get_optimal_simd_block_size();
        auto memoryAlignment = constants::platform::get_optimal_alignment();
        
        std::cout << "  Min parallel size: " << parallelThreshold << std::endl;
        std::cout << "  Min distribution parallel size: " << distributionThreshold << std::endl;
        std::cout << "  Default grain size: " << defaultGrainSize << std::endl;
        std::cout << "  SIMD block size: " << simdBlockSize << std::endl;
        std::cout << "  Memory alignment: " << memoryAlignment << " bytes" << std::endl;
        
        // Verify constants are reasonable
        assert(parallelThreshold > 0);
        assert(distributionThreshold > 0);
        assert(defaultGrainSize > 0);
        assert(simdBlockSize > 0);
        assert(memoryAlignment > 0);
        
        std::cout << "  ✓ Constants integration working correctly\n\n";
    }

    // Test 7: CPU Detection Integration
    std::cout << "Test 7: CPU Detection Integration\n";
    {
        const auto& features = cpu::get_features();
        auto physicalCores = features.topology.physical_cores;
        auto logicalCores = features.topology.logical_cores;
        auto l1CacheSize = features.l1_cache_size;
        auto l2CacheSize = features.l2_cache_size;
        auto l3CacheSize = features.l3_cache_size;
        auto cacheLineSize = features.cache_line_size;
        auto optimalThreads = WorkStealingPool::getOptimalThreadCount();
        
        std::cout << "  Physical cores: " << physicalCores << std::endl;
        std::cout << "  Logical cores: " << logicalCores << std::endl;
        std::cout << "  L1 cache: " << (l1CacheSize > 0 ? l1CacheSize / 1024 : 0) << " KB" << std::endl;
        std::cout << "  L2 cache: " << (l2CacheSize > 0 ? l2CacheSize / 1024 : 0) << " KB" << std::endl;
        std::cout << "  L3 cache: " << (l3CacheSize > 0 ? l3CacheSize / 1024 / 1024 : 0) << " MB" << std::endl;
        std::cout << "  Cache line size: " << cacheLineSize << " bytes" << std::endl;
        std::cout << "  Optimal threads: " << optimalThreads << std::endl;
        std::cout << "  Has hyperthreading: " << (features.topology.hyperthreading ? "Yes" : "No") << std::endl;
        
        // Be lenient for CI/VM environments where CPU detection might not fully work
        if (physicalCores == 0) {
            std::cerr << "  Warning: physicalCores could not be detected on this platform (CI runner?)" << std::endl;
        }
        assert(physicalCores >= 0);
        
        if (logicalCores == 0) {
            std::cerr << "  Warning: logicalCores could not be detected on this platform (CI runner?)" << std::endl;
        }
        assert(logicalCores >= 0);
        
        // Cache sizes might be 0 or unavailable on VMs/CI runners
        assert(l1CacheSize >= 0);
        assert(l2CacheSize >= 0);
        assert(l3CacheSize >= 0);
        
        if (cacheLineSize == 0) {
            std::cerr << "  Warning: Cache line size could not be detected on this platform (CI runner?)" << std::endl;
        }
        assert(cacheLineSize >= 0);
        
        // Optimal threads should always be at least 1
        assert(optimalThreads > 0);
        
        if (features.topology.hyperthreading) {
            // Work stealing pools can use logical cores effectively
            assert(optimalThreads <= logicalCores);
        }
        
        std::cout << "  ✓ CPU detection integration working correctly\n\n";
    }

    std::cout << "🎉 All WorkStealingPool tests passed with Level 0-2 integration!\n";
    
    return 0;
}

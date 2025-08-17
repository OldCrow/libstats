#include "../include/platform/cache_platform.h"  // For cache functionality
#include <iostream>
#include <chrono>
#include <thread>

using namespace libstats::cache;

int main() {
    std::cout << "=== Adaptive Cache Grain Size Diagnostic Tool ===\n\n";
    
    // Create cache with default configuration
    AdaptiveCache<std::string, double> cache;
    
    // Test scenarios that were causing performance issues
    std::vector<size_t> test_sizes = {50, 500, 5000, 50000};
    std::vector<std::string> operation_types = {"poisson_pdf", "poisson_logpdf", "poisson_cdf"};
    
    std::cout << "Testing getOptimalGrainSize() method:\n";
    std::cout << "Size\t\tOperation\t\tGrain Size\tExpected Threads\n";
    std::cout << "----------------------------------------------------------------\n";
    
    for (size_t data_size : test_sizes) {
        for (const std::string& op_type : operation_types) {
            // Get grain size from current implementation
            size_t grain_size = cache.getOptimalGrainSize(data_size, op_type);
            size_t expected_threads = (grain_size > 0) ? (data_size + grain_size - 1) / grain_size : data_size;
            
            std::cout << data_size << "\t\t" << op_type << "\t\t" 
                      << grain_size << "\t\t" << expected_threads << "\n";
        }
        std::cout << "\n";
    }
    
    // Test with different cache states to simulate the original issue
    std::cout << "\n=== Testing with different cache hit rates ===\n";
    
    // Create cache with some artificial metrics to simulate poor performance
    AdaptiveCache<std::string, double> cache_with_poor_hit_rate;
    
    // Artificially set poor hit rate by forcing some cache operations
    // First put some values
    for (int i = 0; i < 50; ++i) {
        cache_with_poor_hit_rate.put("test_key_" + std::to_string(i), 42.0);
    }
    
    // Now create hits and many misses to simulate poor hit rate
    for (int i = 0; i < 100; ++i) {
        // Some hits (existing keys)
        if (i < 50) {
            cache_with_poor_hit_rate.get("test_key_" + std::to_string(i));
        }
        // Many misses (non-existent keys)
        cache_with_poor_hit_rate.get("unique_miss_key_" + std::to_string(i));
    }
    
    auto metrics = cache_with_poor_hit_rate.getMetrics();
    std::cout << "Cache hit rate: " << metrics.hit_rate.load() << "\n\n";
    
    std::cout << "Grain sizes with poor hit rate:\n";
    std::cout << "Size\t\tOperation\t\tGrain Size\tExpected Threads\n";
    std::cout << "----------------------------------------------------------------\n";
    
    for (size_t data_size : test_sizes) {
        for (const std::string& op_type : operation_types) {
            size_t grain_size = cache_with_poor_hit_rate.getOptimalGrainSize(data_size, op_type);
            size_t expected_threads = (grain_size > 0) ? (data_size + grain_size - 1) / grain_size : data_size;
            
            std::cout << data_size << "\t\t" << op_type << "\t\t" 
                      << grain_size << "\t\t" << expected_threads << "\n";
        }
        std::cout << "\n";
    }
    
    // Test the specific case mentioned in the roadmap (50K elements)
    std::cout << "\n=== CRITICAL TEST: 50K elements scenario ===\n";
    size_t critical_size = 50000;
    std::string critical_op = "poisson_pdf";
    
    size_t critical_grain_size = cache.getOptimalGrainSize(critical_size, critical_op);
    size_t critical_threads = (critical_grain_size > 0) ? (critical_size + critical_grain_size - 1) / critical_grain_size : critical_size;
    
    std::cout << "Data size: " << critical_size << "\n";
    std::cout << "Operation: " << critical_op << "\n";
    std::cout << "Calculated grain size: " << critical_grain_size << "\n";
    std::cout << "Expected thread count: " << critical_threads << "\n";
    
    if (critical_threads > 1000) {
        std::cout << "🚨 ISSUE DETECTED: Thread count exceeds reasonable limit!\n";
        std::cout << "This would cause significant performance degradation.\n";
    } else {
        std::cout << "✅ Thread count appears reasonable.\n";
    }
    
    // Test time to see if there are other performance bottlenecks
    std::cout << "\n=== Performance timing test ===\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        cache.getOptimalGrainSize(50000, "poisson_pdf");
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time for 1000 getOptimalGrainSize() calls: " << duration.count() << "μs\n";
    std::cout << "Average time per call: " << (duration.count() / 1000.0) << "μs\n";
    
    if (duration.count() > 50000) { // > 50ms total
        std::cout << "🚨 PERFORMANCE ISSUE: getOptimalGrainSize() is too slow!\n";
    } else {
        std::cout << "✅ getOptimalGrainSize() performance appears acceptable.\n";
    }
    
    return 0;
}

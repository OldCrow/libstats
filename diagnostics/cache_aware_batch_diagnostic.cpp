#include "../include/cache/adaptive_cache.h"
#include "../include/platform/thread_pool.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <span>
#include <thread>
#include <cmath>

using namespace libstats::cache;

// Simulate the problematic cache-aware batch operation from v0.8.3
void simulateCacheAwarePoissonPMFBatch(
    std::span<const double> input_values,
    std::span<double> output_results,
    AdaptiveCache<std::string, double>& cache_manager) {
    
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const size_t count = input_values.size();
    if (count == 0) return;
    
    // Replicate the cache key generation from v0.8.3
    const std::string cache_key = "poisson_pdf_batch_" + std::to_string(count);
    
    // Replicate the grain size calculation
    const size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, "poisson_pdf");
    
    std::cout << "  Cache key: " << cache_key << "\n";
    std::cout << "  Optimal grain size: " << optimal_grain_size << "\n";
    std::cout << "  Expected threads: " << (count + optimal_grain_size - 1) / optimal_grain_size << "\n";
    
    // Simulate the parallel processing with cache operations that might be the issue
    auto start = std::chrono::high_resolution_clock::now();
    
    // This was the pattern in v0.8.3 - parallel processing with custom grain size
    libstats::ParallelUtils::parallelFor(size_t{0}, count, [&](size_t i) {
        // Simulate Poisson PMF computation with potential cache operations
        // This might be where the performance issue was hiding
        
        double k = input_values[i];
        if (k < 0.0) {
            output_results[i] = 0.0;
            return;
        }
        
        // The problem might be here - excessive cache lookups during parallel execution
        std::string element_key = "pmf_" + std::to_string(i) + "_" + std::to_string(k);
        
        auto cached_result = cache_manager.get(element_key);
        if (cached_result.has_value()) {
            output_results[i] = cached_result.value();
            return;
        }
        
        // Simulate expensive computation (actual Poisson PMF calculation)
        double lambda = 3.5;
        int int_k = static_cast<int>(std::round(k));
        
        double result;
        if (int_k == 0) {
            result = std::exp(-lambda);
        } else {
            // Simulate log-space computation
            double log_factorial = 0.0;
            for (int j = 1; j <= int_k; ++j) {
                log_factorial += std::log(j);
            }
            double log_pmf = int_k * std::log(lambda) - lambda - log_factorial;
            result = std::exp(log_pmf);
        }
        
        // Cache the computed result - THIS MIGHT BE THE BOTTLENECK
        cache_manager.put(element_key, result);
        output_results[i] = result;
        
    }, optimal_grain_size);  // Use the adaptive grain size from cache manager
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Batch processing time: " << duration.count() << "μs\n";
    
    // Record performance metrics like v0.8.3 did
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

int main() {
    std::cout << "=== Cache-Aware Batch Performance Diagnostic ===\n\n";
    
    // Test progressively larger sizes to identify where performance degrades
    std::vector<size_t> test_sizes = {100, 500, 1000, 5000, 10000, 25000};
    
    AdaptiveCache<std::string, double> cache;
    
    for (size_t test_size : test_sizes) {
        std::cout << "\n=== Testing batch size: " << test_size << " ===\n";
        
        std::vector<double> test_values(test_size);
        std::vector<double> results(test_size);
        
        // Generate test data (discrete values for Poisson)
        for (size_t i = 0; i < test_size; ++i) {
            test_values[i] = static_cast<double>(i % 15);  // Values 0-14
        }
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Warm up run
        std::cout << "--- Warm-up run ---\n";
        simulateCacheAwarePoissonPMFBatch(
            std::span<const double>(test_values), 
            std::span<double>(results), 
            cache
        );
        
        // Performance run (this should hit cache more)
        std::cout << "--- Performance run ---\n";
        auto overall_start = std::chrono::high_resolution_clock::now();
        
        simulateCacheAwarePoissonPMFBatch(
            std::span<const double>(test_values), 
            std::span<double>(results), 
            cache
        );
        
        auto overall_end = std::chrono::high_resolution_clock::now();
        auto overall_duration = std::chrono::duration_cast<std::chrono::microseconds>(overall_end - overall_start);
        
        std::cout << "Performance run time: " << overall_duration.count() << "μs\n";
        
        // Performance analysis for this batch size
        if (overall_duration.count() > 50000) {  // > 50ms
            std::cout << "🚨 PERFORMANCE ISSUE DETECTED at batch size " << test_size << "!\n";
            std::cout << "Time: " << overall_duration.count() << "μs (exceeds reasonable threshold)\n";
            std::cout << "This matches the reported issue from the roadmap.\n";
            
            // If we find the issue, we can break early or continue to see how bad it gets
            if (overall_duration.count() > 100000) {  // > 100ms
                std::cout << "Performance is severely degraded. Stopping further size tests.\n";
                break;
            }
        } else {
            std::cout << "✅ Performance appears acceptable for batch size " << test_size << "\n";
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
        std::cout << "Total time for batch size " << test_size << ": " << total_duration.count() << "μs\n";
    }
    
    // Final cache metrics
    auto metrics = cache.getMetrics();
    std::cout << "\n=== Final Cache Metrics ===\n";
    std::cout << "Hits: " << metrics.hits.load() << "\n";
    std::cout << "Misses: " << metrics.misses.load() << "\n";
    std::cout << "Hit rate: " << metrics.hit_rate.load() << "\n";
    std::cout << "Cache size: " << metrics.cache_size.load() << "\n";
    std::cout << "Memory usage: " << metrics.memory_usage.load() << " bytes\n";
    
    std::cout << "\n=== Diagnostic Summary ===\n";
    std::cout << "This diagnostic tested progressive batch sizes to identify where\n";
    std::cout << "the cache-aware performance regression occurs. The v0.8.3 issue\n";
    std::cout << "showed >100x slowdown around 50K elements (102,369μs).\n";
    std::cout << "\nIf no severe performance degradation was observed, the issue\n";
    std::cout << "might be in different code paths or interaction patterns.\n";
    
    return 0;
}

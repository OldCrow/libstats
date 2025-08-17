#include "../include/cache/adaptive_cache.h"
#include "../include/platform/thread_pool.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <span>
#include <thread>
#include <cmath>
#include <random>

using namespace libstats::cache;

// Simulate cache-aware batch operation with continuous distributions
void simulateCacheAwareContinuousBatch(
    std::span<const double> input_values,
    std::span<double> output_results,
    AdaptiveCache<std::string, double>& cache_manager,
    const std::string& distribution_type) {
    
    if (input_values.size() != output_results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const size_t count = input_values.size();
    if (count == 0) return;
    
    // Replicate the cache key generation from v0.8.3
    const std::string cache_key = distribution_type + "_pdf_batch_" + std::to_string(count);
    
    // Replicate the grain size calculation
    const size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, distribution_type + "_pdf");
    
    std::cout << "  Cache key: " << cache_key << "\n";
    std::cout << "  Optimal grain size: " << optimal_grain_size << "\n";
    std::cout << "  Expected threads: " << (count + optimal_grain_size - 1) / optimal_grain_size << "\n";
    
    // Simulate the parallel processing with cache operations
    auto start = std::chrono::high_resolution_clock::now();
    
    // This was the pattern in v0.8.3 - parallel processing with custom grain size
    libstats::ParallelUtils::parallelFor(size_t{0}, count, [&](size_t i) {
        double x = input_values[i];
        
        // Create unique cache keys for continuous values (very low hit rate expected)
        std::string element_key = distribution_type + "_pdf_" + std::to_string(i) + "_" + std::to_string(x);
        
        auto cached_result = cache_manager.get(element_key);
        if (cached_result.has_value()) {
            output_results[i] = cached_result.value();
            return;
        }
        
        double result = 0.0;
        
        if (distribution_type == "gaussian") {
            // Standard normal PDF computation
            const double mu = 0.0;
            const double sigma = 1.0;
            const double sqrt_2pi = std::sqrt(2.0 * M_PI);
            
            double z = (x - mu) / sigma;
            result = std::exp(-0.5 * z * z) / (sigma * sqrt_2pi);
            
        } else if (distribution_type == "exponential") {
            // Exponential PDF computation
            const double lambda = 1.0;  // rate parameter
            
            if (x >= 0.0) {
                result = lambda * std::exp(-lambda * x);
            } else {
                result = 0.0;  // Exponential is zero for x < 0
            }
        }
        
        // Cache the computed result - THIS IS WHERE CONTENTION OCCURS
        cache_manager.put(element_key, result);
        output_results[i] = result;
        
    }, optimal_grain_size);  // Use the adaptive grain size from cache manager
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "  Batch processing time: " << duration.count() << "μs\n";
    
    // Record performance metrics like v0.8.3 did
    cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
}

void testDistribution(const std::string& dist_name, 
                     const std::vector<size_t>& test_sizes,
                     std::function<std::vector<double>(size_t)> value_generator) {
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TESTING " << dist_name << " DISTRIBUTION\n";
    std::cout << std::string(60, '=') << "\n";
    
    AdaptiveCache<std::string, double> cache;
    
    for (size_t test_size : test_sizes) {
        std::cout << "\n=== Testing " << dist_name << " batch size: " << test_size << " ===\n";
        
        // Generate unique continuous values 
        std::vector<double> test_values = value_generator(test_size);
        std::vector<double> results(test_size);
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Single performance run (no warm-up needed since cache hits will be rare)
        std::cout << "--- Performance run ---\n";
        auto run_start = std::chrono::high_resolution_clock::now();
        
        simulateCacheAwareContinuousBatch(
            std::span<const double>(test_values), 
            std::span<double>(results), 
            cache,
            dist_name
        );
        
        auto run_end = std::chrono::high_resolution_clock::now();
        auto run_duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_start);
        
        std::cout << "Performance run time: " << run_duration.count() << "μs\n";
        
        // Performance analysis for this batch size
        if (run_duration.count() > 50000) {  // > 50ms
            std::cout << "🚨 PERFORMANCE ISSUE DETECTED at batch size " << test_size << "!\n";
            std::cout << "Time: " << run_duration.count() << "μs (exceeds reasonable threshold)\n";
            std::cout << "This matches the reported issue from the roadmap.\n";
            
            // If we find severe degradation, we can break early
            if (run_duration.count() > 100000) {  // > 100ms
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
    
    // Cache metrics for this distribution
    auto metrics = cache.getMetrics();
    std::cout << "\n=== " << dist_name << " Cache Metrics ===\n";
    std::cout << "Hits: " << metrics.hits.load() << "\n";
    std::cout << "Misses: " << metrics.misses.load() << "\n";
    std::cout << "Hit rate: " << metrics.hit_rate.load() << "\n";
    std::cout << "Cache size: " << metrics.cache_size.load() << "\n";
    std::cout << "Memory usage: " << metrics.memory_usage.load() << " bytes\n";
}

int main() {
    std::cout << "=== Cache-Aware Continuous Distribution Performance Diagnostic ===\n";
    std::cout << "Testing with Gaussian and Exponential distributions to minimize cache hits\n";
    std::cout << "and isolate parallel cache contention issues.\n";
    
    // Test progressively larger sizes to identify where performance degrades
    std::vector<size_t> test_sizes = {100, 500, 1000, 2500, 5000, 10000};
    
    // Random number generator for unique continuous values
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Test Gaussian distribution
    testDistribution("gaussian", test_sizes, [&](size_t count) -> std::vector<double> {
        std::normal_distribution<double> dist(0.0, 1.0);
        std::vector<double> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = dist(gen);
        }
        return values;
    });
    
    // Test Exponential distribution  
    testDistribution("exponential", test_sizes, [&](size_t count) -> std::vector<double> {
        std::exponential_distribution<double> dist(1.0);  // lambda = 1.0
        std::vector<double> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = dist(gen);
        }
        return values;
    });
    
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "=== DIAGNOSTIC SUMMARY ===\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "This diagnostic tested continuous distributions with unique values\n";
    std::cout << "to minimize cache hits and isolate parallel cache write contention.\n";
    std::cout << "\nWith continuous distributions, cache hit rates should be near 0%,\n";
    std::cout << "making this a pure test of parallel cache write performance.\n";
    std::cout << "\nThe v0.8.3 issue showed >100x slowdown. If the issue persists\n";
    std::cout << "with continuous distributions, it confirms the problem is in the\n";
    std::cout << "AdaptiveCache's parallel write synchronization mechanism.\n";
    
    return 0;
}

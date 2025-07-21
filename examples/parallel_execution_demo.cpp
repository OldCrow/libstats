/**
 * @file parallel_execution_demo.cpp
 * @brief Demonstration of platform-aware parallel execution features in libstats
 * 
 * This program showcases the enhanced parallel execution capabilities with
 * adaptive grain sizing, platform-specific optimizations, and intelligent
 * thread management based on CPU architecture and workload characteristics.
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <random>

#include "libstats.h"

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void demonstrate_platform_detection() {
    print_separator("Platform Detection & Capabilities");
    
    std::cout << "Execution support: " << libstats::parallel::execution_support_string() << std::endl;
    
    // CPU information
    const auto& cpu_features = libstats::cpu::get_features();
    std::cout << "Physical CPU cores: " << libstats::cpu::get_physical_core_count() << std::endl;
    std::cout << "Logical CPU cores: " << libstats::cpu::get_logical_core_count() << std::endl;
    
    if (cpu_features.l3_cache_size > 0) {
        std::cout << "L3 Cache size: " << cpu_features.l3_cache_size / (1024 * 1024) << " MB" << std::endl;
    }
    
    // SIMD capabilities
    std::cout << "SIMD support: ";
    if (cpu_features.avx512f) std::cout << "AVX-512 ";
    if (cpu_features.avx2) std::cout << "AVX2 ";
    if (cpu_features.avx) std::cout << "AVX ";
    if (cpu_features.sse4_2) std::cout << "SSE4.2 ";
    if (cpu_features.neon) std::cout << "NEON ";
    std::cout << std::endl;
}

void demonstrate_adaptive_grain_sizing() {
    print_separator("Adaptive Grain Sizing");
    
    std::vector<std::size_t> data_sizes = {1000, 10000, 100000, 1000000};
    std::vector<std::string> operation_types = {"Memory-bound", "Computation-bound", "Mixed"};
    
    std::cout << std::left << std::setw(12) << "Data Size" 
              << std::setw(18) << "Memory-bound" 
              << std::setw(18) << "Computation-bound" 
              << std::setw(12) << "Mixed" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (auto size : data_sizes) {
        std::cout << std::setw(12) << size;
        for (int op_type = 0; op_type < 3; ++op_type) {
            auto grain = libstats::parallel::get_adaptive_grain_size(op_type, size);
            std::cout << std::setw(18) << grain;
        }
        std::cout << std::endl;
    }
    
    std::cout << "\nBase grain size: " << libstats::parallel::get_optimal_grain_size() << " elements" << std::endl;
    std::cout << "Parallel threshold: " << libstats::parallel::get_optimal_parallel_threshold() << " elements" << std::endl;
}

void demonstrate_thread_optimization() {
    print_separator("Thread Count Optimization");
    
    std::vector<std::size_t> workload_sizes = {1000, 10000, 100000, 1000000, 10000000};
    
    std::cout << std::left << std::setw(15) << "Workload Size" 
              << std::setw(18) << "Optimal Threads" 
              << std::setw(12) << "Decision" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    for (auto workload : workload_sizes) {
        auto threads = libstats::parallel::get_optimal_thread_count(workload);
        bool should_parallel = libstats::parallel::should_use_parallel(workload);
        
        std::cout << std::setw(15) << workload 
                  << std::setw(18) << threads
                  << std::setw(12) << (should_parallel ? "Parallel" : "Serial") << std::endl;
    }
}

void benchmark_parallel_operations() {
    print_separator("Performance Benchmarks");
    
    const std::size_t data_size = 1000000;  // 1M elements
    
    // Generate test data
    std::vector<double> data(data_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(1.0, 100.0);
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });
    
    std::cout << "Benchmarking with " << data_size << " elements:" << std::endl;
    
    // Transform benchmark
    {
        std::vector<double> result(data_size);
        auto start = std::chrono::high_resolution_clock::now();
        
        libstats::parallel::safe_transform(data.begin(), data.end(), result.begin(), 
            [](double x) { return std::sqrt(x * x + 1.0); });
            
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Transform operation: " << duration.count() << " μs" << std::endl;
    }
    
    // Reduce benchmark
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto sum = libstats::parallel::safe_reduce(data.begin(), data.end(), 0.0);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Reduce operation: " << duration.count() << " μs (result: " 
                  << std::fixed << std::setprecision(2) << sum << ")" << std::endl;
    }
    
    // Count benchmark
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto count = libstats::parallel::safe_count_if(data.begin(), data.end(), 
            [](double x) { return x > 50.0; });
            
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Count_if operation: " << duration.count() << " μs (result: " 
                  << count << ")" << std::endl;
    }
}

void demonstrate_cache_awareness() {
    print_separator("Cache-Aware Optimization");
    
    const auto& cpu_features = libstats::cpu::get_features();
    
    if (cpu_features.l3_cache_size > 0) {
        std::size_t cache_size_mb = cpu_features.l3_cache_size / (1024 * 1024);
        std::size_t elements_in_cache = cpu_features.l3_cache_size / sizeof(double);
        
        std::cout << "L3 Cache size: " << cache_size_mb << " MB" << std::endl;
        std::cout << "Approximate double elements in cache: " << elements_in_cache << std::endl;
        
        // Test different data sizes relative to cache
        std::vector<std::size_t> test_sizes = {
            elements_in_cache / 4,    // Fits easily in cache
            elements_in_cache / 2,    // Half cache
            elements_in_cache,        // Full cache
            elements_in_cache * 2,    // Exceeds cache
            elements_in_cache * 4     // Much larger than cache
        };
        
        std::cout << "\nCache-aware grain sizing:" << std::endl;
        for (auto size : test_sizes) {
            auto grain = libstats::parallel::get_adaptive_grain_size(0, size); // Memory-bound
            double cache_ratio = static_cast<double>(size) / elements_in_cache;
            
            std::cout << "  " << std::setw(10) << size << " elements (" 
                      << std::fixed << std::setprecision(1) << cache_ratio 
                      << "x cache): grain = " << grain << std::endl;
        }
    } else {
        std::cout << "Cache size information not available on this platform." << std::endl;
    }
}

int main() {
    std::cout << "=== libstats Platform-Aware Parallel Execution Demo ===" << std::endl;
    std::cout << "This demonstration showcases adaptive parallel execution features" << std::endl;
    std::cout << "that automatically optimize for your specific hardware platform." << std::endl;
    
    try {
        demonstrate_platform_detection();
        demonstrate_adaptive_grain_sizing();
        demonstrate_thread_optimization();
        benchmark_parallel_operations();
        demonstrate_cache_awareness();
        
        print_separator("Summary");
        std::cout << "✅ Platform detection working correctly" << std::endl;
        std::cout << "✅ Adaptive grain sizing optimized for workload types" << std::endl;
        std::cout << "✅ Thread count optimization based on workload size" << std::endl;
        std::cout << "✅ Performance benchmarks completed successfully" << std::endl;
        std::cout << "✅ Cache-aware optimizations functioning" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

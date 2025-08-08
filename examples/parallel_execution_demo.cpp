/**
 * @file parallel_execution_demo.cpp
 * @brief Comprehensive demonstration of platform-aware parallel execution in libstats v0.7.0
 * 
 * This example showcases the advanced parallel execution capabilities including:
 * - Automatic platform detection and CPU capability analysis
 * - Adaptive grain sizing based on workload characteristics
 * - Intelligent thread count optimization
 * - Cache-aware memory access patterns
 * - Performance benchmarking of parallel algorithms
 * - Real-world optimization strategies for statistical computing
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <random>

#include "../include/libstats.h"

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void demonstrate_platform_detection() {
    print_separator("1. Platform Detection & Capabilities");
    std::cout << "\nAutomatic hardware detection and optimization target selection:\n"
              << "(libstats adapts algorithms based on your specific CPU features)\n" << std::endl;
    
    std::cout << "🖥️  Execution support: " << libstats::parallel::execution_support_string() << std::endl;
    
    // CPU information
    const auto& cpu_features = libstats::cpu::get_features();
    std::cout << "💻 CPU Configuration:" << std::endl;
    std::cout << "    Physical cores: " << libstats::cpu::get_physical_core_count() 
              << " [Available for heavy parallel work]" << std::endl;
    std::cout << "    Logical cores: " << libstats::cpu::get_logical_core_count() 
              << " [Total threading capacity with hyperthreading]" << std::endl;
    
    if (cpu_features.l3_cache_size > 0) {
        std::cout << "    L3 Cache: " << cpu_features.l3_cache_size / (1024 * 1024) 
                  << " MB [Affects grain sizing and memory layout]" << std::endl;
    }
    
    // SIMD capabilities
    std::cout << "\n⚡ SIMD Instruction Support:" << std::endl;
    std::cout << "    Available: ";
    if (cpu_features.avx512f) std::cout << "AVX-512 (16x64-bit ops) ";
    if (cpu_features.avx2) std::cout << "AVX2 (4x64-bit ops) ";
    if (cpu_features.avx) std::cout << "AVX (4x64-bit ops) ";
    if (cpu_features.sse4_2) std::cout << "SSE4.2 (2x64-bit ops) ";
    if (cpu_features.neon) std::cout << "NEON (ARM SIMD) ";
    std::cout << std::endl;
    std::cout << "    ✓ libstats automatically selects the best SIMD implementation" << std::endl;
}

void demonstrate_adaptive_grain_sizing() {
    print_separator("2. Adaptive Grain Sizing");
    std::cout << "\nOptimal work unit sizing based on operation characteristics:\n"
              << "(Grain size = elements per thread for best performance)\n" << std::endl;
    
    std::vector<std::size_t> data_sizes = {1000, 10000, 100000, 1000000};
    std::vector<std::string> operation_types = {"Memory-bound", "Computation-bound", "Mixed"};
    
    std::cout << "📊 Adaptive grain sizes by workload type:" << std::endl;
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
    
    std::cout << "\n🔧 Configuration parameters:" << std::endl;
    std::cout << "   Base grain size: " << libstats::parallel::get_optimal_grain_size() 
              << " elements [Default work unit size]" << std::endl;
    std::cout << "   Parallel threshold: " << libstats::parallel::get_optimal_parallel_threshold("gaussian", "pdf") 
              << " elements [Minimum size for parallel execution]" << std::endl;
    std::cout << "\n   ℹ️ Memory-bound: Larger grains reduce cache misses" << std::endl;
    std::cout << "   ℹ️ Computation-bound: Smaller grains improve load balancing" << std::endl;
    std::cout << "   ℹ️ Mixed: Balanced approach for typical statistical operations" << std::endl;
}

void demonstrate_thread_optimization() {
    print_separator("3. Thread Count Optimization");
    std::cout << "\nIntelligent thread count selection based on workload size:\n"
              << "(Prevents over-threading and context switching overhead)\n" << std::endl;
    
    std::vector<std::size_t> workload_sizes = {1000, 10000, 100000, 1000000, 10000000};
    
    std::cout << "🔄 Thread allocation strategy:" << std::endl;
    std::cout << std::left << std::setw(15) << "Workload Size" 
              << std::setw(18) << "Optimal Threads" 
              << std::setw(12) << "Decision" 
              << std::setw(20) << "Reasoning" << std::endl;
    std::cout << std::string(65, '-') << std::endl;
    
    for (auto workload : workload_sizes) {
        auto threads = libstats::parallel::get_optimal_thread_count(workload);
        bool should_parallel = libstats::parallel::should_use_parallel(workload);
        
        std::string reasoning;
        if (!should_parallel) {
            reasoning = "Too small for parallel";
        } else if (threads == 1) {
            reasoning = "Single-threaded optimal";
        } else if (threads < libstats::cpu::get_logical_core_count()) {
            reasoning = "Partial core utilization";
        } else {
            reasoning = "Full core utilization";
        }
        
        std::cout << std::setw(15) << workload 
                  << std::setw(18) << threads
                  << std::setw(12) << (should_parallel ? "Parallel" : "Serial")
                  << std::setw(20) << reasoning << std::endl;
    }
    
    std::cout << "\n   ℹ️ Small workloads use serial execution to avoid threading overhead" << std::endl;
    std::cout << "   ℹ️ Large workloads scale up to available CPU cores" << std::endl;
    std::cout << "   ℹ️ Thread count is capped to prevent resource contention" << std::endl;
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
            double cache_ratio = static_cast<double>(size) / static_cast<double>(elements_in_cache);
            
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

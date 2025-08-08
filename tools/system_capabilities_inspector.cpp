/**
 * @file system_capabilities_inspector_working.cpp
 * @brief Working system capability inspection tool for Phase 3 performance framework
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>

#include "../include/libstats.h"
#include "../include/core/performance_dispatcher.h"

using namespace std::chrono;

class SystemCapabilitiesInspector {
public:
    void runInspection() {
        std::cout << "=== System Capabilities Inspector ===\n\n";
        
        displayHardwareInfo();
        displayPerformanceMetrics();
        displaySIMDCapabilities();
        displayMemoryHierarchy();
        displayPerformanceBaselines();
        displayDispatcherConfiguration();
    }

private:
    void displayHardwareInfo() {
        std::cout << "--- Hardware Information ---\n";
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        
        std::cout << "Logical Cores: " << capabilities.logical_cores() << "\n";
        std::cout << "Physical Cores: " << capabilities.physical_cores() << "\n";
        std::cout << "Hyperthreading: " << (capabilities.logical_cores() > capabilities.physical_cores() ? "Enabled" : "Disabled") << "\n";
        std::cout << "\n";
    }
    
    void displayPerformanceMetrics() {
        std::cout << "--- Performance Metrics ---\n";
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "SIMD Efficiency: " << capabilities.simd_efficiency() << "\n";
        std::cout << "Memory Bandwidth: " << std::setprecision(2) << capabilities.memory_bandwidth_gb_s() << " GB/s\n";
        std::cout << "Thread Overhead: " << std::setprecision(1) << capabilities.threading_overhead_ns() << " ns\n";
        std::cout << "\n";
    }
    
    void displaySIMDCapabilities() {
        std::cout << "--- SIMD Capabilities ---\n";
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        
        std::cout << std::left << std::setw(12) << "Instruction" << std::setw(10) << "Support" << std::setw(15) << "Vector Width" << "Description\n";
        std::cout << std::string(50, '-') << "\n";
        
        std::cout << std::setw(12) << "SSE2" << std::setw(10) << (capabilities.has_sse2() ? "Yes" : "No") 
                  << std::setw(15) << "128-bit" << "Basic SIMD operations\n";
        std::cout << std::setw(12) << "AVX" << std::setw(10) << (capabilities.has_avx() ? "Yes" : "No") 
                  << std::setw(15) << "256-bit" << "Advanced vector ext\n";
        std::cout << std::setw(12) << "AVX2" << std::setw(10) << (capabilities.has_avx2() ? "Yes" : "No") 
                  << std::setw(15) << "256-bit" << "Integer AVX operations\n";
        std::cout << std::setw(12) << "AVX-512" << std::setw(10) << (capabilities.has_avx512() ? "Yes" : "No") 
                  << std::setw(15) << "512-bit" << "Foundation instructions\n";
        std::cout << std::setw(12) << "NEON" << std::setw(10) << (capabilities.has_neon() ? "Yes" : "No") 
                  << std::setw(15) << "128-bit" << "ARM SIMD instructions\n";
        
        // Display active SIMD level
        std::string active_simd = libstats::simd::VectorOps::get_active_simd_level();
        std::cout << "\nActive SIMD Level: " << active_simd << "\n\n";
    }
    
    void displayMemoryHierarchy() {
        std::cout << "--- Memory Hierarchy ---\n";
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        
        std::cout << std::left << std::setw(15) << "Cache Level" << std::setw(12) << "Size (KB)" 
                  << std::setw(15) << "Line Size" << "Associativity\n";
        std::cout << std::string(55, '-') << "\n";
        
        std::cout << std::setw(15) << "L1 Data" << std::setw(12) << capabilities.l1_cache_size() / 1024 
                  << std::setw(15) << "64 bytes" << "8-way (typical)\n";
        std::cout << std::setw(15) << "L2 Unified" << std::setw(12) << capabilities.l2_cache_size() / 1024 
                  << std::setw(15) << "64 bytes" << "8-way (typical)\n";
        std::cout << std::setw(15) << "L3 Shared" << std::setw(12) << capabilities.l3_cache_size() / 1024 
                  << std::setw(15) << "64 bytes" << "16-way (typical)\n";
        std::cout << "\n";
    }
    
    void displayPerformanceBaselines() {
        std::cout << "--- Performance Baselines ---\n";
        
        // Simple arithmetic throughput test
        const size_t test_size = 1000000;
        std::vector<double> data(test_size, 1.0);
        std::vector<double> result(test_size);
        
        // SIMD throughput
        auto start = high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            libstats::simd::VectorOps::vector_multiply(data.data(), data.data(), result.data(), test_size);
        }
        auto end = high_resolution_clock::now();
        double simd_time = static_cast<double>(duration_cast<microseconds>(end - start).count()) / 10.0;
        
        // Scalar throughput
        start = high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            for (size_t j = 0; j < test_size; ++j) {
                result[j] = data[j] * data[j];
            }
        }
        end = high_resolution_clock::now();
        double scalar_time = static_cast<double>(duration_cast<microseconds>(end - start).count()) / 10.0;
        
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "SIMD Multiply Throughput: " << simd_time << " μs (" << test_size / simd_time << " MOps/s)\n";
        std::cout << "Scalar Multiply Throughput: " << scalar_time << " μs (" << test_size / scalar_time << " MOps/s)\n";
        std::cout << "SIMD Speedup: " << std::setprecision(2) << scalar_time / simd_time << "x\n\n";
    }
    
    void displayDispatcherConfiguration() {
        std::cout << "--- Performance Dispatcher Configuration ---\n";
        
        // Show some example strategy selections
        std::cout << "Example Strategy Selections:\n";
        std::cout << std::left << std::setw(20) << "Batch Size" << std::setw(15) << "Distribution" 
                  << std::setw(15) << "Complexity" << "Strategy\n";
        std::cout << std::string(65, '-') << "\n";
        
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        std::vector<size_t> test_sizes = {100, 1000, 10000, 100000};
        std::vector<libstats::performance::DistributionType> dist_types = {
            libstats::performance::DistributionType::UNIFORM,
            libstats::performance::DistributionType::GAUSSIAN,
            libstats::performance::DistributionType::EXPONENTIAL,
            libstats::performance::DistributionType::POISSON,
            libstats::performance::DistributionType::DISCRETE
        };
        std::vector<libstats::performance::ComputationComplexity> complexities = {
            libstats::performance::ComputationComplexity::SIMPLE,
            libstats::performance::ComputationComplexity::MODERATE,
            libstats::performance::ComputationComplexity::COMPLEX
        };
        
        for (auto size : test_sizes) {
            for (auto dist : dist_types) {
                for (auto complexity : complexities) {
                    libstats::performance::PerformanceDispatcher dispatcher;
                    auto strategy = dispatcher.selectOptimalStrategy(size, dist, complexity, capabilities);
                    
                    std::cout << std::setw(20) << size 
                              << std::setw(15) << distributionTypeToString(dist)
                              << std::setw(15) << complexityToString(complexity)
                              << strategyToString(strategy) << "\n";
                    
                    // Only show first complexity for brevity
                    break;
                }
            }
        }
        std::cout << "\n";
    }
    
    std::string distributionTypeToString(libstats::performance::DistributionType type) {
        switch (type) {
            case libstats::performance::DistributionType::UNIFORM: return "Uniform";
            case libstats::performance::DistributionType::GAUSSIAN: return "Gaussian";
            case libstats::performance::DistributionType::POISSON: return "Poisson";
            case libstats::performance::DistributionType::EXPONENTIAL: return "Exponential";
            case libstats::performance::DistributionType::DISCRETE: return "Discrete";
            case libstats::performance::DistributionType::GAMMA: return "Gamma";
            default: return "Unknown";
        }
    }
    
    std::string complexityToString(libstats::performance::ComputationComplexity complexity) {
        switch (complexity) {
            case libstats::performance::ComputationComplexity::SIMPLE: return "Simple";
            case libstats::performance::ComputationComplexity::MODERATE: return "Moderate";
            case libstats::performance::ComputationComplexity::COMPLEX: return "Complex";
            default: return "Unknown";
        }
    }
    
    std::string strategyToString(libstats::performance::Strategy strategy) {
        switch (strategy) {
            case libstats::performance::Strategy::SCALAR: return "Scalar";
            case libstats::performance::Strategy::SIMD_BATCH: return "SIMD";
            case libstats::performance::Strategy::PARALLEL_SIMD: return "Parallel+SIMD";
            case libstats::performance::Strategy::WORK_STEALING: return "Work-Stealing";
            case libstats::performance::Strategy::CACHE_AWARE: return "Cache-Aware";
            default: return "Unknown";
        }
    }
};

int main() {
    try {
        SystemCapabilitiesInspector inspector;
        inspector.runInspection();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during system inspection: " << e.what() << std::endl;
        return 1;
    }
}

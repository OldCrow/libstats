/**
 * @file performance_benchmark.cpp
 * @brief Enhanced performance benchmark for libstats parallel operations with PerformanceDispatcher integration
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <string>
#include <iomanip>

#include "../include/libstats.h"
#include "../include/core/performance_dispatcher.h"
#include "../include/core/performance_history.h"

class SimpleBenchmark {
private:
    std::mt19937 rng_;
    
    std::string strategyToString(libstats::performance::Strategy strategy) {
        switch (strategy) {
            case libstats::performance::Strategy::SCALAR:
                return "SCALAR";
            case libstats::performance::Strategy::SIMD_BATCH:
                return "SIMD_BATCH";
            case libstats::performance::Strategy::PARALLEL_SIMD:
                return "PARALLEL_SIMD";
            case libstats::performance::Strategy::WORK_STEALING:
                return "WORK_STEALING";
            case libstats::performance::Strategy::CACHE_AWARE:
                return "CACHE_AWARE";
            default:
                return "UNKNOWN";
        }
    }
    
public:
    SimpleBenchmark() : rng_(std::random_device{}()) {}
    
    void runBenchmarks() {
        std::cout << "=== Enhanced libstats Performance Benchmark with PerformanceHistory ===" << std::endl;
        
        // Print system info using both old and new APIs
        const auto& features = libstats::cpu::get_features();
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        const std::string simd_level = libstats::simd::VectorOps::get_active_simd_level();
        const auto thread_count = std::thread::hardware_concurrency();
        
        std::cout << "\n--- System Configuration ---" << std::endl;
        std::cout << "SIMD Level: " << simd_level << std::endl;
        std::cout << "Hardware Threads: " << thread_count << std::endl;
        std::cout << "Logical Cores: " << capabilities.logical_cores() << std::endl;
        std::cout << "Physical Cores: " << capabilities.physical_cores() << std::endl;
        std::cout << "L1 Cache: " << features.l1_cache_size / 1024 << " KB" << std::endl;
        std::cout << "L2 Cache: " << features.l2_cache_size / 1024 << " KB" << std::endl;
        std::cout << "L3 Cache: " << features.l3_cache_size / 1024 << " KB" << std::endl;
        std::cout << "Cache Line Size: " << features.cache_line_size << " bytes" << std::endl;
        std::cout << "Memory Bandwidth: " << std::fixed << std::setprecision(2) 
                  << capabilities.memory_bandwidth_gb_s() << " GB/s" << std::endl;
        std::cout << "SIMD Efficiency: " << std::setprecision(3) 
                  << capabilities.simd_efficiency() << std::endl;
        
        // Test different data sizes
        std::vector<std::size_t> data_sizes = {
            1000, 10000, 100000, 1000000, 10000000
        };
        
        for (auto size : data_sizes) {
            std::cout << "\n--- Testing with " << size << " elements ---" << std::endl;
            
            // Generate test data
            std::vector<double> data(size);
            std::uniform_real_distribution<double> dis(1.0, 100.0);
            std::generate(data.begin(), data.end(), [&]() { return dis(rng_); });
            
            benchmarkTransform(data);
            benchmarkReduce(data);
            benchmarkCountIf(data);
        }
        
        // Show performance learning results
        showPerformanceLearning();
    }
    
private:
    void benchmarkTransform(const std::vector<double>& data) {
        const int num_runs = 5;
        
        // Serial transform
        auto serial_time = timeOperation([&]() {
            std::vector<double> result(data.size());
            std::transform(data.begin(), data.end(), result.begin(), 
                          [](double x) { return x * 2.0; });
        }, num_runs);
        
        // Parallel transform
        auto parallel_time = timeOperation([&]() {
            std::vector<double> result(data.size());
            libstats::parallel::safe_transform(data.begin(), data.end(), result.begin(),
                                             [](double x) { return x * 2.0; });
        }, num_runs);
        
        printResults("Transform", serial_time, parallel_time);
        
        // Record performance data in PerformanceHistory
        recordPerformanceData(data.size(), "Transform", serial_time, parallel_time);
    }
    
    void benchmarkReduce(const std::vector<double>& data) {
        const int num_runs = 10;
        volatile double sink = 0.0;  // Prevent optimization
        
        // Serial reduce
        auto serial_time = timeOperation([&]() {
            auto result = std::accumulate(data.begin(), data.end(), 0.0);
            sink += result;  // Force use of result
        }, num_runs);
        
        // Parallel reduce
        auto parallel_time = timeOperation([&]() {
            auto result = libstats::parallel::safe_reduce(data.begin(), data.end(), 0.0);
            sink += result;  // Force use of result
        }, num_runs);
        
        printResults("Reduce", serial_time, parallel_time);
    }
    
    void benchmarkCountIf(const std::vector<double>& data) {
        const int num_runs = 10;
        volatile size_t sink = 0;  // Prevent optimization
        
        // Serial count_if
        auto serial_time = timeOperation([&]() {
            auto result = std::count_if(data.begin(), data.end(), 
                                       [](double x) { return x > 50.0; });
            sink += result;  // Force use of result
        }, num_runs);
        
        // Parallel count_if
        auto parallel_time = timeOperation([&]() {
            auto result = libstats::parallel::safe_count_if(data.begin(), data.end(),
                                                          [](double x) { return x > 50.0; });
            sink += result;  // Force use of result
        }, num_runs);
        
        printResults("Count_if", serial_time, parallel_time);
        
        // Record performance data in PerformanceHistory
        recordPerformanceData(data.size(), "Count_if", serial_time, parallel_time);
    }
    
    void recordPerformanceData(size_t data_size, const std::string& /* operation */, 
                              double serial_time, double parallel_time) {
        // Map operation names to PerformanceDispatcher enums
        libstats::performance::DistributionType dist_type = libstats::performance::DistributionType::UNIFORM;  // Generic for parallel ops
        
        // Record serial strategy performance (simulated as SIMD since we're not doing true serial)
        libstats::performance::PerformanceDispatcher::recordPerformance(
            libstats::performance::Strategy::SIMD_BATCH, 
            dist_type, 
            data_size,
            static_cast<uint64_t>(serial_time * 1000)  // Convert μs to ns
        );
        
        // Record parallel strategy performance
        libstats::performance::PerformanceDispatcher::recordPerformance(
            libstats::performance::Strategy::PARALLEL_SIMD, 
            dist_type, 
            data_size,
            static_cast<uint64_t>(parallel_time * 1000)  // Convert μs to ns
        );
    }
    
    void showPerformanceLearning() {
        std::cout << "\n=== Performance Learning Results ===" << std::endl;
        
        const auto& history = libstats::performance::PerformanceDispatcher::getPerformanceHistory();
        
        std::cout << "Total measurements recorded: " << history.getTotalExecutions() << std::endl;
        
        // Show learned thresholds for the distribution we've been testing
        auto thresholds = history.learnOptimalThresholds(libstats::performance::DistributionType::UNIFORM);
        if (thresholds.has_value()) {
            std::cout << "\nLearned Optimal Thresholds (from benchmark data):" << std::endl;
            std::cout << "  SIMD threshold: " << thresholds->first << " elements" << std::endl;
            std::cout << "  Parallel threshold: " << thresholds->second << " elements" << std::endl;
        } else {
            std::cout << "\nInsufficient data for threshold learning (need more varied workload sizes)" << std::endl;
        }
        
        // Show strategy recommendations for key sizes
        std::cout << "\nStrategy Recommendations Based on Measurements:" << std::endl;
        std::vector<size_t> test_sizes = {1000, 10000, 100000, 1000000};
        for (auto size : test_sizes) {
            auto recommendation = history.getBestStrategy(libstats::performance::DistributionType::UNIFORM, size);
            if (recommendation.has_sufficient_data) {
                std::ostringstream confidence_str;
                confidence_str << std::fixed << std::setprecision(0) << (recommendation.confidence_score * 100) << "%";
                
                std::cout << "  Size " << size << ": " << strategyToString(recommendation.recommended_strategy)
                          << " (confidence: " << confidence_str.str() << ")" << std::endl;
            }
        }
        
        std::cout << "\nThis learning data will help optimize future strategy selections!" << std::endl;
    }
    
    void printResults(const std::string& operation, double serial_time, double parallel_time) {
        // Format time with appropriate precision and units
        auto formatTime = [](double time_us) -> std::string {
            if (time_us < 1.0) {
                return std::to_string(static_cast<int>(time_us * 1000)) + "ns";
            } else if (time_us < 1000.0) {
                return std::to_string(static_cast<int>(time_us)) + "μs";
            } else {
                return std::to_string(time_us / 1000.0) + "ms";
            }
        };
        
        std::string serial_str = formatTime(serial_time);
        std::string parallel_str = formatTime(parallel_time);
        
        // Calculate speedup with zero-division protection
        std::string speedup_str;
        if (parallel_time < 0.001) {
            speedup_str = "inf";
        } else if (serial_time < 0.001) {
            speedup_str = "0x";
        } else {
            double speedup = serial_time / parallel_time;
            speedup_str = std::to_string(speedup).substr(0, 5) + "x";
        }
        
        std::cout << "  " << operation << ": Serial=" << serial_str 
                  << ", Parallel=" << parallel_str 
                  << ", Speedup=" << speedup_str << std::endl;
    }
    
    template<typename Func>
    double timeOperation(Func func, int num_runs) {
        std::vector<double> times;
        
        // Warm-up runs
        for (int i = 0; i < 2; ++i) {
            func();
        }
        
        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(static_cast<double>(duration.count()) / 1000.0); // Convert to microseconds
        }
        
        // Return median time
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    }
};

int main() {
    SimpleBenchmark benchmark;
    
    try {
        benchmark.runBenchmarks();
    } catch (const std::exception& e) {
        std::cerr << "Error during benchmark: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

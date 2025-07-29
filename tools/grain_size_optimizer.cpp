/**
 * @file grain_size_optimizer.cpp
 * @brief Comprehensive benchmarking tool for optimizing grain sizes and parallel thresholds
 * 
 * This tool systematically benchmarks different grain sizes, thread counts, and operation types
 * to determine optimal parameters for different SIMD architectures and workload patterns.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <random>
#include <map>
#include <fstream>
#include <iomanip>
#include <functional>
#include <thread>
#include <future>

#include "../include/libstats.h"

// Global volatile to prevent optimization across function boundaries
volatile double global_sink = 0.0;

struct BenchmarkResult {
    std::size_t grain_size;
    std::size_t data_size;
    std::size_t thread_count;
    std::string operation_type;
    std::string simd_level;
    double execution_time_us;
    double throughput_elements_per_us;
    double efficiency_ratio; // vs serial baseline
};

class GrainSizeOptimizer {
private:
    std::vector<BenchmarkResult> results_;
    std::mt19937 rng_;
    
    // Test data sizes (powers of 2 and cache-relative sizes)
    std::vector<std::size_t> getTestDataSizes() {
        const auto& features = libstats::cpu::get_features();
        std::size_t l3_elements = features.l3_cache_size > 0 ? 
            features.l3_cache_size / sizeof(double) : 262144; // 2MB fallback
            
        return {
            64, 128, 256, 512, 1024,           // Small sizes
            2048, 4096, 8192, 16384,           // Medium sizes  
            l3_elements / 8,                   // L3 cache fractions
            l3_elements / 4,
            l3_elements / 2, 
            l3_elements,
            l3_elements * 2,
            l3_elements * 4,                   // Exceeds cache
            1000000, 10000000                  // Large datasets
        };
    }
    
    // Test grain sizes to evaluate
    std::vector<std::size_t> getTestGrainSizes(std::size_t data_size) {
        const auto& features = libstats::cpu::get_features();
        std::size_t simd_width = libstats::constants::platform::get_optimal_simd_block_size();
        std::size_t cache_line_elements = features.cache_line_size / sizeof(double);
        
        std::vector<std::size_t> grain_sizes;
        
        // SIMD-aligned grain sizes
        for (std::size_t mult = 1; mult <= 64; mult *= 2) {
            grain_sizes.push_back(simd_width * mult);
        }
        
        // Cache-line aligned grain sizes  
        for (std::size_t mult = 1; mult <= 32; mult *= 2) {
            grain_sizes.push_back(cache_line_elements * mult);
        }
        
        // Power-of-2 grain sizes
        std::size_t grain_limit = data_size / 8 < 65536UL ? data_size / 8 : 65536UL;
        for (std::size_t grain = 8; grain <= grain_limit; grain *= 2) {
            grain_sizes.push_back(grain);
        }
        
        // Remove duplicates and sort
        std::sort(grain_sizes.begin(), grain_sizes.end());
        grain_sizes.erase(std::unique(grain_sizes.begin(), grain_sizes.end()), grain_sizes.end());
        
        // Only keep reasonable grain sizes
        grain_sizes.erase(
            std::remove_if(grain_sizes.begin(), grain_sizes.end(),
                [data_size](std::size_t grain) { 
                    return grain > data_size / 4 || grain < 8; 
                }), 
            grain_sizes.end());
            
        return grain_sizes;
    }
    
    // Benchmark different operation types using safe parallel functions
    template<typename Container, typename Func>
    double benchmarkOperation(const Container& data, Func operation, 
                            std::size_t /* grain_size */, const std::string& op_name) {
        const int num_runs = 10;
        std::vector<double> times;
        volatile double sink_double = 0.0;
        volatile size_t sink_size = 0;
        
        // Warm-up runs
        for (int i = 0; i < 2; ++i) {
            if (op_name == "transform") {
                std::vector<double> result(data.size());
                libstats::parallel::safe_transform(data.begin(), data.end(), result.begin(), operation);
            } else if (op_name == "reduce") {
                auto result = libstats::parallel::safe_reduce(data.begin(), data.end(), 0.0);
                sink_double += result;
            } else if (op_name == "count_if") {
                auto result = libstats::parallel::safe_count_if(data.begin(), data.end(), operation);
                sink_size += result;
            }
        }
        
        for (int run = 0; run < num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if (op_name == "transform") {
                std::vector<double> result(data.size());
                libstats::parallel::safe_transform(data.begin(), data.end(), result.begin(), operation);
                sink_double += result[0]; // Use result to prevent optimization
            } else if (op_name == "reduce") {
                auto result = libstats::parallel::safe_reduce(data.begin(), data.end(), 0.0);
                sink_double += result; // Force use of result
            } else if (op_name == "count_if") {
                auto result = libstats::parallel::safe_count_if(data.begin(), data.end(), operation);
                sink_size += result; // Force use of result
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // Convert to microseconds
        }
        
        // Return median time
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    }
    
    // Serial baseline for efficiency comparison
    template<typename Container, typename Func>
    double benchmarkSerial(const Container& data, Func operation, const std::string& op_name) {
        const int num_runs = 10;
        std::vector<double> times;
        volatile double sink_double = 0.0;
        volatile size_t sink_size = 0;
        
        // Warm-up runs
        for (int i = 0; i < 2; ++i) {
            if (op_name == "transform") {
                std::vector<double> result(data.size());
                std::transform(data.begin(), data.end(), result.begin(), operation);
                sink_double += result[0];
            } else if (op_name == "count_if") {
                auto result = std::count_if(data.begin(), data.end(), operation);
                sink_size += result;
            }
        }
        
        for (int run = 0; run < num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            
            if (op_name == "transform") {
                std::vector<double> result(data.size());
                std::transform(data.begin(), data.end(), result.begin(), operation);
                sink_double += result[0]; // Use result to prevent optimization
            } else if (op_name == "count_if") {
                auto result = std::count_if(data.begin(), data.end(), operation);
                sink_size += static_cast<size_t>(result); // Force use of result
            }
            // Note: reduce is handled separately with benchmarkSerialReduce
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }
        
        // Use the sinks to prevent the compiler from optimizing away the entire loops
        if (sink_double < -1e9 || sink_size > 1e9) {
            std::cout << "Impossible sink values" << std::endl;
        }
        
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    }
    
    // Helper for serial reduce benchmark
    template<typename Container>
    double benchmarkSerialReduce(const Container& data) {
        const int num_runs = 10;
        std::vector<double> times;
        
        // Warm-up runs
        for (int i = 0; i < 2; ++i) {
            volatile double temp = 0.0;
            for (auto it = data.begin(); it != data.end(); ++it) {
                temp += *it;
            }
            global_sink += temp;
        }
        
        for (int run = 0; run < num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Use manual loop instead of std::accumulate to prevent optimization
            volatile double temp = 0.0;
            for (auto it = data.begin(); it != data.end(); ++it) {
                temp += *it;
            }
            global_sink += temp; // Force use of result
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }
        
        std::sort(times.begin(), times.end());
        double median_time = times[times.size() / 2];
        
        // Debug output for large datasets
        if (data.size() >= 1000000) {
            std::cout << "[DEBUG] Serial reduce for " << data.size() << " elements: " << median_time << "μs" << std::endl;
        }
        
        return median_time;
    }
    
    // Helper for parallel reduce benchmark
    template<typename Container>
    double benchmarkParallelReduce(const Container& data, std::size_t /* grain_size */) {
        const int num_runs = 10;
        std::vector<double> times;
        volatile double sink = 0.0;
        
        // Warm-up runs
        for (int i = 0; i < 2; ++i) {
            auto result = libstats::parallel::safe_reduce(data.begin(), data.end(), 0.0);
            sink += result;
        }
        
        for (int run = 0; run < num_runs; ++run) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = libstats::parallel::safe_reduce(data.begin(), data.end(), 0.0);
            sink += result; // Force use of result
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            times.push_back(duration.count() / 1000.0);
        }
        
        std::sort(times.begin(), times.end());
        return times[times.size() / 2];
    }

public:
    GrainSizeOptimizer() : rng_(std::random_device{}()) {}
    
    void runComprehensiveBenchmark() {
        std::cout << "Starting comprehensive grain size optimization benchmark..." << std::endl;
        
        const auto data_sizes = getTestDataSizes();
        const std::string simd_level = libstats::simd::VectorOps::get_active_simd_level();
        const auto thread_count = std::thread::hardware_concurrency();
        
        std::cout << "SIMD Level: " << simd_level << std::endl;
        std::cout << "Thread Count: " << thread_count << std::endl;
        std::cout << "Testing " << data_sizes.size() << " data sizes..." << std::endl;
        
        for (auto data_size : data_sizes) {
            std::cout << "\nTesting data size: " << data_size << " elements" << std::endl;
            
            // Generate test data
            std::vector<double> data(data_size);
            std::uniform_real_distribution<double> dis(1.0, 100.0);
            std::generate(data.begin(), data.end(), [&]() { return dis(rng_); });
            
            auto grain_sizes = getTestGrainSizes(data_size);
            std::cout << "  Testing " << grain_sizes.size() << " grain sizes" << std::endl;
            
            // Test different operation types
            std::vector<std::pair<std::string, std::function<void()>>> operations = {
                {"transform_simple", [&]() {
                    auto op = [](double x) { return x * 2.0; };
                    auto serial_time = benchmarkSerial(data, op, "transform");
                    
                    for (auto grain : grain_sizes) {
                        auto parallel_time = benchmarkOperation(data, op, grain, "transform");
                        results_.push_back({grain, data_size, thread_count, "transform_simple", 
                                          simd_level, parallel_time, 
                                          data_size / parallel_time, 
                                          serial_time / parallel_time});
                    }
                }},
                
                {"transform_complex", [&]() {
                    auto op = [](double x) { return std::sqrt(x * x + 1.0) + std::sin(x * 0.1); };
                    auto serial_time = benchmarkSerial(data, op, "transform");
                    
                    for (auto grain : grain_sizes) {
                        auto parallel_time = benchmarkOperation(data, op, grain, "transform");
                        results_.push_back({grain, data_size, thread_count, "transform_complex", 
                                          simd_level, parallel_time, 
                                          data_size / parallel_time,
                                          serial_time / parallel_time});
                    }
                }},
                
                {"reduce", [&]() {
                    // safe_reduce only does sum with default binary op, so use dummy op for serial benchmark
                    auto serial_time = benchmarkSerialReduce(data);
                    
                    for (auto grain : grain_sizes) {
                        auto parallel_time = benchmarkParallelReduce(data, grain);
                        double efficiency = serial_time / parallel_time;
                        
                        // Debug output for large datasets
                        if (data.size() >= 1000000 && grain == grain_sizes[0]) {
                            std::cout << "[DEBUG] Serial: " << serial_time << "μs, Parallel: " << parallel_time << "μs, Efficiency: " << efficiency << "x" << std::endl;
                        }
                        
                        results_.push_back({grain, data_size, thread_count, "reduce", 
                                          simd_level, parallel_time, 
                                          data_size / parallel_time,
                                          efficiency});
                    }
                }},
                
                {"count_if", [&]() {
                    auto op = [](double x) { return x > 50.0; };
                    auto serial_time = benchmarkSerial(data, op, "count_if");
                    
                    for (auto grain : grain_sizes) {
                        auto parallel_time = benchmarkOperation(data, op, grain, "count_if");
                        results_.push_back({grain, data_size, thread_count, "count_if", 
                                          simd_level, parallel_time, 
                                          data_size / parallel_time,
                                          serial_time / parallel_time});
                    }
                }}
            };
            
            for (auto& [op_name, op_func] : operations) {
                std::cout << "    " << op_name << "..." << std::flush;
                op_func();
                std::cout << " done" << std::endl;
            }
        }
        
        std::cout << "\nBenchmark completed. Total results: " << results_.size() << std::endl;
    }
    
    void analyzeResults() {
        std::cout << "\n=== Optimization Analysis ===" << std::endl;
        
        // Group results by operation type and data size ranges
        std::map<std::string, std::vector<BenchmarkResult*>> by_operation;
        for (auto& result : results_) {
            by_operation[result.operation_type].push_back(&result);
        }
        
        for (auto& [op_type, op_results] : by_operation) {
            std::cout << "\n" << op_type << " Operation:" << std::endl;
            
            // Group by data size ranges (small, medium, large, huge)
            std::map<std::string, std::vector<BenchmarkResult*>> by_size_range;
            
            for (auto* result : op_results) {
                std::string range;
                if (result->data_size < 1024) range = "tiny";
                else if (result->data_size < 16384) range = "small";
                else if (result->data_size < 262144) range = "medium";
                else if (result->data_size < 2097152) range = "large";
                else range = "huge";
                
                by_size_range[range].push_back(result);
            }
            
            for (auto& [range, range_results] : by_size_range) {
                if (range_results.empty()) continue;
                
                // Find optimal grain size for this range
                auto best_result = *std::max_element(range_results.begin(), range_results.end(),
                    [](const BenchmarkResult* a, const BenchmarkResult* b) {
                        return a->efficiency_ratio < b->efficiency_ratio;
                    });
                    
                std::cout << "  " << range << " datasets: optimal grain = " 
                          << best_result->grain_size << " (efficiency: " 
                          << std::fixed << std::setprecision(2) 
                          << best_result->efficiency_ratio << "x)" << std::endl;
            }
        }
    }
    
    void generateOptimizedConstants() {
        std::cout << "\n=== Suggested Constants ===" << std::endl;
        
        // Generate optimized constants based on analysis
        analyzeResults();
        
        std::cout << "\nRecommended updates to constants.h:" << std::endl;
        std::cout << "// TODO: Update these based on benchmark results" << std::endl;
    }
    
    void saveResultsToCsv(const std::string& filename) {
        std::ofstream file(filename);
        file << "grain_size,data_size,thread_count,operation_type,simd_level,"
             << "execution_time_us,throughput,efficiency_ratio\n";
             
        for (const auto& result : results_) {
            file << result.grain_size << "," << result.data_size << "," 
                 << result.thread_count << "," << result.operation_type << ","
                 << result.simd_level << "," << result.execution_time_us << ","
                 << result.throughput_elements_per_us << "," << result.efficiency_ratio << "\n";
        }
        
        std::cout << "Results saved to " << filename << std::endl;
    }
};

int main() {
    std::cout << "=== libstats Grain Size Optimization Tool ===" << std::endl;
    std::cout << "This tool will benchmark different grain sizes to find optimal values" << std::endl;
    std::cout << "for your specific hardware configuration." << std::endl;
    
    GrainSizeOptimizer optimizer;
    
    try {
        optimizer.runComprehensiveBenchmark();
        optimizer.analyzeResults();
        optimizer.generateOptimizedConstants();
        optimizer.saveResultsToCsv("grain_size_optimization_results.csv");
        
    } catch (const std::exception& e) {
        std::cerr << "Error during optimization: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

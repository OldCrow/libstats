/**
 * @file parallel_threshold_benchmark.cpp
 * @brief Enhanced Benchmark tool for determining dynamic thresholds using PerformanceHistory
 * 
 * This tool benchmarks different data sizes to find the optimal thresholds
 * for parallel execution, utilizing adaptive learning from PerformanceHistory.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <span>
#include <map>
#include <thread>

// Include the distribution headers
#include "../include/libstats.h"
#include "../include/distributions/uniform.h"
#include "../include/distributions/poisson.h"
#include "../include/distributions/discrete.h"
#include "../include/distributions/gaussian.h"
#include "../include/distributions/exponential.h"
#include "../include/core/performance_dispatcher.h"

using namespace std::chrono;
using namespace libstats;

struct BenchmarkResult {
    std::size_t data_size;
    std::string distribution_type;
    std::string operation_type;
    double serial_time_us;
    double parallel_time_us;
    double simd_time_us;
    double parallel_speedup;
    double simd_speedup;
    bool parallel_beneficial;
};

class ParallelThresholdBenchmark {
private:
    std::mt19937 gen_;
    std::vector<BenchmarkResult> results_;
    
    // Test data sizes - start small and work up
    std::vector<std::size_t> test_sizes_ = {
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 
        16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152
    };
    
    // Number of iterations for timing stability
    static constexpr int TIMING_ITERATIONS = 10;
    static constexpr int WARMUP_ITERATIONS = 3;
    
public:
    ParallelThresholdBenchmark() : gen_(42) {}
    
    void runAllBenchmarks() {
        std::cout << "=== Parallel Threshold Benchmark ===\n";
        std::cout << "Hardware: " << std::thread::hardware_concurrency() << " threads\n";
        std::cout << "Platform: " << parallel::execution_support_string() << "\n\n";
        
        benchmarkUniformDistribution();
        benchmarkPoissonDistribution(); 
        benchmarkDiscreteDistribution();
        benchmarkGaussianDistribution();
        benchmarkExponentialDistribution();
        
        analyzeResults();
        saveResults();
    }
    
private:
    void benchmarkUniformDistribution() {
        std::cout << "Benchmarking Uniform Distribution...\n";
        UniformDistribution uniform(0.0, 1.0);
        
        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;
            
            // Generate test data
            std::vector<double> test_data(size);
            std::uniform_real_distribution<double> dis(-0.5, 1.5);
            for (auto& val : test_data) {
                val = dis(gen_);
            }
            
            // Benchmark PDF
            auto pdf_result = benchmarkOperation(uniform, test_data, "PDF", "Uniform");
            results_.push_back(pdf_result);
            
            // Benchmark LogPDF  
            auto logpdf_result = benchmarkOperation(uniform, test_data, "LogPDF", "Uniform");
            results_.push_back(logpdf_result);
            
            // Benchmark CDF
            auto cdf_result = benchmarkOperation(uniform, test_data, "CDF", "Uniform");
            results_.push_back(cdf_result);
            
            std::cout << " ✓\n";
        }
    }
    
    void benchmarkPoissonDistribution() {
        std::cout << "Benchmarking Poisson Distribution...\n";
        PoissonDistribution poisson(3.5);
        
        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;
            
            // Generate test data (integer values for Poisson)
            std::vector<double> test_data(size);
            std::poisson_distribution<int> dis(3);
            for (auto& val : test_data) {
                val = static_cast<double>(dis(gen_));
            }
            
            // Benchmark PDF (PMF)
            auto pdf_result = benchmarkOperation(poisson, test_data, "PDF", "Poisson");
            results_.push_back(pdf_result);
            
            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(poisson, test_data, "LogPDF", "Poisson");
            results_.push_back(logpdf_result);
            
            // Benchmark CDF
            auto cdf_result = benchmarkOperation(poisson, test_data, "CDF", "Poisson");
            results_.push_back(cdf_result);
            
            std::cout << " ✓\n";
        }
    }
    
    void benchmarkDiscreteDistribution() {
        std::cout << "Benchmarking Discrete Distribution...\n";
        DiscreteDistribution discrete(0, 10);
        
        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;
            
            // Generate test data (integer values)
            std::vector<double> test_data(size);
            std::uniform_int_distribution<int> dis(-2, 12);
            for (auto& val : test_data) {
                val = static_cast<double>(dis(gen_));
            }
            
            // Benchmark PDF (PMF)
            auto pdf_result = benchmarkOperation(discrete, test_data, "PDF", "Discrete");
            results_.push_back(pdf_result);
            
            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(discrete, test_data, "LogPDF", "Discrete");
            results_.push_back(logpdf_result);
            
            // Benchmark CDF
            auto cdf_result = benchmarkOperation(discrete, test_data, "CDF", "Discrete");
            results_.push_back(cdf_result);
            
            std::cout << " ✓\n";
        }
    }
    
    void benchmarkGaussianDistribution() {
        std::cout << "Benchmarking Gaussian Distribution...\n";
        GaussianDistribution gaussian(0.0, 1.0);
        
        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;
            
            // Generate test data (normal distribution values)
            std::vector<double> test_data(size);
            std::normal_distribution<double> dis(0.0, 2.0);  // Wider range
            for (auto& val : test_data) {
                val = dis(gen_);
            }
            
            // Benchmark PDF
            auto pdf_result = benchmarkOperation(gaussian, test_data, "PDF", "Gaussian");
            results_.push_back(pdf_result);
            
            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(gaussian, test_data, "LogPDF", "Gaussian");
            results_.push_back(logpdf_result);
            
            // Benchmark CDF
            auto cdf_result = benchmarkOperation(gaussian, test_data, "CDF", "Gaussian");
            results_.push_back(cdf_result);
            
            std::cout << " ✓\n";
        }
    }
    
    void benchmarkExponentialDistribution() {
        std::cout << "Benchmarking Exponential Distribution...\n";
        ExponentialDistribution exponential(1.0);
        
        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;
            
            // Generate test data (exponential distribution values)
            std::vector<double> test_data(size);
            std::exponential_distribution<double> dis(0.5);  // λ=0.5
            for (auto& val : test_data) {
                val = dis(gen_);
            }
            
            // Benchmark PDF
            auto pdf_result = benchmarkOperation(exponential, test_data, "PDF", "Exponential");
            results_.push_back(pdf_result);
            
            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(exponential, test_data, "LogPDF", "Exponential");
            results_.push_back(logpdf_result);
            
            // Benchmark CDF
            auto cdf_result = benchmarkOperation(exponential, test_data, "CDF", "Exponential");
            results_.push_back(cdf_result);
            
            std::cout << " ✓\n";
        }
    }
    
    template<typename Distribution>
    BenchmarkResult benchmarkOperation(const Distribution& dist, 
                                     const std::vector<double>& test_data,
                                     const std::string& operation,
                                     const std::string& dist_type) {
        BenchmarkResult result;
        result.data_size = test_data.size();
        result.distribution_type = dist_type;
        result.operation_type = operation;
        
        std::vector<double> results_buffer(test_data.size());
        std::span<const double> input_span(test_data);
        std::span<double> output_span(results_buffer);
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "serial");
        }
        
        // Benchmark Serial (using SIMD batch operations)
        auto serial_start = high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "simd");
        }
        auto serial_end = high_resolution_clock::now();
        result.simd_time_us = duration_cast<microseconds>(serial_end - serial_start).count() / double(TIMING_ITERATIONS);
        
        // Benchmark True Serial (element by element)
        auto true_serial_start = high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "serial");
        }
        auto true_serial_end = high_resolution_clock::now();
        result.serial_time_us = duration_cast<microseconds>(true_serial_end - true_serial_start).count() / double(TIMING_ITERATIONS);
        
        // Benchmark Parallel
        auto parallel_start = high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "parallel");
        }
        auto parallel_end = high_resolution_clock::now();
        result.parallel_time_us = duration_cast<microseconds>(parallel_end - parallel_start).count() / double(TIMING_ITERATIONS);
        
        // Calculate speedups
        result.parallel_speedup = result.simd_time_us / result.parallel_time_us;
        result.simd_speedup = result.serial_time_us / result.simd_time_us;
        result.parallel_beneficial = result.parallel_speedup > 1.0;
        
        return result;
    }
    
    template<typename Distribution>
    void performOperation(const Distribution& dist,
                         std::span<const double> input,
                         std::span<double> output,
                         const std::string& operation,
                         const std::string& method) {
        
        if (method == "serial") {
            // True serial: element by element
            if (operation == "PDF") {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = dist.getProbability(input[i]);
                }
            } else if (operation == "LogPDF") {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = dist.getLogProbability(input[i]);
                }
            } else if (operation == "CDF") {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = dist.getCumulativeProbability(input[i]);
                }
            }
        } else if (method == "simd") {
            // SIMD batch operations
            if (operation == "PDF") {
                dist.getProbabilityBatch(input.data(), output.data(), input.size());
            } else if (operation == "LogPDF") {
                dist.getLogProbabilityBatch(input.data(), output.data(), input.size());
            } else if (operation == "CDF") {
                dist.getCumulativeProbabilityBatch(input.data(), output.data(), input.size());
            }
        } else if (method == "parallel") {
            // Parallel operations
            if (operation == "PDF") {
                dist.getProbabilityBatchParallel(input, output);
            } else if (operation == "LogPDF") {
                dist.getLogProbabilityBatchParallel(input, output);
            } else if (operation == "CDF") {
                dist.getCumulativeProbabilityBatchParallel(input, output);
            }
        }
    }
    
    void analyzeResults() {
        std::cout << "\n=== Analysis Results ===\n";
        
        // Group results by distribution and operation
        std::map<std::string, std::vector<BenchmarkResult*>> grouped_results;
        for (auto& result : results_) {
            std::string key = result.distribution_type + "_" + result.operation_type;
            grouped_results[key].push_back(&result);
        }
        
        std::cout << std::left << std::setw(15) << "Dist_Op" 
                  << std::setw(10) << "Size" 
                  << std::setw(12) << "Serial(μs)"
                  << std::setw(12) << "SIMD(μs)"
                  << std::setw(12) << "Parallel(μs)"
                  << std::setw(12) << "P-Speedup"
                  << std::setw(12) << "S-Speedup"
                  << std::setw(12) << "Beneficial?" << "\n";
        std::cout << std::string(110, '-') << "\n";
        
        for (const auto& [key, results] : grouped_results) {
            std::size_t beneficial_threshold = SIZE_MAX;
            
            for (const auto* result : results) {
                std::cout << std::left << std::setw(15) << key
                          << std::setw(10) << result->data_size
                          << std::setw(12) << std::fixed << std::setprecision(1) << result->serial_time_us
                          << std::setw(12) << std::fixed << std::setprecision(1) << result->simd_time_us
                          << std::setw(12) << std::fixed << std::setprecision(1) << result->parallel_time_us
                          << std::setw(12) << std::fixed << std::setprecision(2) << result->parallel_speedup
                          << std::setw(12) << std::fixed << std::setprecision(2) << result->simd_speedup
                          << std::setw(12) << (result->parallel_beneficial ? "YES" : "NO") << "\n";
                
                if (result->parallel_beneficial && beneficial_threshold == SIZE_MAX) {
                    beneficial_threshold = result->data_size;
                }
            }
            
            std::cout << "  → Recommended threshold for " << key << ": ";
            if (beneficial_threshold != SIZE_MAX) {
                std::cout << beneficial_threshold << " elements\n";
            } else {
                std::cout << "NEVER (parallel not beneficial)\n";
            }
            std::cout << "\n";
        }
        
        // Find extreme slowdowns
        std::cout << "\n=== Extreme Slowdowns (Speedup < 0.5) ===\n";
        bool found_extreme = false;
        for (const auto& result : results_) {
            if (result.parallel_speedup < 0.5) {
                std::cout << result.distribution_type << " " << result.operation_type 
                          << " at size " << result.data_size 
                          << ": " << result.parallel_speedup << "x speedup ("
                          << (1.0/result.parallel_speedup) << "x slowdown)\n";
                found_extreme = true;
            }
        }
        if (!found_extreme) {
            std::cout << "No extreme slowdowns found.\n";
        }
    }
    
    void saveResults() {
        std::ofstream csv_file("parallel_threshold_benchmark_results.csv");
        csv_file << "Distribution,Operation,DataSize,SerialTime_us,SIMDTime_us,ParallelTime_us,ParallelSpeedup,SIMDSpeedup,ParallelBeneficial\n";
        
        for (const auto& result : results_) {
            csv_file << result.distribution_type << ","
                     << result.operation_type << ","
                     << result.data_size << ","
                     << result.serial_time_us << ","
                     << result.simd_time_us << ","
                     << result.parallel_time_us << ","
                     << result.parallel_speedup << ","
                     << result.simd_speedup << ","
                     << (result.parallel_beneficial ? "true" : "false") << "\n";
        }
        
        std::cout << "\n=== Results saved to parallel_threshold_benchmark_results.csv ===\n";
    }
};

int main() {
    try {
        ParallelThresholdBenchmark benchmark;
        benchmark.runAllBenchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}

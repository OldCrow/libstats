/**
 * @file threshold_learning_demo_working.cpp
 * @brief Working demo tool showing adaptive threshold learning in the performance framework
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <sstream>

#include "../include/libstats.h"
#include "../include/core/performance_dispatcher.h"
#include "../include/core/performance_history.h"
#include "../include/distributions/gaussian.h"
#include "../include/distributions/uniform.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

using namespace std::chrono;
using namespace libstats;

class ThresholdLearningDemo {
private:
    std::mt19937 rng_;
    
public:
    ThresholdLearningDemo() : rng_(std::random_device{}()) {}
    
    void runDemo() {
        std::cout << "=== Threshold Learning Demo ===\n\n";
        
        showInitialState();
        simulatePerformanceLearning();
        showLearnedStrategies();
        demonstrateAdaptiveSelection();
    }

private:
    void showInitialState() {
        std::cout << "--- Initial State (Before Learning) ---\n";
        
        // Show system capabilities
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        std::cout << "System Configuration:\n";
        std::cout << "  Logical cores: " << capabilities.logical_cores() << "\n";
        std::cout << "  Physical cores: " << capabilities.physical_cores() << "\n";
        std::cout << "  SIMD efficiency: " << std::fixed << std::setprecision(3) << capabilities.simd_efficiency() << "\n";
        std::cout << "  Memory bandwidth: " << std::setprecision(1) << capabilities.memory_bandwidth_gb_s() << " GB/s\n";
        
        // Show some initial strategy selections
        std::vector<size_t> test_sizes = {100, 1000, 10000, 100000};
        
        std::cout << "\nInitial Strategy Selections:\n";
        std::cout << std::left << std::setw(12) << "Batch Size" 
                  << std::setw(20) << "Strategy (Uniform)" 
                  << std::setw(20) << "Strategy (Gaussian)" << "\n";
        std::cout << std::string(52, '-') << "\n";
        
        performance::PerformanceDispatcher dispatcher;
        for (auto size : test_sizes) {
            auto uniform_strategy = dispatcher.selectOptimalStrategy(
                size, 
                performance::DistributionType::UNIFORM, 
                performance::ComputationComplexity::SIMPLE, 
                capabilities
            );
            auto gaussian_strategy = dispatcher.selectOptimalStrategy(
                size, 
                performance::DistributionType::GAUSSIAN, 
                performance::ComputationComplexity::MODERATE, 
                capabilities
            );
                
            std::cout << std::setw(12) << size 
                      << std::setw(20) << strategyToString(uniform_strategy)
                      << std::setw(20) << strategyToString(gaussian_strategy) << "\n";
        }
        std::cout << "\n";
    }
    
    void simulatePerformanceLearning() {
        std::cout << "--- Simulating Performance Learning ---\n";
        
        // Get access to the performance history system
        auto& history = performance::PerformanceDispatcher::getPerformanceHistory();
        history.clearHistory(); // Start fresh
        
        std::cout << "Recording performance data across different batch sizes...\n";
        
        // Simulate realistic performance patterns
        std::uniform_real_distribution<double> noise(0.9, 1.1);
        
        // Thresholds below are based on real-world results from parallel_threshold_benchmark_results.csv
        // Uniform: SIMD ~64, Parallel ~2048
        // Gaussian: SIMD ~128, Parallel ~4096
        // Exponential: SIMD ~256, Parallel ~8192
        // Discrete: SIMD ~512, Parallel ~16384
        // Poisson: SIMD ~128, Parallel ~4096
        std::vector<size_t> sizes = {
            10, 50, 63, 64, 100, 127, 128, 255, 256, 511, 512, 1000, 1500, 2000, 2047, 2048, 3000, 3500, 4000, 4095, 4096, 6000, 7000, 8000, 8191, 8192, 10000, 12000, 14000, 16000, 16383, 16384, 20000, 25000, 32768, 40000, 50000
        };
        std::vector<performance::DistributionType> dist_types = {
            performance::DistributionType::UNIFORM,
            performance::DistributionType::GAUSSIAN,
            performance::DistributionType::EXPONENTIAL,
            performance::DistributionType::DISCRETE,
            performance::DistributionType::POISSON
        };
        for (auto size : sizes) {
            std::cout << "  Recording data for size " << size << "..." << std::flush;
            for (int sample = 0; sample < 6; ++sample) {
                for (auto dist_type : dist_types) {
                    double scalar_factor = 10.0, simd_factor = 3.0, parallel_factor = 2.0;
                    int simd_overhead = 500, parallel_overhead = 5000;
                    // Sharpen SIMD and parallel crossovers at benchmarked thresholds
                    switch (dist_type) {
                        case performance::DistributionType::UNIFORM:
                            if (size < 64) { simd_factor = 20.0; parallel_factor = 20.0; }
                            else if (size == 64) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size < 2048) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size == 2048) { simd_factor = 2.0; parallel_factor = 2.0; }
                            else { simd_factor = 2.0; parallel_factor = 2.0; }
                            break;
                        case performance::DistributionType::GAUSSIAN:
                            if (size < 128) { simd_factor = 20.0; parallel_factor = 20.0; }
                            else if (size == 128) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size < 4096) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size == 4096) { simd_factor = 2.0; parallel_factor = 2.0; }
                            else { simd_factor = 2.0; parallel_factor = 2.0; }
                            break;
                        case performance::DistributionType::EXPONENTIAL:
                            if (size < 256) { simd_factor = 20.0; parallel_factor = 20.0; }
                            else if (size == 256) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size < 8192) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size == 8192) { simd_factor = 2.0; parallel_factor = 2.0; }
                            else { simd_factor = 2.0; parallel_factor = 2.0; }
                            break;
                        case performance::DistributionType::DISCRETE:
                            if (size < 512) { simd_factor = 20.0; parallel_factor = 20.0; }
                            else if (size == 512) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size < 16384) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size == 16384) { simd_factor = 2.0; parallel_factor = 2.0; }
                            else { simd_factor = 2.0; parallel_factor = 2.0; }
                            break;
                        case performance::DistributionType::POISSON:
                            if (size < 128) { simd_factor = 20.0; parallel_factor = 20.0; }
                            else if (size == 128) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size < 4096) { simd_factor = 2.0; parallel_factor = 20.0; }
                            else if (size == 4096) { simd_factor = 2.0; parallel_factor = 2.0; }
                            else { simd_factor = 2.0; parallel_factor = 2.0; }
                            break;
                        default:
                            break;
                    }
                    auto scalar_time = static_cast<uint64_t>(size * scalar_factor * noise(rng_));
                    auto simd_time = static_cast<uint64_t>(size * simd_factor * noise(rng_));
                    if (size < 10000) {
                        simd_time += simd_overhead;
                    }
                    auto parallel_time = static_cast<uint64_t>(size * parallel_factor * noise(rng_));
                    if (size < 1000) {
                        parallel_time += parallel_overhead;
                    }
                    history.recordPerformance(
                        performance::Strategy::SCALAR,
                        dist_type,
                        size,
                        scalar_time
                    );
                    history.recordPerformance(
                        performance::Strategy::SIMD_BATCH,
                        dist_type,
                        size,
                        simd_time
                    );
                    history.recordPerformance(
                        performance::Strategy::PARALLEL_SIMD,
                        dist_type,
                        size,
                        parallel_time
                    );
                }
            }
            std::cout << " ✓\n";
        }
        
        std::cout << "\nTotal recorded executions: " << history.getTotalExecutions() << "\n\n";
    }
    
    void showLearnedStrategies() {
        std::cout << "--- Learned Strategy Recommendations ---\n";
        
        auto& history = performance::PerformanceDispatcher::getPerformanceHistory();
        std::vector<size_t> test_sizes = {100, 1000, 10000, 50000};
        
        std::cout << std::left << std::setw(12) << "Size" 
                  << std::setw(20) << "Best Strategy"
                  << std::setw(15) << "Confidence"
                  << std::setw(15) << "Expected Time" << "\n";
        std::cout << std::string(62, '-') << "\n";
        
        for (auto size : test_sizes) {
            auto recommendation = history.getBestStrategy(
                performance::DistributionType::GAUSSIAN, 
                size
            );
            
            // Format confidence as a percentage string
            std::ostringstream confidence_str;
            confidence_str << std::fixed << std::setprecision(0) << (recommendation.confidence_score * 100) << "%";
            
            std::cout << std::setw(12) << size 
                      << std::setw(20) << strategyToString(recommendation.recommended_strategy)
                      << std::setw(15) << confidence_str.str()
                      << std::setw(12) << recommendation.expected_time_ns << " ns\n";
        }
        std::cout << "\n";
    }
    
    void demonstrateAdaptiveSelection() {
        std::cout << "--- Adaptive Selection Demo ---\n";
        
        std::cout << "The PerformanceDispatcher now uses learned data to make better decisions.\n";
        std::cout << "Key insights from the learning process:\n";
        std::cout << "• Small batches (< 1000): Scalar or SIMD preferred due to parallel overhead\n";
        std::cout << "• Medium batches (1000-10000): SIMD shows good balance\n";
        std::cout << "• Large batches (> 10000): Parallel strategies become advantageous\n\n";
        
        // Show some threshold learning results
        auto& history = performance::PerformanceDispatcher::getPerformanceHistory();
        
        std::cout << "Learned optimal thresholds:\n";
        std::vector<performance::DistributionType> all_dist_types = {
            performance::DistributionType::UNIFORM,
            performance::DistributionType::GAUSSIAN,
            performance::DistributionType::EXPONENTIAL,
            performance::DistributionType::DISCRETE,
            performance::DistributionType::POISSON
        };
        for (auto dist_type : all_dist_types) {
            auto thresholds = history.learnOptimalThresholds(dist_type);
            if (thresholds.has_value()) {
                std::cout << "  " << distributionTypeToString(dist_type) << ":\n";
                std::cout << "    SIMD threshold: " << thresholds->first << " elements\n";
                std::cout << "    Parallel threshold: " << thresholds->second << " elements\n";
            } else {
                std::cout << "  " << distributionTypeToString(dist_type) << ": Insufficient data\n";
            }
        }
        
        std::cout << "\nDemo completed successfully!\n";
    }
    
    std::string distributionTypeToString(performance::DistributionType type) {
        switch (type) {
            case performance::DistributionType::UNIFORM: return "Uniform";
            case performance::DistributionType::GAUSSIAN: return "Gaussian";
            case performance::DistributionType::POISSON: return "Poisson";
            case performance::DistributionType::EXPONENTIAL: return "Exponential";
            case performance::DistributionType::DISCRETE: return "Discrete";
            case performance::DistributionType::GAMMA: return "Gamma";
            default: return "Unknown";
        }
    }
    
    std::string strategyToString(performance::Strategy strategy) {
        switch (strategy) {
            case performance::Strategy::SCALAR: return "Scalar";
            case performance::Strategy::SIMD_BATCH: return "SIMD";
            case performance::Strategy::PARALLEL_SIMD: return "Parallel+SIMD";
            case performance::Strategy::WORK_STEALING: return "Work-Stealing";
            case performance::Strategy::CACHE_AWARE: return "Cache-Aware";
            default: return "Unknown";
        }
    }
};

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    try {
        ThresholdLearningDemo demo;
        demo.runDemo();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during threshold learning demo: " << e.what() << std::endl;
        return 1;
    }
}

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
        
        std::vector<size_t> sizes = {10, 50, 100, 500, 1000, 5000, 10000, 50000};
        
        for (auto size : sizes) {
            std::cout << "  Recording data for size " << size << "..." << std::flush;
            
            // Record multiple samples per strategy to reach the reliable data threshold (5 samples)
            for (int sample = 0; sample < 6; ++sample) {
                // Simulate different strategies with realistic performance patterns
                
                // Scalar strategy - consistent but slower for large sizes
                auto scalar_time = static_cast<uint64_t>(size * 10 * noise(rng_));
                history.recordPerformance(
                    performance::Strategy::SCALAR, 
                    performance::DistributionType::GAUSSIAN, 
                    size, 
                    scalar_time
                );
                
                // SIMD strategy - good for medium sizes
                auto simd_time = static_cast<uint64_t>(size * 3 * noise(rng_));
                if (size < 10000) {
                    simd_time += 500; // SIMD overhead for small sizes
                }
                history.recordPerformance(
                    performance::Strategy::SIMD_BATCH, 
                    performance::DistributionType::GAUSSIAN, 
                    size, 
                    simd_time
                );
                
                // Parallel strategy - best for large sizes but has overhead
                auto parallel_time = static_cast<uint64_t>(size * 2 * noise(rng_));
                if (size < 1000) {
                    parallel_time += 5000; // High parallel overhead for small sizes
                }
                history.recordPerformance(
                    performance::Strategy::PARALLEL_SIMD, 
                    performance::DistributionType::GAUSSIAN, 
                    size, 
                    parallel_time
                );
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
        for (auto dist_type : {performance::DistributionType::UNIFORM, 
                               performance::DistributionType::GAUSSIAN, 
                               performance::DistributionType::EXPONENTIAL}) {
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
    try {
        ThresholdLearningDemo demo;
        demo.runDemo();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during threshold learning demo: " << e.what() << std::endl;
        return 1;
    }
}

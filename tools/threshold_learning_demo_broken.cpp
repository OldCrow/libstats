/**
 * @file threshold_learning_demo.cpp
 * @brief Demo tool showing adaptive threshold learning in the performance framework
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <thread>

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
        runLearningWorkloads();
        showLearnedThresholds();
        demonstrateAdaptiveSelection();
    }

private:
    void showInitialState() {
        std::cout << "--- Initial State (Before Learning) ---\n";
        
        // Show some initial strategy selections
        std::vector<size_t> test_sizes = {500, 1000, 2000, 5000, 10000, 20000};
        
        std::cout << std::left << std::setw(12) << "Batch Size" 
                  << std::setw(20) << "Strategy (Uniform)" 
                  << std::setw(20) << "Strategy (Gaussian)" << "\n";
        std::cout << std::string(52, '-') << "\n";
        
        for (auto size : test_sizes) {
            auto uniform_strategy = PerformanceDispatcher::selectStrategy(
                size, DistributionType::Uniform, ComputationComplexity::Simple);
            auto gaussian_strategy = PerformanceDispatcher::selectStrategy(
                size, DistributionType::Gaussian, ComputationComplexity::Moderate);
                
            std::cout << std::setw(12) << size 
                      << std::setw(20) << strategyToString(uniform_strategy)
                      << std::setw(20) << strategyToString(gaussian_strategy) << "\n";
        }
        std::cout << "\n";
    }
    
    void runLearningWorkloads() {
        std::cout << "--- Running Learning Workloads ---\n";
        
        // Create distributions for testing
        UniformDistribution uniform(0.0, 1.0);
        GaussianDistribution gaussian(0.0, 1.0);
        
        std::vector<size_t> workload_sizes = {
            100, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000
        };
        
        for (auto size : workload_sizes) {
            std::cout << "Learning from workload size: " << size << "..." << std::flush;
            
            // Generate test data
            std::vector<double> data(size);
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            for (auto& val : data) {
                val = dis(rng_);
            }
            
            // Run uniform distribution workloads with timing
            runTimedWorkload(uniform, data, DistributionType::Uniform, ComputationComplexity::Simple);
            runTimedWorkload(gaussian, data, DistributionType::Gaussian, ComputationComplexity::Moderate);
            
            std::cout << " ✓\n";
        }
        std::cout << "\n";
    }
    
    void runTimedWorkload(const auto& distribution, const std::vector<double>& data,
                         DistributionType dist_type, ComputationComplexity complexity) {
        std::vector<double> results(data.size());
        
        // Try different strategies and record timings
        std::vector<ExecutionStrategy> strategies = {
            ExecutionStrategy::Serial,
            ExecutionStrategy::SIMD,
            ExecutionStrategy::Parallel
        };
        
        for (auto strategy : strategies) {
            auto start = high_resolution_clock::now();
            
            // Simulate workload execution based on strategy
            switch (strategy) {
                case ExecutionStrategy::Serial:
                    for (size_t i = 0; i < data.size(); ++i) {
                        results[i] = distribution.getProbability(data[i]);
                    }
                    break;
                    
                case ExecutionStrategy::SIMD:
                    distribution.getProbabilityBatch(data.data(), results.data(), data.size());
                    break;
                    
                case ExecutionStrategy::Parallel:
                    // Simulate parallel execution with some overhead
                    if (data.size() > 1000) {
                        std::this_thread::sleep_for(std::chrono::microseconds(10)); // Thread overhead
                        distribution.getProbabilityBatch(data.data(), results.data(), data.size());
                    } else {
                        // Small data - parallel is likely slower
                        std::this_thread::sleep_for(std::chrono::microseconds(50));
                        for (size_t i = 0; i < data.size(); ++i) {
                            results[i] = distribution.getProbability(data[i]);
                        }
                    }
                    break;
                    
                default:
                    break;
            }
            
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start).count();
            
            // Record the performance data
            PerformanceDispatcher::recordPerformance(data.size(), dist_type, complexity, strategy, duration);
        }
    }
    
    void showLearnedThresholds() {
        std::cout << "--- Learned Performance Statistics ---\n";
        
        // Get performance statistics for different workloads
        std::vector<std::tuple<size_t, DistributionType, ComputationComplexity>> workloads = {
            {1000, DistributionType::Uniform, ComputationComplexity::Simple},
            {5000, DistributionType::Uniform, ComputationComplexity::Simple},
            {10000, DistributionType::Uniform, ComputationComplexity::Simple},
            {1000, DistributionType::Gaussian, ComputationComplexity::Moderate},
            {5000, DistributionType::Gaussian, ComputationComplexity::Moderate},
            {10000, DistributionType::Gaussian, ComputationComplexity::Moderate}
        };
        
        std::cout << std::left << std::setw(8) << "Size" 
                  << std::setw(12) << "Distribution" 
                  << std::setw(12) << "Complexity"
                  << std::setw(12) << "Best Strategy"
                  << std::setw(15) << "Avg Time (μs)" << "\n";
        std::cout << std::string(70, '-') << "\n";
        
        for (const auto& [size, dist_type, complexity] : workloads) {
            auto stats = PerformanceDispatcher::getPerformanceHistory().getPerformanceStats(size, dist_type, complexity);
            if (stats.has_value()) {
                auto recommended = PerformanceDispatcher::getPerformanceHistory().recommendStrategy(size, dist_type, complexity);
                
                std::cout << std::setw(8) << size 
                          << std::setw(12) << distributionTypeToString(dist_type)
                          << std::setw(12) << complexityToString(complexity)
                          << std::setw(12) << strategyToString(recommended)
                          << std::setw(15) << std::fixed << std::setprecision(1) << stats->avg_execution_time.load() << "\n";
            }
        }
        std::cout << "\n";
    }
    
    void demonstrateAdaptiveSelection() {
        std::cout << "--- Adaptive Strategy Selection (After Learning) ---\n";
        
        std::vector<size_t> test_sizes = {500, 1000, 2000, 5000, 10000, 20000};
        
        std::cout << std::left << std::setw(12) << "Batch Size" 
                  << std::setw(20) << "Strategy (Uniform)" 
                  << std::setw(20) << "Strategy (Gaussian)" 
                  << std::setw(15) << "Confidence" << "\n";
        std::cout << std::string(67, '-') << "\n";
        
        for (auto size : test_sizes) {
            auto uniform_strategy = PerformanceDispatcher::selectStrategy(
                size, DistributionType::Uniform, ComputationComplexity::Simple);
            auto gaussian_strategy = PerformanceDispatcher::selectStrategy(
                size, DistributionType::Gaussian, ComputationComplexity::Moderate);
            
            // Check if we have learned data for this size
            auto uniform_stats = PerformanceDispatcher::getPerformanceHistory().getPerformanceStats(
                size, DistributionType::Uniform, ComputationComplexity::Simple);
            
            std::string confidence = uniform_stats.has_value() ? "High" : "Fallback";
            
            std::cout << std::setw(12) << size 
                      << std::setw(20) << strategyToString(uniform_strategy)
                      << std::setw(20) << strategyToString(gaussian_strategy)
                      << std::setw(15) << confidence << "\n";
        }
        std::cout << "\n";
        
        showPerformanceSummary();
    }
    
    void showPerformanceSummary() {
        std::cout << "--- Performance Learning Summary ---\n";
        
        auto& history = PerformanceDispatcher::getPerformanceHistory();
        
        std::cout << "Total recorded measurements: " << history.getTotalMeasurements() << "\n";
        std::cout << "Unique workload configurations: " << history.getUniqueConfigurations() << "\n";
        
        // Show adaptation examples
        std::cout << "\nAdaptation Examples:\n";
        std::cout << "• Small workloads (< 1000 elements): Prefer Serial/SIMD to avoid thread overhead\n";
        std::cout << "• Medium workloads (1000-10000 elements): Balance between SIMD and Parallel\n";
        std::cout << "• Large workloads (> 10000 elements): Prefer Parallel+SIMD for maximum throughput\n";
        std::cout << "• Complex distributions: Higher threshold for parallelization due to computation cost\n";
        std::cout << "\n";
    }
    
    std::string distributionTypeToString(DistributionType type) {
        switch (type) {
            case DistributionType::Uniform: return "Uniform";
            case DistributionType::Gaussian: return "Gaussian";
            case DistributionType::Poisson: return "Poisson";
            case DistributionType::Exponential: return "Exponential";
            case DistributionType::Discrete: return "Discrete";
            default: return "Unknown";
        }
    }
    
    std::string complexityToString(ComputationComplexity complexity) {
        switch (complexity) {
            case ComputationComplexity::Simple: return "Simple";
            case ComputationComplexity::Moderate: return "Moderate";
            case ComputationComplexity::Complex: return "Complex";
            default: return "Unknown";
        }
    }
    
    std::string strategyToString(ExecutionStrategy strategy) {
        switch (strategy) {
            case ExecutionStrategy::Serial: return "Serial";
            case ExecutionStrategy::SIMD: return "SIMD";
            case ExecutionStrategy::Parallel: return "Parallel";
            case ExecutionStrategy::ParallelSIMD: return "Parallel+SIMD";
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

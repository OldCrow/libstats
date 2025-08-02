/**
 * @file performance_dispatcher_tool.cpp
 * @brief Interactive tool to test and analyze the PerformanceDispatcher system
 * 
 * This tool demonstrates the Phase 3 performance optimization framework including:
 * - SystemCapabilities detection and benchmarking  
 * - PerformanceDispatcher strategy selection
 * - PerformanceHistory learning and adaptation
 * - Real-time threshold optimization
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>
#include <map>
#include <algorithm>
#include <thread>
#include <sstream>

#include "../include/core/performance_dispatcher.h"
#include "../include/core/performance_history.h"

using namespace libstats::performance;
using namespace std::chrono;

class PerformanceDispatcherTool {
private:
    PerformanceDispatcher dispatcher_;
    const SystemCapabilities& system_;
    std::mt19937 rng_;
    
public:
    PerformanceDispatcherTool() : system_(SystemCapabilities::current()), rng_(42) {}
    
    void run() {
        std::cout << "=== LibStats Performance Dispatcher Tool ===\n\n";
        
        showSystemCapabilities();
        std::cout << "\n" << std::string(60, '=') << "\n\n";
        
        demonstrateStrategySelection();
        std::cout << "\n" << std::string(60, '=') << "\n\n";
        
        demonstratePerformanceLearning();
        std::cout << "\n" << std::string(60, '=') << "\n\n";
        
        runInteractiveMode();
    }
    
private:
    void showSystemCapabilities() {
        std::cout << "SYSTEM CAPABILITIES ANALYSIS\n";
        std::cout << std::string(30, '-') << "\n";
        
        // CPU Information
        std::cout << "CPU Cores:\n";
        std::cout << "  Logical:  " << system_.logical_cores() << "\n";
        std::cout << "  Physical: " << system_.physical_cores() << "\n";
        
        // Cache Information  
        std::cout << "\nCache Hierarchy:\n";
        std::cout << "  L1: " << system_.l1_cache_size() / 1024 << " KB\n";
        std::cout << "  L2: " << system_.l2_cache_size() / 1024 << " KB\n";
        std::cout << "  L3: " << system_.l3_cache_size() / 1024 << " KB\n";
        
        // SIMD Capabilities
        std::cout << "\nSIMD Support:\n";
        std::cout << "  SSE2:     " << (system_.has_sse2() ? "✓" : "✗") << "\n";
        std::cout << "  AVX:      " << (system_.has_avx() ? "✓" : "✗") << "\n";
        std::cout << "  AVX2:     " << (system_.has_avx2() ? "✓" : "✗") << "\n";
        std::cout << "  AVX-512:  " << (system_.has_avx512() ? "✓" : "✗") << "\n";
        std::cout << "  NEON:     " << (system_.has_neon() ? "✓" : "✗") << "\n";
        
        // Performance Characteristics
        std::cout << "\nPerformance Metrics (Benchmarked):\n";
        std::cout << "  SIMD Efficiency:     " << std::fixed << std::setprecision(4) 
                  << system_.simd_efficiency() << "\n";
        std::cout << "  Threading Overhead:  " << std::fixed << std::setprecision(1)
                  << system_.threading_overhead_ns() << " ns\n";
        std::cout << "  Memory Bandwidth:    " << std::fixed << std::setprecision(2)
                  << system_.memory_bandwidth_gb_s() << " GB/s\n";
    }
    
    void demonstrateStrategySelection() {
        std::cout << "STRATEGY SELECTION DEMONSTRATION\n";
        std::cout << std::string(35, '-') << "\n";
        
        // Test different batch sizes and show strategy selection
        std::vector<size_t> test_sizes = {10, 100, 1000, 10000, 100000, 1000000};
        std::vector<DistributionType> distributions = {
            DistributionType::UNIFORM, DistributionType::GAUSSIAN, 
            DistributionType::EXPONENTIAL, DistributionType::POISSON,
            DistributionType::DISCRETE
        };
        
        std::cout << std::left << std::setw(12) << "Batch Size"
                  << std::setw(14) << "Distribution" 
                  << std::setw(15) << "Complexity"
                  << std::setw(18) << "Selected Strategy" << "\n";
        std::cout << std::string(62, '-') << "\n";
        
        for (auto size : test_sizes) {
            for (auto dist : distributions) {
                for (auto complexity : {ComputationComplexity::SIMPLE, ComputationComplexity::COMPLEX}) {
                    auto strategy = dispatcher_.selectOptimalStrategy(size, dist, complexity, system_);
                    
                    std::cout << std::setw(12) << size
                              << std::setw(14) << distributionName(dist)
                              << std::setw(15) << complexityName(complexity)
                              << std::setw(18) << strategyName(strategy) << "\n";
                }
            }
        }
    }
    
    void demonstratePerformanceLearning() {
        std::cout << "PERFORMANCE LEARNING DEMONSTRATION\n";
        std::cout << std::string(37, '-') << "\n";
        
        auto& history = PerformanceDispatcher::getPerformanceHistory();
        history.clearHistory(); // Start fresh for demonstration
        
        std::cout << "Simulating performance data collection...\n";
        
        // Simulate collecting performance data over time
        simulatePerformanceData(history);
        
        std::cout << "\nTotal recorded executions: " << history.getTotalExecutions() << "\n";
        
        // Show learned thresholds
        std::cout << "\nLearned Optimal Thresholds:\n";
        for (auto dist : {DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL, DistributionType::UNIFORM}) {
            auto thresholds = history.learnOptimalThresholds(dist);
            if (thresholds.has_value()) {
                std::cout << "  " << distributionName(dist) << ":\n";
                std::cout << "    SIMD threshold:     " << thresholds->first << "\n";
                std::cout << "    Parallel threshold: " << thresholds->second << "\n";
            } else {
                std::cout << "  " << distributionName(dist) << ": Insufficient data\n";
            }
        }
        
        // Show strategy recommendations
        std::cout << "\nStrategy Recommendations (with confidence):\n";
        std::vector<size_t> test_sizes = {100, 1000, 10000};
        for (auto size : test_sizes) {
            auto recommendation = history.getBestStrategy(DistributionType::GAUSSIAN, size);
            // Format confidence as a percentage string
            std::ostringstream confidence_str;
            confidence_str << std::fixed << std::setprecision(0) << (recommendation.confidence_score * 100) << "%";
            
            std::cout << "  Batch size " << size << ": " 
                      << strategyName(recommendation.recommended_strategy)
                      << " (confidence: " << confidence_str.str() << ")\n";
        }
    }
    
    void simulatePerformanceData(PerformanceHistory& history) {
        // Simulate realistic performance patterns
        std::uniform_real_distribution<double> noise(0.9, 1.1);
        
        // Small batches: scalar is typically faster
        for (int i = 0; i < 20; ++i) {
            auto scalar_time = static_cast<uint64_t>(100 * noise(rng_)); // Fast
            auto simd_time = static_cast<uint64_t>(150 * noise(rng_));   // Slower due to overhead
            
            history.recordPerformance(Strategy::SCALAR, DistributionType::GAUSSIAN, 50, scalar_time);
            history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN, 50, simd_time);
        }
        
        // Medium batches: SIMD becomes advantageous
        for (int i = 0; i < 20; ++i) {
            auto scalar_time = static_cast<uint64_t>(5000 * noise(rng_));
            auto simd_time = static_cast<uint64_t>(2000 * noise(rng_));   // Much faster
            
            history.recordPerformance(Strategy::SCALAR, DistributionType::GAUSSIAN, 1000, scalar_time);
            history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN, 1000, simd_time);
        }
        
        // Large batches: parallel becomes best
        for (int i = 0; i < 20; ++i) {
            auto simd_time = static_cast<uint64_t>(15000 * noise(rng_));
            auto parallel_time = static_cast<uint64_t>(8000 * noise(rng_)); // Even faster
            
            history.recordPerformance(Strategy::SIMD_BATCH, DistributionType::GAUSSIAN, 10000, simd_time);
            history.recordPerformance(Strategy::PARALLEL_SIMD, DistributionType::GAUSSIAN, 10000, parallel_time);
        }
        
        // Add some data for other distributions
        for (auto dist : {DistributionType::EXPONENTIAL, DistributionType::UNIFORM}) {
            for (int i = 0; i < 10; ++i) {
                history.recordPerformance(Strategy::SCALAR, dist, 100, static_cast<uint64_t>(200 * noise(rng_)));
                history.recordPerformance(Strategy::SIMD_BATCH, dist, 1000, static_cast<uint64_t>(1800 * noise(rng_)));
                history.recordPerformance(Strategy::PARALLEL_SIMD, dist, 10000, static_cast<uint64_t>(7500 * noise(rng_)));
            }
        }
    }
    
    void runInteractiveMode() {
        std::cout << "INTERACTIVE MODE\n";
        std::cout << std::string(16, '-') << "\n";
        std::cout << "Enter batch sizes to test strategy selection (0 to exit):\n";
        
        size_t batch_size;
        while (std::cout << "> " && std::cin >> batch_size && batch_size != 0) {
            std::cout << "\nTesting batch size: " << batch_size << "\n";
            
            for (auto dist : {DistributionType::UNIFORM, DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL}) {
                for (auto complexity : {ComputationComplexity::SIMPLE, ComputationComplexity::COMPLEX}) {
                    auto strategy = dispatcher_.selectOptimalStrategy(batch_size, dist, complexity, system_);
                    std::cout << "  " << distributionName(dist) << " (" << complexityName(complexity) << "): "
                              << strategyName(strategy) << "\n";
                }
            }
            std::cout << "\n";
        }
        
        std::cout << "Interactive mode ended.\n";
    }
    
    // Helper functions for pretty printing
    const char* strategyName(Strategy strategy) {
        switch (strategy) {
            case Strategy::SCALAR: return "SCALAR";
            case Strategy::SIMD_BATCH: return "SIMD_BATCH";
            case Strategy::PARALLEL_SIMD: return "PARALLEL_SIMD";
            case Strategy::WORK_STEALING: return "WORK_STEALING";
            case Strategy::CACHE_AWARE: return "CACHE_AWARE";
            default: return "UNKNOWN";
        }
    }
    
    const char* distributionName(DistributionType dist) {
        switch (dist) {
            case DistributionType::UNIFORM: return "Uniform";
            case DistributionType::GAUSSIAN: return "Gaussian";
            case DistributionType::EXPONENTIAL: return "Exponential";
            case DistributionType::POISSON: return "Poisson";
            case DistributionType::GAMMA: return "Gamma";
            case DistributionType::DISCRETE: return "Discrete";
            default: return "Unknown";
        }
    }
    
    const char* complexityName(ComputationComplexity complexity) {
        switch (complexity) {
            case ComputationComplexity::SIMPLE: return "Simple";
            case ComputationComplexity::MODERATE: return "Moderate";
            case ComputationComplexity::COMPLEX: return "Complex";
            default: return "Unknown";
        }
    }
};

int main() {
    try {
        PerformanceDispatcherTool tool;
        tool.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

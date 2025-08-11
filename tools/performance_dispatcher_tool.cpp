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
#include "tool_utils.h"

using namespace libstats::performance;
using namespace std::chrono;
using namespace libstats::constants;

// Tool-specific simulation constants
namespace {
    constexpr int DEMO_SEED = 42;
    constexpr double SIMULATION_NOISE_MIN = 0.9;
    constexpr double SIMULATION_NOISE_MAX = 1.1;
    
    // Realistic performance simulation parameters (matching threshold_learning_demo)
    namespace timing_simulation {
        // Performance scaling factors for different strategies
        constexpr double SCALAR_PERFORMANCE_FACTOR = 10.0;
        constexpr double SIMD_PERFORMANCE_FACTOR = 3.0;
        constexpr double PARALLEL_PERFORMANCE_FACTOR = 2.0;
        
        // Strategy overhead constants  
        constexpr uint64_t SIMD_SMALL_OVERHEAD = 500;     // Additional time for small SIMD operations
        constexpr uint64_t PARALLEL_BASE_OVERHEAD = 8000; // Base threading overhead
        
        // Size thresholds for overhead application
        constexpr size_t SIMD_OVERHEAD_THRESHOLD = 10000;
    }
    
    namespace batch_sizes {
        // Batch sizes reserved for future interactive testing features
        [[maybe_unused]] constexpr size_t SMALL_BATCH = 50;
        [[maybe_unused]] constexpr size_t MEDIUM_BATCH = 1000;
        [[maybe_unused]] constexpr size_t LARGE_BATCH = 10000;
        [[maybe_unused]] constexpr size_t OTHER_DIST_BATCH = 100;
        [[maybe_unused]] constexpr size_t OTHER_DIST_MEDIUM_BATCH = 1000;
        [[maybe_unused]] constexpr size_t OTHER_DIST_LARGE_BATCH = 10000;
    }
    
    // Sample counts for simulation - reserved for future use
    [[maybe_unused]] constexpr int SAMPLES_PER_STRATEGY = 20;
    [[maybe_unused]] constexpr int OTHER_DIST_SAMPLES = 10;
}

class PerformanceDispatcherTool {
private:
    PerformanceDispatcher dispatcher_;
    const SystemCapabilities& system_;
    std::mt19937 rng_;
    
public:
    PerformanceDispatcherTool() : system_(SystemCapabilities::current()), rng_(DEMO_SEED) {}
    
    void run() {
        using namespace libstats::tools;
        
        // Display tool header with system information
        system_info::displayToolHeader("Performance Dispatcher Tool", 
                                       "Interactive analysis of performance optimization framework");
        
        // Display major sections
        system_info::displaySystemCapabilities();
        demonstrateStrategySelection();
        demonstratePerformanceLearning();
        runInteractiveMode();
        
        std::cout << "Performance dispatcher analysis completed successfully.\n";
    }
    
private:
    void demonstrateStrategySelection() {
        using namespace libstats::tools;
        
        display::sectionHeader("Strategy Selection Demonstration");
        
        // Test different batch sizes and show strategy selection
        std::vector<size_t> test_sizes = {10, 100, 1000, 10000, 100000, 1000000};
        std::vector<DistributionType> distributions = {
            DistributionType::UNIFORM, DistributionType::GAUSSIAN, 
            DistributionType::EXPONENTIAL, DistributionType::POISSON,
            DistributionType::DISCRETE, DistributionType::GAMMA
        };
        
        table::ColumnFormatter formatter({12, 14, 15, 18});
        std::cout << formatter.formatRow({"Batch Size", "Distribution", "Complexity", "Selected Strategy"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        for (auto size : test_sizes) {
            for (auto dist : distributions) {
                for (auto complexity : {ComputationComplexity::SIMPLE, ComputationComplexity::COMPLEX}) {
                    auto strategy = dispatcher_.selectOptimalStrategy(size, dist, complexity, system_);
                    
                    std::cout << formatter.formatRow({std::to_string(size),
                                                    strings::distributionTypeToString(dist),
                                                    strings::complexityToString(complexity),
                                                    strings::strategyToString(strategy)}) << "\n";
                }
            }
        }
        std::cout << "\n";
    }
    
    void demonstratePerformanceLearning() {
        using namespace libstats::tools;
        
        display::sectionHeader("Performance Learning Demonstration");
        
        auto& history = PerformanceDispatcher::getPerformanceHistory();
        history.clearHistory(); // Start fresh for demonstration
        
        std::cout << "Simulating performance data collection...\n\n";
        
        // Simulate collecting performance data over time
        simulatePerformanceData(history);
        
        std::cout << "Total recorded executions: " << history.getTotalExecutions() << "\n\n";
        
        // Show learned thresholds
        display::subsectionHeader("Learned Optimal Thresholds");
        
        table::ColumnFormatter threshold_formatter({15, 20, 20});
        std::cout << threshold_formatter.formatRow({"Distribution", "SIMD Threshold", "Parallel Threshold"}) << "\n";
        std::cout << threshold_formatter.getSeparator() << "\n";
        
        for (auto dist : {DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL, DistributionType::UNIFORM,
                          DistributionType::DISCRETE, DistributionType::POISSON, DistributionType::GAMMA}) {
            auto thresholds = history.learnOptimalThresholds(dist);
            if (thresholds.has_value()) {
                std::cout << threshold_formatter.formatRow({strings::distributionTypeToString(dist),
                                                           std::to_string(thresholds->first),
                                                           std::to_string(thresholds->second)}) << "\n";
            } else {
                std::cout << threshold_formatter.formatRow({strings::distributionTypeToString(dist),
                                                           "Insufficient data", 
                                                           "Insufficient data"}) << "\n";
            }
        }
        
        // Show strategy recommendations
        display::subsectionHeader("Strategy Recommendations (with confidence)");
        
        table::ColumnFormatter rec_formatter({12, 15, 22, 12});
        std::cout << rec_formatter.formatRow({"Batch Size", "Distribution", "Recommended Strategy", "Confidence"}) << "\n";
        std::cout << rec_formatter.getSeparator() << "\n";
        
        std::vector<size_t> test_sizes = {100, 1000, 10000};
        std::vector<DistributionType> rec_distributions = {DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL, DistributionType::UNIFORM,
                                                          DistributionType::DISCRETE, DistributionType::POISSON, DistributionType::GAMMA};
        
        for (auto size : test_sizes) {
            for (auto dist : rec_distributions) {
                auto recommendation = history.getBestStrategy(dist, size);
                std::string confidence_str = format::confidenceToString(recommendation.confidence_score);
                
                std::cout << rec_formatter.formatRow({std::to_string(size),
                                                    strings::distributionTypeToString(dist),
                                                    strings::strategyToDisplayString(recommendation.recommended_strategy),
                                                    confidence_str}) << "\n";
            }
        }
        std::cout << "\n";
    }
    
    void simulatePerformanceData(PerformanceHistory& history) {
        // Simulate realistic performance patterns using the same modeling as threshold_learning_demo
        std::uniform_real_distribution<double> noise(SIMULATION_NOISE_MIN, SIMULATION_NOISE_MAX);
        
        // Performance complexity factors for different distributions
        std::map<DistributionType, double> complexity_factors = {
            {DistributionType::UNIFORM, 1.0},      // Simple - just random scaling
            {DistributionType::DISCRETE, 1.5},    // Simple integer operations
            {DistributionType::EXPONENTIAL, 2.5}, // Moderate - requires exp/log
            {DistributionType::GAUSSIAN, 3.0},    // Moderate - Box-Muller transform
            {DistributionType::POISSON, 4.0},     // Complex - iterative algorithms
            {DistributionType::GAMMA, 5.0}        // Most complex - special functions
        };
        
        // Distribution-specific efficiency characteristics
        std::map<DistributionType, std::pair<double, double>> efficiency_characteristics = {
            {DistributionType::UNIFORM,    {0.40, 0.25}}, // Good SIMD/Parallel efficiency - simple ops
            {DistributionType::DISCRETE,   {0.35, 0.22}}, // Decent efficiency
            {DistributionType::EXPONENTIAL, {0.28, 0.18}}, // Moderate efficiency - transcendental
            {DistributionType::GAUSSIAN,   {0.25, 0.15}}, // Lower efficiency - complex transform
            {DistributionType::POISSON,    {0.22, 0.12}}, // Poor efficiency - iterative
            {DistributionType::GAMMA,      {0.20, 0.10}}  // Worst efficiency - special functions
        };
        
        // More granular sizes around potential crossover points for better threshold learning
        std::vector<size_t> sizes = {10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 25000, 50000};
        
        // All distribution types to simulate
        std::vector<DistributionType> distributions = {
            DistributionType::UNIFORM, DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL,
            DistributionType::DISCRETE, DistributionType::POISSON, DistributionType::GAMMA
        };
        
        for (auto dist_type : distributions) {
            double complexity = complexity_factors[dist_type];
            auto [simd_efficiency, parallel_efficiency] = efficiency_characteristics[dist_type];
            
            for (auto size : sizes) {
                // Record multiple samples per strategy to reach the reliable data threshold (>=5 samples)
                for (int sample = 0; sample < SAMPLES_PER_STRATEGY / 4; ++sample) { // Use fewer samples per size for broader coverage
                    // Scalar strategy - affected by computational complexity
                    auto scalar_time = static_cast<uint64_t>(static_cast<double>(size) * timing_simulation::SCALAR_PERFORMANCE_FACTOR * complexity * noise(rng_));
                    history.recordPerformance(Strategy::SCALAR, dist_type, size, scalar_time);
                    
                    // SIMD strategy - use distribution-specific efficiency with overhead
                    auto simd_time = static_cast<uint64_t>(static_cast<double>(size) * timing_simulation::SIMD_PERFORMANCE_FACTOR * complexity * simd_efficiency * noise(rng_));
                    if (size < timing_simulation::SIMD_OVERHEAD_THRESHOLD) {
                        simd_time += timing_simulation::SIMD_SMALL_OVERHEAD; // SIMD overhead for small sizes
                    }
                    history.recordPerformance(Strategy::SIMD_BATCH, dist_type, size, simd_time);
                    
                    // Parallel strategy - use distribution-specific efficiency with realistic overhead model
                    auto parallel_time = static_cast<uint64_t>(static_cast<double>(size) * timing_simulation::PARALLEL_PERFORMANCE_FACTOR * complexity * parallel_efficiency * noise(rng_));
                    
                    // More realistic parallel overhead model - decreases with complexity and size
                    double complexity_factor = complexity;
                    double overhead_reduction = std::max(1.0, static_cast<double>(size) / 1000.0); // Overhead reduces with size
                    
                    // Base overhead varies by complexity:
                    // - Simple distributions (Uniform): High overhead, needs ~10k+ elements  
                    // - Complex distributions (Gamma): Lower overhead, benefits earlier
                    uint64_t base_overhead = static_cast<uint64_t>(timing_simulation::PARALLEL_BASE_OVERHEAD / complexity_factor / overhead_reduction);
                    parallel_time += base_overhead;
                    history.recordPerformance(Strategy::PARALLEL_SIMD, dist_type, size, parallel_time);
                }
            }
        }
    }
    
    void runInteractiveMode() {
        using namespace libstats::tools;
        
        display::sectionHeader("Interactive Mode");
        
        std::cout << "Enter batch sizes to test strategy selection (0 to exit):\n";
        
        size_t batch_size;
        while (std::cout << "> " && std::cin >> batch_size && batch_size != 0) {
            display::subsectionHeader("Testing batch size: " + std::to_string(batch_size));
            
            table::ColumnFormatter formatter({15, 12, 18});
            std::cout << formatter.formatRow({"Distribution", "Complexity", "Selected Strategy"}) << "\n";
            std::cout << formatter.getSeparator() << "\n";
            
            for (auto dist : {DistributionType::UNIFORM, DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL, 
                              DistributionType::DISCRETE, DistributionType::POISSON, DistributionType::GAMMA}) {
                for (auto complexity : {ComputationComplexity::SIMPLE, ComputationComplexity::COMPLEX}) {
                    auto strategy = dispatcher_.selectOptimalStrategy(batch_size, dist, complexity, system_);
                    std::cout << formatter.formatRow({strings::distributionTypeToString(dist),
                                                    strings::complexityToString(complexity),
                                                    strings::strategyToDisplayString(strategy)}) << "\n";
                }
            }
            std::cout << "\n";
        }
        
        std::cout << "Interactive mode ended.\n";
    }
};

int main() {
    using namespace libstats::tools;
    
    // Use the standard tool runner pattern
    return tool_utils::runTool("Performance Dispatcher Tool", []() {
        PerformanceDispatcherTool tool;
        tool.run();
    });
}

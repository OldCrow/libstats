/**
 * @file empirical_characteristics_demo.cpp
 * @brief Demonstration of empirical distribution characteristics integration
 * 
 * This tool showcases how the performance dispatcher now uses empirically-derived
 * distribution characteristics instead of hardcoded assumptions.
 */

#include "../include/libstats.h"
#include "../include/core/distribution_characteristics.h"
#include "../include/core/performance_dispatcher.h"
#include "tool_utils.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>

using namespace libstats;
using namespace libstats::performance;
using namespace libstats::performance::characteristics;
using namespace libstats::tools::display;
using namespace libstats::tools::strings;

namespace {

void displayCharacteristics() {
    sectionHeader("Empirical Distribution Characteristics");
    
    std::vector<std::pair<std::string, DistributionType>> distributions = {
        {"Uniform", DistributionType::UNIFORM},
        {"Gaussian", DistributionType::GAUSSIAN},
        {"Exponential", DistributionType::EXPONENTIAL},
        {"Discrete", DistributionType::DISCRETE},
        {"Poisson", DistributionType::POISSON},
        {"Gamma", DistributionType::GAMMA}
    };
    
    // Table headers
    std::cout << std::left 
              << std::setw(13) << "Distribution"
              << std::setw(12) << "Complexity"
              << std::setw(12) << "SIMD Eff"
              << std::setw(12) << "Parallel"
              << std::setw(12) << "SIMD Thresh"
              << std::setw(12) << "Par Thresh"
              << std::setw(12) << "Memory"
              << std::setw(12) << "Branching"
              << "\n";
    
    std::cout << std::string(96, '-') << "\n";
    
    for (const auto& [name, dist_type] : distributions) {
        const auto& chars = getCharacteristics(dist_type);
        
        std::cout << std::left
                  << std::setw(13) << name
                  << std::setw(12) << std::fixed << std::setprecision(1) << chars.base_complexity
                  << std::setw(12) << std::fixed << std::setprecision(2) << chars.vectorization_efficiency
                  << std::setw(12) << std::fixed << std::setprecision(2) << chars.parallelization_efficiency
                  << std::setw(12) << chars.min_simd_threshold
                  << std::setw(12) << chars.min_parallel_threshold
                  << std::setw(12) << std::fixed << std::setprecision(2) << chars.memory_access_pattern
                  << std::setw(12) << std::fixed << std::setprecision(2) << chars.branch_prediction_cost
                  << "\n";
    }
    
    std::cout << "\n";
    std::cout << "Key:\n";
    std::cout << "  Complexity:    Computational cost relative to uniform (1.0 = baseline)\n";
    std::cout << "  SIMD Eff:      Vectorization efficiency (0.0-1.0, higher is better)\n";
    std::cout << "  Parallel:      Parallelization efficiency (0.0-1.0, higher is better)\n";
    std::cout << "  SIMD Thresh:   Minimum elements where SIMD becomes beneficial\n";
    std::cout << "  Par Thresh:    Minimum elements where parallelization helps\n";
    std::cout << "  Memory:        Memory access efficiency (1.0 = perfect locality)\n";
    std::cout << "  Branching:     Branch prediction cost factor (1.0 = no branching)\n";
}

void displayScalingFactors() {
    sectionHeader("Expected Performance Scaling");
    
    std::vector<std::pair<std::string, DistributionType>> distributions = {
        {"Uniform", DistributionType::UNIFORM},
        {"Gaussian", DistributionType::GAUSSIAN},
        {"Exponential", DistributionType::EXPONENTIAL},
        {"Discrete", DistributionType::DISCRETE},
        {"Poisson", DistributionType::POISSON},
        {"Gamma", DistributionType::GAMMA}
    };
    
    std::vector<size_t> thread_counts = {2, 4, 8, 16};
    
    std::cout << std::left 
              << std::setw(13) << "Distribution"
              << std::setw(12) << "SIMD (4x)"
              << std::setw(11) << "2 threads"
              << std::setw(11) << "4 threads"
              << std::setw(11) << "8 threads"
              << std::setw(12) << "16 threads"
              << "\n";
    
    std::cout << std::string(76, '-') << "\n";
    
    for (const auto& [name, dist_type] : distributions) {
        const auto& chars = getCharacteristics(dist_type);
        
        std::cout << std::left << std::setw(13) << name;
        
        // SIMD speedup
        double simd_speedup = scaling::calculateSIMDSpeedup(chars);
        std::ostringstream simd_stream;
        simd_stream << std::fixed << std::setprecision(2) << simd_speedup << "x";
        std::cout << std::setw(12) << simd_stream.str();
        
        // Parallel speedups for different thread counts
        for (size_t threads : thread_counts) {
            double parallel_speedup = scaling::calculateParallelSpeedup(chars, threads);
            std::ostringstream parallel_stream;
            parallel_stream << std::fixed << std::setprecision(1) << parallel_speedup << "x";
            std::cout << std::setw(11) << parallel_stream.str();
        }
        
        std::cout << "\n";
    }
    
    std::cout << "\nNote: These are theoretical maximums based on algorithmic analysis.\n";
    std::cout << "      Actual performance depends on system capabilities and data patterns.\n";
}

void demonstrateStrategySelection() {
    sectionHeader("Strategy Selection with Empirical Data");
    
    PerformanceDispatcher dispatcher;
    SystemCapabilities system = SystemCapabilities::current();
    
    std::vector<std::pair<std::string, DistributionType>> distributions = {
        {"Uniform", DistributionType::UNIFORM},
        {"Gaussian", DistributionType::GAUSSIAN},
        {"Exponential", DistributionType::EXPONENTIAL},
        {"Discrete", DistributionType::DISCRETE},
        {"Poisson", DistributionType::POISSON},
        {"Gamma", DistributionType::GAMMA}
    };
    
    std::vector<size_t> batch_sizes = {100, 1000, 10000, 100000};
    
    std::cout << std::left 
              << std::setw(13) << "Distribution"
              << std::setw(12) << "Size=100"
              << std::setw(12) << "Size=1K"
              << std::setw(12) << "Size=10K"
              << std::setw(12) << "Size=100K"
              << "\n";
    
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& [name, dist_type] : distributions) {
        std::cout << std::left << std::setw(13) << name;
        
        for (size_t batch_size : batch_sizes) {
            Strategy strategy = dispatcher.selectOptimalStrategy(
                batch_size, dist_type, ComputationComplexity::MODERATE, system
            );
            
            std::string strategy_str = strategyToString(strategy);
            // Abbreviate for table display
            if (strategy_str == "PARALLEL_SIMD") strategy_str = "PAR_SIMD";
            else if (strategy_str == "SIMD_BATCH") strategy_str = "SIMD";
            else if (strategy_str == "WORK_STEALING") strategy_str = "WORK_ST";
            else if (strategy_str == "CACHE_AWARE") strategy_str = "CACHE";
            else if (strategy_str == "SCALAR") strategy_str = "SCALAR";
            
            std::cout << std::setw(12) << strategy_str;
        }
        std::cout << "\n";
    }
    
    std::cout << "\nStrategy Selection Rationale:\n";
    std::cout << "  • Simple distributions (Uniform, Discrete) prefer SIMD at medium sizes\n";
    std::cout << "  • Complex distributions (Gaussian, Poisson, Gamma) benefit from parallelization earlier\n";
    std::cout << "  • Distributions with poor vectorization (Poisson, Gamma) avoid SIMD strategies\n";
    std::cout << "  • Decisions account for algorithmic complexity, not just batch size\n";
}

void demonstrateAdaptiveLearning() {
    sectionHeader("Adaptive Learning Integration");
    
    // Show how empirical characteristics can be refined
    auto base_chars = getCharacteristics(DistributionType::GAUSSIAN);
    
    std::cout << "Base Gaussian Characteristics:\n";
    std::cout << "  SIMD Efficiency: " << std::fixed << std::setprecision(2) 
              << base_chars.vectorization_efficiency << "\n";
    std::cout << "  Parallel Efficiency: " << std::fixed << std::setprecision(2) 
              << base_chars.parallelization_efficiency << "\n";
    std::cout << "  Base Complexity: " << std::fixed << std::setprecision(1) 
              << base_chars.base_complexity << "\n";
    
    // Simulate learned refinements
    adaptive::LearnedRefinements refinements;
    refinements.simd_efficiency_multiplier = 1.2;      // 20% better than expected
    refinements.parallel_efficiency_multiplier = 0.85; // 15% worse than expected
    refinements.complexity_adjustment = 0.9;           // 10% less complex than expected
    refinements.simd_threshold_offset = -8;            // SIMD beneficial 8 elements earlier
    refinements.learning_confidence = 0.7;             // 70% confidence in learned values
    
    auto refined_chars = adaptive::applyRefinements(base_chars, refinements);
    
    std::cout << "\nAfter Adaptive Learning (70% confidence):\n";
    std::cout << "  SIMD Efficiency: " << std::fixed << std::setprecision(2) 
              << refined_chars.vectorization_efficiency 
              << " (+" << std::fixed << std::setprecision(1) 
              << ((refined_chars.vectorization_efficiency / base_chars.vectorization_efficiency - 1.0) * 100) << "%)\n";
    std::cout << "  Parallel Efficiency: " << std::fixed << std::setprecision(2) 
              << refined_chars.parallelization_efficiency 
              << " (" << std::fixed << std::setprecision(1) 
              << ((refined_chars.parallelization_efficiency / base_chars.parallelization_efficiency - 1.0) * 100) << "%)\n";
    std::cout << "  Base Complexity: " << std::fixed << std::setprecision(1) 
              << refined_chars.base_complexity 
              << " (" << std::fixed << std::setprecision(1) 
              << ((refined_chars.base_complexity / base_chars.base_complexity - 1.0) * 100) << "%)\n";
    std::cout << "  SIMD Threshold: " << refined_chars.min_simd_threshold 
              << " (was " << base_chars.min_simd_threshold << ")\n";
    
    std::cout << "\nAdaptive Learning Benefits:\n";
    std::cout << "  • Starts with empirically-derived baselines instead of assumptions\n";
    std::cout << "  • Learns system-specific refinements over time\n";
    std::cout << "  • Blends empirical knowledge with measured performance\n";
    std::cout << "  • Confidence-weighted adjustments prevent over-fitting\n";
}

} // anonymous namespace

int main() {
    // Initialize performance systems
    libstats::initialize_performance_systems();
    
    sectionHeader("Empirical Distribution Characteristics Demo");
    std::cout << "This demo shows how libstats now uses empirically-derived distribution\n";
    std::cout << "characteristics instead of hardcoded performance assumptions.\n";
    
    displayCharacteristics();
    displayScalingFactors();
    demonstrateStrategySelection();
    demonstrateAdaptiveLearning();
    
    sectionHeader("Summary");
    std::cout << "The empirical characteristics system provides:\n\n";
    std::cout << "1. Data-Driven Baselines:\n";
    std::cout << "   • Characteristics derived from algorithmic analysis\n";
    std::cout << "   • No more magic numbers or arbitrary assumptions\n";
    std::cout << "   • Performance models grounded in computational reality\n\n";
    
    std::cout << "2. Distribution-Aware Strategy Selection:\n";
    std::cout << "   • Considers vectorization efficiency per distribution\n";
    std::cout << "   • Accounts for branch prediction and memory access patterns\n";
    std::cout << "   • Scales thresholds by computational complexity\n\n";
    
    std::cout << "3. Adaptive Learning Integration:\n";
    std::cout << "   • Starts with empirical baselines, not zero knowledge\n";
    std::cout << "   • Learns system-specific refinements over time\n";
    std::cout << "   • Confidence-weighted blending prevents over-correction\n\n";
    
    std::cout << "This foundation enables more accurate performance predictions and\n";
    std::cout << "better strategy selection across different distribution types.\n";
    
    return 0;
}

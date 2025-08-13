/**
 * @file system_inspector.cpp
 * @brief Consolidated system analysis tool combining constants and capabilities inspection
 * 
 * This unified tool consolidates the functionality of:
 * - constants_inspector.cpp (architecture-specific constants analysis)
 * - system_capabilities_inspector.cpp (comprehensive system capability analysis)
 * 
 * The tool provides multiple modes to serve different use cases while eliminating code duplication:
 * - Full mode: Complete comprehensive analysis (default)
 * - Constants mode: Focus on constants and architecture comparison
 * - Performance mode: Focus on capabilities and performance baselines
 * - Quick mode: Basic system information only
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
// Use consolidated header for complete library functionality
#include "tool_utils.h"

using namespace std::chrono;

// Tool-specific constants
namespace {
    constexpr size_t BASELINE_TEST_SIZE = 1000000;
    constexpr int BASELINE_ITERATIONS = 10;
    constexpr int MAX_COMPLEXITY_DEMOS = 1;  // Only show first complexity for brevity
}

// Mode enumeration
enum class InspectionMode {
    FULL,         // Complete analysis (default)
    CONSTANTS,    // Constants-focused analysis
    PERFORMANCE,  // Performance-focused analysis
    QUICK         // Basic system info only
};

class SystemInspector {
public:
    SystemInspector(InspectionMode mode) : mode_(mode) {}
    
    void runInspection() {
        using namespace libstats::tools;
        
        // Display tool header with system information (shared across all modes)
        displayToolHeader();
        
        // Execute mode-specific analysis
        switch (mode_) {
            case InspectionMode::QUICK:
                runQuickInspection();
                break;
            case InspectionMode::CONSTANTS:
                runConstantsInspection();
                break;
            case InspectionMode::PERFORMANCE:
                runPerformanceInspection();
                break;
            case InspectionMode::FULL:
            default:
                runFullInspection();
                break;
        }
        
        // Common validation (shared across all modes)
        tool_utils::validateAndWarnFeatureConsistency();
        
        std::cout << "System inspection completed successfully.\n";
    }

private:
    InspectionMode mode_;
    
    void displayToolHeader() {
        using namespace libstats::tools;
        
        std::string title;
        std::string description;
        
        switch (mode_) {
            case InspectionMode::QUICK:
                title = "System Inspector - Quick Mode";
                description = "Basic system information and capabilities overview";
                break;
            case InspectionMode::CONSTANTS:
                title = "System Inspector - Constants Mode";
                description = "Architecture-specific constants analysis and platform optimizations";
                break;
            case InspectionMode::PERFORMANCE:
                title = "System Inspector - Performance Mode";
                description = "System capabilities analysis with performance measurements";
                break;
            case InspectionMode::FULL:
            default:
                title = "System Inspector - Complete Analysis";
                description = "Comprehensive system analysis: constants, capabilities, and performance";
                break;
        }
        
        system_info::displayToolHeader(title, description);
    }
    
    void runQuickInspection() {
        using namespace libstats::tools;
        
        // Basic system overview
        system_info::displaySystemCapabilities();
        system_info::displayCPUFeatures();
        system_info::displayCacheInfo();
        
        std::cout << "\nUse --constants, --performance, or --full for detailed analysis.\n\n";
    }
    
    void runConstantsInspection() {
        using namespace libstats::tools;
        
        // Shared basic info
        displayBasicSystemInfo();
        
        // Constants-specific sections
        displaySelectedArchitecture();
        system_info::displayAdaptiveConstants();
        displayAdditionalAdaptiveConstants();
        system_info::displayPlatformConstants();
        displayCacheThresholds();
        displayArchitectureComparison();
    }
    
    void runPerformanceInspection() {
        using namespace libstats::tools;
        
        // Shared basic info
        displayBasicSystemInfo();
        
        // Performance-specific sections
        displayCPUTopology();
        displaySIMDCapabilitiesDetailed();
        displayPerformanceBaselines();
        displayDispatcherConfiguration();
        system_info::displayPlatformConstants();
        system_info::displayAdaptiveConstants();
    }
    
    void runFullInspection() {
        using namespace libstats::tools;
        
        display::sectionHeader("COMPLETE SYSTEM ANALYSIS", '=');
        
        // Part 1: System Overview
        display::sectionHeader("System Overview", '-');
        system_info::displaySystemCapabilities();
        displayCPUTopology();
        
        // Part 2: Hardware Details
        display::sectionHeader("Hardware Analysis", '-');
        system_info::displayCacheInfo();
        system_info::displayCPUFeatures();
        displaySIMDCapabilitiesDetailed();
        
        // Part 3: Performance Analysis
        display::sectionHeader("Performance Analysis", '-');
        displayPerformanceBaselines();
        displayDispatcherConfiguration();
        
        // Part 4: Constants Analysis
        display::sectionHeader("Constants Analysis", '-');
        displaySelectedArchitecture();
        system_info::displayAdaptiveConstants();
        displayAdditionalAdaptiveConstants();
        system_info::displayPlatformConstants();
        displayCacheThresholds();
        displayArchitectureComparison();
        
        std::cout << "\n";
        display::sectionHeader("Analysis Complete", '=');
    }
    
    // ========================================================================
    // SHARED UTILITY FUNCTIONS (eliminate duplication)
    // ========================================================================
    
    void displayBasicSystemInfo() {
        using namespace libstats::tools;
        system_info::displayCPUFeatures();
        system_info::displayCacheInfo();
    }
    
    // ========================================================================
    // CONSTANTS-SPECIFIC FUNCTIONS (from constants_inspector)
    // ========================================================================
    
    void displaySelectedArchitecture() {
        using namespace libstats::tools;
        
        display::subsectionHeader("Selected Architecture");
        std::cout << "Active Architecture: " << system_info::getActiveArchitecture() << "\n";
        system_info::displaySIMDLevel();
        std::cout << "\n";
    }
    
    void displayAdditionalAdaptiveConstants() {
        using namespace libstats::tools;
        using namespace libstats::constants;
        
        display::subsectionHeader("Additional Adaptive Constants");
        
        table::ColumnFormatter formatter({40, 15});
        std::cout << formatter.formatRow({"Constant", "Value"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        std::cout << formatter.formatRow({"Min Elements for Distribution Parallel", std::to_string(parallel::adaptive::min_elements_for_distribution_parallel())}) << "\n";
        std::cout << formatter.formatRow({"Min Elements for Simple Dist Parallel", std::to_string(parallel::adaptive::min_elements_for_simple_distribution_parallel())}) << "\n";
        std::cout << formatter.formatRow({"Monte Carlo Grain Size", std::to_string(parallel::adaptive::monte_carlo_grain_size())}) << "\n";
        std::cout << formatter.formatRow({"Max Grain Size", std::to_string(parallel::adaptive::max_grain_size())}) << "\n";
        
        std::cout << "\n";
    }
    
    void displayCacheThresholds() {
        using namespace libstats::tools;
        using namespace libstats::constants;
        
        display::subsectionHeader("Cache Thresholds");
        
        auto cache_thresholds = platform::get_cache_thresholds();
        
        table::ColumnFormatter formatter({25, 15});
        std::cout << formatter.formatRow({"Threshold", "Value"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        std::cout << formatter.formatRow({"L1 Optimal Size", std::to_string(cache_thresholds.l1_optimal_size) + " doubles"}) << "\n";
        std::cout << formatter.formatRow({"L2 Optimal Size", std::to_string(cache_thresholds.l2_optimal_size) + " doubles"}) << "\n";
        std::cout << formatter.formatRow({"L3 Optimal Size", std::to_string(cache_thresholds.l3_optimal_size) + " doubles"}) << "\n";
        std::cout << formatter.formatRow({"Blocking Size", std::to_string(cache_thresholds.blocking_size) + " doubles"}) << "\n";
        
        std::cout << "\n";
    }
    
    void displayArchitectureComparison() {
        using namespace libstats::tools;
        using namespace libstats::constants;
        
        display::subsectionHeader("Architecture-Specific Constants Comparison");
        
        table::ColumnFormatter formatter({25, 15, 15, 15});
        std::cout << formatter.formatRow({"Architecture", "Min Parallel", "Grain Size", "Simple Grain"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        // Display all architecture constants
        std::cout << formatter.formatRow({"SSE", 
                                         std::to_string(parallel::sse::MIN_ELEMENTS_FOR_PARALLEL),
                                         std::to_string(parallel::sse::DEFAULT_GRAIN_SIZE),
                                         std::to_string(parallel::sse::SIMPLE_OPERATION_GRAIN_SIZE)}) << "\n";
        
        std::cout << formatter.formatRow({"AVX", 
                                         std::to_string(parallel::avx::MIN_ELEMENTS_FOR_PARALLEL),
                                         std::to_string(parallel::avx::DEFAULT_GRAIN_SIZE),
                                         std::to_string(parallel::avx::SIMPLE_OPERATION_GRAIN_SIZE)}) << "\n";
        
        std::cout << formatter.formatRow({"AVX2", 
                                         std::to_string(parallel::avx2::MIN_ELEMENTS_FOR_PARALLEL),
                                         std::to_string(parallel::avx2::DEFAULT_GRAIN_SIZE),
                                         std::to_string(parallel::avx2::SIMPLE_OPERATION_GRAIN_SIZE)}) << "\n";
        
        std::cout << formatter.formatRow({"AVX-512", 
                                         std::to_string(parallel::avx512::MIN_ELEMENTS_FOR_PARALLEL),
                                         std::to_string(parallel::avx512::DEFAULT_GRAIN_SIZE),
                                         std::to_string(parallel::avx512::SIMPLE_OPERATION_GRAIN_SIZE)}) << "\n";
        
        std::cout << formatter.formatRow({"NEON", 
                                         std::to_string(parallel::neon::MIN_ELEMENTS_FOR_PARALLEL),
                                         std::to_string(parallel::neon::DEFAULT_GRAIN_SIZE),
                                         std::to_string(parallel::neon::SIMPLE_OPERATION_GRAIN_SIZE)}) << "\n";
        
        std::cout << formatter.formatRow({"Fallback", 
                                         std::to_string(parallel::fallback::MIN_ELEMENTS_FOR_PARALLEL),
                                         std::to_string(parallel::fallback::DEFAULT_GRAIN_SIZE),
                                         std::to_string(parallel::fallback::SIMPLE_OPERATION_GRAIN_SIZE)}) << "\n";
        
        std::cout << formatter.getSeparator() << "\n";
        
        // Show selected (adaptive) constants
        std::cout << formatter.formatRow({"SELECTED (adaptive)", 
                                         std::to_string(parallel::adaptive::min_elements_for_parallel()),
                                         std::to_string(parallel::adaptive::grain_size()),
                                         std::to_string(parallel::adaptive::simple_operation_grain_size())}) << "\n";
        
        std::cout << "\n";
    }
    
    // ========================================================================
    // PERFORMANCE-SPECIFIC FUNCTIONS (from system_capabilities_inspector)
    // ========================================================================
    
    void displayCPUTopology() {
        using namespace libstats::tools;
        
        display::subsectionHeader("CPU Topology");
        
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        
        std::cout << std::left << std::setw(25) << "Hardware Threads:" 
                  << std::thread::hardware_concurrency() << "\n";
        std::cout << std::setw(25) << "Logical Cores:" 
                  << capabilities.logical_cores() << "\n";
        std::cout << std::setw(25) << "Physical Cores:" 
                  << capabilities.physical_cores() << "\n";
        std::cout << std::setw(25) << "Hyperthreading:" 
                  << (capabilities.logical_cores() > capabilities.physical_cores() ? "Enabled" : "Disabled") << "\n";
        
        std::cout << "\n";
    }
    
    void displaySIMDCapabilitiesDetailed() {
        using namespace libstats::tools;
        
        display::subsectionHeader("SIMD Capabilities");
        
        const auto& capabilities = libstats::performance::SystemCapabilities::current();
        
        table::ColumnFormatter formatter({12, 10, 15, 25});
        std::cout << formatter.formatRow({"Instruction", "Support", "Vector Width", "Description"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        std::cout << formatter.formatRow({"SSE2", capabilities.has_sse2() ? "Yes" : "No", "128-bit", "Basic SIMD operations"}) << "\n";
        std::cout << formatter.formatRow({"AVX", capabilities.has_avx() ? "Yes" : "No", "256-bit", "Advanced vector ext"}) << "\n";
        std::cout << formatter.formatRow({"AVX2", capabilities.has_avx2() ? "Yes" : "No", "256-bit", "Integer AVX operations"}) << "\n";
        std::cout << formatter.formatRow({"AVX-512", capabilities.has_avx512() ? "Yes" : "No", "512-bit", "Foundation instructions"}) << "\n";
        std::cout << formatter.formatRow({"NEON", capabilities.has_neon() ? "Yes" : "No", "128-bit", "ARM SIMD instructions"}) << "\n";
        
        // Display active SIMD level
        std::cout << "\nActive SIMD Level: " << libstats::simd::VectorOps::get_active_simd_level() << "\n\n";
    }
    
    void displayPerformanceBaselines() {
        using namespace libstats::tools;
        
        display::subsectionHeader("Performance Baselines");
        
        // Simple arithmetic throughput test
        const size_t test_size = BASELINE_TEST_SIZE;
        std::vector<double> data(test_size, libstats::constants::math::ONE);
        std::vector<double> result(test_size);
        
        // SIMD throughput
        auto start = high_resolution_clock::now();
        for (int i = 0; i < BASELINE_ITERATIONS; ++i) {
            libstats::simd::VectorOps::vector_multiply(data.data(), data.data(), result.data(), test_size);
        }
        auto end = high_resolution_clock::now();
        double simd_time = static_cast<double>(duration_cast<microseconds>(end - start).count()) / static_cast<double>(BASELINE_ITERATIONS);
        
        // Scalar throughput
        start = high_resolution_clock::now();
        for (int i = 0; i < BASELINE_ITERATIONS; ++i) {
            for (size_t j = 0; j < test_size; ++j) {
                result[j] = data[j] * data[j];
            }
        }
        end = high_resolution_clock::now();
        double scalar_time = static_cast<double>(duration_cast<microseconds>(end - start).count()) / static_cast<double>(BASELINE_ITERATIONS);
        
        // Display results using utility
        table::ColumnFormatter formatter({25, 15, 20});
        std::cout << formatter.formatRow({"Operation Type", "Time (μs)", "Throughput (MOps/s)"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        std::cout << std::fixed << std::setprecision(1);
        std::cout << formatter.formatRow({"SIMD Multiply", 
                                        std::to_string(static_cast<int>(simd_time)), 
                                        std::to_string(static_cast<int>(test_size / simd_time))}) << "\n";
        std::cout << formatter.formatRow({"Scalar Multiply", 
                                        std::to_string(static_cast<int>(scalar_time)), 
                                        std::to_string(static_cast<int>(test_size / scalar_time))}) << "\n";
        
        std::cout << "\nSIMD Speedup: " << std::setprecision(2) << scalar_time / simd_time << "x\n\n";
    }
    
    void displayDispatcherConfiguration() {
        using namespace libstats::tools;
        
        display::subsectionHeader("Performance Dispatcher Configuration");
        
        // Show some example strategy selections
        std::cout << "Example Strategy Selections:\n";
        
        table::ColumnFormatter formatter({20, 15, 15, 20});
        std::cout << formatter.formatRow({"Batch Size", "Distribution", "Complexity", "Strategy"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
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
                int complexity_count = 0;
                for (auto complexity : complexities) {
                    libstats::performance::PerformanceDispatcher dispatcher;
                    auto strategy = dispatcher.selectOptimalStrategy(size, dist, complexity, capabilities);
                    
                    std::cout << formatter.formatRow({std::to_string(size), 
                                                    strings::distributionTypeToString(dist),
                                                    strings::complexityToString(complexity),
                                                    strings::strategyToDisplayString(strategy)}) << "\n";
                    
                    // Only show first complexity for brevity
                    if (++complexity_count >= MAX_COMPLEXITY_DEMOS) break;
                }
            }
        }
        std::cout << "\n";
    }
};

void showUsage(const std::string& program_name) {
    std::cout << "LIBSTATS SYSTEM INSPECTOR\n";
    std::cout << "=========================\n\n";
    std::cout << "Comprehensive system analysis tool with multiple inspection modes.\n\n";
    std::cout << "Usage: " << program_name << " [MODE]\n\n";
    std::cout << "Modes:\n";
    std::cout << "  --full         Complete comprehensive analysis (default)\n";
    std::cout << "  --constants    Architecture-specific constants analysis\n";
    std::cout << "  --performance  System capabilities and performance analysis\n";
    std::cout << "  --quick        Basic system information only\n";
    std::cout << "  --help, -h     Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                    # Complete analysis\n";
    std::cout << "  " << program_name << " --constants         # Constants focus\n";
    std::cout << "  " << program_name << " --performance       # Performance focus\n";
    std::cout << "  " << program_name << " --quick             # Quick overview\n\n";
    std::cout << "This tool consolidates the functionality of constants_inspector\n";
    std::cout << "and system_capabilities_inspector into a single comprehensive\n";
    std::cout << "system analysis tool with mode-specific focus areas.\n\n";
}

InspectionMode parseMode(const std::string& arg) {
    if (arg == "--constants" || arg == "-c") {
        return InspectionMode::CONSTANTS;
    } else if (arg == "--performance" || arg == "-p") {
        return InspectionMode::PERFORMANCE;
    } else if (arg == "--quick" || arg == "-q") {
        return InspectionMode::QUICK;
    } else if (arg == "--full" || arg == "-f") {
        return InspectionMode::FULL;
    } else {
        return InspectionMode::FULL; // default
    }
}

int main(int argc, char* argv[]) {
    using namespace libstats::tools;
    
    // Parse command line arguments
    InspectionMode mode = InspectionMode::FULL; // default
    
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            showUsage(argv[0]);
            return 0;
        } else {
            mode = parseMode(arg);
        }
    }
    
    // Use the standard tool runner pattern with initialization
    return tool_utils::runTool("System Inspector Tool", [mode]() {
        // Initialize performance systems for accurate inspection
        libstats::initialize_performance_systems();
        
        SystemInspector inspector(mode);
        inspector.runInspection();
    });
}

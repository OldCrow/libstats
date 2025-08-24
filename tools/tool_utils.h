/**
 * @file tool_utils.h
 * @brief Shared utilities for libstats tools to eliminate code duplication
 *
 * This header provides common functionality used across multiple libstats tools
 * including string formatting, performance data helpers, and display utilities.
 */

#pragma once

// Use libstats.h for complete library functionality
#define LIBSTATS_FULL_INTERFACE
#include "../include/libstats.h"

// Additional standard library includes for tool-specific functionality
#include <iomanip>
#include <iostream>
#include <sstream>

namespace stats {
namespace detail {  // tools utilities

// Time formatting utilities
namespace detail {  // time utilities
/**
 * @brief Formats a duration in nanoseconds to human-readable string
 * @param ns Duration in nanoseconds
 * @return Formatted string with appropriate units (ns, μs, ms, s)
 */
inline std::string formatDuration(std::chrono::nanoseconds ns) {
    constexpr long NANOSECONDS_TO_MICROSECONDS = 1000;
    constexpr long NANOSECONDS_TO_MILLISECONDS = 1000000;
    constexpr long NANOSECONDS_TO_SECONDS = 1000000000;

    if (ns.count() < NANOSECONDS_TO_MICROSECONDS) {
        return std::to_string(ns.count()) + "ns";
    }
    if (ns.count() < NANOSECONDS_TO_MILLISECONDS) {
        return std::to_string(ns.count() / NANOSECONDS_TO_MICROSECONDS) + "μs";
    }
    if (ns.count() < NANOSECONDS_TO_SECONDS) {
        return std::to_string(ns.count() / NANOSECONDS_TO_MILLISECONDS) + "ms";
    }
    return std::to_string(ns.count() / NANOSECONDS_TO_SECONDS) + "s";
}

/**
 * @brief Formats microseconds with appropriate unit selection
 * @param time_us Time in microseconds
 * @return Formatted time string with units
 */
inline std::string formatMicroseconds(double time_us) {
    constexpr double MICROSECONDS_TO_NANOSECONDS = 1000.0;
    constexpr double NANOSECOND_THRESHOLD = 1.0;      // μs
    constexpr double MILLISECOND_THRESHOLD = 1000.0;  // μs

    if (time_us < NANOSECOND_THRESHOLD) {
        return std::to_string(static_cast<int>(time_us * MICROSECONDS_TO_NANOSECONDS)) + "ns";
    } else if (time_us < MILLISECOND_THRESHOLD) {
        return std::to_string(static_cast<int>(time_us)) + "μs";
    } else {
        return std::to_string(time_us / MICROSECONDS_TO_NANOSECONDS) + "ms";
    }
}
}  // namespace detail

// String conversion utilities for enums
namespace detail {  // strings utilities
/**
 * @brief Converts Strategy enum to string representation
 * @param strategy The strategy enum value
 * @return String representation of the strategy
 */
inline std::string strategyToString(stats::detail::Strategy strategy) {
    switch (strategy) {
        case stats::detail::Strategy::SCALAR:
            return "SCALAR";
        case stats::detail::Strategy::SIMD_BATCH:
            return "SIMD_BATCH";
        case stats::detail::Strategy::PARALLEL_SIMD:
            return "PARALLEL_SIMD";
        case stats::detail::Strategy::WORK_STEALING:
            return "WORK_STEALING";
        case stats::detail::Strategy::GPU_ACCELERATED:
            return "GPU_ACCELERATED";
        default:
            return "UNKNOWN";
    }
}

/**
 * @brief Converts Strategy enum to display-friendly string
 * @param strategy The strategy enum value
 * @return Display-friendly string representation
 */
inline std::string strategyToDisplayString(stats::detail::Strategy strategy) {
    switch (strategy) {
        case stats::detail::Strategy::SCALAR:
            return "Scalar";
        case stats::detail::Strategy::SIMD_BATCH:
            return "SIMD";
        case stats::detail::Strategy::PARALLEL_SIMD:
            return "Parallel+SIMD";
        case stats::detail::Strategy::WORK_STEALING:
            return "Work-Stealing";
        case stats::detail::Strategy::GPU_ACCELERATED:
            return "GPU-Accelerated";
        default:
            return "Unknown";
    }
}

/**
 * @brief Converts DistributionType enum to string representation
 * @param type The distribution type enum value
 * @return String representation of the distribution type
 */
inline std::string distributionTypeToString(stats::detail::DistributionType type) {
    switch (type) {
        case stats::detail::DistributionType::UNIFORM:
            return "Uniform";
        case stats::detail::DistributionType::GAUSSIAN:
            return "Gaussian";
        case stats::detail::DistributionType::POISSON:
            return "Poisson";
        case stats::detail::DistributionType::EXPONENTIAL:
            return "Exponential";
        case stats::detail::DistributionType::DISCRETE:
            return "Discrete";
        case stats::detail::DistributionType::GAMMA:
            return "Gamma";
        default:
            return "Unknown";
    }
}

/**
 * @brief Converts ComputationComplexity enum to string representation
 * @param complexity The computation complexity enum value
 * @return String representation of the complexity
 */
inline std::string complexityToString(stats::detail::ComputationComplexity complexity) {
    switch (complexity) {
        case stats::detail::ComputationComplexity::SIMPLE:
            return "Simple";
        case stats::detail::ComputationComplexity::MODERATE:
            return "Moderate";
        case stats::detail::ComputationComplexity::COMPLEX:
            return "Complex";
        default:
            return "Unknown";
    }
}
}  // namespace detail

// Formatting utilities
namespace detail {  // format utilities
/**
 * @brief Formats confidence score as percentage string
 * @param confidence_score Confidence value between 0.0 and 1.0
 * @param precision Number of decimal places (default: 0)
 * @return Formatted percentage string
 */
inline std::string confidenceToString(double confidence_score, int precision = 0) {
    constexpr double PERCENTAGE_MULTIPLIER = 100.0;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << (confidence_score * PERCENTAGE_MULTIPLIER)
        << "%";
    return oss.str();
}

/**
 * @brief Creates a separator line of specified width
 * @param width Width of the separator line
 * @param character Character to use for the separator (default: '-')
 * @return String containing the separator line
 */
inline std::string separator(int width, char character = '-') {
    return std::string(static_cast<std::size_t>(width), character);
}

/**
 * @brief Formats memory size from bytes to KB
 * @param bytes Size in bytes
 * @return Formatted string with KB suffix
 */
inline std::string bytesToKB(size_t bytes) {
    constexpr size_t BYTES_TO_KB = 1024;
    return std::to_string(bytes / BYTES_TO_KB) + " KB";
}

/**
 * @brief Formats nanoseconds to microseconds for display
 * @param nanoseconds Time in nanoseconds
 * @return Formatted string with μs suffix
 */
inline std::string nanosecondsToMicroseconds(uint64_t nanoseconds) {
    constexpr uint64_t NANOSECONDS_TO_MICROSECONDS = 1000;
    return std::to_string(nanoseconds / NANOSECONDS_TO_MICROSECONDS) + " μs";
}

/**
 * @brief Formats a double value with specified precision
 * @param value The double value to format
 * @param precision Number of decimal places (default: 2)
 * @return Formatted string representation
 */
inline std::string formatDouble(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}
}  // namespace detail

// Table formatting utilities
namespace detail {  // table utilities
/**
 * @brief Table column formatter for consistent output
 */
class ColumnFormatter {
   public:
    /**
     * @brief Constructor with column widths
     * @param widths Vector of column widths
     */
    explicit ColumnFormatter(const std::vector<int>& widths) : widths_(widths) {}

    /**
     * @brief Format a row with given values
     * @param values Vector of string values for each column
     * @return Formatted row string
     */
    std::string formatRow(const std::vector<std::string>& values) const {
        std::ostringstream oss;
        oss << std::left;
        for (size_t i = 0; i < values.size() && i < widths_.size(); ++i) {
            oss << std::setw(widths_[i]) << values[i];
        }
        return oss.str();
    }

    /**
     * @brief Get total width of all columns
     * @return Sum of all column widths
     */
    int getTotalWidth() const {
        int total = 0;
        for (int width : widths_) {
            total += width;
        }
        return total;
    }

    /**
     * @brief Create separator line for the table
     * @param character Character to use for separator (default: '-')
     * @return Formatted separator string
     */
    std::string getSeparator(char character = '-') const {
        return std::string(static_cast<std::size_t>(getTotalWidth()), character);
    }

   private:
    std::vector<int> widths_;
};
}  // namespace detail

// Performance recording utilities
namespace detail {  // perf_utils utilities
/**
 * @brief Record performance data with automatic unit conversion
 * @param strategy Strategy used
 * @param dist_type Distribution type
 * @param data_size Size of data processed
 * @param time_microseconds Execution time in microseconds
 */
inline void recordPerformanceMicroseconds(stats::detail::Strategy strategy,
                                          stats::detail::DistributionType dist_type,
                                          size_t data_size, double time_microseconds) {
    constexpr double MICROSECONDS_TO_NANOSECONDS = 1000.0;
    stats::detail::PerformanceDispatcher::recordPerformance(
        strategy, dist_type, data_size,
        static_cast<uint64_t>(time_microseconds * MICROSECONDS_TO_NANOSECONDS));
}
}  // namespace detail

// Common display utilities
namespace detail {  // display utilities
/**
 * @brief Display section header with consistent formatting
 * @param title Section title
 * @param separator_char Character for separator lines (default: '=')
 */
inline void sectionHeader(const std::string& title, char separator_char = '=') {
    const int separator_width = static_cast<int>(title.length() + 4);
    std::cout << "\n"
              << std::string(static_cast<std::size_t>(separator_width), separator_char) << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(static_cast<std::size_t>(separator_width), separator_char) << "\n\n";
}

/**
 * @brief Display subsection header with consistent formatting
 * @param title Subsection title
 */
inline void subsectionHeader(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}
}  // namespace detail

// System information display utilities
namespace detail {  // system_info utilities
/**
 * @brief Display CPU features in a consistent format
 */
inline void displayCPUFeatures() {
    const auto& features = stats::arch::get_features();
    detail::subsectionHeader("CPU Features");

    detail::ColumnFormatter formatter({20, 10, 30});
    std::cout << formatter.formatRow({"Feature", "Support", "Description"}) << "\n";
    std::cout << formatter.getSeparator() << "\n";

    std::cout << formatter.formatRow(
                     {"AVX-512", features.avx512f ? "Yes" : "No", "Foundation instructions"})
              << "\n";
    std::cout << formatter.formatRow(
                     {"AVX2", features.avx2 ? "Yes" : "No", "Advanced Vector Ext 2"})
              << "\n";
    std::cout << formatter.formatRow({"AVX", features.avx ? "Yes" : "No", "Advanced Vector Ext"})
              << "\n";
    std::cout << formatter.formatRow({"SSE2", features.sse2 ? "Yes" : "No", "Streaming SIMD Ext 2"})
              << "\n";
    std::cout << formatter.formatRow(
                     {"NEON", features.neon ? "Yes" : "No", "ARM SIMD instructions"})
              << "\n";
    std::cout << formatter.formatRow({"FMA", features.fma ? "Yes" : "No", "Fused Multiply-Add"})
              << "\n";

    std::cout << "\n";
}

/**
 * @brief Display cache information in a consistent format
 */
inline void displayCacheInfo() {
    const auto& features = stats::arch::get_features();
    detail::subsectionHeader("Cache Information");

    detail::ColumnFormatter formatter({15, 12, 15});
    std::cout << formatter.formatRow({"Cache Level", "Size (KB)", "Line Size"}) << "\n";
    std::cout << formatter.getSeparator() << "\n";

    std::cout << formatter.formatRow({"L1", std::to_string(features.l1_cache_size / 1024),
                                      std::to_string(features.cache_line_size) + " bytes"})
              << "\n";
    std::cout << formatter.formatRow({"L2", std::to_string(features.l2_cache_size / 1024),
                                      std::to_string(features.cache_line_size) + " bytes"})
              << "\n";
    std::cout << formatter.formatRow({"L3", std::to_string(features.l3_cache_size / 1024),
                                      std::to_string(features.cache_line_size) + " bytes"})
              << "\n";

    std::cout << "\n";
}

/**
 * @brief Display system capabilities in a consistent format
 */
inline void displaySystemCapabilities() {
    const auto& capabilities = stats::detail::SystemCapabilities::current();
    detail::subsectionHeader("System Capabilities");

    std::cout << std::left << std::setw(25) << "Logical Cores:" << capabilities.logical_cores()
              << "\n";
    std::cout << std::setw(25) << "Physical Cores:" << capabilities.physical_cores() << "\n";
    std::cout << std::setw(25) << "Hyperthreading:"
              << (capabilities.logical_cores() > capabilities.physical_cores() ? "Enabled"
                                                                               : "Disabled")
              << "\n";
    std::cout << std::setw(25) << "SIMD Efficiency:" << std::fixed << std::setprecision(3)
              << capabilities.simd_efficiency() << "\n";
    std::cout << std::setw(25) << "Memory Bandwidth:" << std::setprecision(2)
              << capabilities.memory_bandwidth_gb_s() << " GB/s\n";
    std::cout << std::setw(25) << "Thread Overhead:" << std::setprecision(1)
              << capabilities.threading_overhead_ns() << " ns\n";

    std::cout << "\n";
}

/**
 * @brief Display active SIMD level
 */
inline void displaySIMDLevel() {
    const std::string simd_level = stats::arch::simd::VectorOps::get_active_simd_level();
    std::cout << "Active SIMD Level: " << simd_level << "\n";
}

/**
 * @brief Display compact system summary for tool headers
 */
inline void displayCompactSystemInfo() {
    const auto& features = stats::arch::get_features();
    const auto& capabilities = stats::detail::SystemCapabilities::current();
    const std::string simd_level = stats::arch::simd::VectorOps::get_active_simd_level();

    std::cout << "System: " << capabilities.logical_cores() << " logical cores, " << simd_level
              << " SIMD, " << detail::bytesToKB(features.l3_cache_size) << " L3 cache\n";
}

/**
 * @brief Display tool header with system information
 * @param tool_name Name of the tool
 * @param description Brief description of the tool
 */
inline void displayToolHeader(const std::string& tool_name, const std::string& description = "") {
    detail::sectionHeader(tool_name);
    if (!description.empty()) {
        std::cout << description << "\n\n";
    }
    displayCompactSystemInfo();
    std::cout << "\n";
}

/**
 * @brief Get active CPU architecture name based on features
 * @return String representing the active architecture
 */
inline std::string getActiveArchitecture() {
    const auto& features = stats::arch::get_features();

    if (features.avx512f)
        return "AVX-512";
    if (features.avx2)
        return "AVX2";
    if (features.avx)
        return "AVX";
    if (features.sse2)
        return "SSE2";
    if (features.neon)
        return "NEON";
    return "Fallback";
}

/**
 * @brief Display adaptive constants information
 */
inline void displayAdaptiveConstants() {
    using namespace stats::detail;
    detail::subsectionHeader("Adaptive Constants");

    detail::ColumnFormatter formatter({35, 15});
    std::cout << formatter.formatRow({"Constant", "Value"}) << "\n";
    std::cout << formatter.getSeparator() << "\n";

    std::cout << formatter.formatRow(
                     {"Min Elements for Parallel",
                      std::to_string(arch::parallel::detail::min_elements_for_parallel())})
              << "\n";
    std::cout << formatter.formatRow(
                     {"Default Grain Size", std::to_string(arch::parallel::detail::grain_size())})
              << "\n";
    std::cout << formatter.formatRow(
                     {"Simple Operation Grain Size",
                      std::to_string(arch::parallel::detail::simple_operation_grain_size())})
              << "\n";
    std::cout << formatter.formatRow(
                     {"Complex Operation Grain Size",
                      std::to_string(arch::parallel::detail::complex_operation_grain_size())})
              << "\n";

    std::cout << "\n";
}

/**
 * @brief Display platform constants information
 */
inline void displayPlatformConstants() {
    using namespace stats::detail;
    detail::subsectionHeader("Platform Constants");

    detail::ColumnFormatter formatter({35, 15});
    std::cout << formatter.formatRow({"Constant", "Value"}) << "\n";
    std::cout << formatter.getSeparator() << "\n";

    std::cout << formatter.formatRow(
                     {"SIMD Block Size",
                      std::to_string(stats::arch::get_optimal_simd_block_size()) + " doubles"})
              << "\n";
    std::cout << formatter.formatRow(
                     {"Memory Alignment",
                      std::to_string(stats::arch::get_optimal_alignment()) + " bytes"})
              << "\n";
    std::cout << formatter.formatRow(
                     {"Min SIMD Size",
                      std::to_string(stats::arch::get_min_simd_size()) + " elements"})
              << "\n";
    std::cout << formatter.formatRow(
                     {"Optimal Grain Size",
                      std::to_string(stats::arch::get_optimal_grain_size()) + " elements"})
              << "\n";
    std::cout << formatter.formatRow({"Fast Transcendental Support",
                                      stats::arch::supports_fast_transcendental() ? "Yes" : "No"})
              << "\n";

    std::cout << "\n";
}
}  // namespace detail

// Common tool utilities
namespace detail {  // tool_utils utilities
/**
 * @brief Initialize libstats with error handling
 * @return true if initialization was successful, false otherwise
 */
inline bool initializeLibstats() {
    try {
        stats::initialize_performance_systems();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize libstats: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Standard tool runner with error handling
 * @param tool_name Name of the tool (for error messages)
 * @param func Function to run the tool logic
 * @return Exit code (0 for success, 1 for error)
 */
template <typename Func>
inline int runTool(const std::string& tool_name, Func&& func) {
    try {
        if (!initializeLibstats()) {
            return 1;
        }
        func();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in " << tool_name << ": " << e.what() << std::endl;
        return 1;
    }
}

/**
 * @brief Validate CPU feature consistency and print warnings
 */
inline void validateAndWarnFeatureConsistency() {
    const auto& features = stats::arch::get_features();

    // Check for logical inconsistencies
    if (features.avx512f && !features.avx2) {
        std::cout
            << "Warning: AVX-512 detected but AVX2 not reported - may indicate detection issues\n";
    }
    if (features.avx2 && !features.avx) {
        std::cout
            << "Warning: AVX2 detected but AVX not reported - may indicate detection issues\n";
    }
    if (features.avx && !features.sse2) {
        std::cout
            << "Warning: AVX detected but SSE2 not reported - may indicate detection issues\n";
    }

    // Check for unusual cache configurations
    if (features.l1_cache_size > features.l2_cache_size) {
        std::cout << "Warning: L1 cache larger than L2 cache - unusual configuration\n";
    }
    if (features.l2_cache_size > features.l3_cache_size) {
        std::cout << "Warning: L2 cache larger than L3 cache - may be normal on some systems\n";
    }
}

/**
 * @brief Calculate and display speedup with proper formatting
 * @param serial_time Time for serial execution
 * @param parallel_time Time for parallel execution
 * @return Formatted speedup string
 */
inline std::string formatSpeedup(double serial_time, double parallel_time) {
    constexpr double MIN_TIME_THRESHOLD = 0.001;

    if (parallel_time < MIN_TIME_THRESHOLD) {
        return "inf";
    }
    if (serial_time < MIN_TIME_THRESHOLD) {
        return "0x";
    }

    double speedup = serial_time / parallel_time;
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << speedup << "x";
    return oss.str();
}

/**
 * @brief Print benchmark results in consistent format
 * @param operation_name Name of the operation
 * @param serial_time Serial execution time
 * @param parallel_time Parallel execution time
 * @param time_unit Unit of time measurements (default: "μs")
 */
inline void printBenchmarkResults(const std::string& operation_name, double serial_time,
                                  double parallel_time, const std::string& time_unit = "μs") {
    std::string speedup_str = formatSpeedup(serial_time, parallel_time);

    std::cout << "  " << operation_name << ": "
              << "Serial=" << static_cast<int>(serial_time) << time_unit << ", "
              << "Parallel=" << static_cast<int>(parallel_time) << time_unit << ", "
              << "Speedup=" << speedup_str << std::endl;
}
}  // namespace detail

}  // namespace detail
}  // namespace stats

#pragma once

#include <cstddef>

/**
 * @file core/benchmark_constants.h
 * @brief Benchmark and performance testing constants for libstats
 * 
 * This header contains constants used for benchmarking, performance testing,
 * and algorithm thresholds throughout the library.
 */

namespace libstats {
namespace constants {

/// Benchmark and performance testing constants
namespace benchmark {
    /// Default number of benchmark iterations
    inline constexpr std::size_t DEFAULT_ITERATIONS = 100;
    
    /// Default number of warmup runs
    inline constexpr std::size_t DEFAULT_WARMUP_RUNS = 10;
    
    /// Minimum number of benchmark iterations
    inline constexpr std::size_t MIN_ITERATIONS = 10;
    
    /// Minimum number of warmup runs
    inline constexpr std::size_t MIN_WARMUP_RUNS = 5;
    
    /// Maximum number of benchmark iterations (for safety)
    inline constexpr std::size_t MAX_ITERATIONS = 100000;
    
    /// Maximum number of warmup runs (for safety)
    inline constexpr std::size_t MAX_WARMUP_RUNS = 1000;
    
    /// Minimum execution time threshold (seconds)
    inline constexpr double MIN_EXECUTION_TIME = 1.0e-9;
    
    /// Maximum execution time threshold (seconds)
    inline constexpr double MAX_EXECUTION_TIME = 3600.0;
    
    /// Statistical significance threshold for performance comparisons
    inline constexpr double PERFORMANCE_SIGNIFICANCE_THRESHOLD = 0.05;
    
    /// Coefficient of variation threshold for stable measurements
    inline constexpr double CV_THRESHOLD = 0.1;
}

/// Algorithm-specific thresholds
namespace thresholds {
    /// Minimum scale factor for scaled algorithms
    inline constexpr double MIN_SCALE_FACTOR = 1.0e-100;
    
    /// Maximum scale factor for scaled algorithms
    inline constexpr double MAX_SCALE_FACTOR = 1.0e100;
    
    /// Threshold for switching to log-space computation
    inline constexpr double LOG_SPACE_THRESHOLD = 1.0e-50;
    
    /// Maximum parameter value for distribution fitting
    inline constexpr double MAX_DISTRIBUTION_PARAMETER = 1.0e6;
    
    /// Minimum parameter value for distribution fitting
    inline constexpr double MIN_DISTRIBUTION_PARAMETER = 1.0e-6;
}

} // namespace constants
} // namespace libstats

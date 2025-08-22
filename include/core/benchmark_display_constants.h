#pragma once

#include <cstddef>

/**
 * @file core/benchmark_display_constants.h
 * @brief Display and formatting constants for benchmark results
 *
 * This header contains constants used for displaying and comparing
 * benchmark results, including thresholds for performance comparisons
 * and formatting options.
 */

namespace libstats {
namespace constants {
namespace benchmark {

/// Display and formatting constants
namespace display {
/// Number of raw times to show in detailed output
inline constexpr std::size_t MAX_RAW_TIMES_TO_SHOW = 10;

/// Minimum number of CSV tokens for valid baseline entry
inline constexpr std::size_t MIN_CSV_TOKENS = 8;

/// Percentage conversion factor
inline constexpr double PERCENT_FACTOR = 100.0;
}  // namespace display

/// Performance comparison thresholds
namespace comparison {
/// Speedup threshold to consider result as "FASTER"
inline constexpr double FASTER_THRESHOLD = 1.05;

/// Speedup threshold to consider result as "SLOWER"
inline constexpr double SLOWER_THRESHOLD = 0.95;
}  // namespace comparison

/// Test data generation constants
namespace test_data {
/// Default mean for normal distribution in test data
inline constexpr double NORMAL_DIST_MEAN = 0.0;

/// Default standard deviation for normal distribution in test data
inline constexpr double NORMAL_DIST_STDDEV = 1.0;

/// Scalar value for addition benchmarks
inline constexpr double SCALAR_ADD_VALUE = 1.0;

/// Scalar value for multiplication benchmarks
inline constexpr double SCALAR_MUL_VALUE = 2.0;

/// Divisor for dot product operation count
inline constexpr double DOT_PRODUCT_OPS_DIVISOR = 2.0;
}  // namespace test_data

/// CPU-based optimization thresholds
namespace cpu_optimization {
/// High core count threshold (cores)
inline constexpr std::size_t HIGH_CORE_COUNT = 16;

/// Low core count threshold (cores)
inline constexpr std::size_t LOW_CORE_COUNT = 4;

/// Large L3 cache threshold (bytes)
inline constexpr std::size_t LARGE_L3_CACHE = 16 * 1024 * 1024;  // 16MB

/// Small L3 cache threshold (bytes)
inline constexpr std::size_t SMALL_L3_CACHE = 4 * 1024 * 1024;  // 4MB

/// Iteration multiplier for high-core systems
inline constexpr double HIGH_CORE_ITER_MULTIPLIER = 1.5;

/// Warmup multiplier for high-core systems
inline constexpr double HIGH_CORE_WARMUP_MULTIPLIER = 1.2;

/// Iteration multiplier for low-core systems
inline constexpr double LOW_CORE_ITER_MULTIPLIER = 0.8;

/// Warmup multiplier for low-core systems
inline constexpr double LOW_CORE_WARMUP_MULTIPLIER = 0.8;

/// Iteration multiplier for large cache systems
inline constexpr double LARGE_CACHE_ITER_MULTIPLIER = 1.2;

/// Iteration multiplier for small cache systems
inline constexpr double SMALL_CACHE_ITER_MULTIPLIER = 0.9;

/// Warmup multiplier for hyperthreaded systems
inline constexpr double HYPERTHREAD_WARMUP_MULTIPLIER = 1.3;
}  // namespace cpu_optimization

}  // namespace benchmark
}  // namespace constants
}  // namespace libstats

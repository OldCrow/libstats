#pragma once

#include <cstddef>

/**
 * @file core/performance_constants.h
 * @brief Benchmarking and performance testing constants for libstats
 *
 * This header contains constants used exclusively for measuring and
 * validating performance: iteration counts, warmup runs, timing bounds,
 * and statistical significance thresholds for benchmark comparisons.
 *
 * These are tool-level constants — they govern how the library measures
 * itself, not how it computes statistical results. Distributions and
 * core algorithms should not need this header.
 *
 * For mathematical and precision constants, see math_constants.h.
 * For statistical constants, see statistical_constants.h.
 */

namespace stats {
namespace detail {

// =============================================================================
// BENCHMARK ITERATION COUNTS
// =============================================================================

inline constexpr std::size_t DEFAULT_ITERATIONS = 100;
inline constexpr std::size_t DEFAULT_WARMUP_RUNS = 10;
inline constexpr std::size_t MIN_ITERATIONS = 10;
inline constexpr std::size_t MIN_WARMUP_RUNS = 5;
inline constexpr std::size_t MAX_ITERATIONS = 100000;
inline constexpr std::size_t MAX_WARMUP_RUNS = 1000;

// =============================================================================
// TIMING BOUNDS
// =============================================================================

/// Minimum plausible execution time (seconds) — shorter times are measurement noise
inline constexpr double MIN_EXECUTION_TIME = 1.0e-9;

/// Maximum plausible execution time (seconds) — longer times indicate a hang
inline constexpr double MAX_EXECUTION_TIME = 3600.0;

// =============================================================================
// BENCHMARK STATISTICAL QUALITY
// =============================================================================

/// Minimum p-value to declare a performance difference statistically significant
inline constexpr double PERFORMANCE_SIGNIFICANCE_THRESHOLD = 0.05;

/// Maximum coefficient of variation for a measurement to be considered stable
inline constexpr double CV_THRESHOLD = 0.1;

}  // namespace detail
}  // namespace stats

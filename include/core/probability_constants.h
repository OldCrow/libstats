#pragma once

#include <limits>

/**
 * @file core/probability_constants.h
 * @brief Probability bounds and safety limits for libstats
 *
 * This header contains probability bounds, safety limits, and related constants
 * used for safe probability calculations throughout the library.
 */

namespace stats {
namespace detail {
/// Minimum probability value to prevent underflow
inline constexpr double MIN_PROBABILITY = 1.0e-300;

/// Maximum probability value to prevent overflow
inline constexpr double MAX_PROBABILITY = 1.0 - 1.0e-15;

/// Minimum log probability to prevent -infinity
inline constexpr double MIN_LOG_PROBABILITY = -4605.0;

/// Maximum log probability (log(1.0) = 0.0)
inline constexpr double MAX_LOG_PROBABILITY = 0.0;

/// Negative infinity for initialization
inline constexpr double NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();

/// Epsilon for safe log probability computations
inline constexpr double LOG_PROBABILITY_EPSILON = 1.0e-300;
}  // namespace detail
}  // namespace stats

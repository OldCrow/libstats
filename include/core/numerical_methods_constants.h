#pragma once

/**
 * @file core/numerical_methods_constants.h
 * @brief Constants for numerical integration and mathematical methods
 *
 * This header contains constants used in numerical integration methods,
 * such as Simpson's rule coefficients and other numerical method parameters.
 */

namespace libstats {
namespace constants {
namespace numerical {

/// Simpson's rule coefficients
namespace simpson {
/// Coefficient for Simpson's 1/6 rule divisor
inline constexpr double SIMPSON_DIVISOR = 6.0;

/// Coefficient for midpoint weight in Simpson's rule
inline constexpr double MIDPOINT_WEIGHT = 4.0;
}  // namespace simpson

/// Richardson extrapolation constants
namespace richardson {
/// Error threshold multiplier for Richardson extrapolation
inline constexpr double ERROR_THRESHOLD_MULTIPLIER = 15.0;

/// Richardson extrapolation divisor
inline constexpr double EXTRAPOLATION_DIVISOR = 15.0;
}  // namespace richardson

/// Adaptive integration constants
namespace adaptive {
/// Tolerance reduction factor for recursive subdivision
inline constexpr double TOLERANCE_REDUCTION_FACTOR = 2.0;
}  // namespace adaptive

/// Gamma function computation thresholds
namespace gamma {
/// Threshold for choosing between series expansion and continued fraction
inline constexpr double SERIES_VS_FRACTION_THRESHOLD = 1.0;
}  // namespace gamma

/// Beta function computation thresholds
namespace beta {
/// Maximum iterations for beta continued fraction
inline constexpr int MAX_ITERATIONS = 100;

/// Threshold for choosing between direct and complementary computation
/// Formula: x < (a + 1.0) / (a + b + 2.0)
inline constexpr double COMPUTATION_THRESHOLD_NUMERATOR_OFFSET = 1.0;
inline constexpr double COMPUTATION_THRESHOLD_DENOMINATOR_OFFSET = 2.0;
}  // namespace beta

}  // namespace numerical
}  // namespace constants
}  // namespace libstats

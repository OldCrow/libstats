#pragma once

/**
 * @file core/robust_constants.h
 * @brief Robust estimation constants for libstats
 *
 * This header contains constants used for robust statistical estimation,
 * including M-estimator tuning constants and robust scale factors.
 */

namespace libstats {
namespace constants {

/// Robust estimation constants
namespace robust {
/// MAD (Median Absolute Deviation) scaling factor for Gaussian distribution
/// This converts MAD to a robust estimate of the standard deviation
/// Factor = 1/Φ⁻¹(3/4) ≈ 1.4826 for normal distribution consistency
inline constexpr double MAD_SCALING_FACTOR = 1.4826;

/// Default tuning constants for M-estimators
namespace tuning {
/// Huber's M-estimator tuning constant (95% efficiency under normality)
inline constexpr double HUBER_DEFAULT = 1.345;

/// Tukey's bisquare M-estimator tuning constant (95% efficiency)
inline constexpr double TUKEY_DEFAULT = 4.685;

/// Hampel M-estimator tuning constants (a, b, c parameters)
inline constexpr double HAMPEL_A = 1.7;
inline constexpr double HAMPEL_B = 3.4;
inline constexpr double HAMPEL_C = 8.5;
}  // namespace tuning

/// Maximum iterations for robust iterative algorithms
inline constexpr int MAX_ROBUST_ITERATIONS = 50;

/// Convergence tolerance for robust estimation
inline constexpr double ROBUST_CONVERGENCE_TOLERANCE = 1.0e-6;

/// Minimum robust scale factor to prevent numerical issues
inline constexpr double MIN_ROBUST_SCALE = 1.0e-8;
}  // namespace robust

}  // namespace constants
}  // namespace libstats

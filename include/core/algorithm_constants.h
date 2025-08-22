#pragma once

/**
 * @file core/algorithm_constants.h
 * @brief Algorithm-specific constants for numerical methods in libstats
 *
 * This header contains constants specific to various numerical algorithms
 * including approximation coefficients, critical values, and iteration limits.
 */

namespace libstats {
namespace constants {
namespace algorithm {

// Lanczos approximation for gamma function
namespace lanczos {
/// Lanczos g parameter
inline constexpr double G = 7.0;

/// Lanczos coefficients for g=7
inline constexpr double COEFF_0 = 0.99999999999980993;
inline constexpr double COEFF_1 = 676.5203681218851;
inline constexpr double COEFF_2 = -1259.1392167224028;
inline constexpr double COEFF_3 = 771.32342877765313;
inline constexpr double COEFF_4 = -176.61502916214059;
inline constexpr double COEFF_5 = 12.507343278686905;
inline constexpr double COEFF_6 = -0.13857109526572012;
inline constexpr double COEFF_7 = 9.9843695780195716e-6;
inline constexpr double COEFF_8 = 1.5056327351493116e-7;
}  // namespace lanczos

// Anderson-Darling test critical values
namespace anderson_darling {
inline constexpr double CRIT_50 = 0.576;   // α = 0.50
inline constexpr double CRIT_40 = 0.656;   // α = 0.40
inline constexpr double CRIT_30 = 0.787;   // α = 0.30
inline constexpr double CRIT_25 = 1.248;   // α = 0.25
inline constexpr double CRIT_15 = 1.610;   // α = 0.15
inline constexpr double CRIT_10 = 1.933;   // α = 0.10
inline constexpr double CRIT_05 = 2.492;   // α = 0.05
inline constexpr double CRIT_025 = 3.070;  // α = 0.025
inline constexpr double CRIT_01 = 3.857;   // α = 0.01
inline constexpr double CRIT_005 = 4.500;  // α = 0.005

/// Corresponding significance levels for the critical values
inline constexpr double ALPHA_50 = 0.50;
inline constexpr double ALPHA_40 = 0.40;
inline constexpr double ALPHA_30 = 0.30;
inline constexpr double ALPHA_25 = 0.25;
inline constexpr double ALPHA_15 = 0.15;
inline constexpr double ALPHA_10 = 0.10;
inline constexpr double ALPHA_05 = 0.05;
inline constexpr double ALPHA_025 = 0.025;
inline constexpr double ALPHA_01 = 0.01;
inline constexpr double ALPHA_005 = 0.005;
}  // namespace anderson_darling

// Common significance levels
namespace significance {
inline constexpr double ALPHA_001 = 0.001;
inline constexpr double ALPHA_005 = 0.005;
inline constexpr double ALPHA_01 = 0.01;
inline constexpr double ALPHA_025 = 0.025;
inline constexpr double ALPHA_05 = 0.05;
inline constexpr double ALPHA_10 = 0.10;
inline constexpr double ALPHA_15 = 0.15;
inline constexpr double ALPHA_20 = 0.20;
inline constexpr double ALPHA_25 = 0.25;
inline constexpr double ALPHA_30 = 0.30;
inline constexpr double ALPHA_40 = 0.40;
inline constexpr double ALPHA_50 = 0.50;
inline constexpr double ALPHA_60 = 0.60;
inline constexpr double ALPHA_70 = 0.70;
inline constexpr double ALPHA_75 = 0.75;
inline constexpr double ALPHA_80 = 0.80;
inline constexpr double ALPHA_90 = 0.90;
inline constexpr double ALPHA_95 = 0.95;
inline constexpr double ALPHA_99 = 0.99;
inline constexpr double ALPHA_995 = 0.995;
inline constexpr double ALPHA_999 = 0.999;
}  // namespace significance

// Common iteration limits
namespace iterations {
inline constexpr int TINY = 10;
inline constexpr int SMALL = 100;
inline constexpr int MEDIUM = 200;
inline constexpr int LARGE = 500;
inline constexpr int VERY_LARGE = 1000;
inline constexpr int EXTRA_LARGE = 5000;
inline constexpr int MASSIVE = 10000;
}  // namespace iterations

// Convergence tolerances (in addition to those in precision_constants.h)
namespace convergence {
inline constexpr double ULTRA_TIGHT = 1e-15;
inline constexpr double VERY_TIGHT = 1e-12;
inline constexpr double TIGHT = 1e-10;
inline constexpr double STANDARD = 1e-8;
inline constexpr double RELAXED = 1e-6;
inline constexpr double LOOSE = 1e-4;
inline constexpr double VERY_LOOSE = 1e-3;
}  // namespace convergence

// Special threshold values
namespace thresholds {
/// Kolmogorov-Smirnov test threshold for small z
inline constexpr double KS_SMALL_Z = 0.27;

/// Minimum probability to avoid log(0)
inline constexpr double MIN_PROBABILITY = 1e-300;

/// Maximum reasonable standard deviation
inline constexpr double MAX_STD_DEV = 1e10;

/// Minimum count for chi-squared test validity
inline constexpr double MIN_CHI_SQUARED_COUNT = 5.0;
}  // namespace thresholds

}  // namespace algorithm
}  // namespace constants
}  // namespace libstats

#pragma once

#include <cstddef>
#include <limits>

/**
 * @file core/statistical_constants.h
 * @brief Statistical constants, critical values, and algorithm thresholds
 *
 * This header covers everything that is specific to the statistical domain:
 *
 *   - Probability bounds and log-probability limits
 *   - Standard normal, t, chi-squared, and F critical values
 *   - Goodness-of-fit test critical values (KS, AD, Shapiro-Wilk)
 *   - Statistical significance levels and confidence levels
 *   - Effect size, correlation, and power analysis thresholds
 *   - Robust estimation tuning constants (MAD, M-estimators)
 *   - Bayesian prior defaults, bootstrap and cross-validation parameters
 *   - Distribution-specific algorithm thresholds (Poisson, etc.)
 *
 * For pure mathematical constants and numerical precision, see math_constants.h.
 * For benchmarking and performance testing parameters, see performance_constants.h.
 */

namespace stats {
namespace detail {

// =============================================================================
// PROBABILITY BOUNDS AND SAFETY LIMITS
// Prevents underflow, overflow, and log(0) in probability calculations.
// =============================================================================

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

/// Minimum probability for log calculations to avoid log(0)
inline constexpr double LOG_CALCULATION_MIN_PROB = 1.0e-16;

/// Ultra-minimum probability for Anderson-Darling calculations
inline constexpr double ANDERSON_DARLING_MIN_PROB = 1.0e-300;

/// Minimum epsilon for chi-squared calculations
inline constexpr double CHI_SQUARED_EPSILON = 1.0e-12;

// =============================================================================
// SIGNIFICANCE LEVELS AND CONFIDENCE LEVELS
// =============================================================================

inline constexpr double ALPHA_001 = 0.001;
inline constexpr double ALPHA_01 = 0.01;
inline constexpr double ALPHA_05 = 0.05;
inline constexpr double ALPHA_10 = 0.10;

inline constexpr double DEFAULT_SIGNIFICANCE_LEVEL = 0.05;

inline constexpr double CONFIDENCE_90 = 0.90;
inline constexpr double CONFIDENCE_95 = 0.95;
inline constexpr double CONFIDENCE_99 = 0.99;
inline constexpr double CONFIDENCE_999 = 0.999;

// =============================================================================
// STANDARD NORMAL CRITICAL VALUES
// =============================================================================

/// Winitzki's approximation parameter for inverse error function
inline constexpr double WINITZKI_A = 0.147;

inline constexpr double Z_90 = 1.645;   ///< 90% CI (alpha = 0.10)
inline constexpr double Z_95 = 1.96;    ///< 95% CI (alpha = 0.05)
inline constexpr double Z_99 = 2.576;   ///< 99% CI (alpha = 0.01)
inline constexpr double Z_999 = 3.291;  ///< 99.9% CI (alpha = 0.001)
inline constexpr double Z_95_ONE_TAIL = 1.645;
inline constexpr double Z_99_ONE_TAIL = 2.326;

// =============================================================================
// STUDENT'S t CRITICAL VALUES
// =============================================================================

// 95% confidence, two-tailed
inline constexpr double T_95_DF_1 = 12.706;
inline constexpr double T_95_DF_2 = 4.303;
inline constexpr double T_95_DF_3 = 3.182;
inline constexpr double T_95_DF_4 = 2.776;
inline constexpr double T_95_DF_5 = 2.571;
inline constexpr double T_95_DF_10 = 2.228;
inline constexpr double T_95_DF_20 = 2.086;
inline constexpr double T_95_DF_30 = 2.042;
inline constexpr double T_95_DF_INF = 1.96;  // Approaches normal

// 99% confidence, two-tailed
inline constexpr double T_99_DF_1 = 63.657;
inline constexpr double T_99_DF_2 = 9.925;
inline constexpr double T_99_DF_3 = 5.841;
inline constexpr double T_99_DF_4 = 4.604;
inline constexpr double T_99_DF_5 = 4.032;
inline constexpr double T_99_DF_10 = 3.169;
inline constexpr double T_99_DF_20 = 2.845;
inline constexpr double T_99_DF_30 = 2.750;
inline constexpr double T_99_DF_INF = 2.576;  // Approaches normal

// =============================================================================
// CHI-SQUARED CRITICAL VALUES
// =============================================================================

// 95% confidence
inline constexpr double CHI2_95_DF_1 = 3.841;
inline constexpr double CHI2_95_DF_2 = 5.991;
inline constexpr double CHI2_95_DF_3 = 7.815;
inline constexpr double CHI2_95_DF_4 = 9.488;
inline constexpr double CHI2_95_DF_5 = 11.070;
inline constexpr double CHI2_95_DF_10 = 18.307;
inline constexpr double CHI2_95_DF_20 = 31.410;
inline constexpr double CHI2_95_DF_30 = 43.773;

// 99% confidence
inline constexpr double CHI2_99_DF_1 = 6.635;
inline constexpr double CHI2_99_DF_2 = 9.210;
inline constexpr double CHI2_99_DF_3 = 11.345;
inline constexpr double CHI2_99_DF_4 = 13.277;
inline constexpr double CHI2_99_DF_5 = 15.086;
inline constexpr double CHI2_99_DF_10 = 23.209;
inline constexpr double CHI2_99_DF_20 = 37.566;
inline constexpr double CHI2_99_DF_30 = 50.892;

// =============================================================================
// F-DISTRIBUTION CRITICAL VALUES
// =============================================================================

// 95% confidence (alpha = 0.05)
inline constexpr double F_95_DF_1_1 = 161.4;
inline constexpr double F_95_DF_1_5 = 6.61;
inline constexpr double F_95_DF_1_10 = 4.96;
inline constexpr double F_95_DF_1_20 = 4.35;
inline constexpr double F_95_DF_1_INF = 3.84;
inline constexpr double F_95_DF_5_5 = 5.05;
inline constexpr double F_95_DF_5_10 = 3.33;
inline constexpr double F_95_DF_5_20 = 2.71;
inline constexpr double F_95_DF_5_INF = 2.21;
inline constexpr double F_95_DF_10_10 = 2.98;
inline constexpr double F_95_DF_10_20 = 2.35;
inline constexpr double F_95_DF_10_INF = 1.83;

// 99% confidence (alpha = 0.01)
inline constexpr double F_99_DF_1_1 = 4052.0;
inline constexpr double F_99_DF_1_5 = 16.26;
inline constexpr double F_99_DF_1_10 = 10.04;
inline constexpr double F_99_DF_1_20 = 8.10;
inline constexpr double F_99_DF_1_INF = 6.63;
inline constexpr double F_99_DF_5_5 = 11.0;
inline constexpr double F_99_DF_5_10 = 5.64;
inline constexpr double F_99_DF_5_20 = 4.10;
inline constexpr double F_99_DF_5_INF = 3.02;
inline constexpr double F_99_DF_10_10 = 4.85;
inline constexpr double F_99_DF_10_20 = 3.37;
inline constexpr double F_99_DF_10_INF = 2.32;

// =============================================================================
// GOODNESS-OF-FIT CRITICAL VALUES
// =============================================================================

// Kolmogorov-Smirnov: alpha = 0.05
inline constexpr double KS_05_N_5 = 0.565;
inline constexpr double KS_05_N_10 = 0.409;
inline constexpr double KS_05_N_15 = 0.338;
inline constexpr double KS_05_N_20 = 0.294;
inline constexpr double KS_05_N_25 = 0.264;
inline constexpr double KS_05_N_30 = 0.242;
inline constexpr double KS_05_N_50 = 0.188;
inline constexpr double KS_05_N_100 = 0.134;

// Kolmogorov-Smirnov: alpha = 0.01
inline constexpr double KS_01_N_5 = 0.669;
inline constexpr double KS_01_N_10 = 0.490;
inline constexpr double KS_01_N_15 = 0.404;
inline constexpr double KS_01_N_20 = 0.352;
inline constexpr double KS_01_N_25 = 0.317;
inline constexpr double KS_01_N_30 = 0.290;
inline constexpr double KS_01_N_50 = 0.226;
inline constexpr double KS_01_N_100 = 0.161;

/// KS test large-sample approximation coefficients
inline constexpr double KS_APPROX_COEFF_1 = 0.12;
inline constexpr double KS_APPROX_COEFF_2 = 0.11;

// Anderson-Darling critical values for normality test
inline constexpr double AD_15 = 0.576;   ///< alpha = 0.15
inline constexpr double AD_10 = 0.656;   ///< alpha = 0.10
inline constexpr double AD_05 = 0.787;   ///< alpha = 0.05
inline constexpr double AD_025 = 0.918;  ///< alpha = 0.025
inline constexpr double AD_01 = 1.092;   ///< alpha = 0.01

/// Anderson-Darling threshold and p-value reference points
inline constexpr double AD_THRESHOLD_1 = 0.5;
inline constexpr double AD_P_VALUE_HIGH = 0.9;
inline constexpr double AD_P_VALUE_MEDIUM = 0.5;

// Shapiro-Wilk: alpha = 0.05
inline constexpr double SW_05_N_10 = 0.842;
inline constexpr double SW_05_N_15 = 0.881;
inline constexpr double SW_05_N_20 = 0.905;
inline constexpr double SW_05_N_25 = 0.918;
inline constexpr double SW_05_N_30 = 0.927;
inline constexpr double SW_05_N_50 = 0.947;

// Shapiro-Wilk: alpha = 0.01
inline constexpr double SW_01_N_10 = 0.781;
inline constexpr double SW_01_N_15 = 0.835;
inline constexpr double SW_01_N_20 = 0.868;
inline constexpr double SW_01_N_25 = 0.888;
inline constexpr double SW_01_N_30 = 0.900;
inline constexpr double SW_01_N_50 = 0.930;

// =============================================================================
// EFFECT SIZE, CORRELATION, AND POWER THRESHOLDS (Cohen's conventions)
// =============================================================================

inline constexpr double SMALL_EFFECT = 0.2;
inline constexpr double MEDIUM_EFFECT = 0.5;
inline constexpr double LARGE_EFFECT = 0.8;

inline constexpr double WEAK_CORRELATION = 0.3;
inline constexpr double MODERATE_CORRELATION = 0.5;
inline constexpr double STRONG_CORRELATION = 0.7;

inline constexpr double MINIMUM_POWER = 0.80;
inline constexpr double HIGH_POWER = 0.90;
inline constexpr double VERY_HIGH_POWER = 0.95;

inline constexpr double EXCESS_KURTOSIS_OFFSET = 3.0;  ///< Normal distribution baseline

// =============================================================================
// ROBUST ESTIMATION (M-estimators, MAD)
// =============================================================================

/// MAD scaling factor: converts MAD to a std-dev estimate (1/Phi^-1(3/4))
inline constexpr double MAD_SCALING_FACTOR = 1.4826;

/// Huber's M-estimator tuning constant (95% efficiency under normality)
inline constexpr double TUNING_HUBER_DEFAULT = 1.345;

/// Tukey's bisquare M-estimator tuning constant (95% efficiency)
inline constexpr double TUNING_TUKEY_DEFAULT = 4.685;

/// Hampel M-estimator tuning constants (a, b, c parameters)
inline constexpr double TUNING_HAMPEL_A = 1.7;
inline constexpr double TUNING_HAMPEL_B = 3.4;
inline constexpr double TUNING_HAMPEL_C = 8.5;

inline constexpr int MAX_ROBUST_ITERATIONS = 50;
inline constexpr double ROBUST_CONVERGENCE_TOLERANCE = 1.0e-6;
inline constexpr double MIN_ROBUST_SCALE = 1.0e-8;

// =============================================================================
// BAYESIAN ESTIMATION DEFAULTS
// Default parameters for a normal-inverse-gamma conjugate prior.
// =============================================================================

inline constexpr double PRIOR_DEFAULT_MEAN = 0.0;
inline constexpr double PRIOR_DEFAULT_PRECISION = 0.001;
inline constexpr double PRIOR_DEFAULT_SHAPE = 1.0;
inline constexpr double PRIOR_DEFAULT_RATE = 1.0;

// =============================================================================
// BOOTSTRAP AND CROSS-VALIDATION DEFAULTS
// =============================================================================

inline constexpr int BOOTSTRAP_DEFAULT_SAMPLES = 1000;
inline constexpr unsigned int BOOTSTRAP_DEFAULT_RANDOM_SEED = 42;
inline constexpr int CV_DEFAULT_K_FOLDS = 5;

// =============================================================================
// STATISTICAL TEST DATA REQUIREMENTS
// =============================================================================

inline constexpr int MIN_DATA_POINTS_FOR_CHI_SQUARE = 5;
inline constexpr int DEFAULT_EXPECTED_FREQUENCY_THRESHOLD = 5;
inline constexpr std::size_t MIN_DATA_POINTS_FOR_FITTING = 2;
inline constexpr std::size_t MIN_DATA_POINTS_FOR_JB_TEST = 8;
inline constexpr std::size_t MIN_DATA_POINTS_FOR_SW_TEST = 3;
inline constexpr std::size_t MAX_DATA_POINTS_FOR_SW_TEST = 5000;
inline constexpr std::size_t MIN_DATA_POINTS_FOR_LOOCV = 3;

// =============================================================================
// POISSON DISTRIBUTION CONSTANTS
// =============================================================================

/// Maximum lambda for stable Poisson computation
inline constexpr double MAX_POISSON_LAMBDA = 1.0e6;

/// Below this lambda: use direct computation; above: switch to log-space
inline constexpr double SMALL_LAMBDA_THRESHOLD = 10.0;

/// Above this lambda: use normal approximation (rule of thumb: lambda > 10)
inline constexpr double NORMAL_APPROXIMATION_THRESHOLD = 10.0;

/// Upper bound multiplier for quantile bracketing search
inline constexpr double QUANTILE_UPPER_BOUND_MULTIPLIER = 10.0;

/// Largest factorial that fits in a double (170!)
inline constexpr int MAX_FACTORIAL_FOR_DOUBLE = 170;

/// Cache threshold for small-k factorial optimization
inline constexpr int SMALL_K_CACHE_THRESHOLD = 20;

/// Continuity correction for normal approximation of Poisson
inline constexpr double CONTINUITY_CORRECTION = 0.5;

/// Normal approximation range multiplier
inline constexpr double NORMAL_RANGE_MULTIPLIER = 3.0;

}  // namespace detail
}  // namespace stats

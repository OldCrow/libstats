#pragma once

/**
 * @file core/threshold_constants.h
 * @brief Threshold constants for statistical tests and numerical algorithms.
 *
 * This header contains various threshold values used in statistical testing,
 * numerical methods, and robustness checks throughout the library.
 */

namespace libstats {
namespace constants {

/// Threshold constants
namespace thresholds {
/// Common significance levels
inline constexpr double ALPHA_001 = 0.001;
inline constexpr double ALPHA_01 = 0.01;
inline constexpr double ALPHA_05 = 0.05;
inline constexpr double ALPHA_10 = 0.10;
inline constexpr double ALPHA_90 = 0.90;

/// Numerical computation thresholds
/// Exponential overflow threshold for safe_exp function
inline constexpr double LOG_EXP_OVERFLOW_THRESHOLD = 700.0;

/// Large value threshold for log1pexp function
inline constexpr double LOG1PEXP_LARGE_THRESHOLD = 37.0;

/// Small value threshold for log1pexp function
inline constexpr double LOG1PEXP_SMALL_THRESHOLD = -37.0;

/// High condition number threshold for numerical stability warnings
inline constexpr double HIGH_CONDITION_NUMBER_THRESHOLD = 1.0e12;

/// Minimum epsilon for chi-squared calculations
inline constexpr double CHI_SQUARED_EPSILON = 1.0e-12;

/// Minimum probability for log calculations to avoid log(0)
inline constexpr double LOG_CALCULATION_MIN_PROB = 1.0e-16;

/// Ultra-minimum probability for Anderson-Darling calculations
inline constexpr double ANDERSON_DARLING_MIN_PROB = 1.0e-300;

/// Default significance level for statistical tests
inline constexpr double DEFAULT_SIGNIFICANCE_LEVEL = 0.05;

/// Log-space operation constants
/// Log-sum-exp threshold below which terms are considered negligible
inline constexpr double LOG_SUM_EXP_THRESHOLD = -50.0;

/// Lookup table size for log-space operations
inline constexpr std::size_t LOG_SPACE_LOOKUP_TABLE_SIZE = 1024;

/// Excess kurtosis offset (normal distribution baseline)
inline constexpr double EXCESS_KURTOSIS_OFFSET = 3.0;

/// Common confidence levels
inline constexpr double CONFIDENCE_90 = 0.90;
inline constexpr double CONFIDENCE_95 = 0.95;
inline constexpr double CONFIDENCE_99 = 0.99;
inline constexpr double CONFIDENCE_999 = 0.999;

/// Statistical fitting and testing thresholds
inline constexpr int MIN_DATA_POINTS_FOR_CHI_SQUARE = 5;
inline constexpr int DEFAULT_EXPECTED_FREQUENCY_THRESHOLD = 5;

/// Numerical precision thresholds
inline constexpr double STRICT_TOLERANCE = 1e-10;
inline constexpr double RELAXED_TOLERANCE = 1e-9;
inline constexpr double VERY_SMALL_PROBABILITY = 1e-12;

/// Effect size thresholds (Cohen's conventions)
inline constexpr double SMALL_EFFECT = 0.2;
inline constexpr double MEDIUM_EFFECT = 0.5;
inline constexpr double LARGE_EFFECT = 0.8;

/// Variance thresholds
inline constexpr double LOW_VARIANCE_THRESHOLD = 0.0625;  // σ² < 1/16 for low variance detection

/// Correlation strength thresholds
inline constexpr double WEAK_CORRELATION = 0.3;
inline constexpr double MODERATE_CORRELATION = 0.5;
inline constexpr double STRONG_CORRELATION = 0.7;

/// Power analysis thresholds
inline constexpr double MINIMUM_POWER = 0.80;
inline constexpr double HIGH_POWER = 0.90;
inline constexpr double VERY_HIGH_POWER = 0.95;

/// Kolmogorov-Smirnov test approximation coefficients
inline constexpr double KS_APPROX_COEFF_1 = 0.12;
inline constexpr double KS_APPROX_COEFF_2 = 0.11;

/// Anderson-Darling test threshold values and p-values
inline constexpr double AD_THRESHOLD_1 = 0.5;
inline constexpr double AD_P_VALUE_HIGH = 0.9;
inline constexpr double AD_P_VALUE_MEDIUM = 0.5;

/// Anderson-Darling test critical values for uniform distribution
namespace anderson_darling_uniform {
inline constexpr double CRITICAL_VALUE_001 = 3.857;
inline constexpr double CRITICAL_VALUE_005 = 2.492;
inline constexpr double CRITICAL_VALUE_010 = 1.933;
inline constexpr double CRITICAL_VALUE_025 = 1.159;

// P-value calculation thresholds and coefficients
inline constexpr double PVAL_THRESHOLD_1 = 0.2;
inline constexpr double PVAL_THRESHOLD_2 = 0.34;
inline constexpr double PVAL_THRESHOLD_3 = 0.6;

// Range 1: ad_stat < 0.2
inline constexpr double PVAL_RANGE1_COEFF_A = -1.2804;
inline constexpr double PVAL_RANGE1_EXPONENT = -0.5;

// Range 2: 0.2 <= ad_stat < 0.34
inline constexpr double PVAL_RANGE2_COEFF_A = -0.8;
inline constexpr double PVAL_RANGE2_COEFF_B = -0.26;

// Range 3: 0.34 <= ad_stat < 0.6
inline constexpr double PVAL_RANGE3_COEFF_A = -0.9;
inline constexpr double PVAL_RANGE3_COEFF_B = -0.16;

// Range 4: ad_stat >= 0.6
inline constexpr double PVAL_RANGE4_COEFF_A = -1.8;
inline constexpr double PVAL_RANGE4_COEFF_B = 0.258;
}  // namespace anderson_darling_uniform

/// Anderson-Darling test constants for exponential distribution
/// Based on D'Agostino and Stephens (1986) formulas
namespace anderson_darling_exponential {
/// Adjustment factor for sample size correction
inline constexpr double SAMPLE_SIZE_ADJUSTMENT_FACTOR = 0.6;

// Range 1: ad_adjusted < 0.2
inline constexpr double RANGE1_THRESHOLD = 0.2;
inline constexpr double RANGE1_COEFF_A = -13.436;
inline constexpr double RANGE1_COEFF_B = 101.14;
inline constexpr double RANGE1_COEFF_C = -223.73;

// Range 2: 0.2 <= ad_adjusted < 0.34
inline constexpr double RANGE2_THRESHOLD = 0.34;
inline constexpr double RANGE2_COEFF_A = -8.318;
inline constexpr double RANGE2_COEFF_B = 42.796;
inline constexpr double RANGE2_COEFF_C = -59.938;

// Range 3: 0.34 <= ad_adjusted < 0.6
inline constexpr double RANGE3_THRESHOLD = 0.6;
inline constexpr double RANGE3_COEFF_A = 0.9177;
inline constexpr double RANGE3_COEFF_B = -4.279;
inline constexpr double RANGE3_COEFF_C = -1.38;

// Range 4: 0.6 <= ad_adjusted < 2.0
inline constexpr double RANGE4_THRESHOLD = 2.0;
inline constexpr double RANGE4_COEFF_A = 1.2937;
inline constexpr double RANGE4_COEFF_B = -5.709;
inline constexpr double RANGE4_COEFF_C = 0.0186;
}  // namespace anderson_darling_exponential

/// Anderson-Darling test critical values for discrete distributions
namespace anderson_darling_discrete {
inline constexpr double CRITICAL_VALUE_001 = 3.857;
inline constexpr double CRITICAL_VALUE_005 = 2.492;
inline constexpr double CRITICAL_VALUE_010 = 1.933;
inline constexpr double CRITICAL_VALUE_025 = 1.159;

// P-value calculation thresholds and coefficients
inline constexpr double PVAL_THRESHOLD_1 = 0.5;
inline constexpr double PVAL_THRESHOLD_2 = 2.0;

// Range 1: ad_stat < 0.5
inline constexpr double PVAL_RANGE1_COEFF_A = -1.2337;
inline constexpr double PVAL_RANGE1_EXPONENT = -1.0;
inline constexpr double PVAL_RANGE1_COEFF_B = 1.0;

// Range 2: 0.5 <= ad_stat < 2.0
inline constexpr double PVAL_RANGE2_COEFF_A = -0.75;
inline constexpr double PVAL_RANGE2_COEFF_B = -0.5;

// Range 3: ad_stat >= 2.0
inline constexpr double PVAL_RANGE3_COEFF = -1.0;
}  // namespace anderson_darling_discrete

/// Chi-square test constants
namespace chi_square {
// Critical values for different alpha levels and degrees of freedom
inline constexpr double CRITICAL_VALUE_005_DF1 =
    3.841;  // Chi-squared critical value for alpha=0.05, df=1
inline constexpr double CRITICAL_VALUE_001_DF1 =
    6.635;  // Chi-squared critical value for alpha=0.01, df=1
inline constexpr double CRITICAL_VALUE_010_DF1 =
    2.706;  // Chi-squared critical value for alpha=0.10, df=1

// P-value approximations
inline constexpr double PVAL_LOW = 0.01;  // Rough approximation for low p-value
inline constexpr double PVAL_HIGH = 0.5;  // Rough approximation for high p-value
}  // namespace chi_square

/// Z-score constants
namespace z_score {
inline constexpr double Z_SCORE_90 = 1.645;   // 90% confidence (one-tail)
inline constexpr double Z_SCORE_95 = 1.96;    // 95% confidence (two-tail)
inline constexpr double Z_SCORE_97_5 = 1.96;  // 97.5% confidence (one-tail)
inline constexpr double Z_SCORE_99 = 2.576;   // 99% confidence (two-tail)

// P-value approximations
inline constexpr double PVAL_TAIL = 0.025;  // Approximate tail p-value for two-tailed test
}  // namespace z_score

/// Kolmogorov-Smirnov test constants for gamma distribution
namespace kolmogorov_smirnov_gamma {
inline constexpr double CRITICAL_VALUE_COEFFICIENT =
    1.36;  // KS test critical value coefficient at alpha=0.05
inline constexpr double LAMBDA_THRESHOLD_LOW =
    0.27;  // Lower threshold for lambda in p-value calculation
inline constexpr double LAMBDA_THRESHOLD_MID =
    1.0;  // Middle threshold for lambda in p-value calculation
}  // namespace kolmogorov_smirnov_gamma

/// Anderson-Darling test constants for gamma distribution
namespace anderson_darling_gamma {
// Modified statistic coefficients
inline constexpr double MOD_STAT_COEFF_1 = 0.75;  // First coefficient for modified statistic
inline constexpr double MOD_STAT_COEFF_2 = 2.25;  // Second coefficient for modified statistic

// P-value calculation thresholds
inline constexpr double PVAL_THRESHOLD_HIGH = 13.0;  // Upper threshold for p-value calculation
inline constexpr double PVAL_THRESHOLD_MID = 6.0;    // Middle threshold for p-value calculation

// P-value calculation coefficients
inline constexpr double PVAL_COEFF_HIGH = -1.28;  // Coefficient for high modified statistic
inline constexpr double PVAL_COEFF_LOW = -1.8;    // Coefficient for low modified statistic
inline constexpr double PVAL_OFFSET_LOW = 1.5;    // Offset for low modified statistic
}  // namespace anderson_darling_gamma

/// Default number of integration points for numerical methods
inline constexpr int DEFAULT_INTEGRATION_POINTS = 1000;

/// Minimum data points required for fitting distributions
inline constexpr size_t MIN_DATA_POINTS_FOR_FITTING = 2;

/// Minimum data points required for reliable Jarque-Bera test
inline constexpr size_t MIN_DATA_POINTS_FOR_JB_TEST = 8;

/// Minimum data points required for Shapiro-Wilk test
inline constexpr size_t MIN_DATA_POINTS_FOR_SW_TEST = 3;

/// Maximum data points recommended for Shapiro-Wilk test
inline constexpr size_t MAX_DATA_POINTS_FOR_SW_TEST = 5000;

/// Minimum data points required for LOOCV
inline constexpr size_t MIN_DATA_POINTS_FOR_LOOCV = 3;

/// Large degrees of freedom threshold for normal approximation
inline constexpr double LARGE_DF_THRESHOLD = 1000.0;

/// Poisson distribution specific constants
namespace poisson {
/// Maximum lambda value for Poisson distribution (numerical stability)
inline constexpr double MAX_POISSON_LAMBDA = 1.0e6;

/// Threshold for switching from direct computation to log-space computation
inline constexpr double SMALL_LAMBDA_THRESHOLD = 10.0;

/// Threshold for using normal approximation (rule of thumb: λ > 10)
inline constexpr double NORMAL_APPROXIMATION_THRESHOLD = 10.0;

/// Upper bound multiplier for quantile bracketing search (λ + multiplier * √λ)
inline constexpr double QUANTILE_UPPER_BOUND_MULTIPLIER = 10.0;

/// Maximum factorial value before overflow (170! is the largest that fits in double)
inline constexpr int MAX_FACTORIAL_FOR_DOUBLE = 170;

/// Cache threshold for small lambda optimization (k <= threshold uses cached factorial)
inline constexpr int SMALL_K_CACHE_THRESHOLD = 20;

/// Continuity correction factor for normal approximation
inline constexpr double CONTINUITY_CORRECTION = 0.5;

/// Normal approximation standard deviation multiplier for range checking
inline constexpr double NORMAL_RANGE_MULTIPLIER = 3.0;
}  // namespace poisson
}  // namespace thresholds

}  // namespace constants
}  // namespace libstats

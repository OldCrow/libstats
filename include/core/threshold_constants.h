#ifndef LIBSTATS_CORE_THRESHOLD_CONSTANTS_H_
#define LIBSTATS_CORE_THRESHOLD_CONSTANTS_H_

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
    }
}

} // namespace constants
} // namespace libstats

#endif // LIBSTATS_CORE_THRESHOLD_CONSTANTS_H_


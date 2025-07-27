#ifndef LIBSTATS_CORE_CONSTANTS_H_
#define LIBSTATS_CORE_CONSTANTS_H_

#include <cstddef>
#include <climits>
#include <type_traits>
#include <cassert>
#include <cmath>
#include <algorithm>

/**
 * @file core/constants.h
 * @brief Platform-independent mathematical and statistical constants for libstats
 * 
 * This header contains all pure mathematical constants, precision tolerances,
 * and statistical critical values that are independent of platform architecture.
 */

namespace libstats {
namespace constants {

/// Basic precision and tolerance values
namespace precision {
    /// Minimum value considered non-zero throughout libstats
    inline constexpr double ZERO = 1.0e-30;
    
    /// Default convergence tolerance for iterative algorithms
    inline constexpr double DEFAULT_TOLERANCE = 1.0e-8;
    
    /// High precision tolerance for critical numerical operations
    inline constexpr double HIGH_PRECISION_TOLERANCE = 1.0e-12;
    
    /// Ultra-high precision for research and validation purposes
    inline constexpr double ULTRA_HIGH_PRECISION_TOLERANCE = 1.0e-15;
    
    /// Epsilon for safe log probability computations
    inline constexpr double LOG_PROBABILITY_EPSILON = 1.0e-300;
    
    /// Minimum standard deviation for distribution parameters
    inline constexpr double MIN_STD_DEV = 1.0e-6;
    
    /// Machine epsilon variants for different precisions
    inline constexpr double MACHINE_EPSILON = std::numeric_limits<double>::epsilon();
    inline constexpr float MACHINE_EPSILON_FLOAT = std::numeric_limits<float>::epsilon();
    inline constexpr long double MACHINE_EPSILON_LONG_DOUBLE = std::numeric_limits<long double>::epsilon();
    
    /// Numerical differentiation step sizes
    inline constexpr double FORWARD_DIFF_STEP = 1.0e-8;
    inline constexpr double CENTRAL_DIFF_STEP = 1.0e-6;
    inline constexpr double NUMERICAL_DERIVATIVE_STEP = 1.0e-5;
    
    /// Convergence criteria for different numerical methods
    inline constexpr double NEWTON_RAPHSON_TOLERANCE = 1.0e-10;
    inline constexpr double BISECTION_TOLERANCE = 1.0e-12;
    inline constexpr double GRADIENT_DESCENT_TOLERANCE = 1.0e-9;
    inline constexpr double CONJUGATE_GRADIENT_TOLERANCE = 1.0e-10;
    
    /// Maximum iterations for numerical methods
    inline constexpr std::size_t MAX_NEWTON_ITERATIONS = 100;
    inline constexpr std::size_t MAX_BISECTION_ITERATIONS = 1000;
    inline constexpr std::size_t MAX_GRADIENT_DESCENT_ITERATIONS = 10000;
    inline constexpr std::size_t MAX_CONJUGATE_GRADIENT_ITERATIONS = 1000;
    
    /// Maximum iterations for special mathematical functions
    inline constexpr std::size_t MAX_BETA_ITERATIONS = 100;
    inline constexpr std::size_t MAX_GAMMA_SERIES_ITERATIONS = 1000;
    
    /// Numerical integration tolerances
    inline constexpr double INTEGRATION_TOLERANCE = 1.0e-10;
    inline constexpr double ADAPTIVE_INTEGRATION_TOLERANCE = 1.0e-8;
    inline constexpr double MONTE_CARLO_INTEGRATION_TOLERANCE = 1.0e-6;
    
    /// Maximum recursion depth for adaptive Simpson's rule
    inline constexpr int MAX_ADAPTIVE_SIMPSON_DEPTH = 15;
}

/// Probability bounds and safety limits
namespace probability {
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
}

/// Mathematical constants
namespace math {
    /// High-precision value of π
    inline constexpr double PI = 3.141592653589793238462643383279502884;
    
    /// Natural logarithm of 2
    inline constexpr double LN2 = 0.6931471805599453094172321214581766;
    
    /// Natural logarithm of 10
    inline constexpr double LN_10 = 2.302585092994046;
    
    /// Euler's number (e)
    inline constexpr double E = 2.7182818284590452353602874713526625;
    
    /// Square root of π
    inline constexpr double SQRT_PI = 1.7724538509055158819194275219496950;
    
    /// Square root of 2π (used in Gaussian calculations)
    inline constexpr double SQRT_2PI = 2.5066282746310005024157652848110453;
    
    /// Natural logarithm of 2π
    inline constexpr double LN_2PI = 1.8378770664093454835606594728112353;
    
    /// Square root of 2
    inline constexpr double SQRT_2 = 1.4142135623730950488016887242096981;
    
    /// Half of ln(2π)
    inline constexpr double HALF_LN_2PI = 0.9189385332046727417803297364056176;
    
    /// Commonly used fractional constants
    inline constexpr double HALF = 0.5;
    inline constexpr double NEG_HALF = -0.5;
    inline constexpr double QUARTER = 0.25;
    inline constexpr double THREE_QUARTERS = 0.75;
    
    /// Commonly used negative constants for efficiency
    inline constexpr double NEG_ONE = -1.0;
    inline constexpr double NEG_TWO = -2.0;
    
    /// Commonly used integer constants as doubles
    inline constexpr double ZERO_DOUBLE = 0.0;
    inline constexpr double ONE = 1.0;
    inline constexpr double TWO = 2.0;
    inline constexpr double THREE = 3.0;
    inline constexpr double FOUR = 4.0;
    inline constexpr double FIVE = 5.0;
    inline constexpr double TEN = 10.0;
    inline constexpr double HUNDRED = 100.0;
    inline constexpr double THOUSAND = 1000.0;
    inline constexpr double THOUSANDTH = 0.001;
    inline constexpr double TENTH = 0.1;
    
    /// Commonly used integer constants
    inline constexpr int ZERO_INT = 0;
    inline constexpr int ONE_INT = 1;
    inline constexpr int TWO_INT = 2;
    inline constexpr int THREE_INT = 3;
    inline constexpr int FOUR_INT = 4;
    inline constexpr int FIVE_INT = 5;
    inline constexpr int TEN_INT = 10;
    
    /// Additional numerical constants
    inline constexpr double SIX = 6.0;
    inline constexpr double THIRTEEN = 13.0;
    inline constexpr double TWO_TWENTY_FIVE = 225.0;
    inline constexpr double ONE_POINT_TWO_EIGHT = 1.28;
    inline constexpr double ONE_POINT_EIGHT = 1.8;
    inline constexpr double ONE_POINT_FIVE = 1.5;
    
    /// Reciprocal of e (1/e)
    inline constexpr double E_INV = 1.0 / E;
    
    /// Maximum lambda value for Poisson distribution (numerical stability)
    inline constexpr double MAX_POISSON_LAMBDA = 1.0e6;
    
    /// Poisson distribution specific constants
    namespace poisson {
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
    
    /// Precomputed reciprocals to avoid division operations
    inline constexpr double ONE_THIRD = 1.0/3.0;
    inline constexpr double ONE_SIXTH = 1.0/6.0;
    inline constexpr double ONE_TWELFTH = 1.0/12.0;
    
    /// Additional mathematical constants
    /// Golden ratio (φ) = (1 + √5) / 2
    inline constexpr double PHI = 1.6180339887498948482045868343656381;
    
    /// Euler-Mascheroni constant (γ)
    inline constexpr double EULER_MASCHERONI = 0.5772156649015328606065120900824024;
    
    /// Catalan's constant G
    inline constexpr double CATALAN = 0.9159655941772190150546035149323841;
    
    /// Apéry's constant ζ(3)
    inline constexpr double APERY = 1.2020569031595942853997381615114499;
    
    /// Natural logarithm of golden ratio
    inline constexpr double LN_PHI = 0.4812118250596034474977589134243684;
    
    /// Silver ratio (1 + √2)
    inline constexpr double SILVER_RATIO = 2.4142135623730950488016887242096981;
    
    /// Plastic number (real root of x³ - x - 1 = 0)
    inline constexpr double PLASTIC_NUMBER = 1.3247179572447460259609088544780973;
    
    /// Natural logarithm of π
    inline constexpr double LN_PI = 1.1447298858494001741434273513530587;
    
    /// Square root of 3
    inline constexpr double SQRT_3 = 1.7320508075688772935274463415058723;
    
    /// Square root of 5
    inline constexpr double SQRT_5 = 2.2360679774997896964091736687312762;
    
    /// Reciprocal of golden ratio (1/φ)
    inline constexpr double PHI_INV = 1.0 / PHI;
    
    /// Reciprocal of π (1/π)
    inline constexpr double PI_INV = 1.0 / PI;
    
    /// Winitzki's approximation parameter for inverse error function
    inline constexpr double WINITZKI_A = 0.147;
    
    /// Derived mathematical expressions
    inline constexpr double INV_SQRT_2PI = 1.0 / SQRT_2PI;
    inline constexpr double INV_SQRT_2 = 1.0 / SQRT_2;
    inline constexpr double INV_SQRT_3 = 1.0 / SQRT_3;
    inline constexpr double TWO_PI = 2.0 * PI;
    inline constexpr double PI_OVER_2 = PI / 2.0;
    inline constexpr double PI_OVER_3 = PI / 3.0;
    inline constexpr double PI_OVER_4 = PI / 4.0;
    inline constexpr double PI_OVER_6 = PI / 6.0;
    inline constexpr double THREE_PI_OVER_2 = 3.0 * PI / 2.0;
    inline constexpr double FOUR_PI = 4.0 * PI;
    inline constexpr double NEG_HALF_LN_2PI = -0.5 * LN_2PI;
}

/// Statistical critical values and commonly used constants
namespace statistical {
    /// Standard normal distribution critical values
    namespace normal {
        /// 90% confidence interval (α = 0.10)
        inline constexpr double Z_90 = 1.645;
        
        /// 95% confidence interval (α = 0.05)
        inline constexpr double Z_95 = 1.96;
        
        /// 99% confidence interval (α = 0.01)
        inline constexpr double Z_99 = 2.576;
        
        /// 99.9% confidence interval (α = 0.001)
        inline constexpr double Z_999 = 3.291;
        
        /// One-tailed 95% critical value
        inline constexpr double Z_95_ONE_TAIL = 1.645;
        
        /// One-tailed 99% critical value
        inline constexpr double Z_99_ONE_TAIL = 2.326;
    }
    
    /// Student's t-distribution critical values (selected degrees of freedom)
    namespace t_distribution {
        /// t-critical values for 95% confidence (two-tailed)
        inline constexpr double T_95_DF_1 = 12.706;
        inline constexpr double T_95_DF_2 = 4.303;
        inline constexpr double T_95_DF_3 = 3.182;
        inline constexpr double T_95_DF_4 = 2.776;
        inline constexpr double T_95_DF_5 = 2.571;
        inline constexpr double T_95_DF_10 = 2.228;
        inline constexpr double T_95_DF_20 = 2.086;
        inline constexpr double T_95_DF_30 = 2.042;
        inline constexpr double T_95_DF_INF = 1.96;  // Approaches normal distribution
        
        /// t-critical values for 99% confidence (two-tailed)
        inline constexpr double T_99_DF_1 = 63.657;
        inline constexpr double T_99_DF_2 = 9.925;
        inline constexpr double T_99_DF_3 = 5.841;
        inline constexpr double T_99_DF_4 = 4.604;
        inline constexpr double T_99_DF_5 = 4.032;
        inline constexpr double T_99_DF_10 = 3.169;
        inline constexpr double T_99_DF_20 = 2.845;
        inline constexpr double T_99_DF_30 = 2.750;
        inline constexpr double T_99_DF_INF = 2.576;  // Approaches normal distribution
    }
    
    /// Chi-square distribution critical values
    namespace chi_square {
        /// 95% confidence critical values for common degrees of freedom
        inline constexpr double CHI2_95_DF_1 = 3.841;
        inline constexpr double CHI2_95_DF_2 = 5.991;
        inline constexpr double CHI2_95_DF_3 = 7.815;
        inline constexpr double CHI2_95_DF_4 = 9.488;
        inline constexpr double CHI2_95_DF_5 = 11.070;
        inline constexpr double CHI2_95_DF_10 = 18.307;
        inline constexpr double CHI2_95_DF_20 = 31.410;
        inline constexpr double CHI2_95_DF_30 = 43.773;
        
        /// 99% confidence critical values for common degrees of freedom
        inline constexpr double CHI2_99_DF_1 = 6.635;
        inline constexpr double CHI2_99_DF_2 = 9.210;
        inline constexpr double CHI2_99_DF_3 = 11.345;
        inline constexpr double CHI2_99_DF_4 = 13.277;
        inline constexpr double CHI2_99_DF_5 = 15.086;
        inline constexpr double CHI2_99_DF_10 = 23.209;
        inline constexpr double CHI2_99_DF_20 = 37.566;
        inline constexpr double CHI2_99_DF_30 = 50.892;
    }
    
    /// F-distribution critical values (selected numerator/denominator df)
    namespace f_distribution {
        /// F-critical values for 95% confidence (α = 0.05)
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
        
        /// F-critical values for 99% confidence (α = 0.01)
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
    }
    
    /// Commonly used statistical thresholds
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
    }
    
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
        }
        
        /// Maximum iterations for robust iterative algorithms
        inline constexpr int MAX_ROBUST_ITERATIONS = 50;
        
        /// Convergence tolerance for robust estimation
        inline constexpr double ROBUST_CONVERGENCE_TOLERANCE = 1.0e-6;
        
        /// Minimum robust scale factor to prevent numerical issues
        inline constexpr double MIN_ROBUST_SCALE = 1.0e-8;
    }
    
    /// Bayesian estimation default priors
    namespace bayesian {
        /// Default prior parameters for normal-inverse-gamma conjugate prior
        namespace priors {
            /// Default prior mean
            inline constexpr double DEFAULT_PRIOR_MEAN = 0.0;
            
            /// Default prior precision (inverse variance)
            inline constexpr double DEFAULT_PRIOR_PRECISION = 0.001;
            
            /// Default prior shape parameter
            inline constexpr double DEFAULT_PRIOR_SHAPE = 1.0;
            
            /// Default prior rate parameter
            inline constexpr double DEFAULT_PRIOR_RATE = 1.0;
        }
    }
    
    /// Default bootstrap parameters
    namespace bootstrap {
        /// Default number of bootstrap samples
        inline constexpr int DEFAULT_BOOTSTRAP_SAMPLES = 1000;
        
        /// Default random seed for reproducible results
        inline constexpr unsigned int DEFAULT_RANDOM_SEED = 42;
    }
    
    /// Cross-validation defaults
    namespace cross_validation {
        /// Default number of folds for k-fold cross-validation
        inline constexpr int DEFAULT_K_FOLDS = 5;
    }
    
    /// Kolmogorov-Smirnov critical values for goodness-of-fit tests
    namespace kolmogorov_smirnov {
        /// Critical values for α = 0.05 (95% confidence)
        inline constexpr double KS_05_N_5 = 0.565;
        inline constexpr double KS_05_N_10 = 0.409;
        inline constexpr double KS_05_N_15 = 0.338;
        inline constexpr double KS_05_N_20 = 0.294;
        inline constexpr double KS_05_N_25 = 0.264;
        inline constexpr double KS_05_N_30 = 0.242;
        inline constexpr double KS_05_N_50 = 0.188;
        inline constexpr double KS_05_N_100 = 0.134;
        
        /// Critical values for α = 0.01 (99% confidence)
        inline constexpr double KS_01_N_5 = 0.669;
        inline constexpr double KS_01_N_10 = 0.490;
        inline constexpr double KS_01_N_15 = 0.404;
        inline constexpr double KS_01_N_20 = 0.352;
        inline constexpr double KS_01_N_25 = 0.317;
        inline constexpr double KS_01_N_30 = 0.290;
        inline constexpr double KS_01_N_50 = 0.226;
        inline constexpr double KS_01_N_100 = 0.161;
    }
    
    /// Anderson-Darling critical values for normality tests
    namespace anderson_darling {
        /// Critical values for normality test at different significance levels
        inline constexpr double AD_15 = 0.576;  // α = 0.15
        inline constexpr double AD_10 = 0.656;  // α = 0.10
        inline constexpr double AD_05 = 0.787;  // α = 0.05
        inline constexpr double AD_025 = 0.918; // α = 0.025
        inline constexpr double AD_01 = 1.092;  // α = 0.01
    }
    
    /// Shapiro-Wilk critical values for normality tests
    namespace shapiro_wilk {
        /// Critical values for α = 0.05 (selected sample sizes)
        inline constexpr double SW_05_N_10 = 0.842;
        inline constexpr double SW_05_N_15 = 0.881;
        inline constexpr double SW_05_N_20 = 0.905;
        inline constexpr double SW_05_N_25 = 0.918;
        inline constexpr double SW_05_N_30 = 0.927;
        inline constexpr double SW_05_N_50 = 0.947;
        
        /// Critical values for α = 0.01 (selected sample sizes)
        inline constexpr double SW_01_N_10 = 0.781;
        inline constexpr double SW_01_N_15 = 0.835;
        inline constexpr double SW_01_N_20 = 0.868;
        inline constexpr double SW_01_N_25 = 0.888;
        inline constexpr double SW_01_N_30 = 0.900;
        inline constexpr double SW_01_N_50 = 0.930;
    }
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

} // namespace constants
} // namespace libstats

#endif // LIBSTATS_CORE_CONSTANTS_H_

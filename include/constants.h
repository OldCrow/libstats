#ifndef LIBSTATS_CONSTANTS_H_
#define LIBSTATS_CONSTANTS_H_

#include <cstddef>
#include <limits>
#include <type_traits>
#include <cassert>
#include <cmath>
#include <algorithm>

// Forward declaration for platform-specific tuning
#include "cpu_detection.h"

namespace libstats {
    namespace cpu { 
        const Features& get_features(); 
        size_t optimal_double_width();
        size_t optimal_alignment();
    }
}

/**
 * @file constants.h
 * @brief Mathematical and numerical constants for libstats
 * 
 * This header contains all mathematical constants, precision tolerances,
 * and optimization thresholds used throughout the library.
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

/// SIMD optimization parameters and architectural constants
namespace simd {
    /// Default SIMD block size for vectorized operations
    inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;
    
    /// Minimum problem size to benefit from SIMD
    inline constexpr std::size_t MIN_SIMD_SIZE = 4;
    
    /// Maximum block size for cache optimization
    inline constexpr std::size_t MAX_BLOCK_SIZE = 64;
    
    /// SIMD alignment requirement (bytes)
    inline constexpr std::size_t SIMD_ALIGNMENT = 32;
    
    /// Platform-specific SIMD alignment constants
    namespace alignment {
        /// AVX-512: 64-byte alignment for optimal performance
        inline constexpr std::size_t AVX512_ALIGNMENT = 64;
        
        /// AVX/AVX2: 32-byte alignment
        inline constexpr std::size_t AVX_ALIGNMENT = 32;
        
        /// SSE: 16-byte alignment
        inline constexpr std::size_t SSE_ALIGNMENT = 16;
        
        /// ARM NEON: 16-byte alignment
        inline constexpr std::size_t NEON_ALIGNMENT = 16;
        
        /// Generic cache line alignment (64 bytes on most modern systems)
        inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;
        
        /// Minimum safe alignment for all platforms
        inline constexpr std::size_t MIN_SAFE_ALIGNMENT = 8;
    }
    
    /// Matrix operation block sizes for cache-friendly operations
    namespace matrix {
        /// Small matrix block size for L1 cache optimization
        inline constexpr std::size_t L1_BLOCK_SIZE = 64;
        
        /// Medium matrix block size for L2 cache optimization  
        inline constexpr std::size_t L2_BLOCK_SIZE = 256;
        
        /// Large matrix block size for L3 cache optimization
        inline constexpr std::size_t L3_BLOCK_SIZE = 1024;
        
        /// Step size for matrix traversal (optimized for cache lines)
        inline constexpr std::size_t STEP_SIZE = 8;
        
        /// Panel width for matrix decomposition algorithms
        inline constexpr std::size_t PANEL_WIDTH = 64;
        
        /// Minimum matrix size for blocking to be beneficial
        inline constexpr std::size_t MIN_BLOCK_SIZE = 32;
        
        /// Maximum practical block size (memory constraint)
        inline constexpr std::size_t MAX_BLOCK_SIZE = 2048;
    }
    
    /// Platform-specific SIMD register widths (in number of doubles)
    namespace registers {
        /// AVX-512: 8 doubles per register
        inline constexpr std::size_t AVX512_DOUBLES = 8;
        
        /// AVX/AVX2: 4 doubles per register
        inline constexpr std::size_t AVX_DOUBLES = 4;
        
        /// SSE2: 2 doubles per register
        inline constexpr std::size_t SSE_DOUBLES = 2;
        
        /// ARM NEON: 2 doubles per register
        inline constexpr std::size_t NEON_DOUBLES = 2;
        
        /// Scalar: 1 double (no SIMD)
        inline constexpr std::size_t SCALAR_DOUBLES = 1;
    }
    
    /// Loop unrolling factors for different architectures
    namespace unroll {
        /// Unroll factor for AVX-512 (can handle more parallel operations)
        inline constexpr std::size_t AVX512_UNROLL = 4;
        
        /// Unroll factor for AVX/AVX2
        inline constexpr std::size_t AVX_UNROLL = 2;
        
        /// Unroll factor for SSE
        inline constexpr std::size_t SSE_UNROLL = 2;
        
        /// Unroll factor for ARM NEON
        inline constexpr std::size_t NEON_UNROLL = 2;
        
        /// Conservative unroll factor for scalar operations
        inline constexpr std::size_t SCALAR_UNROLL = 1;
    }
    
    /// CPU detection and runtime constants
    namespace cpu {
        /// Maximum backoff time during CPU feature detection (nanoseconds)
        inline constexpr uint64_t MAX_BACKOFF_NANOSECONDS = 1000;
        
        /// Default cache line size fallback (bytes)
        inline constexpr uint32_t DEFAULT_CACHE_LINE_SIZE = 64;
        
        /// Default TSC frequency measurement duration (milliseconds)
        inline constexpr uint32_t DEFAULT_TSC_SAMPLE_MS = 10;
        
        /// Conversion factor from nanoseconds to Hertz
        inline constexpr double NANOSECONDS_TO_HZ = 1e9;
    }
    
    /// SIMD optimization thresholds and platform-specific constants
    namespace optimization {
        /// Medium dataset minimum size for alignment benefits
        inline constexpr std::size_t MEDIUM_DATASET_MIN_SIZE = 32;
        
        /// Minimum threshold for alignment benefit checks
        inline constexpr std::size_t ALIGNMENT_BENEFIT_THRESHOLD = 32;
        
        /// Minimum size for AVX-512 aligned datasets
        inline constexpr std::size_t AVX512_MIN_ALIGNED_SIZE = 8;
        
        /// Aggressive SIMD threshold for Apple Silicon
        inline constexpr std::size_t APPLE_SILICON_AGGRESSIVE_THRESHOLD = 6;
        
        /// Minimum size threshold for AVX-512 small benefit
        inline constexpr std::size_t AVX512_SMALL_BENEFIT_THRESHOLD = 4;
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

/// Platform-specific tuning functions
namespace platform {
    /**
     * @brief Get optimized SIMD block size based on detected CPU features
     * @return Optimal block size for SIMD operations
     */
    inline std::size_t get_optimal_simd_block_size() {
        const auto& features = cpu::get_features();
        
        // AVX-512: 8 doubles per register
        if (features.avx512f) {
            return 8;
        }
        // AVX/AVX2: 4 doubles per register
        else if (features.avx || features.avx2) {
            return 4;
        }
        // SSE2: 2 doubles per register
        else if (features.sse2) {
            return 2;
        }
        // ARM NEON: 2 doubles per register
        else if (features.neon) {
            return 2;
        }
        // No SIMD support
        else {
            return 1;
        }
    }
    
    /**
     * @brief Get optimized memory alignment based on detected CPU features
     * @return Optimal memory alignment in bytes
     */
    inline std::size_t get_optimal_alignment() {
        const auto& features = cpu::get_features();
        
        // AVX-512: 64-byte alignment
        if (features.avx512f) {
            return 64;
        }
        // AVX/AVX2: 32-byte alignment
        else if (features.avx || features.avx2) {
            return 32;
        }
        // SSE2: 16-byte alignment
        else if (features.sse2) {
            return 16;
        }
        // ARM NEON: 16-byte alignment
        else if (features.neon) {
            return 16;
        }
        // Default cache line alignment
        else {
            return features.cache_line_size > 0 ? features.cache_line_size : 64;
        }
    }
    
    /**
     * @brief Get optimized minimum size for SIMD operations
     * @return Minimum size threshold for SIMD benefit
     */
    inline std::size_t get_min_simd_size() {
        const auto& features = cpu::get_features();
        
        // Higher-end SIMD can handle smaller datasets efficiently
        if (features.avx512f) {
            return 4;
        }
        else if (features.avx2 || features.fma) {
            return 6;
        }
        else if (features.avx || features.sse4_2) {
            return 8;
        }
        else if (features.sse2 || features.neon) {
            return 12;
        }
        else {
            return 32;  // No SIMD benefit until larger sizes
        }
    }
    
    /**
     * @brief Get optimized parallel processing thresholds based on CPU features
     * @return Optimal minimum elements for parallel processing
     */
    inline std::size_t get_min_parallel_elements() {
        const auto& features = cpu::get_features();
        
        // More powerful SIMD allows for lower parallel thresholds
        if (features.avx512f) {
            return 256;
        }
        else if (features.avx2 || features.fma) {
            return 384;
        }
        else if (features.avx) {
            return 512;
        }
        else if (features.sse4_2) {
            return 768;
        }
        else if (features.sse2 || features.neon) {
            return 1024;
        }
        else {
            return 2048;  // Higher threshold for scalar operations
        }
    }
    
    /**
     * @brief Get platform-optimized grain size for parallel operations
     * @return Optimal grain size for work distribution
     */
    inline std::size_t get_optimal_grain_size() {
        const auto& features = cpu::get_features();
        const std::size_t optimal_block = get_optimal_simd_block_size();
        
        // Grain size should be a multiple of SIMD block size
        // and account for cache line efficiency
        const std::size_t cache_line_elements = features.cache_line_size / sizeof(double);
        const std::size_t base_grain = std::max(optimal_block * 8, cache_line_elements);
        
        // Adjust based on CPU capabilities
        if (features.avx512f) {
            return base_grain * 2;  // Can handle larger chunks efficiently
        }
        else if (features.avx2 || features.fma) {
            return base_grain * 1.5;
        }
        else {
            return base_grain;
        }
    }
    
    /**
     * @brief Check if platform supports efficient transcendental functions
     * @return True if CPU has hardware support for fast transcendental operations
     */
    inline bool supports_fast_transcendental() {
        const auto& features = cpu::get_features();
        // FMA typically indicates more modern CPU with better transcendental support
        return features.fma || features.avx2 || features.avx512f;
    }
    
    /**
     * @brief Get cache-optimized thresholds for algorithms
     * @return Structure with cache-aware thresholds
     */
    struct CacheThresholds {
        std::size_t l1_optimal_size;    // Optimal size for L1 cache
        std::size_t l2_optimal_size;    // Optimal size for L2 cache
        std::size_t l3_optimal_size;    // Optimal size for L3 cache
        std::size_t blocking_size;      // Optimal blocking size for cache tiling
    };
    
    inline CacheThresholds get_cache_thresholds() {
        const auto& features = cpu::get_features();
        CacheThresholds thresholds;
        
        // Convert cache sizes from bytes to number of doubles
        thresholds.l1_optimal_size = features.l1_cache_size > 0 ? 
            (features.l1_cache_size / sizeof(double)) / 2 : 4096;  // Use half of L1
        
        thresholds.l2_optimal_size = features.l2_cache_size > 0 ? 
            (features.l2_cache_size / sizeof(double)) / 2 : 32768;
        
        thresholds.l3_optimal_size = features.l3_cache_size > 0 ? 
            (features.l3_cache_size / sizeof(double)) / 4 : 262144;
        
        // Blocking size for cache tiling (typically sqrt of L1 size)
        thresholds.blocking_size = static_cast<std::size_t>(
            std::sqrt(static_cast<double>(thresholds.l1_optimal_size))
        );
        
        return thresholds;
    }
}

/// Parallel processing optimization constants
namespace parallel {
    /// Minimum number of elements required to use parallel processing in statistical calculations
    /// Below this threshold, the overhead of thread pool task submission
    /// and synchronization typically outweighs the benefits of parallelization
    inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 512;
    
    /// Minimum number of elements for parallel distribution computations
    /// Distribution calculations can be parallelized at a lower threshold
    /// since they involve more computation per element
    inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 256;
    
    /// Default grain size for parallel statistical computation loops
    /// This determines the minimum number of work items assigned to each thread
    /// Optimized for cache line alignment and SIMD register efficiency
    inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 64;
    
    /// Grain size for simple parallel operations (scaling, element-wise ops)
    /// For very simple operations like scaling or element-wise arithmetic
    inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 32;
    
    /// Minimum dataset size for parallel statistical algorithms
    /// Statistical algorithms benefit from parallelization when
    /// processing large datasets above this threshold
    inline constexpr std::size_t MIN_DATASET_SIZE_FOR_PARALLEL = 1000;
    
    /// Minimum number of bootstrap samples for parallel bootstrap
    /// When performing bootstrap resampling, parallelization
    /// becomes beneficial above this threshold
    inline constexpr std::size_t MIN_BOOTSTRAP_SAMPLES_FOR_PARALLEL = 100;
    
    /// Minimum total work units for parallel Monte Carlo methods
    /// Monte Carlo simulations benefit from parallelization when the total
    /// computational work exceeds this threshold
    inline constexpr std::size_t MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL = 10000;
    
    /// Grain size for parallel Monte Carlo and simulation operations
    /// Optimized for simulation algorithms which typically involve
    /// more computation per work item
    inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 25;
    
    /// Maximum grain size to ensure good load balancing
    /// Prevents any single thread from getting too much work
    inline constexpr std::size_t MAX_GRAIN_SIZE = 1000;
    
    /// Minimum work per thread in parallel reductions
    /// For parallel sum reductions and similar operations
    inline constexpr std::size_t MIN_WORK_PER_THREAD = 100;
    
    /// Batch size for parallel processing of data samples
    /// When processing multiple data samples in statistical algorithms
    inline constexpr std::size_t SAMPLE_BATCH_SIZE = 16;
    
    /// Minimum matrix size for parallel matrix operations
    /// Matrix operations (multiplication, decomposition) benefit from
    /// parallelization above this threshold
    inline constexpr std::size_t MIN_MATRIX_SIZE_FOR_PARALLEL = 256;
    
    /// Minimum number of iterations for parallel iterative algorithms
    /// Iterative algorithms like EM benefit from parallelization
    /// when the number of iterations is large
    inline constexpr std::size_t MIN_ITERATIONS_FOR_PARALLEL = 10;
    
    /// Platform-optimized functions for runtime tuning
    /// These functions provide optimized values based on detected CPU features
    namespace adaptive {
        /// Get platform-optimized minimum elements for parallel processing
        inline std::size_t min_elements_for_parallel() {
            return platform::get_min_parallel_elements();
        }
        
        /// Get platform-optimized grain size
        inline std::size_t grain_size() {
            return platform::get_optimal_grain_size();
        }
        
        /// Get platform-optimized SIMD block size
        inline std::size_t simd_block_size() {
            return platform::get_optimal_simd_block_size();
        }
        
        /// Get platform-optimized memory alignment
        inline std::size_t memory_alignment() {
            return platform::get_optimal_alignment();
        }
    }
}

/// Compile-time validation of mathematical relationships
namespace validation {
    // Mathematical constant relationships
    static_assert(math::PI * 2.0 == math::TWO_PI, "TWO_PI should be twice PI");
    static_assert(math::PI / 2.0 == math::PI_OVER_2, "PI_OVER_2 should be half of PI");
    static_assert(math::E * math::E_INV == math::ONE, "E_INV should be the reciprocal of E");
    static_assert(math::HALF_LN_2PI * 2.0 == math::LN_2PI, "HALF_LN_2PI should be half of LN_2PI");
    static_assert(math::INV_SQRT_2PI * math::SQRT_2PI == math::ONE, "INV_SQRT_2PI should be reciprocal of SQRT_2PI");
    
    // Precision and tolerance relationships
    static_assert(precision::HIGH_PRECISION_TOLERANCE > precision::ULTRA_HIGH_PRECISION_TOLERANCE, 
                  "High precision tolerance should be greater than ultra-high precision");
    static_assert(precision::DEFAULT_TOLERANCE > precision::HIGH_PRECISION_TOLERANCE, 
                  "Default tolerance should be greater than high precision tolerance");
    
    // Probability bounds validation
    static_assert(probability::MIN_PROBABILITY > 0.0, "Minimum probability should be positive");
    static_assert(probability::MAX_PROBABILITY < 1.0, "Maximum probability should be less than 1.0");
    static_assert(probability::MIN_LOG_PROBABILITY < probability::MAX_LOG_PROBABILITY, 
                  "Min log probability should be less than max log probability");
    
    // SIMD parameter validation
    static_assert(simd::MIN_SIMD_SIZE <= simd::DEFAULT_BLOCK_SIZE, 
                  "Minimum SIMD size should not exceed default block size");
    static_assert(simd::DEFAULT_BLOCK_SIZE <= simd::MAX_BLOCK_SIZE, 
                  "Default block size should not exceed maximum block size");
    
    // Statistical critical values validation (ensure they're positive)
    static_assert(statistical::normal::Z_95 > 0.0, "Z_95 should be positive");
    static_assert(statistical::t_distribution::T_95_DF_5 > 0.0, "T_95_DF_5 should be positive");
    static_assert(statistical::chi_square::CHI2_95_DF_1 > 0.0, "CHI2_95_DF_1 should be positive");
    static_assert(statistical::f_distribution::F_95_DF_1_1 > 0.0, "F_95_DF_1_1 should be positive");
    
    // Effect size thresholds validation
    static_assert(statistical::thresholds::SMALL_EFFECT < statistical::thresholds::MEDIUM_EFFECT, 
                  "Small effect should be less than medium effect");
    static_assert(statistical::thresholds::MEDIUM_EFFECT < statistical::thresholds::LARGE_EFFECT, 
                  "Medium effect should be less than large effect");
}

/// Cache hierarchy optimization constants
namespace cache {
    /// Platform-specific cache tuning constants
    namespace platform {
        // Apple Silicon (M1/M2/M3) optimizations
        namespace apple_silicon {
            inline constexpr size_t DEFAULT_MAX_MEMORY_MB = 8;      // Conservative default
            inline constexpr size_t DEFAULT_MAX_ENTRIES = 2048;
            inline constexpr size_t PREFETCH_QUEUE_SIZE = 64;
            inline constexpr double EVICTION_THRESHOLD = 0.75;     // Lower threshold for unified memory
            inline constexpr size_t BATCH_EVICTION_SIZE = 16;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{12000}; // Longer TTL for fast access
            inline constexpr double HIT_RATE_TARGET = 0.88;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.75;
        }
        
        // Intel optimizations (Skylake and newer)
        namespace intel {
            inline constexpr size_t DEFAULT_MAX_MEMORY_MB = 6;
            inline constexpr size_t DEFAULT_MAX_ENTRIES = 1536;
            inline constexpr size_t PREFETCH_QUEUE_SIZE = 48;
            inline constexpr double EVICTION_THRESHOLD = 0.80;
            inline constexpr size_t BATCH_EVICTION_SIZE = 12;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{10000};
            inline constexpr double HIT_RATE_TARGET = 0.85;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.72;
        }
        
        // AMD optimizations (Zen and newer)
        namespace amd {
            inline constexpr size_t DEFAULT_MAX_MEMORY_MB = 4;
            inline constexpr size_t DEFAULT_MAX_ENTRIES = 1024;
            inline constexpr size_t PREFETCH_QUEUE_SIZE = 32;
            inline constexpr double EVICTION_THRESHOLD = 0.82;
            inline constexpr size_t BATCH_EVICTION_SIZE = 10;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{8000};
            inline constexpr double HIT_RATE_TARGET = 0.82;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.70;
        }
        
        // ARM (general, including Cortex-A series)
        namespace arm {
            inline constexpr size_t DEFAULT_MAX_MEMORY_MB = 2;
            inline constexpr size_t DEFAULT_MAX_ENTRIES = 512;
            inline constexpr size_t PREFETCH_QUEUE_SIZE = 16;
            inline constexpr double EVICTION_THRESHOLD = 0.85;
            inline constexpr size_t BATCH_EVICTION_SIZE = 8;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{6000};
            inline constexpr double HIT_RATE_TARGET = 0.80;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.68;
        }
    }
    
    /// Cache size calculation based on architecture features
    namespace sizing {
        // Multipliers for cache size calculation based on L3 cache
        inline constexpr double L3_CACHE_FRACTION = 0.125;      // Use 1/8 of L3 cache
        inline constexpr double L2_CACHE_FRACTION = 0.25;       // Use 1/4 of L2 cache if no L3
        inline constexpr size_t MIN_CACHE_SIZE_BYTES = 512 * 1024;   // 512KB minimum
        inline constexpr size_t MAX_CACHE_SIZE_BYTES = 32 * 1024 * 1024; // 32MB maximum
        
        // Entry count multipliers
        inline constexpr size_t BYTES_PER_ENTRY_ESTIMATE = 128;  // Average bytes per cache entry
        inline constexpr size_t MIN_ENTRY_COUNT = 32;
        inline constexpr size_t MAX_ENTRY_COUNT = 16384;
    }
    
    /// Cache behavior tuning based on CPU features
    namespace tuning {
        // TTL adjustments based on CPU frequency
        inline constexpr uint64_t HIGH_FREQ_THRESHOLD_HZ = 3000000000ULL; // 3 GHz
        inline constexpr uint64_t ULTRA_HIGH_FREQ_THRESHOLD_HZ = 4000000000ULL; // 4 GHz
        
        inline constexpr std::chrono::milliseconds BASE_TTL{8000};
        inline constexpr std::chrono::milliseconds HIGH_FREQ_TTL{12000};
        inline constexpr std::chrono::milliseconds ULTRA_HIGH_FREQ_TTL{15000};
        
        // Prefetch behavior based on SIMD capabilities
        inline constexpr size_t AVX512_PREFETCH_MULTIPLIER = 2;  // More aggressive prefetch for AVX-512
        inline constexpr size_t AVX2_PREFETCH_MULTIPLIER = 1;    // Standard prefetch for AVX2
        inline constexpr size_t SSE_PREFETCH_MULTIPLIER = 1;     // Conservative prefetch for SSE
        
        // Memory pressure sensitivity
        inline constexpr double HIGH_END_PRESSURE_SENSITIVITY = 0.6;  // Less sensitive on high-end CPUs
        inline constexpr double MID_RANGE_PRESSURE_SENSITIVITY = 0.8;
        inline constexpr double LOW_END_PRESSURE_SENSITIVITY = 0.9;   // Very sensitive on low-end CPUs
    }
    
    /// Access pattern optimization
    namespace patterns {
        inline constexpr size_t MAX_PATTERN_HISTORY = 256;       // Track last 256 accesses
        inline constexpr double SEQUENTIAL_PATTERN_THRESHOLD = 0.8;  // 80% sequential to trigger optimizations
        inline constexpr double RANDOM_PATTERN_THRESHOLD = 0.3;      // <30% sequential = random
        
        // Pattern-specific cache sizing
        inline constexpr double SEQUENTIAL_SIZE_MULTIPLIER = 1.5; // Larger cache for sequential patterns
        inline constexpr double RANDOM_SIZE_MULTIPLIER = 0.8;     // Smaller cache for random patterns
        inline constexpr double MIXED_SIZE_MULTIPLIER = 1.0;      // Default for mixed patterns
    }
    
    /// Performance monitoring thresholds
    namespace monitoring {
        inline constexpr double EXCELLENT_HIT_RATE = 0.95;
        inline constexpr double GOOD_HIT_RATE = 0.85;
        inline constexpr double ACCEPTABLE_HIT_RATE = 0.70;
        inline constexpr double POOR_HIT_RATE = 0.50;
        
        inline constexpr double EXCELLENT_MEMORY_EFFICIENCY = 0.80;
        inline constexpr double GOOD_MEMORY_EFFICIENCY = 0.65;
        inline constexpr double ACCEPTABLE_MEMORY_EFFICIENCY = 0.50;
        
        inline constexpr double FAST_ACCESS_TIME_US = 0.1;        // <0.1 μs is excellent
        inline constexpr double ACCEPTABLE_ACCESS_TIME_US = 1.0;   // <1 μs is acceptable
        inline constexpr double SLOW_ACCESS_TIME_US = 10.0;       // >10 μs needs optimization
        
        // Adaptive tuning triggers
        inline constexpr size_t MIN_SAMPLES_FOR_TUNING = 100;      // Minimum operations before tuning
        inline constexpr std::chrono::seconds TUNING_INTERVAL{30}; // How often to consider tuning
        inline constexpr double SIGNIFICANT_CHANGE_THRESHOLD = 0.05; // 5% change triggers re-evaluation
    }
}

/// Memory access and prefetching optimization constants
namespace memory {
    /// Platform-specific prefetching distance tuning
    namespace prefetch {
        /// Base prefetch distance constants (in cache lines)
        namespace distance {
            /// Conservative prefetch distance for older/low-power CPUs
            inline constexpr std::size_t CONSERVATIVE = 2;
            
            /// Standard prefetch distance for most modern CPUs
            inline constexpr std::size_t STANDARD = 4;
            
            /// Aggressive prefetch distance for high-end CPUs with large caches
            inline constexpr std::size_t AGGRESSIVE = 8;
            
            /// Ultra-aggressive prefetch for specialized workloads
            inline constexpr std::size_t ULTRA_AGGRESSIVE = 16;
        }
        
        /// Platform-specific prefetch distances (in elements, not cache lines)
        namespace platform {
            /// Apple Silicon prefetch tuning
            namespace apple_silicon {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 256;  // Elements ahead
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 64;       // Conservative for random access
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 128;      // Matrix operations
                inline constexpr std::size_t PREFETCH_STRIDE = 8;                 // Stride for strided access
            }
            
            /// Intel prefetch tuning (Skylake+)
            namespace intel {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 192;
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 48;
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 96;
                inline constexpr std::size_t PREFETCH_STRIDE = 4;
            }
            
            /// AMD prefetch tuning (Zen+)
            namespace amd {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 128;
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 32;
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 64;
                inline constexpr std::size_t PREFETCH_STRIDE = 4;
            }
            
            /// ARM prefetch tuning
            namespace arm {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 64;
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 16;
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 32;
                inline constexpr std::size_t PREFETCH_STRIDE = 2;
            }
        }
        
        /// Prefetch strategies based on access patterns
        namespace strategy {
            /// Sequential access prefetch multipliers
            inline constexpr double SEQUENTIAL_MULTIPLIER = 2.0;     // More aggressive for sequential
            inline constexpr double RANDOM_MULTIPLIER = 0.5;         // Conservative for random
            inline constexpr double STRIDED_MULTIPLIER = 1.5;        // Moderate for strided access
            
            /// Minimum elements before prefetching becomes beneficial
            inline constexpr std::size_t MIN_PREFETCH_SIZE = 32;
            
            /// Maximum practical prefetch distance (memory bandwidth constraint)
            inline constexpr std::size_t MAX_PREFETCH_DISTANCE = 1024;
            
            /// Prefetch granularity (align prefetch to cache line boundaries)
            inline constexpr std::size_t PREFETCH_GRANULARITY = 8;   // 64-byte cache line / 8-byte double
        }
        
        /// Software prefetch instruction timing
        namespace timing {
            /// Memory latency estimates for prefetch scheduling (in CPU cycles)
            inline constexpr std::size_t L1_LATENCY_CYCLES = 4;      // L1 cache hit
            inline constexpr std::size_t L2_LATENCY_CYCLES = 12;     // L2 cache hit
            inline constexpr std::size_t L3_LATENCY_CYCLES = 36;     // L3 cache hit
            inline constexpr std::size_t DRAM_LATENCY_CYCLES = 300;  // Main memory access
            
            /// Prefetch lead time (how far ahead to prefetch based on expected latency)
            inline constexpr std::size_t L2_PREFETCH_LEAD = 32;      // Elements ahead for L2 prefetch
            inline constexpr std::size_t L3_PREFETCH_LEAD = 128;     // Elements ahead for L3 prefetch
            inline constexpr std::size_t DRAM_PREFETCH_LEAD = 512;   // Elements ahead for DRAM prefetch
        }
    }
    
    /// Memory access pattern optimization
    namespace access {
        /// Cache line utilization constants
        inline constexpr std::size_t CACHE_LINE_SIZE_BYTES = 64;     // Standard cache line size
        inline constexpr std::size_t DOUBLES_PER_CACHE_LINE = 8;     // 64 bytes / 8 bytes per double
        inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;      // Alignment requirement
        
        /// Memory bandwidth optimization
        namespace bandwidth {
            /// Optimal burst sizes for different memory types
            inline constexpr std::size_t DDR4_BURST_SIZE = 64;      // Optimal DDR4 burst
            inline constexpr std::size_t DDR5_BURST_SIZE = 128;     // Optimal DDR5 burst
            inline constexpr std::size_t HBM_BURST_SIZE = 256;      // High Bandwidth Memory burst
            
            /// Memory channel utilization targets
            inline constexpr double TARGET_BANDWIDTH_UTILIZATION = 0.8;  // Aim for 80% bandwidth usage
            inline constexpr double MAX_BANDWIDTH_UTILIZATION = 0.95;    // Maximum before thrashing
        }
        
        /// Memory layout optimization
        namespace layout {
            /// Array-of-Structures vs Structure-of-Arrays thresholds
            inline constexpr std::size_t AOS_TO_SOA_THRESHOLD = 1000;    // Switch to SOA for larger sizes
            
            /// Memory pool and alignment settings
            inline constexpr std::size_t MEMORY_POOL_ALIGNMENT = 4096;   // Page-aligned pools
            inline constexpr std::size_t SMALL_ALLOCATION_THRESHOLD = 256; // Use pool for smaller allocations
            inline constexpr std::size_t LARGE_PAGE_THRESHOLD = 2097152; // 2MB huge page threshold
        }
        
        /// Non-Uniform Memory Access (NUMA) optimization
        namespace numa {
            /// NUMA-aware allocation thresholds
            inline constexpr std::size_t NUMA_AWARE_THRESHOLD = 1048576; // 1MB threshold for NUMA awareness
            
            /// Thread affinity and memory locality settings
            inline constexpr std::size_t NUMA_LOCAL_THRESHOLD = 65536;   // Prefer local memory below this size
            inline constexpr double NUMA_MIGRATION_COST = 0.1;           // Cost factor for NUMA migration
        }
    }
    
    /// Memory allocation strategy constants
    namespace allocation {
        /// Pool-based allocation sizes
        inline constexpr std::size_t SMALL_POOL_SIZE = 4096;        // 4KB pools
        inline constexpr std::size_t MEDIUM_POOL_SIZE = 65536;      // 64KB pools
        inline constexpr std::size_t LARGE_POOL_SIZE = 1048576;     // 1MB pools
        
        /// Allocation alignment requirements
        inline constexpr std::size_t MIN_ALLOCATION_ALIGNMENT = 8;   // Minimum 8-byte alignment
        inline constexpr std::size_t SIMD_ALLOCATION_ALIGNMENT = 32; // SIMD-friendly alignment
        inline constexpr std::size_t PAGE_ALLOCATION_ALIGNMENT = 4096; // Page alignment
        
        /// Memory growth strategies
        namespace growth {
            inline constexpr double EXPONENTIAL_GROWTH_FACTOR = 1.5; // 50% growth per expansion
            inline constexpr double LINEAR_GROWTH_FACTOR = 1.2;      // 20% growth for large allocations
            inline constexpr std::size_t GROWTH_THRESHOLD = 1048576; // Switch to linear above 1MB
        }
    }
}

} // namespace constants
} // namespace libstats

#endif // LIBSTATS_CONSTANTS_H_

#pragma once

#include <cstddef>
#include <limits>

/**
 * @file core/math_constants.h
 * @brief Mathematical and numerical precision constants for libstats
 *
 * This header covers two tightly related concerns:
 *
 *   1. Pure mathematical values — π, e, √2, and derived expressions.
 *      These are facts of mathematics, independent of any application domain.
 *
 *   2. Numerical precision and computation limits — machine epsilon,
 *      convergence tolerances, iteration counts, and differentiation steps.
 *      These govern *how precisely* we compute, which is inseparable from
 *      the mathematical operations themselves.
 *
 * Include this header whenever you need mathematical constants, tolerances,
 * or numerical method parameters.
 *
 * For statistical critical values, probability bounds, and algorithm
 * thresholds, see statistical_constants.h.
 * For benchmarking and performance testing parameters, see performance_constants.h.
 */

namespace stats {
namespace detail {

// =============================================================================
// FUNDAMENTAL MATHEMATICAL CONSTANTS
// =============================================================================

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

/// Square root of 2π  (used in Gaussian normalization)
inline constexpr double SQRT_2PI = 2.5066282746310005024157652848110453;

/// Natural logarithm of 2π
inline constexpr double LN_2PI = 1.8378770664093454835606594728112353;

/// Square root of 2
inline constexpr double SQRT_2 = 1.4142135623730950488016887242096981;

/// Square root of 3
inline constexpr double SQRT_3 = 1.7320508075688772935274463415058723;

/// Square root of 5
inline constexpr double SQRT_5 = 2.2360679774997896964091736687312762;

/// Half of ln(2π)
inline constexpr double HALF_LN_2PI = 0.9189385332046727417803297364056176;

/// Golden ratio φ = (1 + √5) / 2
inline constexpr double PHI = 1.6180339887498948482045868343656381;

/// Euler–Mascheroni constant γ
inline constexpr double EULER_MASCHERONI = 0.5772156649015328606065120900824024;

/// Catalan's constant G
inline constexpr double CATALAN = 0.9159655941772190150546035149323841;

/// Apéry's constant ζ(3)
inline constexpr double APERY = 1.2020569031595942853997381615114499;

/// Natural logarithm of φ
inline constexpr double LN_PHI = 0.4812118250596034474977589134243684;

/// Silver ratio (1 + √2)
inline constexpr double SILVER_RATIO = 2.4142135623730950488016887242096981;

/// Plastic number (real root of x³ − x − 1 = 0)
inline constexpr double PLASTIC_NUMBER = 1.3247179572447460259609088544780973;

/// Natural logarithm of π
inline constexpr double LN_PI = 1.1447298858494001741434273513530587;

// =============================================================================
// COMMON NUMERIC CONVENIENCES
// These avoid magic literals throughout the codebase and make intent legible.
// =============================================================================

inline constexpr double HALF = 0.5;
inline constexpr double NEG_HALF = -0.5;
inline constexpr double QUARTER = 0.25;
inline constexpr double THREE_QUARTERS = 0.75;
inline constexpr double NEG_ONE = -1.0;
inline constexpr double NEG_TWO = -2.0;

inline constexpr double ZERO_DOUBLE = 0.0;
inline constexpr double ONE = 1.0;
inline constexpr double TWO = 2.0;
inline constexpr double THREE = 3.0;
inline constexpr double FOUR = 4.0;
inline constexpr double FIVE = 5.0;
inline constexpr double SIX = 6.0;
inline constexpr double SEVEN = 7.0;
inline constexpr double EIGHT = 8.0;
inline constexpr double NINE = 9.0;
inline constexpr double TEN = 10.0;
inline constexpr double TWELVE = 12.0;
inline constexpr double THIRTEEN = 13.0;
inline constexpr double FIFTY = 50.0;
inline constexpr double HUNDRED = 100.0;
inline constexpr double THOUSAND = 1000.0;
inline constexpr double THOUSANDTH = 0.001;
inline constexpr double TENTH = 0.1;
inline constexpr double TWO_TWENTY_FIVE = 225.0;
inline constexpr double ONE_POINT_TWO_EIGHT = 1.28;
inline constexpr double ONE_POINT_EIGHT = 1.8;
inline constexpr double ONE_POINT_FIVE = 1.5;

inline constexpr int ZERO_INT = 0;
inline constexpr int ONE_INT = 1;
inline constexpr int TWO_INT = 2;
inline constexpr int THREE_INT = 3;
inline constexpr int FOUR_INT = 4;
inline constexpr int FIVE_INT = 5;
inline constexpr int SIX_INT = 6;
inline constexpr int TEN_INT = 10;

// =============================================================================
// DERIVED EXPRESSIONS
// =============================================================================

/// Precomputed reciprocals to avoid repeated division
inline constexpr double ONE_THIRD = 1.0 / 3.0;
inline constexpr double ONE_SIXTH = 1.0 / 6.0;
inline constexpr double ONE_TWELFTH = 1.0 / 12.0;

inline constexpr double E_INV = 1.0 / E;
inline constexpr double PHI_INV = 1.0 / PHI;
inline constexpr double PI_INV = 1.0 / PI;
inline constexpr double INV_LN2 = 1.0 / LN2;
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

// =============================================================================
// MACHINE PRECISION
// =============================================================================

inline constexpr double MACHINE_EPSILON = std::numeric_limits<double>::epsilon();
inline constexpr float MACHINE_EPSILON_FLOAT = std::numeric_limits<float>::epsilon();
inline constexpr long double MACHINE_EPSILON_LONG_DOUBLE =
    std::numeric_limits<long double>::epsilon();

// =============================================================================
// NUMERICAL TOLERANCES
// These govern when iterative algorithms and comparisons consider results
// converged or values equal.  Choose the tightest tolerance that remains
// numerically stable for the operation at hand.
// =============================================================================

inline constexpr double ZERO = 1.0e-30;
inline constexpr double DEFAULT_TOLERANCE = 1.0e-8;
inline constexpr double HIGH_PRECISION_TOLERANCE = 1.0e-12;
inline constexpr double ULTRA_HIGH_PRECISION_TOLERANCE = 1.0e-15;
inline constexpr double LOG_PROBABILITY_EPSILON_PRECISION = 1.0e-300;
inline constexpr double MIN_STD_DEV = 1.0e-6;
inline constexpr double HIGH_PRECISION_UPPER_BOUND = 1.0e12;
inline constexpr double MAX_STANDARD_DEVIATION = 1.0e10;

inline constexpr double STRICT_TOLERANCE = 1e-10;
inline constexpr double RELAXED_TOLERANCE = 1e-9;
inline constexpr double VERY_SMALL_PROBABILITY = 1e-12;

// =============================================================================
// NUMERICAL DIFFERENTIATION
// =============================================================================

inline constexpr double FORWARD_DIFF_STEP = 1.0e-8;
inline constexpr double CENTRAL_DIFF_STEP = 1.0e-6;
inline constexpr double NUMERICAL_DERIVATIVE_STEP = 1.0e-5;

// =============================================================================
// CONVERGENCE CRITERIA FOR NUMERICAL METHODS
// =============================================================================

inline constexpr double NEWTON_RAPHSON_TOLERANCE = 1.0e-10;
inline constexpr double BISECTION_TOLERANCE = 1.0e-12;
inline constexpr double GRADIENT_DESCENT_TOLERANCE = 1.0e-9;
inline constexpr double CONJUGATE_GRADIENT_TOLERANCE = 1.0e-10;

inline constexpr std::size_t MAX_NEWTON_ITERATIONS = 100;
inline constexpr std::size_t MAX_BISECTION_ITERATIONS = 1000;
inline constexpr std::size_t MAX_GRADIENT_DESCENT_ITERATIONS = 10000;
inline constexpr std::size_t MAX_CONJUGATE_GRADIENT_ITERATIONS = 1000;
inline constexpr std::size_t MAX_BETA_ITERATIONS = 100;
inline constexpr std::size_t MAX_GAMMA_SERIES_ITERATIONS = 1000;
inline constexpr std::size_t MAX_CONTINUED_FRACTION_ITERATIONS = 1000;

// =============================================================================
// NUMERICAL INTEGRATION
// =============================================================================

inline constexpr double INTEGRATION_TOLERANCE = 1.0e-10;
inline constexpr double ADAPTIVE_INTEGRATION_TOLERANCE = 1.0e-8;
inline constexpr double MONTE_CARLO_INTEGRATION_TOLERANCE = 1.0e-6;
inline constexpr int MAX_ADAPTIVE_SIMPSON_DEPTH = 15;
inline constexpr int DEFAULT_INTEGRATION_POINTS = 1000;

// =============================================================================
// LOG-SPACE COMPUTATION LIMITS
// Thresholds for safely switching between linear and log-space arithmetic.
// =============================================================================

/// exp() overflows above this value; clamp inputs before calling std::exp
inline constexpr double LOG_EXP_OVERFLOW_THRESHOLD = 700.0;

/// log1p(exp(x)) switches algorithm above this threshold
inline constexpr double LOG1PEXP_LARGE_THRESHOLD = 37.0;

/// log1p(exp(x)) switches algorithm below this threshold
inline constexpr double LOG1PEXP_SMALL_THRESHOLD = -37.0;

/// log-sum-exp: terms smaller than this are negligible
inline constexpr double LOG_SUM_EXP_THRESHOLD = -50.0;

/// Switch from linear to log-space when values drop below this
inline constexpr double LOG_SPACE_THRESHOLD = 1.0e-50;

/// Lookup table size for log-space operations
inline constexpr std::size_t LOG_SPACE_LOOKUP_TABLE_SIZE = 1024;

/// Continued fraction: treat as "large" above this value
inline constexpr double LARGE_CONTINUED_FRACTION_VALUE = 1e30;

/// Values below this are treated as zero in numerical operations
inline constexpr double ULTRA_SMALL_THRESHOLD = 1e-30;

/// High condition number — warns of potential numerical instability
inline constexpr double HIGH_CONDITION_NUMBER_THRESHOLD = 1.0e12;

// =============================================================================
// ALGORITHM SCALE BOUNDS
// =============================================================================

inline constexpr double MIN_SCALE_FACTOR = 1.0e-100;
inline constexpr double MAX_SCALE_FACTOR = 1.0e100;
inline constexpr double MAX_DISTRIBUTION_PARAMETER = 1.0e6;
inline constexpr double MIN_DISTRIBUTION_PARAMETER = 1.0e-6;

}  // namespace detail
}  // namespace stats

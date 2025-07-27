#ifndef LIBSTATS_CORE_MATHEMATICAL_CONSTANTS_H_
#define LIBSTATS_CORE_MATHEMATICAL_CONSTANTS_H_

/**
 * @file core/mathematical_constants.h
 * @brief Fundamental mathematical constants for libstats
 * 
 * This header contains pure mathematical constants such as π, e, and other
 * well-known mathematical values used throughout the library.
 */

namespace libstats {
namespace constants {

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
    
    /// Square root of 3
    inline constexpr double SQRT_3 = 1.7320508075688772935274463415058723;
    
    /// Square root of 5
    inline constexpr double SQRT_5 = 2.2360679774997896964091736687312762;
    
    /// Half of ln(2π)
    inline constexpr double HALF_LN_2PI = 0.9189385332046727417803297364056176;
    
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
    inline constexpr double SIX = 6.0;
    inline constexpr double TEN = 10.0;
    inline constexpr double THIRTEEN = 13.0;
    inline constexpr double HUNDRED = 100.0;
    inline constexpr double THOUSAND = 1000.0;
    inline constexpr double THOUSANDTH = 0.001;
    inline constexpr double TENTH = 0.1;
    inline constexpr double TWO_TWENTY_FIVE = 225.0;
    inline constexpr double ONE_POINT_TWO_EIGHT = 1.28;
    inline constexpr double ONE_POINT_EIGHT = 1.8;
    inline constexpr double ONE_POINT_FIVE = 1.5;
    
    /// Commonly used integer constants
    inline constexpr int ZERO_INT = 0;
    inline constexpr int ONE_INT = 1;
    inline constexpr int TWO_INT = 2;
    inline constexpr int THREE_INT = 3;
    inline constexpr int FOUR_INT = 4;
    inline constexpr int FIVE_INT = 5;
    inline constexpr int TEN_INT = 10;
    
    /// Precomputed reciprocals to avoid division operations
    inline constexpr double ONE_THIRD = 1.0/3.0;
    inline constexpr double ONE_SIXTH = 1.0/6.0;
    inline constexpr double ONE_TWELFTH = 1.0/12.0;
    
    /// Reciprocal constants
    inline constexpr double E_INV = 1.0 / E;
    inline constexpr double PHI_INV = 1.0 / PHI;
    inline constexpr double PI_INV = 1.0 / PI;
    inline constexpr double INV_SQRT_2PI = 1.0 / SQRT_2PI;
    inline constexpr double INV_SQRT_2 = 1.0 / SQRT_2;
    inline constexpr double INV_SQRT_3 = 1.0 / SQRT_3;
    
    /// Derived mathematical expressions
    inline constexpr double TWO_PI = 2.0 * PI;
    inline constexpr double PI_OVER_2 = PI / 2.0;
    inline constexpr double PI_OVER_3 = PI / 3.0;
    inline constexpr double PI_OVER_4 = PI / 4.0;
    inline constexpr double PI_OVER_6 = PI / 6.0;
    inline constexpr double THREE_PI_OVER_2 = 3.0 * PI / 2.0;
    inline constexpr double FOUR_PI = 4.0 * PI;
    inline constexpr double NEG_HALF_LN_2PI = -0.5 * LN_2PI;
    
}

} // namespace constants
} // namespace libstats

#endif // LIBSTATS_CORE_MATHEMATICAL_CONSTANTS_H_

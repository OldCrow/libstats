#pragma once

/**
 * @file core/statistical_constants.h
 * @brief Statistical constants and critical values for libstats
 *
 * This header contains statistical constants and precomputed critical values
 * for common statistical distributions used throughout the library.
 */

namespace libstats {
namespace constants {

/// Statistical critical values and commonly used constants
namespace statistical {
/// Winitzki's approximation parameter for inverse error function
inline constexpr double WINITZKI_A = 0.147;

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
}  // namespace normal

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
}  // namespace t_distribution

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
}  // namespace chi_square

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
}  // namespace f_distribution
}  // namespace statistical

}  // namespace constants
}  // namespace libstats

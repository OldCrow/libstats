#pragma once

/**
 * @file core/goodness_of_fit_constants.h
 * @brief Critical values for goodness-of-fit tests
 * 
 * This header contains precomputed critical values for various
 * goodness-of-fit tests used in distribution fitting and validation.
 */

namespace libstats {
namespace constants {

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

} // namespace constants
} // namespace libstats

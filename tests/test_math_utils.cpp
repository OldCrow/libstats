#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

// Use focused header for math utilities testing
#include "../include/core/math_utils.h"

// Standard library includes
#include <cmath>          // for std::abs, std::isnan, std::isinf, std::log, std::sqrt, M_PI
#include <cstddef>        // for std::size_t
#include <gtest/gtest.h>  // for testing framework

// Use stats::detail namespace but avoid conflicting standard library functions
using namespace stats::detail;

class MathUtilsTest : public ::testing::Test {
   protected:
    static constexpr double TOLERANCE = 1e-10;
    static constexpr double LOOSE_TOLERANCE = 1e-8;
    static constexpr double VERY_LOOSE_TOLERANCE = 1e-6;

    // Helper to check near equality
    static bool near_equal(double a, double b, double tolerance = TOLERANCE) {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (std::isinf(a) && std::isinf(b))
            return a * b > 0;  // Same sign infinity
        return std::abs(a - b) <= tolerance;
    }
};

// =============================================================================
// ERROR FUNCTION TESTS
// =============================================================================

TEST_F(MathUtilsTest, ErfBasicValues) {
    // Test against known values
    EXPECT_TRUE(near_equal(stats::detail::erf(0.0), 0.0));
    EXPECT_TRUE(near_equal(stats::detail::erf(1.0), 0.8427007929, LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(stats::detail::erf(-1.0), -0.8427007929, LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(stats::detail::erf(2.0), 0.9953222650, LOOSE_TOLERANCE));

    // Test symmetry
    EXPECT_TRUE(near_equal(stats::detail::erf(-0.5), -stats::detail::erf(0.5)));

    // Test limits
    EXPECT_TRUE(near_equal(stats::detail::erf(10.0), 1.0, LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(stats::detail::erf(-10.0), -1.0, LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, ErfcBasicValues) {
    // Test against known values
    EXPECT_TRUE(near_equal(stats::detail::erfc(0.0), 1.0));
    EXPECT_TRUE(near_equal(stats::detail::erfc(1.0), 0.1572992071, LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(stats::detail::erfc(-1.0), 1.8427007929, LOOSE_TOLERANCE));

    // Test relationship: erf(x) + erfc(x) = 1
    for (double x = -3.0; x <= 3.0; x += 0.5) {
        EXPECT_TRUE(
            near_equal(stats::detail::erf(x) + stats::detail::erfc(x), 1.0, LOOSE_TOLERANCE));
    }
}

TEST_F(MathUtilsTest, ErfInvBasicValues) {
    // Test against known values (using Winitzki's approximation)
    EXPECT_TRUE(near_equal(erf_inv(0.0), 0.0));

    // Test inverse property: erf_inv(erf(x)) ≈ x for small x
    for (double x = -1.5; x <= 1.5; x += 0.3) {
        double erf_x = stats::detail::erf(x);
        if (std::abs(erf_x) < 0.99) {  // Within valid range
            EXPECT_TRUE(near_equal(erf_inv(erf_x), x, VERY_LOOSE_TOLERANCE));
        }
    }

    // Test edge cases
    EXPECT_TRUE(std::isnan(erf_inv(-1.1)));
    EXPECT_TRUE(std::isnan(erf_inv(1.1)));
    EXPECT_TRUE(std::isinf(erf_inv(1.0)));
    EXPECT_TRUE(std::isinf(erf_inv(-1.0)));
}

// =============================================================================
// GAMMA FUNCTION TESTS
// =============================================================================

TEST_F(MathUtilsTest, LgammaBasicValues) {
    // Test against known values
    EXPECT_TRUE(near_equal(stats::detail::lgamma(1.0), 0.0));            // Γ(1) = 1, ln(1) = 0
    EXPECT_TRUE(near_equal(stats::detail::lgamma(2.0), 0.0));            // Γ(2) = 1, ln(1) = 0
    EXPECT_TRUE(near_equal(stats::detail::lgamma(3.0), std::log(2.0)));  // Γ(3) = 2, ln(2)
    EXPECT_TRUE(near_equal(stats::detail::lgamma(4.0), std::log(6.0)));  // Γ(4) = 6, ln(6)

    // Test half-integer values
    EXPECT_TRUE(near_equal(stats::detail::lgamma(0.5), std::log(std::sqrt(M_PI)), LOOSE_TOLERANCE));
    EXPECT_TRUE(
        near_equal(stats::detail::lgamma(1.5), std::log(std::sqrt(M_PI) / 2.0), LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, GammaPKnownValues) {
    // Test well-known values
    EXPECT_TRUE(near_equal(gamma_p(1.0, 1.0), 1.0 - 1.0 / std::exp(1.0), LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(gamma_p(2.0, 2.0), 1.0 - 3.0 / std::exp(2.0), LOOSE_TOLERANCE));

    // Test boundary conditions
    EXPECT_TRUE(near_equal(gamma_p(1.0, 0.0), 0.0));
    EXPECT_TRUE(near_equal(gamma_p(2.0, 0.0), 0.0));

    // Test with our previously debugged case
    EXPECT_TRUE(near_equal(gamma_p(5.0, 1.6235), 0.025, VERY_LOOSE_TOLERANCE));

    // Test complementary relationship: P(a,x) + Q(a,x) = 1
    for (double a = 0.5; a <= 5.0; a += 0.5) {
        for (double x = 0.1; x <= 8.0; x += 1.0) {
            EXPECT_TRUE(near_equal(gamma_p(a, x) + gamma_q(a, x), 1.0, LOOSE_TOLERANCE));
        }
    }
}

TEST_F(MathUtilsTest, GammaQKnownValues) {
    // Test well-known values
    EXPECT_TRUE(near_equal(gamma_q(1.0, 1.0), 1.0 / std::exp(1.0), LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(gamma_q(2.0, 2.0), 3.0 / std::exp(2.0), LOOSE_TOLERANCE));

    // Test boundary conditions
    EXPECT_TRUE(near_equal(gamma_q(1.0, 0.0), 1.0));
    EXPECT_TRUE(near_equal(gamma_q(2.0, 0.0), 1.0));

    // Test with our debugged case
    EXPECT_TRUE(near_equal(gamma_q(5.0, 1.6235), 0.975, VERY_LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, GammaEdgeCases) {
    // Test invalid inputs
    EXPECT_TRUE(near_equal(gamma_p(-1.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(gamma_p(1.0, -1.0), 0.0));
    EXPECT_TRUE(near_equal(gamma_q(-1.0, 1.0), 1.0));
    EXPECT_TRUE(near_equal(gamma_q(1.0, -1.0), 1.0));
}

// =============================================================================
// BETA FUNCTION TESTS
// =============================================================================

TEST_F(MathUtilsTest, LbetaBasicValues) {
    // Test against known values: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    EXPECT_TRUE(near_equal(lbeta(1.0, 1.0), 0.0));            // B(1,1) = 1, ln(1) = 0
    EXPECT_TRUE(near_equal(lbeta(2.0, 1.0), std::log(0.5)));  // B(2,1) = 1/2
    EXPECT_TRUE(near_equal(lbeta(1.0, 2.0), std::log(0.5)));  // B(1,2) = 1/2

    // Test symmetry: B(a,b) = B(b,a)
    EXPECT_TRUE(near_equal(lbeta(2.5, 3.7), lbeta(3.7, 2.5)));
}

TEST_F(MathUtilsTest, BetaIKnownValues) {
    // Test boundary conditions
    EXPECT_TRUE(near_equal(beta_i(0.0, 1.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(beta_i(1.0, 1.0, 1.0), 1.0));
    EXPECT_TRUE(near_equal(beta_i(0.5, 1.0, 1.0), 0.5));  // Uniform distribution

    // Test known values for specific cases
    EXPECT_TRUE(near_equal(beta_i(0.25, 2.0, 2.0), 0.15625, VERY_LOOSE_TOLERANCE));

    // Test invalid inputs
    EXPECT_TRUE(near_equal(beta_i(-0.1, 1.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(beta_i(1.1, 1.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(beta_i(0.5, 0.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(beta_i(0.5, 1.0, 0.0), 0.0));
}

// =============================================================================
// DISTRIBUTION FUNCTION TESTS
// =============================================================================

TEST_F(MathUtilsTest, NormalCdfBasicValues) {
    // Test known values
    EXPECT_TRUE(near_equal(normal_cdf(0.0), 0.5));
    EXPECT_TRUE(near_equal(normal_cdf(1.0), 0.8413447460, LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(normal_cdf(-1.0), 0.1586552540, LOOSE_TOLERANCE));

    // Test symmetry around 0
    EXPECT_TRUE(near_equal(normal_cdf(-1.5), 1.0 - normal_cdf(1.5), LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, InverseNormalCdfBasicValues) {
    // Test known values
    EXPECT_TRUE(near_equal(inverse_normal_cdf(0.5), 0.0, VERY_LOOSE_TOLERANCE));

    // Test inverse property: inverse_normal_cdf(normal_cdf(x)) ≈ x
    for (double x = -3.0; x <= 3.0; x += 0.5) {
        double cdf_x = normal_cdf(x);
        EXPECT_TRUE(near_equal(inverse_normal_cdf(cdf_x), x, VERY_LOOSE_TOLERANCE));
    }

    // Test edge cases
    EXPECT_TRUE(std::isinf(inverse_normal_cdf(0.0)));
    EXPECT_TRUE(std::isinf(inverse_normal_cdf(1.0)));
    EXPECT_TRUE(std::isnan(inverse_normal_cdf(-0.1)));
    EXPECT_TRUE(std::isnan(inverse_normal_cdf(1.1)));
}

TEST_F(MathUtilsTest, ChiSquaredCdfBasicValues) {
    // Test boundary conditions
    EXPECT_TRUE(near_equal(chi_squared_cdf(0.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(chi_squared_cdf(0.0, 5.0), 0.0));

    // Test known values (SciPy-verified)
    // SciPy: stats.chi2.cdf(3.247, 10) = 0.025000776910264
    EXPECT_TRUE(near_equal(chi_squared_cdf(3.247, 10.0), 0.025000776910264, VERY_LOOSE_TOLERANCE));
    // SciPy: stats.chi2.cdf(20.483, 10) = 0.974998550535649
    EXPECT_TRUE(near_equal(chi_squared_cdf(20.483, 10.0), 0.974998550535649, VERY_LOOSE_TOLERANCE));

    // Test invalid inputs
    EXPECT_TRUE(near_equal(chi_squared_cdf(-1.0, 1.0), 0.0));
    EXPECT_TRUE(std::isnan(chi_squared_cdf(1.0, 0.0)));
}

TEST_F(MathUtilsTest, InverseChiSquaredCdfBasicValues) {
    // Test boundary conditions
    EXPECT_TRUE(near_equal(inverse_chi_squared_cdf(0.0, 1.0), 0.0));
    EXPECT_TRUE(std::isinf(inverse_chi_squared_cdf(1.0, 1.0)));

    // Test inverse property for our debugged case
    double chi_val = inverse_chi_squared_cdf(0.025, 10.0);
    EXPECT_TRUE(near_equal(chi_squared_cdf(chi_val, 10.0), 0.025, VERY_LOOSE_TOLERANCE));

    chi_val = inverse_chi_squared_cdf(0.975, 10.0);
    EXPECT_TRUE(near_equal(chi_squared_cdf(chi_val, 10.0), 0.975, VERY_LOOSE_TOLERANCE));

    // Test invalid inputs
    EXPECT_TRUE(std::isnan(inverse_chi_squared_cdf(-0.1, 1.0)));
    EXPECT_TRUE(std::isnan(inverse_chi_squared_cdf(1.1, 1.0)));
    EXPECT_TRUE(std::isnan(inverse_chi_squared_cdf(0.5, 0.0)));
}

TEST_F(MathUtilsTest, TCdfBasicValues) {
    // Test symmetry around 0
    EXPECT_TRUE(near_equal(t_cdf(0.0, 1.0), 0.5));
    EXPECT_TRUE(near_equal(t_cdf(0.0, 10.0), 0.5));

    // Test that as df → ∞, t-distribution approaches normal
    EXPECT_TRUE(near_equal(t_cdf(1.0, 1000.0), normal_cdf(1.0), LOOSE_TOLERANCE));

    // Test invalid inputs
    EXPECT_TRUE(std::isnan(t_cdf(1.0, 0.0)));
}

TEST_F(MathUtilsTest, FCdfBasicValues) {
    // Test boundary conditions
    EXPECT_TRUE(near_equal(f_cdf(0.0, 1.0, 1.0), 0.0));

    // Test that F(1, df1, df2) = 0.5 when df1 = df2 (by symmetry)
    EXPECT_TRUE(near_equal(f_cdf(1.0, 10.0, 10.0), 0.5, VERY_LOOSE_TOLERANCE));

    // Test invalid inputs
    EXPECT_TRUE(near_equal(f_cdf(-1.0, 1.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(f_cdf(1.0, 0.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(f_cdf(1.0, 1.0, 0.0), 0.0));
}

// =============================================================================
// STATISTICAL UTILITY TESTS
// =============================================================================

TEST_F(MathUtilsTest, SampleMomentsBasicValues) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto moments = sample_moments(data);

    EXPECT_TRUE(near_equal(moments[0], 3.0));                   // mean
    EXPECT_TRUE(near_equal(moments[1], 2.5));                   // variance (sample)
    EXPECT_TRUE(near_equal(moments[2], 0.0, LOOSE_TOLERANCE));  // skewness (symmetric)

    // Test with constant data
    std::vector<double> constant_data(10, 5.0);
    auto const_moments = sample_moments(constant_data);
    EXPECT_TRUE(near_equal(const_moments[0], 5.0));  // mean
    EXPECT_TRUE(near_equal(const_moments[1], 0.0));  // variance
}

TEST_F(MathUtilsTest, CalculateQuantiles) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> quantiles = {0.0, 0.25, 0.5, 0.75, 1.0};

    auto result = calculate_quantiles(data, quantiles);

    EXPECT_EQ(result.size(), quantiles.size());
    EXPECT_TRUE(near_equal(result[0], 1.0));  // min
    EXPECT_TRUE(near_equal(result[2], 3.0));  // median
    EXPECT_TRUE(near_equal(result[4], 5.0));  // max
}

TEST_F(MathUtilsTest, ValidateFittingData) {
    std::vector<double> valid_data = {1.0, 2.0, 3.0};
    EXPECT_TRUE(validate_fitting_data(valid_data));

    std::vector<double> invalid_data = {1.0, std::numeric_limits<double>::infinity(), 3.0};
    EXPECT_FALSE(validate_fitting_data(invalid_data));

    std::vector<double> nan_data = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_FALSE(validate_fitting_data(nan_data));
}

// =============================================================================
// EMPIRICAL CDF TESTS
// =============================================================================

TEST_F(MathUtilsTest, EmpiricalCdf) {
    std::vector<double> data = {3.0, 1.0, 4.0, 2.0, 5.0};
    auto cdf = empirical_cdf(data);

    EXPECT_EQ(cdf.size(), data.size());

    // Should be sorted and increasing
    for (size_t i = 1; i < cdf.size(); ++i) {
        EXPECT_GE(cdf[i], cdf[i - 1]);
    }

    // Last value should be 1.0
    EXPECT_TRUE(near_equal(cdf.back(), 1.0));
}

// =============================================================================
// ALGORITHM INTERNAL PATH TESTS
// =============================================================================

TEST_F(MathUtilsTest, GammaSeriesVsContinuedFraction) {
    // Test that both code paths in gamma_p/gamma_q give consistent results
    double a = 3.0;

    // For x < a+1, should use series
    double x_small = 2.0;
    double p_small = gamma_p(a, x_small);
    double q_small = gamma_q(a, x_small);
    EXPECT_TRUE(near_equal(p_small + q_small, 1.0, LOOSE_TOLERANCE));

    // For x > a+1, should use continued fraction
    double x_large = 6.0;
    double p_large = gamma_p(a, x_large);
    double q_large = gamma_q(a, x_large);
    EXPECT_TRUE(near_equal(p_large + q_large, 1.0, LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, BetaContinuedFractionPaths) {
    // Test both paths in beta_i continued fraction
    double a = 2.0, b = 3.0;

    // Test x < (a+1)/(a+b+2) path
    double x_small = 0.2;
    double result_small = beta_i(x_small, a, b);
    EXPECT_GT(result_small, 0.0);
    EXPECT_LT(result_small, 1.0);

    // Test x >= (a+1)/(a+b+2) path
    double x_large = 0.8;
    double result_large = beta_i(x_large, a, b);
    EXPECT_GT(result_large, 0.0);
    EXPECT_LT(result_large, 1.0);

    // Test monotonicity: beta_i should be increasing in x
    EXPECT_LT(result_small, result_large);
}

TEST_F(MathUtilsTest, InverseChiSquaredBisectionPath) {
    // Test that bisection path is used for extreme probabilities
    double df = 5.0;

    // Very small p (should trigger bisection)
    double small_p = 0.001;
    double chi_small = inverse_chi_squared_cdf(small_p, df);
    EXPECT_GT(chi_small, 0.0);
    EXPECT_TRUE(near_equal(chi_squared_cdf(chi_small, df), small_p, VERY_LOOSE_TOLERANCE));

    // Very large p (should trigger bisection)
    double large_p = 0.999;
    double chi_large = inverse_chi_squared_cdf(large_p, df);
    EXPECT_GT(chi_large, 0.0);
    EXPECT_TRUE(near_equal(chi_squared_cdf(chi_large, df), large_p, VERY_LOOSE_TOLERANCE));
}

// Test main function
TEST_F(MathUtilsTest, ComprehensiveFunctionVerification) {
    // This test ensures all major functions produce reasonable results
    // and don't crash on typical inputs

    // Test a variety of function combinations that might be used together
    double x = 2.5;
    double a = 3.0, b = 4.0;
    double df = 10.0;

    // Chain of calculations that should all work
    double norm_cdf = normal_cdf(x);
    double inv_norm = inverse_normal_cdf(norm_cdf);
    EXPECT_TRUE(near_equal(inv_norm, x, VERY_LOOSE_TOLERANCE));

    double gamma_p_val = gamma_p(a, x);
    double gamma_q_val = gamma_q(a, x);
    EXPECT_TRUE(near_equal(gamma_p_val + gamma_q_val, 1.0, LOOSE_TOLERANCE));

    double beta_val = beta_i(0.3, a, b);
    EXPECT_GT(beta_val, 0.0);
    EXPECT_LT(beta_val, 1.0);

    double chi_cdf = chi_squared_cdf(x, df);
    double chi_inv = inverse_chi_squared_cdf(chi_cdf, df);
    EXPECT_TRUE(near_equal(chi_inv, x, VERY_LOOSE_TOLERANCE));
}

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

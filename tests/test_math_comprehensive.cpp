// Comprehensive Mathematical Utilities Test Suite
// Consolidates test_math_utils and test_vectorized_math
// Tests all mathematical functions and their vectorized variants

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

#include "../include/core/math_utils.h"

// Standard library includes
#include <algorithm>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Test fixture for math utilities tests
class MathUtilsTest : public ::testing::Test {
   protected:
    static constexpr double TOLERANCE = 1e-10;
    static constexpr double LOOSE_TOLERANCE = 1e-8;
    static constexpr double VERY_LOOSE_TOLERANCE = 1e-6;

    static bool near_equal(double a, double b, double tolerance = TOLERANCE) {
        if (std::isnan(a) && std::isnan(b))
            return true;
        if (std::isinf(a) && std::isinf(b))
            return a * b > 0;
        return std::abs(a - b) <= tolerance;
    }

    // Generate test data
    std::vector<double> generate_test_data(size_t size, double min_val = -2.0, double max_val = 2.0,
                                           unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dist(min_val, max_val);

        std::vector<double> data(size);
        for (auto& val : data) {
            val = dist(gen);
        }
        return data;
    }
};

//==============================================================================
// SPECIAL FUNCTION TESTS
//==============================================================================

TEST_F(MathUtilsTest, ErfFunctions) {
    // Test error function implementations
    EXPECT_TRUE(near_equal(stats::detail::erf(0.0), 0.0));
    EXPECT_TRUE(near_equal(stats::detail::erf(1.0), 0.8427007929, LOOSE_TOLERANCE));
    EXPECT_TRUE(near_equal(stats::detail::erf(-1.0), -stats::detail::erf(1.0)));

    EXPECT_TRUE(near_equal(stats::detail::erfc(0.0), 1.0));
    EXPECT_TRUE(near_equal(stats::detail::erfc(1.0), 1.0 - stats::detail::erf(1.0)));

    // Test erf_inv
    double x = 0.5;
    double y = stats::detail::erf(x);
    double x_back = stats::detail::erf_inv(y);
    EXPECT_TRUE(near_equal(x, x_back, VERY_LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, GammaFunctions) {
    // Test gamma function implementations
    EXPECT_TRUE(near_equal(stats::detail::lgamma(1.0), 0.0));
    EXPECT_TRUE(near_equal(stats::detail::lgamma(2.0), 0.0));
    EXPECT_TRUE(near_equal(stats::detail::lgamma(3.0), std::log(2.0)));

    // Test gamma_p and gamma_q relationship
    double a = 2.0, x_g = 1.5;
    double p = stats::detail::gamma_p(a, x_g);
    double q = stats::detail::gamma_q(a, x_g);
    EXPECT_TRUE(near_equal(p + q, 1.0, LOOSE_TOLERANCE));

    // Test known values
    EXPECT_TRUE(
        near_equal(stats::detail::gamma_p(1.0, 1.0), 1.0 - 1.0 / std::exp(1.0), LOOSE_TOLERANCE));
    EXPECT_TRUE(
        near_equal(stats::detail::gamma_p(2.0, 2.0), 1.0 - 3.0 / std::exp(2.0), LOOSE_TOLERANCE));
}

TEST_F(MathUtilsTest, BetaFunctions) {
    // Test beta function implementations
    EXPECT_TRUE(near_equal(stats::detail::lbeta(1.0, 1.0), 0.0));
    EXPECT_TRUE(near_equal(stats::detail::lbeta(2.0, 1.0), std::log(0.5)));
    EXPECT_TRUE(near_equal(stats::detail::lbeta(1.0, 2.0), std::log(0.5)));

    // Test beta function symmetry
    double a_b = 2.5, b_b = 3.7;
    EXPECT_TRUE(near_equal(stats::detail::lbeta(a_b, b_b), stats::detail::lbeta(b_b, a_b)));

    // Test beta_i
    EXPECT_TRUE(near_equal(stats::detail::beta_i(0.5, 1.0, 1.0), 0.5));
}

TEST_F(MathUtilsTest, NormalDistributionFunctions) {
    // Test normal distribution functions
    EXPECT_TRUE(near_equal(stats::detail::normal_cdf(0.0), 0.5));
    EXPECT_TRUE(near_equal(stats::detail::normal_cdf(1.0), 0.8413447460, LOOSE_TOLERANCE));

    // Test inverse relationship
    double p_n = 0.7;
    double x_n = stats::detail::inverse_normal_cdf(p_n);
    double p_back_val = stats::detail::normal_cdf(x_n);
    EXPECT_TRUE(near_equal(p_n, p_back_val, VERY_LOOSE_TOLERANCE));
}

//==============================================================================
// VECTORIZED IMPLEMENTATION TESTS
//==============================================================================

TEST_F(MathUtilsTest, VectorizedErf) {
    const size_t size = 1000;
    std::vector<double> input = generate_test_data(size);
    std::vector<double> output(size);
    std::vector<double> expected(size);

    // Test vector_erf
    stats::detail::vector_erf(input, output);

    bool accurate = true;
    double max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
        expected[i] = stats::detail::erf(input[i]);
        double error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (error > LOOSE_TOLERANCE) {
            accurate = false;
        }
    }

    EXPECT_TRUE(accurate) << "Max error: " << max_error;

    // Test vector_erfc
    stats::detail::vector_erfc(input, output);

    accurate = true;
    max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
        expected[i] = stats::detail::erfc(input[i]);
        double error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (error > LOOSE_TOLERANCE) {
            accurate = false;
        }
    }

    EXPECT_TRUE(accurate) << "Max error: " << max_error;
}

TEST_F(MathUtilsTest, VectorizedGamma) {
    const size_t size = 1000;
    std::vector<double> pos_input = generate_test_data(size, 0.1, 5.0);
    std::vector<double> output(size);
    std::vector<double> expected(size);

    // Test vector_gamma_p
    double a = 2.5;
    stats::detail::vector_gamma_p(a, pos_input, output);

    bool accurate = true;
    double max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
        expected[i] = stats::detail::gamma_p(a, pos_input[i]);
        double error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (error > LOOSE_TOLERANCE) {
            accurate = false;
        }
    }

    EXPECT_TRUE(accurate) << "Max error: " << max_error;

    // Test vector_lgamma
    stats::detail::vector_lgamma(pos_input, output);

    accurate = true;
    max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
        expected[i] = stats::detail::lgamma(pos_input[i]);
        double error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (error > LOOSE_TOLERANCE) {
            accurate = false;
        }
    }

    EXPECT_TRUE(accurate) << "Max error: " << max_error;
}

TEST_F(MathUtilsTest, VectorizedBeta) {
    const size_t size = 1000;
    std::vector<double> a_values = generate_test_data(size, 0.1, 5.0, 123);
    std::vector<double> b_values = generate_test_data(size, 0.1, 5.0, 456);
    std::vector<double> output(size);
    std::vector<double> expected(size);

    stats::detail::vector_lbeta(a_values, b_values, output);

    bool accurate = true;
    double max_error = 0.0;
    for (size_t i = 0; i < size; ++i) {
        expected[i] = stats::detail::lbeta(a_values[i], b_values[i]);
        double error = std::abs(output[i] - expected[i]);
        max_error = std::max(max_error, error);
        if (error > LOOSE_TOLERANCE) {
            accurate = false;
        }
    }

    EXPECT_TRUE(accurate) << "Max error: " << max_error;
}

TEST_F(MathUtilsTest, VectorizedThresholds) {
    size_t threshold = stats::detail::vectorized_math_threshold();
    EXPECT_GT(threshold, 0u);
    EXPECT_LT(threshold, 10000u);

    // Test decision function
    EXPECT_FALSE(stats::detail::should_use_vectorized_math(threshold / 2));
    EXPECT_TRUE(stats::detail::should_use_vectorized_math(threshold * 2));
}

TEST_F(MathUtilsTest, VectorizedEdgeCases) {
    // Empty vectors
    std::vector<double> empty_in, empty_out;
    stats::detail::vector_erf(empty_in, empty_out);
    EXPECT_TRUE(empty_out.empty());

    // Special values
    std::vector<double> special_in = {
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::max(),
        std::numeric_limits<double>::min(),       std::numeric_limits<double>::denorm_min()};
    std::vector<double> special_out(special_in.size());

    // Should handle without crashing
    stats::detail::vector_erf(special_in, special_out);
    stats::detail::vector_erfc(special_in, special_out);
}

//==============================================================================
// PERFORMANCE BENCHMARKS
//==============================================================================

TEST_F(MathUtilsTest, ErfPerformance) {
    const size_t size = 10000;
    std::vector<double> input = generate_test_data(size);
    std::vector<double> output(size);

    // Scalar implementation
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < size; ++i) {
        output[i] = stats::detail::erf(input[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Vectorized implementation
    start = std::chrono::high_resolution_clock::now();
    stats::detail::vector_erf(input, output);
    end = std::chrono::high_resolution_clock::now();
    auto vector_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double speedup = static_cast<double>(scalar_time.count()) / vector_time.count();

    // We expect at least no significant slowdown
    EXPECT_GT(speedup, 0.5) << "Vectorized should not be significantly slower than scalar";

    std::cout << "Erf speedup: " << std::fixed << std::setprecision(2) << speedup << "x"
              << std::endl;
}

TEST_F(MathUtilsTest, GammaPerformance) {
    const size_t size = 10000;
    std::vector<double> pos_input = generate_test_data(size, 0.1, 5.0);
    std::vector<double> output(size);
    double a = 2.5;

    // Scalar gamma_p
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < size; ++i) {
        output[i] = stats::detail::gamma_p(a, pos_input[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Vectorized gamma_p
    start = std::chrono::high_resolution_clock::now();
    stats::detail::vector_gamma_p(a, pos_input, output);
    end = std::chrono::high_resolution_clock::now();
    auto vector_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double speedup = static_cast<double>(scalar_time.count()) / vector_time.count();

    // We expect at least no significant slowdown
    EXPECT_GT(speedup, 0.5) << "Vectorized should not be significantly slower than scalar";

    std::cout << "Gamma_p speedup: " << std::fixed << std::setprecision(2) << speedup << "x"
              << std::endl;
}

//==============================================================================
// ACCURACY TESTS
//==============================================================================

TEST_F(MathUtilsTest, ErfAccuracy) {
    // Test a range of values for accuracy
    std::vector<double> test_values = {-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
                                       0.5,  1.0,  1.5,  2.0,  2.5,  3.0};

    for (double x : test_values) {
        double erf_val = stats::detail::erf(x);
        double erfc_val = stats::detail::erfc(x);

        // Test complementary relationship
        EXPECT_TRUE(near_equal(erf_val + erfc_val, 1.0, LOOSE_TOLERANCE)) << "Failed at x = " << x;

        // Test symmetry
        EXPECT_TRUE(near_equal(stats::detail::erf(-x), -stats::detail::erf(x), TOLERANCE))
            << "Failed symmetry at x = " << x;
    }
}

TEST_F(MathUtilsTest, InverseAccuracy) {
    // Test inverse functions accuracy
    std::vector<double> p_values = {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99};

    for (double p : p_values) {
        // Test erf_inv
        double x = stats::detail::erf_inv(2.0 * p - 1.0);
        double p_recovered = (stats::detail::erf(x) + 1.0) / 2.0;
        EXPECT_TRUE(near_equal(p, p_recovered, VERY_LOOSE_TOLERANCE)) << "Failed at p = " << p;

        // Test inverse_normal_cdf
        double z = stats::detail::inverse_normal_cdf(p);
        double p_norm = stats::detail::normal_cdf(z);
        EXPECT_TRUE(near_equal(p, p_norm, VERY_LOOSE_TOLERANCE)) << "Failed at p = " << p;
    }
}

//==============================================================================
// CROSS-VALIDATION TESTS
//==============================================================================

TEST_F(MathUtilsTest, CrossValidateFunctions) {
    // Cross-validate our implementations against standard library where available
    std::vector<double> test_vals = {0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0};

    for (double x : test_vals) {
        // Compare lgamma with std::lgamma
        double our_val = stats::detail::lgamma(x);
        double std_val = std::lgamma(x);
        EXPECT_TRUE(near_equal(our_val, std_val, LOOSE_TOLERANCE))
            << "lgamma(" << x << ") failed: ours=" << our_val << " std=" << std_val;
    }
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

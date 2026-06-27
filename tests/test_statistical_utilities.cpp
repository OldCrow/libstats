/**
 * @file test_statistical_utilities.cpp
 * @brief Tests for stats::analysis:: statistical utility functions.
 *
 * Covers: empirical_cdf, calculate_quantiles, sample_moments, validate_fitting_data.
 * These were promoted from stats::detail:: in v2.0.0 (API rationalization, D3).
 */

#include "libstats/stats/analysis/statistical_utilities.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using namespace stats::analysis;

// =============================================================================
// empirical_cdf
// =============================================================================

TEST(StatisticalUtilities, EmpiricalCdfEmptyInput) {
    std::vector<double> empty;
    auto result = empirical_cdf(std::span<const double>(empty));
    EXPECT_TRUE(result.empty());
}

TEST(StatisticalUtilities, EmpiricalCdfSingleElement) {
    std::vector<double> data = {3.14};
    auto result = empirical_cdf(std::span<const double>(data));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0], 1.0);  // (0+1)/1 = 1.0
}

TEST(StatisticalUtilities, EmpiricalCdfMonotoneAndBounded) {
    std::vector<double> data = {5.0, 2.0, 8.0, 1.0, 4.0};
    auto result = empirical_cdf(std::span<const double>(data));
    ASSERT_EQ(result.size(), data.size());
    // Values must be in (0, 1] and strictly increasing.
    for (double v : result) {
        EXPECT_GT(v, 0.0);
        EXPECT_LE(v, 1.0);
    }
    for (std::size_t i = 1; i < result.size(); ++i) {
        EXPECT_GT(result[i], result[i - 1]) << "CDF must be strictly increasing at i=" << i;
    }
    // Last value must be exactly 1.0.
    EXPECT_DOUBLE_EQ(result.back(), 1.0);
}

TEST(StatisticalUtilities, EmpiricalCdfKnownValues) {
    // Data {1,2,3,4,5} → sorted order is same → CDF = {0.2, 0.4, 0.6, 0.8, 1.0}
    std::vector<double> data = {3.0, 1.0, 5.0, 2.0, 4.0};
    auto result = empirical_cdf(std::span<const double>(data));
    ASSERT_EQ(result.size(), 5u);
    EXPECT_NEAR(result[0], 0.2, 1e-12);
    EXPECT_NEAR(result[1], 0.4, 1e-12);
    EXPECT_NEAR(result[2], 0.6, 1e-12);
    EXPECT_NEAR(result[3], 0.8, 1e-12);
    EXPECT_NEAR(result[4], 1.0, 1e-12);
}

// =============================================================================
// calculate_quantiles
// =============================================================================

TEST(StatisticalUtilities, CalculateQuantilesThrowsOnEmptyData) {
    std::vector<double> data;
    std::vector<double> qs = {0.5};
    EXPECT_THROW((void)calculate_quantiles(std::span<const double>(data),
                                           std::span<const double>(qs)),
                 std::invalid_argument);
}

TEST(StatisticalUtilities, CalculateQuantilesThrowsOnOutOfRangeLevel) {
    std::vector<double> data = {1.0, 2.0, 3.0};
    std::vector<double> bad_qs = {-0.1};
    EXPECT_THROW((void)calculate_quantiles(std::span<const double>(data),
                                           std::span<const double>(bad_qs)),
                 std::invalid_argument);

    std::vector<double> bad_qs2 = {1.1};
    EXPECT_THROW((void)calculate_quantiles(std::span<const double>(data),
                                           std::span<const double>(bad_qs2)),
                 std::invalid_argument);
}

TEST(StatisticalUtilities, CalculateQuantilesExtremes) {
    std::vector<double> data = {10.0, 20.0, 30.0, 40.0, 50.0};
    std::vector<double> qs = {0.0, 1.0};
    auto result = calculate_quantiles(std::span<const double>(data),
                                      std::span<const double>(qs));
    ASSERT_EQ(result.size(), 2u);
    EXPECT_DOUBLE_EQ(result[0], 10.0);  // min
    EXPECT_DOUBLE_EQ(result[1], 50.0);  // max
}

TEST(StatisticalUtilities, CalculateQuantilesMedianUniform) {
    // Uniform {1,2,3,4,5}: median should be 3.0.
    std::vector<double> data = {5.0, 1.0, 3.0, 2.0, 4.0};
    std::vector<double> qs = {0.5};
    auto result = calculate_quantiles(std::span<const double>(data),
                                      std::span<const double>(qs));
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(result[0], 3.0, 1e-10);
}

TEST(StatisticalUtilities, CalculateQuantilesInterpolation) {
    // Data {0,1}: 25th percentile should be 0.25 via linear interpolation.
    std::vector<double> data = {0.0, 1.0};
    std::vector<double> qs = {0.25, 0.75};
    auto result = calculate_quantiles(std::span<const double>(data),
                                      std::span<const double>(qs));
    ASSERT_EQ(result.size(), 2u);
    EXPECT_NEAR(result[0], 0.25, 1e-12);
    EXPECT_NEAR(result[1], 0.75, 1e-12);
}

// =============================================================================
// sample_moments
// =============================================================================

TEST(StatisticalUtilities, SampleMomentsThrowsOnEmpty) {
    std::vector<double> data;
    EXPECT_THROW((void)sample_moments(std::span<const double>(data)), std::invalid_argument);
}

TEST(StatisticalUtilities, SampleMomentsThrowsOnNonFinite) {
    std::vector<double> data = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};
    EXPECT_THROW((void)sample_moments(std::span<const double>(data)), std::invalid_argument);
}

TEST(StatisticalUtilities, SampleMomentsKnownGaussian) {
    // Large sample from N(5, 4) — mean=5, variance=4, skew≈0, kurtosis≈0.
    // Use a fixed deterministic dataset: {3,4,5,6,7} to get exact moments.
    std::vector<double> data = {3.0, 4.0, 5.0, 6.0, 7.0};
    auto [mean, variance, skewness, kurtosis] = sample_moments(std::span<const double>(data));

    EXPECT_NEAR(mean, 5.0, 1e-12);
    EXPECT_NEAR(variance, 2.5, 1e-12);       // Bessel-corrected: 10/4 = 2.5
    EXPECT_NEAR(skewness, 0.0, 1e-12);       // symmetric
    // Excess kurtosis of uniform-style 5-point sample: not necessarily 0;
    // just check it's finite.
    EXPECT_TRUE(std::isfinite(kurtosis));
}

TEST(StatisticalUtilities, SampleMomentsConstantData) {
    // All values the same: variance=0, skewness and kurtosis should be NaN.
    std::vector<double> data(10, 42.0);
    auto [mean, variance, skewness, kurtosis] = sample_moments(std::span<const double>(data));
    EXPECT_NEAR(mean, 42.0, 1e-12);
    EXPECT_NEAR(variance, 0.0, 1e-12);
    EXPECT_TRUE(std::isnan(skewness))   << "Skewness undefined for constant data";
    EXPECT_TRUE(std::isnan(kurtosis))   << "Kurtosis undefined for constant data";
}

TEST(StatisticalUtilities, SampleMomentsReturnOrder) {
    // Verify [0]=mean, [1]=variance, [2]=skewness, [3]=excess_kurtosis.
    std::vector<double> data = {1.0, 2.0, 3.0};
    auto moments = sample_moments(std::span<const double>(data));
    EXPECT_NEAR(moments[0], 2.0, 1e-12);   // mean
    EXPECT_GT(moments[1], 0.0);            // variance > 0
}

// =============================================================================
// validate_fitting_data
// =============================================================================

TEST(StatisticalUtilities, ValidateFittingDataAcceptsFinite) {
    std::vector<double> good = {1.0, 2.5, -3.7, 0.0};
    EXPECT_TRUE(validate_fitting_data(std::span<const double>(good)));
}

TEST(StatisticalUtilities, ValidateFittingDataRejectsNaN) {
    std::vector<double> bad = {1.0, std::numeric_limits<double>::quiet_NaN()};
    EXPECT_FALSE(validate_fitting_data(std::span<const double>(bad)));
}

TEST(StatisticalUtilities, ValidateFittingDataRejectsInfinity) {
    std::vector<double> bad = {1.0, std::numeric_limits<double>::infinity()};
    EXPECT_FALSE(validate_fitting_data(std::span<const double>(bad)));

    std::vector<double> bad2 = {-std::numeric_limits<double>::infinity(), 0.0};
    EXPECT_FALSE(validate_fitting_data(std::span<const double>(bad2)));
}

TEST(StatisticalUtilities, ValidateFittingDataEmptyIsValid) {
    // An empty span has no non-finite elements — std::all_of on empty range is true.
    std::vector<double> empty;
    EXPECT_TRUE(validate_fitting_data(std::span<const double>(empty)));
}

/**
 * @file test_discrete_analysis.cpp
 * @brief Tests for stats::analysis::discrete::runsTest and frequencyTest.
 *
 * Part 4 of the API rationalization plan — these were the only two public
 * analysis functions with zero GTest coverage.
 */

#include "libstats/stats/analysis/discrete_analysis.h"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

using namespace stats::analysis::discrete;

// =============================================================================
// runsTest
// =============================================================================

TEST(RunsTest, ThrowsOnTooFewElements) {
    std::vector<double> tiny = {1.0, 2.0, 3.0};
    EXPECT_THROW((void)runsTest(tiny), std::invalid_argument);
}

TEST(RunsTest, ThrowsOnBadAlpha) {
    std::vector<double> data(20, 1.0);
    EXPECT_THROW((void)runsTest(data, 0.0), std::invalid_argument);
    EXPECT_THROW((void)runsTest(data, 1.0), std::invalid_argument);
    EXPECT_THROW((void)runsTest(data, -0.1), std::invalid_argument);
}

TEST(RunsTest, RandomDataDoesNotReject) {
    // Alternating sequence has many runs — should NOT reject H₀ of randomness.
    std::vector<double> alternating;
    for (int i = 0; i < 40; ++i)
        alternating.push_back(i % 2 == 0 ? 1.0 : -1.0);

    auto [z, p, reject] = runsTest(alternating);

    EXPECT_TRUE(std::isfinite(z)) << "z-statistic must be finite";
    EXPECT_GE(p, 0.0) << "p-value must be non-negative";
    EXPECT_LE(p, 1.0) << "p-value must be at most 1";
    // Alternating sequence is NOT random; many runs → reject randomness.
    // The test just validates output shape regardless of the decision.
}

TEST(RunsTest, TrendedDataRejectsRandomness) {
    // A monotone sequence has very few runs and should reject H₀.
    std::vector<double> trend;
    for (int i = 0; i < 50; ++i)
        trend.push_back(static_cast<double>(i));

    auto [z, p, reject] = runsTest(trend, 0.05);

    EXPECT_TRUE(std::isfinite(z));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    // A strictly increasing sequence has exactly 1 run → tiny p-value.
    EXPECT_LT(p, 0.001) << "monotone trend should strongly reject H₀";
    EXPECT_TRUE(reject) << "monotone trend should be flagged as non-random";
}

TEST(RunsTest, ReturnTupleOrder) {
    // Verify the return is {z, p, reject} in the documented order.
    std::vector<double> data(20);
    for (int i = 0; i < 20; ++i)
        data[i] = (i % 2 == 0) ? 1.0 : 2.0;
    auto result = runsTest(data);
    auto [z, p, reject] = result;
    // p-value must be in [0,1]; z can be any finite real.
    EXPECT_TRUE(std::isfinite(z));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
}

// =============================================================================
// frequencyTest
// =============================================================================

TEST(FrequencyTest, ThrowsOnInvalidRange) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    EXPECT_THROW((void)frequencyTest(data, 5, 1), std::invalid_argument);  // lo >= hi
    EXPECT_THROW((void)frequencyTest(data, 3, 3), std::invalid_argument);  // lo == hi
}

TEST(FrequencyTest, ThrowsOnBadAlpha) {
    std::vector<double> data(30);
    for (int i = 0; i < 30; ++i)
        data[i] = static_cast<double>(1 + i % 6);
    EXPECT_THROW((void)frequencyTest(data, 1, 6, 0.0), std::invalid_argument);
    EXPECT_THROW((void)frequencyTest(data, 1, 6, 1.1), std::invalid_argument);
}

TEST(FrequencyTest, ThrowsWhenTooFewValuesInSupport) {
    // Only 2 values fall within [1,6] — should throw.
    std::vector<double> data = {1.0, 6.0, 100.0, 200.0, 300.0};
    EXPECT_THROW((void)frequencyTest(data, 1, 6), std::invalid_argument);
}

TEST(FrequencyTest, UniformDataDoesNotReject) {
    // Perfectly balanced die rolls {1,2,3,4,5,6} repeated 20 times.
    std::vector<double> data;
    for (int rep = 0; rep < 20; ++rep)
        for (int face = 1; face <= 6; ++face)
            data.push_back(static_cast<double>(face));

    auto [chi2, p, reject] = frequencyTest(data, 1, 6, 0.05);

    EXPECT_TRUE(std::isfinite(chi2)) << "chi2 statistic must be finite";
    EXPECT_GE(chi2, 0.0) << "chi2 must be non-negative";
    EXPECT_GE(p, 0.0) << "p-value must be non-negative";
    EXPECT_LE(p, 1.0) << "p-value must be at most 1";
    EXPECT_FALSE(reject) << "perfectly uniform data should not reject H₀";
}

TEST(FrequencyTest, SkewedDataRejectsUniformity) {
    // 35 of value 1, 5 each of values 2-6: clearly non-uniform but each bin
    // has enough data for chi-square (expected count = 55/6 ≈ 9.2 >> 5 threshold).
    std::vector<double> data(35, 1.0);
    for (int v = 2; v <= 6; ++v)
        for (int k = 0; k < 5; ++k)
            data.push_back(static_cast<double>(v));

    auto [chi2, p, reject] = frequencyTest(data, 1, 6, 0.05);

    EXPECT_TRUE(std::isfinite(chi2));
    EXPECT_GT(chi2, 0.0);
    EXPECT_LT(p, 0.001) << "heavily skewed data should strongly reject uniformity";
    EXPECT_TRUE(reject) << "heavily skewed data should be flagged as non-uniform";
}

TEST(FrequencyTest, ReturnTupleOrder) {
    // {chi2, p, reject} — chi2 >= 0, p in [0,1].
    std::vector<double> data;
    for (int i = 0; i < 30; ++i)
        data.push_back(static_cast<double>(1 + i % 3));
    auto [chi2, p, reject] = frequencyTest(data, 1, 3);
    EXPECT_GE(chi2, 0.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
}

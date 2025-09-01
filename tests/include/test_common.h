// test_common.h
// Common includes and utilities for test files
// This header reduces test compilation time by consolidating frequently used test includes

#pragma once

#ifndef LIBSTATS_TEST_COMMON_H
    #define LIBSTATS_TEST_COMMON_H

    // Google Test essentials
    #include <gtest/gtest.h>

    // Standard library includes commonly used in tests
    #include <algorithm>
    #include <cmath>
    #include <limits>
    #include <memory>
    #include <random>
    #include <sstream>
    #include <string>
    #include <vector>

    // Core libstats headers needed for testing
    // Use relative paths from tests/include/ to include/
    #include "../../include/core/mathematical_constants.h"
    #include "../../include/core/precision_constants.h"
    #include "../../include/core/threshold_constants.h"

    // Forward declarations for heavy headers
    #include "../../include/common/cpu_detection_fwd.h"
    #include "../../include/common/platform_constants_fwd.h"

namespace stats {
namespace test {

// Common test utilities and helper functions

// Tolerance for floating-point comparisons in tests
constexpr double DEFAULT_TEST_TOLERANCE = 1e-10;
constexpr double RELAXED_TEST_TOLERANCE = 1e-6;
constexpr double STRICT_TEST_TOLERANCE = 1e-14;

// Helper for comparing floating-point values with tolerance
inline bool nearly_equal(double a, double b, double tolerance = DEFAULT_TEST_TOLERANCE) {
    const double diff = std::abs(a - b);
    const double largest = std::max(std::abs(a), std::abs(b));

    // Handle the case where both values are very close to zero
    if (largest < tolerance) {
        return diff < tolerance;
    }

    return diff <= tolerance * largest;
}

// Helper for generating random test data
template <typename RNG = std::mt19937>
class TestDataGenerator {
   public:
    explicit TestDataGenerator(unsigned int seed = 42) : rng_(seed) {}

    std::vector<double> generate_uniform(size_t n, double min = 0.0, double max = 1.0) {
        std::uniform_real_distribution<double> dist(min, max);
        std::vector<double> data(n);
        std::generate(data.begin(), data.end(), [&]() { return dist(rng_); });
        return data;
    }

    std::vector<double> generate_normal(size_t n, double mean = 0.0, double stddev = 1.0) {
        std::normal_distribution<double> dist(mean, stddev);
        std::vector<double> data(n);
        std::generate(data.begin(), data.end(), [&]() { return dist(rng_); });
        return data;
    }

    std::vector<int> generate_poisson(size_t n, double lambda = 1.0) {
        std::poisson_distribution<int> dist(lambda);
        std::vector<int> data(n);
        std::generate(data.begin(), data.end(), [&]() { return dist(rng_); });
        return data;
    }

   private:
    RNG rng_;
};

    // Macro for parameterized test ranges
    #define LIBSTATS_TEST_RANGE(start, end, step) testing::Range(start, end, step)

    // Macro for common test assertions with better error messages
    #define EXPECT_NEAR_REL(actual, expected, rel_tol)                                             \
        EXPECT_NEAR(actual, expected, std::abs(expected) * rel_tol)                                \
            << "Relative tolerance: " << rel_tol

// Test fixture base class with common setup
class LibStatsTestBase : public ::testing::Test {
   protected:
    void SetUp() override {
        // Common setup for all tests
        // Can be extended by derived test fixtures
    }

    void TearDown() override {
        // Common cleanup for all tests
    }

    // Seed for reproducible random tests
    static constexpr unsigned int DEFAULT_SEED = 12345;
};

}  // namespace test
}  // namespace stats

// Make test utilities available in the global namespace for convenience
using stats::test::LibStatsTestBase;
using stats::test::nearly_equal;
using stats::test::TestDataGenerator;

#endif  // LIBSTATS_TEST_COMMON_H

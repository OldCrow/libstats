#pragma once

/**
 * @file tests/tests.h
 * @brief Unified test infrastructure header
 *
 * This header provides a single point of access to all test infrastructure
 * utilities organized under the stats::tests:: namespace hierarchy. It includes
 * constants, fixtures, validators, and benchmarks for comprehensive test support.
 *
 * Phase 3E: Test Infrastructure Namespace - Unified Access Point
 *
 * Usage:
 *   #include "../include/tests/tests.h"
 *
 *   using namespace stats::tests;
 *   // Now you have access to:
 *   // - constants:: namespace for test constants
 *   // - fixtures:: namespace for test utilities and data generators
 *   // - validators:: namespace for adaptive validation
 *   // - benchmarks:: namespace for specialized benchmarking
 */

// Core test infrastructure components
#include "benchmarks.h"
#include "constants.h"
#include "fixtures.h"
#include "validators.h"

namespace stats {
namespace tests {

//==============================================================================
// Unified Test Infrastructure Interface
//==============================================================================

/**
 * @brief Main test infrastructure interface providing convenient access
 */
class TestInfrastructure {
   public:
    // Type aliases for convenience
    using BenchmarkConfig = benchmarks::TestBenchmarkConfig;
    using QuickBench = benchmarks::QuickBenchmark;
    using PerfValidator = validators::PerformanceValidator;
    using TestFormatter = fixtures::BasicTestFormatter;
    using DataGenerator = fixtures::TestDataGenerators;

    /**
     * @brief Initialize test infrastructure with optimal settings
     * @param quick_mode Use quick/minimal configurations for faster testing
     */
    static void initialize(bool quick_mode = false);

    /**
     * @brief Get recommended test configuration for current environment
     * @param test_category Category of test ("unit", "integration", "performance", "regression")
     * @return Appropriate configuration for the test category
     */
    static BenchmarkConfig getRecommendedConfig(const std::string& test_category);

    /**
     * @brief Quick performance validation with architecture-aware thresholds
     * @param measured_simd_speedup Measured SIMD speedup
     * @param measured_parallel_speedup Measured parallel speedup
     * @param batch_size Batch size tested
     * @param distribution_name Name of distribution tested
     * @return True if performance meets adaptive expectations
     */
    static bool validatePerformance(double measured_simd_speedup, double measured_parallel_speedup,
                                    std::size_t batch_size, const std::string& distribution_name);

    /**
     * @brief Generate test data appropriate for a specific distribution
     * @param distribution_name Name of the distribution
     * @param size Number of data points to generate
     * @param seed Random seed for reproducibility
     * @return Generated test data vector
     */
    static std::vector<double> generateTestData(const std::string& distribution_name,
                                                std::size_t size,
                                                uint32_t seed = constants::DEFAULT_RANDOM_SEED);

    /**
     * @brief Quick benchmark comparison between two functions
     * @param baseline_name Name of baseline implementation
     * @param baseline_func Baseline function
     * @param comparison_name Name of comparison implementation
     * @param comparison_func Comparison function
     * @param iterations Number of iterations (0 = auto-detect)
     * @return Speedup factor (baseline_time / comparison_time)
     */
    static double quickCompare(const std::string& baseline_name,
                               std::function<void()> baseline_func,
                               const std::string& comparison_name,
                               std::function<void()> comparison_func, std::size_t iterations = 0);

    /**
     * @brief Print standardized test section header
     * @param section_name Name of the test section
     */
    static void printSectionHeader(const std::string& section_name);

    /**
     * @brief Print test completion summary
     * @param test_name Name of the completed test
     * @param num_passed Number of passed assertions
     * @param num_failed Number of failed assertions
     */
    static void printTestSummary(const std::string& test_name, std::size_t num_passed,
                                 std::size_t num_failed);
};

//==============================================================================
// Convenience Macros for Common Test Patterns
//==============================================================================

// Quick access to common test utilities
#define TEST_CONSTANTS stats::tests::constants
#define TEST_FIXTURES stats::tests::fixtures
#define TEST_VALIDATORS stats::tests::validators
#define TEST_BENCHMARKS stats::tests::benchmarks

// Convenience macros for architecture-aware validation
#define EXPECT_SIMD_SPEEDUP(measured, batch_size, is_complex)                                      \
    EXPECT_TRUE(TEST_VALIDATORS::validateSIMDSpeedup((measured), (batch_size), (is_complex)))      \
        << "SIMD speedup " << (measured) << "x should exceed adaptive threshold for batch size "   \
        << (batch_size)

#define EXPECT_PARALLEL_SPEEDUP(measured, batch_size, is_complex)                                  \
    EXPECT_TRUE(TEST_VALIDATORS::validateParallelSpeedup((measured), (batch_size), (is_complex)))  \
        << "Parallel speedup " << (measured)                                                       \
        << "x should exceed adaptive threshold for batch size " << (batch_size)

// Quick benchmark comparison macro
#define QUICK_BENCHMARK_COMPARE(baseline, comparison, iterations)                                  \
    TestInfrastructure::quickCompare(                                                              \
        #baseline, [&]() { baseline; }, #comparison, [&]() { comparison; }, (iterations))

//==============================================================================
// Backward Compatibility Support
//==============================================================================

namespace legacy {
// Provide aliases to old class names for easier migration
using StandardizedBasicTest = fixtures::BasicTestFormatter;
using StandardizedBenchmark = fixtures::BenchmarkFormatter;
using StatisticalTestUtils = fixtures::StatisticalTestUtils;
using ThreadSafetyTester = fixtures::ThreadSafetyTester<void>;  // Template requires specialization

// Legacy constants for backward compatibility
static constexpr double SIMD_SPEEDUP_MIN_EXPECTED = 1.5;
static constexpr double PARALLEL_SPEEDUP_MIN_EXPECTED = 2.0;
static constexpr std::size_t DEFAULT_BENCHMARK_ITERATIONS = 1000;

/**
 * @brief Legacy initialization function
 * @deprecated Use TestInfrastructure::initialize() instead
 */
[[deprecated("Use TestInfrastructure::initialize() instead")]]
inline void initializeTestInfrastructure() {
    TestInfrastructure::initialize(false);
}
}  // namespace legacy

}  // namespace tests
}  // namespace stats

//==============================================================================
// Global Using Declarations for Convenience (Optional)
//==============================================================================

// Uncomment these if you want global access to test utilities
// using TestConstants = stats::tests::constants;
// using TestFixtures = stats::tests::fixtures;
// using TestValidators = stats::tests::validators;
// using TestBenchmarks = stats::tests::benchmarks;

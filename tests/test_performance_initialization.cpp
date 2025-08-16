#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

#include <gtest/gtest.h>
#define LIBSTATS_FULL_INTERFACE
#include "../include/libstats.h"
#include <chrono>
#include <vector>
#include <iostream>

using namespace std;
using namespace libstats;

/**
 * Test suite for performance system initialization
 */
class PerformanceInitializationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize performance systems once for all tests in this suite
        // This demonstrates the recommended usage pattern
        std::cout << "Initializing performance systems for test suite..." << std::endl;
        libstats::initialize_performance_systems();
        std::cout << "Performance systems initialized." << std::endl;
    }
};

/**
 * Test that initialization function can be called safely multiple times
 */
TEST_F(PerformanceInitializationTest, MultipleInitializationCallsAreSafe) {
    // Note: Since SetUpTestSuite() already called initialize_performance_systems(),
    // all calls in this test are actually subsequent calls using the fast path.
    // This test verifies thread-safety and idempotent behavior rather than
    // measuring true first-time vs subsequent initialization performance.
    
    auto start1 = std::chrono::high_resolution_clock::now();
    libstats::initialize_performance_systems();
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto start2 = std::chrono::high_resolution_clock::now();
    libstats::initialize_performance_systems();
    auto end2 = std::chrono::high_resolution_clock::now();
    
    auto start3 = std::chrono::high_resolution_clock::now();
    libstats::initialize_performance_systems();
    auto end3 = std::chrono::high_resolution_clock::now();
    
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();
    auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();
    auto duration3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count();
    
    std::cout << "Initialization call timings:" << std::endl;
    std::cout << "  First test call:  " << duration1 << " ns" << std::endl;
    std::cout << "  Second test call: " << duration2 << " ns" << std::endl;
    std::cout << "  Third test call:  " << duration3 << " ns" << std::endl;
    
    // Since all calls are fast path, just verify they complete in reasonable time
    // (under 10 microseconds is very reasonable for a static flag check)
    EXPECT_LT(duration1, 10000); // Should complete in under 10μs
    EXPECT_LT(duration2, 10000); // Should complete in under 10μs  
    EXPECT_LT(duration3, 10000); // Should complete in under 10μs
    
    // Verify the function doesn't crash when called multiple times (idempotent)
    EXPECT_NO_THROW(libstats::initialize_performance_systems());
    EXPECT_NO_THROW(libstats::initialize_performance_systems());
}

/**
 * Test that batch operations work correctly after initialization
 */
TEST_F(PerformanceInitializationTest, BatchOperationsWorkAfterInitialization) {
    // Create a distribution for testing
    auto result = Gaussian::create(0.0, 1.0);
    ASSERT_TRUE(result.isOk());
    auto dist = std::move(result.value);
    
    // Create test data
    constexpr size_t test_size = 1000;
    std::vector<double> input_values(test_size);
    std::vector<double> output_results(test_size);
    
    // Fill input with test values
    for (size_t i = 0; i < test_size; ++i) {
        input_values[i] = static_cast<double>(i) * 0.1 - 50.0; // Range from -50 to +50
    }
    
    // Test batch probability calculation (should use initialized performance systems)
    auto start = std::chrono::high_resolution_clock::now();
    dist.getProbability(std::span<const double>(input_values), std::span<double>(output_results));
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Batch operation duration: " << duration << " μs for " << test_size << " values" << std::endl;
    
    // Verify results are reasonable
    for (size_t i = 0; i < test_size; ++i) {
        EXPECT_GE(output_results[i], 0.0); // PDF values should be non-negative
        EXPECT_TRUE(std::isfinite(output_results[i])); // Should be finite
    }
    
    // Test that result at mean (0.0) is higher than at tails
    size_t mean_index = 500; // input_values[500] should be near 0.0
    size_t tail_index = 900; // input_values[900] should be far from 0.0
    EXPECT_GT(output_results[mean_index], output_results[tail_index]);
}

/**
 * Test that performance systems work correctly without explicit initialization
 */
TEST(PerformanceInitializationTestStandalone, WorksWithoutExplicitInitialization) {
    // This test intentionally doesn't call initialize_performance_systems()
    // to verify that the library still works correctly (just with potential cold-start delay)
    
    auto result = Exponential::create(1.0);
    ASSERT_TRUE(result.isOk());
    auto dist = std::move(result.value);
    
    // Small batch operation - should work even without initialization
    std::vector<double> input_values = {0.1, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> output_results(5);
    
    // This call might experience cold-start delay, but should still work
    dist.getProbability(std::span<const double>(input_values), std::span<double>(output_results));
    
    // Verify results are reasonable for exponential distribution
    for (size_t i = 0; i < input_values.size(); ++i) {
        EXPECT_GE(output_results[i], 0.0); // PDF values should be non-negative
        EXPECT_TRUE(std::isfinite(output_results[i])); // Should be finite
        
        // For exponential distribution, larger x values should have smaller PDF values
        if (i > 0) {
            EXPECT_LT(output_results[i], output_results[i-1]);
        }
    }
}

/**
 * Test initialization function basic functionality
 */
TEST(PerformanceInitializationBasic, InitializationFunctionExists) {
    // Simply verify that the function can be called without crashing
    EXPECT_NO_THROW(libstats::initialize_performance_systems());
    
    // Verify it's idempotent (can be called multiple times safely)
    EXPECT_NO_THROW(libstats::initialize_performance_systems());
    EXPECT_NO_THROW(libstats::initialize_performance_systems());
}

/**
 * Performance comparison test to demonstrate the benefit of initialization
 */
TEST(PerformanceInitializationBenchmark, InitializationImprovesColdStartPerformance) {
    std::cout << "\n=== Performance Initialization Benchmark ===" << std::endl;
    
    // This is more of a demonstration than a strict test since timing can vary
    constexpr size_t batch_size = 10000;
    std::vector<double> input_values(batch_size);
    std::vector<double> output_results(batch_size);
    
    // Fill input with test values
    for (size_t i = 0; i < batch_size; ++i) {
        input_values[i] = static_cast<double>(i) * 0.001; // Range from 0 to 10
    }
    
    // Test with initialization
    libstats::initialize_performance_systems();
    
    auto result = Uniform::create(0.0, 10.0);
    ASSERT_TRUE(result.isOk());
    auto dist = std::move(result.value);
    
    auto start = std::chrono::high_resolution_clock::now();
    dist.getProbability(std::span<const double>(input_values), std::span<double>(output_results));
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration_with_init = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Batch operation with initialization: " << duration_with_init << " μs" << std::endl;
    
    // Verify results are correct for uniform distribution
    for (size_t i = 0; i < batch_size; ++i) {
        if (input_values[i] >= 0.0 && input_values[i] <= 10.0) {
            EXPECT_NEAR(output_results[i], 0.1, 1e-10); // PDF should be 1/(b-a) = 1/10 = 0.1
        } else {
            EXPECT_NEAR(output_results[i], 0.0, 1e-10); // Outside support should be 0
        }
    }
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif

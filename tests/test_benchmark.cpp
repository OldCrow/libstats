/**
 * @file test_benchmark.cpp
 * @brief Comprehensive test suite for platform/benchmark.h infrastructure
 *
 * Tests the Timer, BenchmarkStats, BenchmarkResult, and Benchmark classes
 * with command-line options for selective testing:
 * --all/-a           Test all benchmark components (default)
 * --timer/-t         Test Timer class functionality
 * --stats/-s         Test BenchmarkStats calculations
 * --benchmark/-b     Test Benchmark class operations
 * --comparison/-c    Test benchmark result comparisons
 * --throughput/-T    Test throughput measurements
 * --stress/-S        Run stress tests with large iterations
 * --help/-h          Show this help
 */

#include "libstats/platform/benchmark.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/laplace.h"
#include "libstats/distributions/cauchy.h"

// Standard library includes
#include <algorithm>  // for std::min, std::max
#include <chrono>     // for std::chrono::high_resolution_clock, std::chrono::milliseconds
#include <cmath>      // for std::abs
#include <cstddef>    // for std::size_t
#include <gtest/gtest.h>
#include <iostream>  // for std::cout, std::cerr, std::endl
#include <string>    // for std::string
#include <thread>    // for std::this_thread::sleep_for
#include <vector>    // for std::vector

using namespace stats;

//==============================================================================
// COMMAND-LINE ARGUMENT PARSING
//==============================================================================

;

//==============================================================================
// TEST FUNCTIONS
//==============================================================================

void test_timer_functionality() {
    std::cout << "[Timer Class Tests]\n";

    // Test 1: Basic timer functionality
    {
        Timer timer;

        // Timer should not be running initially
        EXPECT_TRUE(!timer.isRunning());
        EXPECT_TRUE(timer.elapsed() == 0.0);

        // Start timer
        timer.start();
        EXPECT_TRUE(timer.isRunning());

        // Small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Check elapsed time
        [[maybe_unused]] double elapsed1 = timer.elapsed();
        EXPECT_TRUE(elapsed1 > 0.0);
        EXPECT_TRUE(timer.isRunning());  // Should still be running

        // Another small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Elapsed should increase
        [[maybe_unused]] double elapsed2 = timer.elapsed();
        EXPECT_TRUE(elapsed2 > elapsed1);

        // Stop timer
        [[maybe_unused]] double final_time = timer.stop();
        EXPECT_TRUE(!timer.isRunning());
        EXPECT_TRUE(final_time >= elapsed2);

        std::cout << "   ✓ Basic timer functionality passed\n";
    }

    // Test 2: Multiple start/stop cycles
    {
        Timer timer;
        std::vector<double> times;

        for (int i = 0; i < 3; ++i) {
            timer.start();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            double time = timer.stop();
            times.push_back(time);
            EXPECT_TRUE(time > 0.0);
        }

        // All times should be positive and reasonably consistent
        for ([[maybe_unused]] auto time : times) {
            EXPECT_TRUE(time > 0.0);
            EXPECT_TRUE(time < 1.0);  // Should be less than 1 second
        }

        std::cout << "   ✓ Multiple timer cycles passed\n";
    }

    // Test 3: Timer precision and accuracy
    {
        Timer timer;
        timer.start();

        // Precise delay
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto end = std::chrono::high_resolution_clock::now();

        double timer_elapsed = timer.stop();
        double reference_elapsed = std::chrono::duration<double>(end - start).count();

        // Timer should be reasonably accurate (within 20% due to scheduler variance)
        double relative_error = std::abs(timer_elapsed - reference_elapsed) / reference_elapsed;
        EXPECT_TRUE(relative_error < 0.5);  // Allow 50% variance for CI environments

        std::cout << "   ✓ Timer precision test passed (error: " << (relative_error * 100)
                  << "%)\n";
    }

    std::cout << "\n";
}

void test_benchmark_stats() {
    std::cout << "[BenchmarkStats Tests]\n";

    // Test 1: Basic stats structure
    {
        BenchmarkStats stats;
        stats.mean = 1.5;
        stats.median = 1.4;
        stats.stddev = 0.2;
        stats.min = 1.0;
        stats.max = 2.0;
        stats.samples = 100;
        stats.throughput = 1000.0;

        // Test toString functionality
        std::string stats_str = stats.toString();
        EXPECT_TRUE(!stats_str.empty());
        EXPECT_TRUE(stats_str.find("1.5") != std::string::npos);  // Should contain mean
        EXPECT_TRUE(stats_str.find("100") != std::string::npos);  // Should contain sample count

        std::cout << "   ✓ BenchmarkStats structure test passed\n";
    }

    // Test 2: Edge cases
    {
        BenchmarkStats stats;
        // Test with zero/minimal values
        stats.samples = 1;
        stats.mean = 0.0;
        stats.throughput = 0.0;

        std::string stats_str = stats.toString();
        EXPECT_TRUE(!stats_str.empty());  // Should handle edge cases gracefully

        std::cout << "   ✓ BenchmarkStats edge cases passed\n";
    }

    std::cout << "\n";
}

void test_benchmark_class() {
    std::cout << "[Benchmark Class Tests]\n";

    // Test 1: Basic benchmark creation and setup
    {
        Benchmark bench(true, 10, 3);  // Enable warmup, 10 iterations, 3 warmup runs

        // Add a simple test
        bench.addTest(
            "simple_add",
            []() {
                volatile double a = 1.0;
                volatile double b = 2.0;
                volatile double c = a + b;
                (void)c;  // Suppress unused variable warning
            },
            10, 1.0);

        // Run benchmark
        auto results = bench.runAll();
        EXPECT_TRUE(!results.empty());
        EXPECT_TRUE(results.size() == 1);
        EXPECT_TRUE(results[0].name == "simple_add");
        EXPECT_TRUE(results[0].stats.samples > 0);
        EXPECT_TRUE(results[0].stats.mean > 0.0);

        std::cout << "   ✓ Basic benchmark functionality passed\n";
    }

    // Test 2: Multiple tests
    {
        Benchmark bench(false, 5, 0);  // No warmup for faster testing

        // Add multiple tests
        bench.addTest(
            "test1", []() { std::this_thread::sleep_for(std::chrono::microseconds(100)); }, 5);

        bench.addTest(
            "test2", []() { std::this_thread::sleep_for(std::chrono::microseconds(200)); }, 5);

        auto results = bench.runAll();
        EXPECT_TRUE(results.size() == 2);

        // test2 should generally take longer than test1
        // (though this might be flaky in CI environments)
        [[maybe_unused]] bool found_test1 = false, found_test2 = false;
        for (const auto& result : results) {
            if (result.name == "test1")
                found_test1 = true;
            if (result.name == "test2")
                found_test2 = true;
        }
        EXPECT_TRUE(found_test1 && found_test2);

        std::cout << "   ✓ Multiple benchmark tests passed\n";
    }

    // Test 3: Setup and teardown functions
    {
        Benchmark bench(false, 3, 0);

        int setup_count = 0;
        int teardown_count = 0;

        bench.addTest(
            "with_setup",
            []() {
                // Simple test operation
                volatile int x = 42;
                (void)x;
            },
            3, 1.0, [&setup_count]() { setup_count++; },  // Setup
            [&teardown_count]() { teardown_count++; }     // Teardown
        );

        auto results = bench.runAll();
        EXPECT_TRUE(!results.empty());
        EXPECT_TRUE(setup_count == 3);     // Should run setup 3 times
        EXPECT_TRUE(teardown_count == 3);  // Should run teardown 3 times

        std::cout << "   ✓ Setup/teardown functionality passed\n";
    }

    std::cout << "\n";
}

void test_benchmark_comparison() {
    std::cout << "[Benchmark Comparison Tests]\n";

    // Test 1: Result comparison functionality
    {
        // Create two sets of results
        std::vector<BenchmarkResult> baseline, comparison;

        // Baseline results
        BenchmarkResult base1;
        base1.name = "test1";
        base1.stats.mean = 1.0;
        base1.stats.stddev = 0.1;
        base1.stats.samples = 10;
        baseline.push_back(base1);

        // Comparison results (should be faster)
        BenchmarkResult comp1;
        comp1.name = "test1";
        comp1.stats.mean = 0.8;  // 20% faster
        comp1.stats.stddev = 0.08;
        comp1.stats.samples = 10;
        comparison.push_back(comp1);

        // Test comparison (should not crash)
        std::ostringstream oss;
        Benchmark::compareResults(baseline, comparison, oss);

        std::string comparison_output = oss.str();
        EXPECT_TRUE(!comparison_output.empty());

        std::cout << "   ✓ Benchmark comparison functionality passed\n";
    }

    std::cout << "\n";
}

void test_throughput_measurements() {
    std::cout << "[Throughput Measurement Tests]\n";

    // Test 1: Throughput calculation
    {
        Benchmark bench(false, 5, 0);

        const double operations_per_iteration = 1000.0;

        bench.addTest(
            "throughput_test",
            []() {
                // Simulate processing 1000 operations
                for (int i = 0; i < 1000; ++i) {
                    volatile double x = std::sqrt(i);
                    (void)x;
                }
            },
            5, operations_per_iteration);

        auto results = bench.runAll();
        EXPECT_TRUE(!results.empty());

        const auto& result = results[0];
        EXPECT_TRUE(result.stats.throughput > 0.0);

        // Throughput should be operations per second
        double expected_ops_per_sec = operations_per_iteration / result.stats.mean;
        [[maybe_unused]] double relative_error =
            std::abs(result.stats.throughput - expected_ops_per_sec) / expected_ops_per_sec;
        EXPECT_TRUE(relative_error < 0.1);  // Within 10%

        std::cout << "   ✓ Throughput calculation test passed\n";
        std::cout << "     Throughput: " << result.stats.throughput << " ops/sec\n";
    }

    std::cout << "\n";
}

void test_stress_conditions() {
    std::cout << "[Stress Test Conditions]\n";

    // Test 1: Large number of iterations
    {
        Benchmark bench(false, 100, 0);  // 100 iterations

        bench.addTest(
            "stress_test",
            []() {
                // Very lightweight operation
                volatile int x = 1;
                x = x + 1;  // Avoid deprecated volatile increment
                (void)x;
            },
            100);

        Timer stress_timer;
        stress_timer.start();
        auto results = bench.runAll();
        double total_time = stress_timer.stop();

        EXPECT_TRUE(!results.empty());
        EXPECT_TRUE(results[0].stats.samples == 100);
        EXPECT_TRUE(total_time < 10.0);  // Should complete in reasonable time

        std::cout << "   ✓ High iteration stress test passed (" << total_time << "s)\n";
    }

    // Test 2: Memory stress
    {
        Benchmark bench(false, 10, 0);

        bench.addTest(
            "memory_stress",
            []() {
                // Allocate and deallocate memory
                std::vector<double> data(1000, 1.0);
                double sum = 0.0;
                for (auto val : data) {
                    sum += val;
                }
                volatile double result = sum;
                (void)result;
            },
            10);

        auto results = bench.runAll();
        EXPECT_TRUE(!results.empty());
        EXPECT_TRUE(results[0].stats.mean > 0.0);
        EXPECT_TRUE(results[0].stats.stddev >= 0.0);

        std::cout << "   ✓ Memory allocation stress test passed\n";
    }

    std::cout << "\n";
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

TEST(BenchmarkInfrastructure, TimerFunctionality) {
    test_timer_functionality();
}
TEST(BenchmarkInfrastructure, BenchmarkStats) {
    test_benchmark_stats();
}
TEST(BenchmarkInfrastructure, BenchmarkClass) {
    test_benchmark_class();
}
TEST(BenchmarkInfrastructure, Comparison) {
    test_benchmark_comparison();
}
TEST(BenchmarkInfrastructure, ThroughputMeasurements) {
    test_throughput_measurements();
}
TEST(BenchmarkInfrastructure, StressConditions) {
    test_stress_conditions();
}

//==============================================================================
// DISTRIBUTION THROUGHPUT BENCHMARKS
//==============================================================================

TEST(DistributionBenchmark, GeometricPDFLogPDFCDF) {
    auto dist = GeometricDistribution::create(0.3).value;
    constexpr size_t N = 10000;
    std::vector<double> xs(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 50 + 1);

    Benchmark bench(false, 5);
    bench.addTest("Geometric PDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getProbability(xs[i]);
    });
    bench.addTest("Geometric LogPDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getLogProbability(xs[i]);
    });
    bench.addTest("Geometric CDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getCumulativeProbability(xs[i]);
    });
    auto results = bench.runAll();
    EXPECT_EQ(results.size(), 3u);
    for (const auto& r : results)
        EXPECT_GT(r.stats.mean, 0.0);
}

TEST(DistributionBenchmark, LaplacePDFLogPDFCDF) {
    auto dist = LaplaceDistribution::create(0.0, 1.0).value;
    constexpr size_t N = 10000;
    std::vector<double> xs(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -5.0 + 10.0 * static_cast<double>(i) / static_cast<double>(N);

    Benchmark bench(false, 5);
    bench.addTest("Laplace PDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getProbability(xs[i]);
    });
    bench.addTest("Laplace LogPDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getLogProbability(xs[i]);
    });
    bench.addTest("Laplace CDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getCumulativeProbability(xs[i]);
    });
    auto results = bench.runAll();
    EXPECT_EQ(results.size(), 3u);
    for (const auto& r : results)
        EXPECT_GT(r.stats.mean, 0.0);
}

TEST(DistributionBenchmark, CauchyPDFLogPDFCDF) {
    auto dist = CauchyDistribution::create(0.0, 1.0).value;
    constexpr size_t N = 10000;
    std::vector<double> xs(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -5.0 + 10.0 * static_cast<double>(i) / static_cast<double>(N);

    Benchmark bench(false, 5);
    bench.addTest("Cauchy PDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getProbability(xs[i]);
    });
    bench.addTest("Cauchy LogPDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getLogProbability(xs[i]);
    });
    bench.addTest("Cauchy CDF", [&]() {
        volatile double sink = 0.0;
        for (size_t i = 0; i < N; ++i)
            sink += dist.getCumulativeProbability(xs[i]);
    });
    auto results = bench.runAll();
    EXPECT_EQ(results.size(), 3u);
    for (const auto& r : results)
        EXPECT_GT(r.stats.mean, 0.0);
}

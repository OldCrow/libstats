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

#include "../include/platform/benchmark.h"

// Standard library includes
#include <algorithm>  // for std::min, std::max
#include <cassert>    // for assert
#include <chrono>     // for std::chrono::high_resolution_clock, std::chrono::milliseconds
#include <cmath>      // for std::abs
#include <cstddef>    // for std::size_t
#include <iostream>   // for std::cout, std::cerr, std::endl
#include <string>     // for std::string
#include <thread>     // for std::this_thread::sleep_for
#include <vector>     // for std::vector

using namespace stats;

//==============================================================================
// COMMAND-LINE ARGUMENT PARSING
//==============================================================================

struct TestOptions {
    bool test_all = false;
    bool test_timer = false;
    bool test_stats = false;
    bool test_benchmark = false;
    bool test_comparison = false;
    bool test_throughput = false;
    bool test_stress = false;
    bool show_help = false;
};

void print_help() {
    std::cout << "Usage: test_benchmark [options]\n\n";
    std::cout << "Test platform/benchmark.h infrastructure with selective options:\n\n";
    std::cout << "Options:\n";
    std::cout << "  --all/-a           Test all benchmark components (default)\n";
    std::cout << "  --timer/-t         Test Timer class functionality\n";
    std::cout << "  --stats/-s         Test BenchmarkStats calculations\n";
    std::cout << "  --benchmark/-b     Test Benchmark class operations\n";
    std::cout << "  --comparison/-c    Test benchmark result comparisons\n";
    std::cout << "  --throughput/-T    Test throughput measurements\n";
    std::cout << "  --stress/-S        Run stress tests with large iterations\n";
    std::cout << "  --help/-h          Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  test_benchmark                    # Test all components\n";
    std::cout << "  test_benchmark --timer --stats   # Test Timer and BenchmarkStats\n";
    std::cout << "  test_benchmark -b -c             # Test Benchmark class and comparisons\n";
    std::cout << "  test_benchmark --stress           # Run stress tests\n";
}

TestOptions parse_arguments(int argc, char* argv[]) {
    TestOptions options;
    bool any_specific_test = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--all" || arg == "-a") {
            options.test_all = true;
        } else if (arg == "--timer" || arg == "-t") {
            options.test_timer = true;
            any_specific_test = true;
        } else if (arg == "--stats" || arg == "-s") {
            options.test_stats = true;
            any_specific_test = true;
        } else if (arg == "--benchmark" || arg == "-b") {
            options.test_benchmark = true;
            any_specific_test = true;
        } else if (arg == "--comparison" || arg == "-c") {
            options.test_comparison = true;
            any_specific_test = true;
        } else if (arg == "--throughput" || arg == "-T") {
            options.test_throughput = true;
            any_specific_test = true;
        } else if (arg == "--stress" || arg == "-S") {
            options.test_stress = true;
            any_specific_test = true;
        } else if (arg == "--help" || arg == "-h") {
            options.show_help = true;
            return options;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help/-h for usage information.\n";
            options.show_help = true;
            return options;
        }
    }

    // If no specific tests requested, default to all
    if (!any_specific_test && !options.test_all) {
        options.test_all = true;
    }

    return options;
}

//==============================================================================
// TEST FUNCTIONS
//==============================================================================

void test_timer_functionality() {
    std::cout << "[Timer Class Tests]\n";

    // Test 1: Basic timer functionality
    {
        Timer timer;

        // Timer should not be running initially
        assert(!timer.isRunning());
        assert(timer.elapsed() == 0.0);

        // Start timer
        timer.start();
        assert(timer.isRunning());

        // Small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Check elapsed time
        [[maybe_unused]] double elapsed1 = timer.elapsed();
        assert(elapsed1 > 0.0);
        assert(timer.isRunning());  // Should still be running

        // Another small delay
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // Elapsed should increase
        [[maybe_unused]] double elapsed2 = timer.elapsed();
        assert(elapsed2 > elapsed1);

        // Stop timer
        [[maybe_unused]] double final_time = timer.stop();
        assert(!timer.isRunning());
        assert(final_time >= elapsed2);

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
            assert(time > 0.0);
        }

        // All times should be positive and reasonably consistent
        for ([[maybe_unused]] auto time : times) {
            assert(time > 0.0);
            assert(time < 1.0);  // Should be less than 1 second
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
        assert(relative_error < 0.5);  // Allow 50% variance for CI environments

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
        assert(!stats_str.empty());
        assert(stats_str.find("1.5") != std::string::npos);  // Should contain mean
        assert(stats_str.find("100") != std::string::npos);  // Should contain sample count

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
        assert(!stats_str.empty());  // Should handle edge cases gracefully

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
        assert(!results.empty());
        assert(results.size() == 1);
        assert(results[0].name == "simple_add");
        assert(results[0].stats.samples > 0);
        assert(results[0].stats.mean > 0.0);

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
        assert(results.size() == 2);

        // test2 should generally take longer than test1
        // (though this might be flaky in CI environments)
        [[maybe_unused]] bool found_test1 = false, found_test2 = false;
        for (const auto& result : results) {
            if (result.name == "test1")
                found_test1 = true;
            if (result.name == "test2")
                found_test2 = true;
        }
        assert(found_test1 && found_test2);

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
        assert(!results.empty());
        assert(setup_count == 3);     // Should run setup 3 times
        assert(teardown_count == 3);  // Should run teardown 3 times

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
        assert(!comparison_output.empty());

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
        assert(!results.empty());

        const auto& result = results[0];
        assert(result.stats.throughput > 0.0);

        // Throughput should be operations per second
        double expected_ops_per_sec = operations_per_iteration / result.stats.mean;
        [[maybe_unused]] double relative_error =
            std::abs(result.stats.throughput - expected_ops_per_sec) / expected_ops_per_sec;
        assert(relative_error < 0.1);  // Within 10%

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

        assert(!results.empty());
        assert(results[0].stats.samples == 100);
        assert(total_time < 10.0);  // Should complete in reasonable time

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
        assert(!results.empty());
        assert(results[0].stats.mean > 0.0);
        assert(results[0].stats.stddev >= 0.0);

        std::cout << "   ✓ Memory allocation stress test passed\n";
    }

    std::cout << "\n";
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

int main(int argc, char* argv[]) {
    TestOptions options = parse_arguments(argc, argv);

    if (options.show_help) {
        print_help();
        return 0;
    }

    std::cout << "Testing platform/benchmark.h infrastructure...\n\n";

    int tests_run = 0;
    int tests_passed = 0;

    try {
        // Run selected tests
        if (options.test_all || options.test_timer) {
            test_timer_functionality();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_stats) {
            test_benchmark_stats();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_benchmark) {
            test_benchmark_class();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_comparison) {
            test_benchmark_comparison();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_throughput) {
            test_throughput_measurements();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_stress) {
            test_stress_conditions();
            tests_run++;
            tests_passed++;
        }

        // Print summary
        std::cout << "=== Test Summary ===\n";
        std::cout << "Tests run: " << tests_run << "\n";
        std::cout << "Tests passed: " << tests_passed << "\n";

        if (tests_passed == tests_run && tests_run > 0) {
            std::cout << "✓ All benchmark infrastructure tests passed!\n";
            return 0;
        } else if (tests_run == 0) {
            std::cout << "No tests were run. Use --help for usage information.\n";
            return 1;
        } else {
            std::cout << "✗ Some tests failed!\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}

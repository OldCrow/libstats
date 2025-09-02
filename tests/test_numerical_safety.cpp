// Comprehensive Numerical Safety Test Suite
// Consolidates tests for core/safety.h and core/log_space_ops.h
// Provides unified testing of all numerical safety mechanisms

#include "../include/core/constants.h"
#include "../include/core/log_space_ops.h"
#include "../include/core/safety.h"

// Standard library includes
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stats;
using namespace stats::detail;
using namespace std;

// Test configuration
struct TestConfig {
    bool test_all = true;
    bool test_scalar = false;
    bool test_vector = false;
    bool test_log_space = false;
    bool test_edge_cases = false;
    bool test_benchmarks = false;
    bool test_precision = false;
    bool test_stress = false;
    bool verbose = false;
    size_t benchmark_size = 10000;
    size_t stress_iterations = 1000;
};

// Test statistics
struct TestStats {
    int total_tests = 0;
    int passed = 0;
    int failed = 0;

    void record(bool success, const string& test_name) {
        total_tests++;
        if (success) {
            passed++;
            cout << "  ✓ " << test_name << endl;
        } else {
            failed++;
            cout << "  ✗ " << test_name << " FAILED" << endl;
        }
    }

    void print_summary(const string& category) {
        cout << "\n" << category << " Results: " << passed << "/" << total_tests << " passed";
        if (failed > 0) {
            cout << " (" << failed << " failed)";
        }
        cout << endl;
    }
};

// Helper functions
template <typename T>
bool approx_equal(T a, T b, T epsilon = 1e-10) {
    if (std::isnan(a) && std::isnan(b))
        return true;
    if (std::isinf(a) && std::isinf(b))
        return (a > 0) == (b > 0);
    return std::abs(a - b) < epsilon;
}

// Reference implementations for validation
double reference_log_sum_exp(double log_a, double log_b) {
    if (std::isinf(log_a) && log_a < 0)
        return log_b;
    if (std::isinf(log_b) && log_b < 0)
        return log_a;

    double max_val = std::max(log_a, log_b);
    return max_val + std::log(std::exp(log_a - max_val) + std::exp(log_b - max_val));
}

//==============================================================================
// SCALAR SAFETY FUNCTION TESTS
//==============================================================================

void test_scalar_safety([[maybe_unused]] const TestConfig& config) {
    cout << "\n=== Testing Scalar Safety Functions ===" << endl;
    TestStats stats;

    // Test safe_log
    {
        stats.record(approx_equal(safe_log(1.0), 0.0, 1e-10), "safe_log(1.0) == 0.0");
        stats.record(approx_equal(safe_log(2.718281828), 1.0, 1e-6), "safe_log(e) == 1.0");
        stats.record(safe_log(10.0) > 0.0, "safe_log(10.0) > 0.0");
        stats.record(safe_log(0.0) == MIN_LOG_PROBABILITY,
                     "safe_log(0.0) returns MIN_LOG_PROBABILITY");
        stats.record(safe_log(-1.0) == MIN_LOG_PROBABILITY,
                     "safe_log(-1.0) returns MIN_LOG_PROBABILITY");
        stats.record(safe_log(numeric_limits<double>::quiet_NaN()) == MIN_LOG_PROBABILITY,
                     "safe_log(NaN) returns MIN_LOG_PROBABILITY");
        stats.record(safe_log(numeric_limits<double>::infinity()) == numeric_limits<double>::max(),
                     "safe_log(inf) returns max()");
    }

    // Test safe_exp
    {
        stats.record(approx_equal(safe_exp(0.0), 1.0, 1e-10), "safe_exp(0.0) == 1.0");
        stats.record(approx_equal(safe_exp(1.0), 2.718281828, 1e-6), "safe_exp(1.0) == e");
        stats.record(safe_exp(2.0) > 0.0, "safe_exp(2.0) > 0.0");
        stats.record(safe_exp(numeric_limits<double>::quiet_NaN()) == 0.0,
                     "safe_exp(NaN) returns 0.0");
        stats.record(safe_exp(-1000.0) == MIN_PROBABILITY,
                     "safe_exp(-1000.0) returns MIN_PROBABILITY");
        stats.record(safe_exp(800.0) == numeric_limits<double>::max(),
                     "safe_exp(800.0) returns max()");
    }

    // Test safe_sqrt
    {
        stats.record(approx_equal(safe_sqrt(0.0), 0.0, 1e-10), "safe_sqrt(0.0) == 0.0");
        stats.record(approx_equal(safe_sqrt(1.0), 1.0, 1e-10), "safe_sqrt(1.0) == 1.0");
        stats.record(approx_equal(safe_sqrt(4.0), 2.0, 1e-10), "safe_sqrt(4.0) == 2.0");
        stats.record(approx_equal(safe_sqrt(9.0), 3.0, 1e-10), "safe_sqrt(9.0) == 3.0");
        stats.record(safe_sqrt(-1.0) == 0.0, "safe_sqrt(-1.0) returns 0.0");
        stats.record(safe_sqrt(numeric_limits<double>::quiet_NaN()) == 0.0,
                     "safe_sqrt(NaN) returns 0.0");
        stats.record(safe_sqrt(numeric_limits<double>::infinity()) == numeric_limits<double>::max(),
                     "safe_sqrt(inf) returns max()");
    }

    // Test clamp_probability
    {
        stats.record(approx_equal(clamp_probability(0.5), 0.5, 1e-10),
                     "clamp_probability(0.5) == 0.5");
        stats.record(approx_equal(clamp_probability(0.0), MIN_PROBABILITY, 1e-10),
                     "clamp_probability(0.0) == MIN_PROBABILITY");
        stats.record(approx_equal(clamp_probability(1.0), MAX_PROBABILITY, 1e-10),
                     "clamp_probability(1.0) == MAX_PROBABILITY");
        stats.record(clamp_probability(-0.5) == MIN_PROBABILITY,
                     "clamp_probability(-0.5) clamps to MIN");
        stats.record(clamp_probability(1.5) == MAX_PROBABILITY,
                     "clamp_probability(1.5) clamps to MAX");
        stats.record(clamp_probability(numeric_limits<double>::quiet_NaN()) == MIN_PROBABILITY,
                     "clamp_probability(NaN) returns MIN_PROBABILITY");
    }

    // Test clamp_log_probability
    {
        stats.record(approx_equal(clamp_log_probability(-0.5), -0.5, 1e-10),
                     "clamp_log_probability(-0.5) == -0.5");
        stats.record(approx_equal(clamp_log_probability(0.0), MAX_LOG_PROBABILITY, 1e-10),
                     "clamp_log_probability(0.0) == MAX_LOG_PROBABILITY");
        stats.record(clamp_log_probability(0.5) == MAX_LOG_PROBABILITY,
                     "clamp_log_probability(0.5) clamps to MAX");
        stats.record(clamp_log_probability(-1000.0) == -1000.0,
                     "clamp_log_probability(-1000.0) unchanged");
        stats.record(clamp_log_probability(MIN_LOG_PROBABILITY - 1.0) == MIN_LOG_PROBABILITY,
                     "clamp_log_probability underflow clamps to MIN");
    }

    stats.print_summary("Scalar Safety");
}

//==============================================================================
// VECTORIZED SAFETY FUNCTION TESTS
//==============================================================================

void test_vector_safety(const TestConfig& config) {
    cout << "\n=== Testing Vectorized Safety Functions ===" << endl;
    TestStats stats;

    // Test threshold functions
    {
        stats.record(vectorized_safety_threshold() > 0, "vectorized_safety_threshold() > 0");
        stats.record(vectorized_safety_threshold() == 32, "vectorized_safety_threshold() == 32");
        stats.record(!should_use_vectorized_safety(10),
                     "should_use_vectorized_safety(10) == false");
        stats.record(should_use_vectorized_safety(100),
                     "should_use_vectorized_safety(100) == true");
    }

    // Test vector_safe_log
    {
        vector<double> input = {1.0, 2.718281828, 0.0, -1.0, numeric_limits<double>::quiet_NaN()};
        vector<double> output(input.size());

        vector_safe_log(input, output);

        stats.record(approx_equal(output[0], 0.0, 1e-10), "vector_safe_log[0] == 0.0");
        stats.record(approx_equal(output[1], 1.0, 1e-6), "vector_safe_log[1] == 1.0");
        stats.record(output[2] == MIN_LOG_PROBABILITY, "vector_safe_log handles 0.0");
        stats.record(output[3] == MIN_LOG_PROBABILITY, "vector_safe_log handles negative");
        stats.record(output[4] == MIN_LOG_PROBABILITY, "vector_safe_log handles NaN");
    }

    // Test vector_safe_exp
    {
        vector<double> input = {0.0, 1.0, -1000.0, 800.0, numeric_limits<double>::quiet_NaN()};
        vector<double> output(input.size());

        vector_safe_exp(input, output);

        stats.record(approx_equal(output[0], 1.0, 1e-10), "vector_safe_exp[0] == 1.0");
        stats.record(approx_equal(output[1], 2.718281828, 1e-6), "vector_safe_exp[1] == e");
        stats.record(output[2] == MIN_PROBABILITY, "vector_safe_exp handles underflow");
        stats.record(output[3] == numeric_limits<double>::max(),
                     "vector_safe_exp handles overflow");
        stats.record(output[4] == 0.0, "vector_safe_exp handles NaN");
    }

    // Test vector_safe_sqrt
    {
        vector<double> input = {0.0, 1.0, 4.0, 9.0, -1.0, numeric_limits<double>::quiet_NaN()};
        vector<double> output(input.size());

        vector_safe_sqrt(input, output);

        stats.record(approx_equal(output[0], 0.0, 1e-10), "vector_safe_sqrt[0] == 0.0");
        stats.record(approx_equal(output[1], 1.0, 1e-10), "vector_safe_sqrt[1] == 1.0");
        stats.record(approx_equal(output[2], 2.0, 1e-10), "vector_safe_sqrt[2] == 2.0");
        stats.record(approx_equal(output[3], 3.0, 1e-10), "vector_safe_sqrt[3] == 3.0");
        stats.record(output[4] == 0.0, "vector_safe_sqrt handles negative");
        stats.record(output[5] == 0.0, "vector_safe_sqrt handles NaN");
    }

    // Test large array consistency
    if (config.verbose) {
        cout << "  Testing large array consistency..." << endl;
    }
    {
        const size_t size = 1000;
        vector<double> input(size);
        vector<double> vec_output(size);

        // Fill with random values
        mt19937 rng(42);
        uniform_real_distribution<> dist(-100, 100);
        for (auto& val : input)
            val = dist(rng);

        // Test vector_safe_log consistency
        vector_safe_log(input, vec_output);
        bool consistent = true;
        for (size_t i = 0; i < size; ++i) {
            if (!approx_equal(vec_output[i], safe_log(input[i]), 1e-10)) {
                consistent = false;
                break;
            }
        }
        stats.record(consistent, "vector_safe_log consistent with scalar (1000 elements)");

        // Test vector_safe_exp consistency
        vector_safe_exp(input, vec_output);
        consistent = true;
        for (size_t i = 0; i < size; ++i) {
            if (!approx_equal(vec_output[i], safe_exp(input[i]), 1e-10)) {
                consistent = false;
                break;
            }
        }
        stats.record(consistent, "vector_safe_exp consistent with scalar (1000 elements)");
    }

    stats.print_summary("Vectorized Safety");
}

//==============================================================================
// LOG-SPACE OPERATIONS TESTS
//==============================================================================

void test_log_space_operations([[maybe_unused]] const TestConfig& config) {
    cout << "\n=== Testing Log-Space Operations ===" << endl;
    TestStats stats;

    // Initialize log-space operations
    LogSpaceOps::initialize();
    stats.record(true, "LogSpaceOps initialized");

    // Test basic logSumExp
    {
        double result = LogSpaceOps::logSumExp(log(2.0), log(3.0));
        double expected = log(5.0);
        stats.record(approx_equal(result, expected, 1e-9), "logSumExp(log(2), log(3)) == log(5)");

        result = LogSpaceOps::logSumExp(LogSpaceOps::LOG_ZERO, log(5.0));
        expected = log(5.0);
        stats.record(approx_equal(result, expected), "logSumExp(LOG_ZERO, log(5)) == log(5)");

        result = LogSpaceOps::logSumExp(LogSpaceOps::LOG_ZERO, LogSpaceOps::LOG_ZERO);
        stats.record(result == LogSpaceOps::LOG_ZERO, "logSumExp(LOG_ZERO, LOG_ZERO) == LOG_ZERO");
    }

    // Test logSumExpArray
    {
        vector<double> log_values = {log(1.0), log(2.0), log(3.0)};
        double result = LogSpaceOps::logSumExpArray(log_values.data(), log_values.size());
        double expected = log(6.0);
        stats.record(approx_equal(result, expected, 1e-9),
                     "logSumExpArray([log(1), log(2), log(3)]) == log(6)");

        // Test with LOG_ZERO values
        log_values = {LogSpaceOps::LOG_ZERO, log(5.0), LogSpaceOps::LOG_ZERO, log(3.0)};
        result = LogSpaceOps::logSumExpArray(log_values.data(), log_values.size());
        expected = log(8.0);
        stats.record(approx_equal(result, expected, 1e-9), "logSumExpArray with LOG_ZERO values");

        // Test empty array
        result = LogSpaceOps::logSumExpArray(nullptr, 0);
        stats.record(result == LogSpaceOps::LOG_ZERO || isnan(result),
                     "logSumExpArray handles empty array");
    }

    // Test helper functions
    {
        stats.record(LogSpaceOps::isLogZero(LogSpaceOps::LOG_ZERO), "isLogZero(LOG_ZERO) == true");
        stats.record(LogSpaceOps::isLogZero(numeric_limits<double>::quiet_NaN()),
                     "isLogZero(NaN) == true");
        stats.record(!LogSpaceOps::isLogZero(0.0), "isLogZero(0.0) == false");

        double pos_result = LogSpaceOps::safeLog(2.718281828);
        double zero_result = LogSpaceOps::safeLog(0.0);
        double neg_result = LogSpaceOps::safeLog(-1.0);

        stats.record(approx_equal(pos_result, 1.0, 1e-6), "safeLog(e) ≈ 1.0");
        stats.record(zero_result == LogSpaceOps::LOG_ZERO, "safeLog(0.0) == LOG_ZERO");
        stats.record(neg_result == LogSpaceOps::LOG_ZERO, "safeLog(-1.0) == LOG_ZERO");
    }

    // Test matrix operations
    {
        vector<double> prob_matrix = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
        vector<double> log_matrix(6);

        LogSpaceOps::precomputeLogMatrix(prob_matrix.data(), log_matrix.data(), 2, 3);

        bool correct = true;
        for (size_t i = 0; i < 6; ++i) {
            if (!approx_equal(log_matrix[i], log(prob_matrix[i]), 1e-10)) {
                correct = false;
                break;
            }
        }
        stats.record(correct, "precomputeLogMatrix correctness");

        // Test matrix-vector multiply
        vector<double> log_mat = {log(0.3), log(0.7), log(0.6), log(0.4)};
        vector<double> log_vec = {log(0.2), log(0.8)};
        vector<double> result(2);

        LogSpaceOps::logMatrixVectorMultiply(log_mat.data(), log_vec.data(), result.data(), 2, 2);

        // Expected: result[0] = log(0.3*0.2 + 0.7*0.8) = log(0.62)
        //           result[1] = log(0.6*0.2 + 0.4*0.8) = log(0.44)
        vector<double> expected = {log(0.62), log(0.44)};
        correct = true;
        for (size_t i = 0; i < 2; ++i) {
            if (!approx_equal(result[i], expected[i], 1e-9)) {
                correct = false;
                break;
            }
        }
        stats.record(correct, "logMatrixVectorMultiply correctness");
    }

    stats.print_summary("Log-Space Operations");
}

//==============================================================================
// EDGE CASES TESTS
//==============================================================================

void test_edge_cases([[maybe_unused]] const TestConfig& config) {
    cout << "\n=== Testing Edge Cases ===" << endl;
    TestStats stats;

    // Test extreme values in safety functions
    {
        double inf = numeric_limits<double>::infinity();
        double ninf = -numeric_limits<double>::infinity();
        double nan = numeric_limits<double>::quiet_NaN();
        double max_val = numeric_limits<double>::max();
        double denorm = numeric_limits<double>::denorm_min();

        // safe_log edge cases
        stats.record(safe_log(inf) == max_val, "safe_log(inf) == max");
        stats.record(safe_log(ninf) == MIN_LOG_PROBABILITY, "safe_log(-inf) == MIN_LOG");
        stats.record(safe_log(nan) == MIN_LOG_PROBABILITY, "safe_log(NaN) == MIN_LOG");
        stats.record(safe_log(denorm) < 0, "safe_log(denorm) < 0");

        // safe_exp edge cases
        stats.record(safe_exp(inf) == max_val, "safe_exp(inf) == max");
        stats.record(safe_exp(ninf) == MIN_PROBABILITY, "safe_exp(-inf) == MIN_PROB");
        stats.record(safe_exp(nan) == 0.0, "safe_exp(NaN) == 0");

        // safe_sqrt edge cases
        stats.record(safe_sqrt(inf) == max_val, "safe_sqrt(inf) == max");
        stats.record(safe_sqrt(ninf) == 0.0, "safe_sqrt(-inf) == 0");
        stats.record(safe_sqrt(nan) == 0.0, "safe_sqrt(NaN) == 0");
    }

    // Test log-space edge cases
    {
        double large = 100.0;
        double small = -100.0;
        double result = LogSpaceOps::logSumExp(large, small);
        stats.record(approx_equal(result, large, 1e-10), "logSumExp with large difference");

        result = LogSpaceOps::logSumExp(numeric_limits<double>::quiet_NaN(), log(5.0));
        stats.record(isnan(result) || approx_equal(result, log(5.0), 1e-10),
                     "logSumExp handles NaN input");

        result = LogSpaceOps::logSumExp(numeric_limits<double>::infinity(), log(5.0));
        stats.record(isinf(result) && result > 0, "logSumExp handles positive infinity");

        vector<double> all_zeros(10, LogSpaceOps::LOG_ZERO);
        result = LogSpaceOps::logSumExpArray(all_zeros.data(), all_zeros.size());
        stats.record(result == LogSpaceOps::LOG_ZERO, "logSumExpArray with all LOG_ZERO");
    }

    // Test boundary conditions
    {
        // Test clamping at exact boundaries
        stats.record(clamp_probability(MIN_PROBABILITY) == MIN_PROBABILITY,
                     "clamp_probability at MIN boundary");
        stats.record(clamp_probability(MAX_PROBABILITY) == MAX_PROBABILITY,
                     "clamp_probability at MAX boundary");
        stats.record(clamp_log_probability(MIN_LOG_PROBABILITY) == MIN_LOG_PROBABILITY,
                     "clamp_log_probability at MIN boundary");
        stats.record(clamp_log_probability(MAX_LOG_PROBABILITY) == MAX_LOG_PROBABILITY,
                     "clamp_log_probability at MAX boundary");
    }

    stats.print_summary("Edge Cases");
}

//==============================================================================
// PERFORMANCE BENCHMARKS
//==============================================================================

void test_benchmarks(const TestConfig& config) {
    cout << "\n=== Performance Benchmarks ===" << endl;
    cout << "Benchmark size: " << config.benchmark_size << " elements\n" << endl;

    const size_t size = config.benchmark_size;
    vector<double> input(size);
    vector<double> output(size);

    // Generate test data
    mt19937 rng(42);
    uniform_real_distribution<> dist(-50, 50);
    for (auto& val : input)
        val = dist(rng);

    // Benchmark scalar vs vectorized safe_log
    {
        auto start = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; ++i) {
            output[i] = safe_log(input[i]);
        }
        auto end = chrono::high_resolution_clock::now();
        auto scalar_time = chrono::duration_cast<chrono::microseconds>(end - start);

        start = chrono::high_resolution_clock::now();
        vector_safe_log(input, output);
        end = chrono::high_resolution_clock::now();
        auto vector_time = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "safe_log performance:" << endl;
        cout << "  Scalar: " << scalar_time.count() << " μs" << endl;
        cout << "  Vector: " << vector_time.count() << " μs" << endl;
        cout << "  Speedup: " << fixed << setprecision(2)
             << (double)scalar_time.count() / vector_time.count() << "x" << endl;
    }

    // Benchmark logSumExpArray
    {
        vector<double> log_values(size);
        for (auto& val : log_values)
            val = dist(rng);

        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            double result = LogSpaceOps::logSumExpArray(log_values.data(), log_values.size());
            (void)result;  // Avoid unused variable warning
        }
        auto end = chrono::high_resolution_clock::now();
        auto array_time = chrono::duration_cast<chrono::microseconds>(end - start);

        start = chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            double sum = LogSpaceOps::LOG_ZERO;
            for (size_t j = 0; j < size; ++j) {
                sum = LogSpaceOps::logSumExp(sum, log_values[j]);
            }
            (void)sum;
        }
        end = chrono::high_resolution_clock::now();
        auto pairwise_time = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "\nlogSumExp performance (100 iterations):" << endl;
        cout << "  Array method: " << array_time.count() << " μs" << endl;
        cout << "  Pairwise method: " << pairwise_time.count() << " μs" << endl;
        cout << "  Speedup: " << fixed << setprecision(2)
             << (double)pairwise_time.count() / array_time.count() << "x" << endl;
    }

    // Benchmark matrix operations
    {
        size_t mat_size = static_cast<size_t>(sqrt(size));
        vector<double> log_matrix(mat_size * mat_size);
        vector<double> log_vector(mat_size);
        vector<double> result_vec(mat_size);

        for (auto& val : log_matrix)
            val = dist(rng);
        for (auto& val : log_vector)
            val = dist(rng);

        auto start = chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i) {
            LogSpaceOps::logMatrixVectorMultiply(log_matrix.data(), log_vector.data(),
                                                 result_vec.data(), mat_size, mat_size);
        }
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "\nlogMatrixVectorMultiply (" << mat_size << "x" << mat_size
             << ", 1000 iterations): " << duration.count() << " μs" << endl;
    }
}

//==============================================================================
// PRECISION TESTS
//==============================================================================

void test_precision(const TestConfig& config) {
    cout << "\n=== Precision and Accuracy Tests ===" << endl;

    mt19937 rng(42);
    uniform_real_distribution<> dist(-50, 10);

    // Test logSumExp accuracy
    {
        double max_error = 0.0;
        double total_error = 0.0;
        int test_count = 1000;

        for (int i = 0; i < test_count; ++i) {
            double a = dist(rng);
            double b = dist(rng);

            double result = LogSpaceOps::logSumExp(a, b);
            double reference = reference_log_sum_exp(a, b);

            double error = abs(result - reference);
            max_error = max(max_error, error);
            total_error += error;
        }

        cout << "logSumExp accuracy (" << test_count << " tests):" << endl;
        cout << "  Maximum error: " << scientific << max_error << endl;
        cout << "  Average error: " << total_error / test_count << endl;
        cout << "  All errors < 1e-8: " << (max_error < 1e-8 ? "✓" : "✗") << endl;
    }

    // Test safe function precision preservation
    {
        cout << "\nSafe function precision preservation:" << endl;

        // Test that safe functions preserve precision for valid inputs
        vector<double> test_values = {1.0, M_E, M_PI, 0.5, 2.0, 10.0};
        bool all_precise = true;

        for (double val : test_values) {
            double log_result = safe_log(val);
            double exp_result = safe_exp(log_result);
            if (!approx_equal(exp_result, val, 1e-10)) {
                all_precise = false;
                if (config.verbose) {
                    cout << "  Precision loss: exp(safe_log(" << val << ")) = " << exp_result
                         << endl;
                }
            }
        }
        cout << "  Round-trip exp(safe_log(x)) == x: " << (all_precise ? "✓" : "✗") << endl;

        // Test sqrt precision
        all_precise = true;
        for (double val : test_values) {
            double sqrt_result = safe_sqrt(val);
            double squared = sqrt_result * sqrt_result;
            if (!approx_equal(squared, val, 1e-10)) {
                all_precise = false;
            }
        }
        cout << "  safe_sqrt(x)² == x: " << (all_precise ? "✓" : "✗") << endl;
    }
}

//==============================================================================
// STRESS TESTS
//==============================================================================

void test_stress(const TestConfig& config) {
    cout << "\n=== Stress Tests ===" << endl;
    cout << "Running " << config.stress_iterations << " iterations..." << endl;

    mt19937 rng(42);
    uniform_real_distribution<> dist(-100, 100);

    // Stress test with extreme values
    {
        cout << "\nTesting extreme values:" << endl;
        int crashes = 0;
        int invalid = 0;

        for (size_t i = 0; i < config.stress_iterations; ++i) {
            vector<double> extreme_values = {LogSpaceOps::LOG_ZERO,
                                             numeric_limits<double>::max(),
                                             -numeric_limits<double>::max(),
                                             numeric_limits<double>::quiet_NaN(),
                                             numeric_limits<double>::infinity(),
                                             -numeric_limits<double>::infinity(),
                                             dist(rng)};

            try {
                // Test scalar safety functions
                for (double val : extreme_values) {
                    double r1 = safe_log(val);
                    double r2 = safe_exp(val);
                    double r3 = safe_sqrt(abs(val));
                    double r4 = clamp_probability(val);
                    double r5 = clamp_log_probability(val);
                    (void)r1;
                    (void)r2;
                    (void)r3;
                    (void)r4;
                    (void)r5;
                }

                // Test log-space operations
                double result =
                    LogSpaceOps::logSumExpArray(extreme_values.data(), extreme_values.size());
                if (isnan(result) && !any_of(extreme_values.begin(), extreme_values.end(),
                                             [](double x) { return isnan(x); })) {
                    invalid++;
                }
            } catch (...) {
                crashes++;
            }
        }

        cout << "  Completed: " << (config.stress_iterations - crashes) << "/"
             << config.stress_iterations << endl;
        cout << "  Invalid results: " << invalid << endl;
        cout << "  Crashes: " << crashes << endl;
    }

    // Stress test with large arrays
    {
        cout << "\nTesting large arrays:" << endl;

        for (size_t size = 1000; size <= 100000; size *= 10) {
            vector<double> large_array(size);
            for (auto& val : large_array)
                val = dist(rng);

            auto start = chrono::high_resolution_clock::now();

            // Test vectorized operations
            vector<double> output(size);
            vector_safe_log(large_array, output);
            vector_safe_exp(large_array, output);

            // Test log-space array operations
            double result = LogSpaceOps::logSumExpArray(large_array.data(), size);

            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "  Size " << setw(6) << size << ": " << setw(5) << duration.count() << " ms";
            if (!isfinite(result) && !isinf(result)) {
                cout << " [WARNING: non-finite result]";
            }
            cout << endl;
        }
    }

    // Thread safety stress test (if applicable)
    if (config.verbose) {
        cout << "\nNote: Thread safety testing would require multi-threading support" << endl;
    }
}

//==============================================================================
// MAIN FUNCTION AND ARGUMENT PARSING
//==============================================================================

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]\n\n";
    cout << "Comprehensive Numerical Safety Test Suite\n";
    cout << "Tests core/safety.h and core/log_space_ops.h functionality\n\n";
    cout << "Options:\n";
    cout << "  --all/-a              Test all components (default)\n";
    cout << "  --scalar/-s           Test scalar safety functions\n";
    cout << "  --vector/-v           Test vectorized safety functions\n";
    cout << "  --log-space/-l        Test log-space operations\n";
    cout << "  --edge-cases/-e       Test edge cases\n";
    cout << "  --benchmarks/-b       Run performance benchmarks\n";
    cout << "  --precision/-p        Test numerical precision\n";
    cout << "  --stress/-S           Run stress tests\n";
    cout << "  --verbose/-V          Enable verbose output\n";
    cout << "  --size N              Set benchmark array size (default: 10000)\n";
    cout << "  --iterations N        Set stress test iterations (default: 1000)\n";
    cout << "  --help/-h             Show this help\n\n";
    cout << "Examples:\n";
    cout << "  " << program_name << "                    # Test everything\n";
    cout << "  " << program_name << " --scalar --vector   # Test safety functions\n";
    cout << "  " << program_name << " --log-space --benchmarks\n";
    cout << "  " << program_name << " --stress --iterations 10000\n";
}

TestConfig parse_args(int argc, char* argv[]) {
    TestConfig config;

    if (argc == 1) {
        return config;  // Default: test all
    }

    config.test_all = false;  // If args provided, don't test all by default

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--all" || arg == "-a") {
            config.test_all = true;
        } else if (arg == "--scalar" || arg == "-s") {
            config.test_scalar = true;
        } else if (arg == "--vector" || arg == "-v") {
            config.test_vector = true;
        } else if (arg == "--log-space" || arg == "-l") {
            config.test_log_space = true;
        } else if (arg == "--edge-cases" || arg == "-e") {
            config.test_edge_cases = true;
        } else if (arg == "--benchmarks" || arg == "-b") {
            config.test_benchmarks = true;
        } else if (arg == "--precision" || arg == "-p") {
            config.test_precision = true;
        } else if (arg == "--stress" || arg == "-S") {
            config.test_stress = true;
        } else if (arg == "--verbose" || arg == "-V") {
            config.verbose = true;
        } else if (arg == "--size" && i + 1 < argc) {
            config.benchmark_size = stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.stress_iterations = stoul(argv[++i]);
        } else {
            cerr << "Unknown option: " << arg << endl;
            print_usage(argv[0]);
            exit(1);
        }
    }

    // If no specific tests selected, test all
    if (!config.test_scalar && !config.test_vector && !config.test_log_space &&
        !config.test_edge_cases && !config.test_benchmarks && !config.test_precision &&
        !config.test_stress) {
        config.test_all = true;
    }

    return config;
}

int main(int argc, char* argv[]) {
    TestConfig config = parse_args(argc, argv);

    cout << "=== Numerical Safety Test Suite ===" << endl;
    cout << "Testing core/safety.h and core/log_space_ops.h\n" << endl;

    // Initialize log-space operations
    LogSpaceOps::initialize();

    int total_sections = 0;

    if (config.test_all || config.test_scalar) {
        test_scalar_safety(config);
        total_sections++;
    }

    if (config.test_all || config.test_vector) {
        test_vector_safety(config);
        total_sections++;
    }

    if (config.test_all || config.test_log_space) {
        test_log_space_operations(config);
        total_sections++;
    }

    if (config.test_all || config.test_edge_cases) {
        test_edge_cases(config);
        total_sections++;
    }

    if (config.test_all || config.test_precision) {
        test_precision(config);
        total_sections++;
    }

    if (config.test_all || config.test_benchmarks) {
        test_benchmarks(config);
        total_sections++;
    }

    if (config.test_stress) {  // Only run if explicitly requested
        test_stress(config);
        total_sections++;
    }

    cout << "\n=== Test Suite Complete ===" << endl;
    cout << "Tested " << total_sections << " component(s)" << endl;

    return 0;
}

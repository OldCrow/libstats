#include "../include/core/math_utils.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;
using namespace stats::detail;

// Helper function to check if two values are nearly equal
bool is_nearly_equal(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) <= tolerance;
}

void test_vector_erf() {
    cout << "Testing vector_erf..." << endl;

    // Create test data
    [[maybe_unused]] std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    const size_t test_size = 1000;
    std::vector<double> input_data(test_size);
    std::vector<double> output_data(test_size);
    std::vector<double> expected_data(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        input_data[i] = dis(gen);
    }

    // Test vector_erf against scalar erf
    vector_erf(input_data, output_data);

    // Calculate expected results using scalar function
    for (std::size_t i = 0; i < test_size; ++i) {
        expected_data[i] = std::erf(input_data[i]);
    }

    // Compare results
    for (std::size_t i = 0; i < test_size; ++i) {
        assert(is_nearly_equal(output_data[i], expected_data[i]));
    }

    cout << "   ✓ vector_erf correctness test passed" << endl;
}

void test_vector_erfc() {
    cout << "Testing vector_erfc..." << endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    const size_t test_size = 1000;
    std::vector<double> input_data(test_size);
    std::vector<double> output_data(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        input_data[i] = dis(gen);
    }

    vector_erfc(input_data, output_data);

    for (std::size_t i = 0; i < test_size; ++i) {
        assert(is_nearly_equal(output_data[i], std::erfc(input_data[i])));
    }

    cout << "   ✓ vector_erfc correctness test passed" << endl;
}

void test_vector_gamma_p() {
    cout << "Testing vector_gamma_p..." << endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> pos_dis(0.1, 5.0);

    const size_t test_size = 100;  // Smaller size for gamma functions
    std::vector<double> input_data(test_size);
    std::vector<double> output_data(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        input_data[i] = pos_dis(gen);
    }

    double a = 2.5;
    vector_gamma_p(a, input_data, output_data);

    for (std::size_t i = 0; i < test_size; ++i) {
        assert(is_nearly_equal(output_data[i], gamma_p(a, input_data[i]), 1e-8));
    }

    cout << "   ✓ vector_gamma_p correctness test passed" << endl;
}

void test_vector_beta_i() {
    cout << "Testing vector_beta_i..." << endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> unit_dis(0.01, 0.99);

    const size_t test_size = 100;
    std::vector<double> input_data(test_size);
    std::vector<double> output_data(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        input_data[i] = unit_dis(gen);
    }

    double a = 2.0, b = 3.0;
    vector_beta_i(input_data, a, b, output_data);

    for (std::size_t i = 0; i < test_size; ++i) {
        assert(is_nearly_equal(output_data[i], beta_i(input_data[i], a, b), 1e-8));
    }

    cout << "   ✓ vector_beta_i correctness test passed" << endl;
}

void test_vector_lgamma() {
    cout << "Testing vector_lgamma..." << endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> pos_dis(0.1, 10.0);

    const size_t test_size = 1000;
    std::vector<double> input_data(test_size);
    std::vector<double> output_data(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        input_data[i] = pos_dis(gen);
    }

    vector_lgamma(input_data, output_data);

    for (std::size_t i = 0; i < test_size; ++i) {
        assert(is_nearly_equal(output_data[i], std::lgamma(input_data[i])));
    }

    cout << "   ✓ vector_lgamma correctness test passed" << endl;
}

void test_vector_lbeta() {
    cout << "Testing vector_lbeta..." << endl;

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> pos_dis(0.1, 5.0);

    const size_t test_size = 100;
    std::vector<double> a_values(test_size);
    std::vector<double> b_values(test_size);
    std::vector<double> output_data(test_size);

    for (std::size_t i = 0; i < test_size; ++i) {
        a_values[i] = pos_dis(gen);
        b_values[i] = pos_dis(gen);
    }

    vector_lbeta(a_values, b_values, output_data);

    for (std::size_t i = 0; i < test_size; ++i) {
        assert(is_nearly_equal(output_data[i], lbeta(a_values[i], b_values[i])));
    }

    cout << "   ✓ vector_lbeta correctness test passed" << endl;
}

void test_threshold_functions() {
    cout << "Testing threshold functions..." << endl;

    // Test threshold functions
    assert(should_use_vectorized_math(1000));
    assert(!should_use_vectorized_math(1));

    [[maybe_unused]] std::size_t threshold = vectorized_math_threshold();
    assert(threshold > 0);
    assert(threshold < 1000);

    cout << "   ✓ Threshold functions test passed" << endl;
}

void test_empty_and_mismatched_inputs() {
    cout << "Testing empty and mismatched inputs..." << endl;

    // Test empty inputs
    std::vector<double> empty_input;
    std::vector<double> empty_output;

    vector_erf(empty_input, empty_output);
    assert(empty_output.empty());

    // Test mismatched sizes
    std::vector<double> small_input(5, 1.0);
    std::vector<double> large_output(10, 0.0);

    vector_erf(small_input, large_output);
    // Should not crash, output should remain unchanged
    assert(large_output.size() == 10);

    cout << "   ✓ Empty and mismatched inputs test passed" << endl;
}

void performance_benchmark() {
    cout << "Performance benchmark..." << endl;

    // Simple performance comparison
    const std::size_t large_size = 100000;
    std::vector<double> large_input(large_size);
    std::vector<double> large_output(large_size);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-2.0, 2.0);

    for (std::size_t i = 0; i < large_size; ++i) {
        large_input[i] = dis(gen);
    }

    // Time vectorized version
    auto start = std::chrono::high_resolution_clock::now();
    vector_erf(large_input, large_output);
    auto end = std::chrono::high_resolution_clock::now();

    auto vectorized_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Time scalar version
    start = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < large_size; ++i) {
        large_output[i] = std::erf(large_input[i]);
    }
    end = std::chrono::high_resolution_clock::now();

    auto scalar_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Print timing results (for informational purposes)
    cout << "   Vectorized erf time: " << vectorized_time.count() << " μs" << endl;
    cout << "   Scalar erf time: " << scalar_time.count() << " μs" << endl;

    // Just verify both completed successfully
    assert(vectorized_time.count() > 0);
    assert(scalar_time.count() > 0);

    cout << "   ✓ Performance benchmark completed" << endl;
}

int main() {
    cout << "=== Vectorized Math Functions Test ===" << endl;

    try {
        test_vector_erf();
        test_vector_erfc();
        test_vector_gamma_p();
        test_vector_beta_i();
        test_vector_lgamma();
        test_vector_lbeta();
        test_threshold_functions();
        test_empty_and_mismatched_inputs();
        performance_benchmark();

        cout << "\n=== ALL VECTORIZED MATH TESTS PASSED! ===" << endl;
        cout << "✓ vector_erf works correctly" << endl;
        cout << "✓ vector_erfc works correctly" << endl;
        cout << "✓ vector_gamma_p works correctly" << endl;
        cout << "✓ vector_beta_i works correctly" << endl;
        cout << "✓ vector_lgamma works correctly" << endl;
        cout << "✓ vector_lbeta works correctly" << endl;
        cout << "✓ Threshold functions work correctly" << endl;
        cout << "✓ Empty/mismatched input handling works" << endl;
        cout << "✓ Performance benchmarking completed" << endl;

        return 0;

    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

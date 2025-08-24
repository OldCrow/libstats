// Use focused header for SIMD testing
#include "../include/platform/simd.h"

#include <iostream>

using namespace std;
using namespace stats;

/**
 * @brief Comprehensive test for SIMD operations including AVX512 and FMA
 *
 * This test verifies that all SIMD code paths compile correctly and produce
 * accurate results, even if the runtime CPU doesn't support all instruction sets.
 */

// Test helper function to compare floating point results
bool close_enough(double a, double b, double tolerance = 1e-12) {
    // Handle NaN comparisons: both NaN should be considered equal
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    // If only one is NaN, they're not equal
    if (std::isnan(a) || std::isnan(b)) {
        return false;
    }
    // Handle infinite values
    if (std::isinf(a) && std::isinf(b)) {
        return (a > 0) == (b > 0);  // Same sign infinity
    }
    // Regular comparison
    return std::abs(a - b) < tolerance;
}

// Test helper function to compare vectors
bool vectors_equal(const vector<double>& a, const vector<double>& b, double tolerance = 1e-12) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!close_enough(a[i], b[i], tolerance)) {
            cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << endl;
            return false;
        }
    }
    return true;
}

// Test basic vector operations
void test_vector_operations() {
    cout << "\n=== TESTING VECTOR OPERATIONS ===" << endl;

    // Test different sizes to ensure all code paths are tested
    vector<size_t> test_sizes = {1, 2, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 1000};

    for (size_t size : test_sizes) {
        cout << "Testing size " << size << ": ";

        // Generate test data
        vector<double> a(size), b(size), result(size), expected(size);

        // Fill with predictable test data
        for (size_t i = 0; i < size; ++i) {
            a[i] = static_cast<double>(i) + 1.0;
            b[i] = static_cast<double>(i) * 0.5 + 2.0;
        }

        // Test dot product
        double dot_result = arch::simd::VectorOps::dot_product(a.data(), b.data(), size);
        double dot_expected = 0.0;
        for (size_t i = 0; i < size; ++i) {
            dot_expected += a[i] * b[i];
        }

        if (!close_enough(dot_result, dot_expected)) {
            cout << "FAILED (dot product)" << endl;
            continue;
        }

        // Test vector addition
        arch::simd::VectorOps::vector_add(a.data(), b.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = a[i] + b[i];
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector add)" << endl;
            continue;
        }

        // Test vector subtraction
        arch::simd::VectorOps::vector_subtract(a.data(), b.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = a[i] - b[i];
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector subtract)" << endl;
            continue;
        }

        // Test vector multiplication
        arch::simd::VectorOps::vector_multiply(a.data(), b.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = a[i] * b[i];
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector multiply)" << endl;
            continue;
        }

        // Test scalar multiplication
        double scalar = 3.14159;
        arch::simd::VectorOps::scalar_multiply(a.data(), scalar, result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = a[i] * scalar;
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (scalar multiply)" << endl;
            continue;
        }

        // Test scalar addition
        arch::simd::VectorOps::scalar_add(a.data(), scalar, result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = a[i] + scalar;
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (scalar add)" << endl;
            continue;
        }

        cout << "PASSED" << endl;
    }
}

// Test transcendental functions
void test_transcendental_functions() {
    cout << "\n=== TESTING TRANSCENDENTAL FUNCTIONS ===" << endl;

    vector<size_t> test_sizes = {1, 8, 16, 32, 100};

    for (size_t size : test_sizes) {
        cout << "Testing size " << size << ": ";

        // Generate test data (small positive values for safety)
        vector<double> values(size), result(size), expected(size);

        for (size_t i = 0; i < size; ++i) {
            values[i] = 0.1 + static_cast<double>(i) * 0.1;
        }

        // Test vector_exp
        arch::simd::VectorOps::vector_exp(values.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = std::exp(values[i]);
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector_exp)" << endl;
            continue;
        }

        // Test vector_log
        arch::simd::VectorOps::vector_log(values.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = std::log(values[i]);
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector_log)" << endl;
            continue;
        }

        // Test vector_pow
        double exponent = 2.5;
        arch::simd::VectorOps::vector_pow(values.data(), exponent, result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = std::pow(values[i], exponent);
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector_pow)" << endl;
            continue;
        }

        // Test vector_erf (with smaller values for better accuracy)
        vector<double> erf_values(size);
        if (size == 1) {
            erf_values[0] = 0.0;  // Use a safe value for size=1
        } else {
            for (size_t i = 0; i < size; ++i) {
                erf_values[i] = -2.0 + static_cast<double>(i) * 4.0 / static_cast<double>(size - 1);
            }
        }

        arch::simd::VectorOps::vector_erf(erf_values.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i) {
            expected[i] = std::erf(erf_values[i]);
        }

        if (!vectors_equal(result, expected)) {
            cout << "FAILED (vector_erf)" << endl;
            continue;
        }

        cout << "PASSED" << endl;
    }
}

// Test SIMD width detection
void test_simd_width_detection() {
    cout << "\n=== TESTING SIMD WIDTH DETECTION ===" << endl;

    cout << "Compile-time SIMD widths:" << endl;
    cout << "  Double vector width: " << arch::simd::double_vector_width() << endl;
    cout << "  Float vector width: " << arch::simd::float_vector_width() << endl;
    cout << "  Optimal alignment: " << arch::simd::optimal_alignment() << " bytes" << endl;
    cout << "  Feature string: " << arch::simd::feature_string() << endl;

    cout << "\nRuntime SIMD widths:" << endl;
    cout << "  Double vector width: " << arch::optimal_double_width() << endl;
    cout << "  Float vector width: " << arch::optimal_float_width() << endl;
    cout << "  Optimal alignment: " << arch::optimal_alignment() << " bytes" << endl;
    cout << "  Best SIMD level: " << arch::best_simd_level() << endl;

    // Test minimum SIMD sizes
    cout << "\nMinimum SIMD sizes:" << endl;
    cout << "  Min SIMD size: " << arch::simd::VectorOps::min_simd_size() << endl;
    cout << "  Should use SIMD (size 1): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(1) ? "YES" : "NO") << endl;
    cout << "  Should use SIMD (size 8): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(8) ? "YES" : "NO") << endl;
    cout << "  Should use SIMD (size 16): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(16) ? "YES" : "NO") << endl;
    cout << "  Should use SIMD (size 32): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(32) ? "YES" : "NO") << endl;
}

// Test compile-time instruction set detection
void test_compile_time_detection() {
    cout << "\n=== TESTING COMPILE-TIME INSTRUCTION SET DETECTION ===" << endl;

    cout << "Compile-time instruction set support:" << endl;

#ifdef LIBSTATS_HAS_SSE2
    cout << "  SSE2: SUPPORTED" << endl;
#else
    cout << "  SSE2: NOT SUPPORTED" << endl;
#endif

#ifdef LIBSTATS_HAS_SSE4_1
    cout << "  SSE4.1: SUPPORTED" << endl;
#else
    cout << "  SSE4.1: NOT SUPPORTED" << endl;
#endif

#ifdef LIBSTATS_HAS_AVX
    cout << "  AVX: SUPPORTED" << endl;
#else
    cout << "  AVX: NOT SUPPORTED" << endl;
#endif

#ifdef LIBSTATS_HAS_AVX2
    cout << "  AVX2: SUPPORTED" << endl;
#else
    cout << "  AVX2: NOT SUPPORTED" << endl;
#endif

#ifdef LIBSTATS_HAS_AVX512
    cout << "  AVX512: SUPPORTED" << endl;
#else
    cout << "  AVX512: NOT SUPPORTED" << endl;
#endif

#ifdef __FMA__
    cout << "  FMA: SUPPORTED" << endl;
#else
    cout << "  FMA: NOT SUPPORTED" << endl;
#endif

#ifdef LIBSTATS_HAS_NEON
    cout << "  NEON: SUPPORTED" << endl;
#else
    cout << "  NEON: NOT SUPPORTED" << endl;
#endif

#ifdef LIBSTATS_APPLE_SILICON
    cout << "  Apple Silicon: SUPPORTED" << endl;
#else
    cout << "  Apple Silicon: NOT SUPPORTED" << endl;
#endif
}

// Test runtime instruction set detection
void test_runtime_detection() {
    cout << "\n=== TESTING RUNTIME INSTRUCTION SET DETECTION ===" << endl;

    cout << "Runtime instruction set support:" << endl;
    cout << "  SSE2: " << (arch::supports_sse2() ? "SUPPORTED" : "NOT SUPPORTED") << endl;
    cout << "  SSE4.1: " << (arch::supports_sse4_1() ? "SUPPORTED" : "NOT SUPPORTED") << endl;
    cout << "  AVX: " << (arch::supports_avx() ? "SUPPORTED" : "NOT SUPPORTED") << endl;
    cout << "  AVX2: " << (arch::supports_avx2() ? "SUPPORTED" : "NOT SUPPORTED") << endl;
    cout << "  AVX512: " << (arch::supports_avx512() ? "SUPPORTED" : "NOT SUPPORTED") << endl;
    cout << "  FMA: " << (arch::supports_fma() ? "SUPPORTED" : "NOT SUPPORTED") << endl;
    cout << "  NEON: " << (arch::supports_neon() ? "SUPPORTED" : "NOT SUPPORTED") << endl;

    // Get detailed CPU information
    const auto& features = arch::get_features();
    cout << "\nCPU Information:" << endl;
    cout << "  Vendor: " << features.vendor << endl;
    cout << "  Brand: " << features.brand << endl;
    cout << "  Family: " << features.family << ", Model: " << features.model
         << ", Stepping: " << features.stepping << endl;
}

// Performance benchmark
void benchmark_performance() {
    cout << "\n=== PERFORMANCE BENCHMARK ===" << endl;

    const size_t size = 1000000;
    const int iterations = 100;

    vector<double> a(size, 1.5);
    vector<double> b(size, 2.5);
    vector<double> result(size);

    // Benchmark vector addition
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        arch::simd::VectorOps::vector_add(a.data(), b.data(), result.data(), size);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Vector addition (" << iterations << " iterations, " << size
         << " elements): " << duration.count() << " microseconds" << endl;

    // Benchmark dot product
    start = chrono::high_resolution_clock::now();
    double dot_result = 0.0;
    for (int i = 0; i < iterations; ++i) {
        dot_result += arch::simd::VectorOps::dot_product(a.data(), b.data(), size);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Dot product (" << iterations << " iterations, " << size
         << " elements): " << duration.count() << " microseconds" << endl;
    cout << "Final dot product result: " << dot_result / iterations << endl;

    // Benchmark transcendental functions
    vector<double> values(size);
    for (size_t i = 0; i < size; ++i) {
        values[i] = 0.1 + static_cast<double>(i) * 0.001;
    }

    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        arch::simd::VectorOps::vector_exp(values.data(), result.data(), size);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Vector exp (" << iterations << " iterations, " << size
         << " elements): " << duration.count() << " microseconds" << endl;
}

int main() {
    cout << "=== SIMD OPERATIONS COMPREHENSIVE TEST ===" << endl;
    cout << "=========================================" << endl;

    try {
        test_compile_time_detection();
        test_runtime_detection();
        test_simd_width_detection();
        test_vector_operations();
        test_transcendental_functions();
        benchmark_performance();

        cout << "\n=== ALL TESTS COMPLETED SUCCESSFULLY ===" << endl;
        return 0;

    } catch (const exception& e) {
        cout << "\nError: " << e.what() << endl;
        return 1;
    }
}

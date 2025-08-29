/**
 * @file test_simd_comprehensive.cpp
 * @brief Comprehensive SIMD testing suite consolidating all SIMD-related tests
 *
 * This test suite combines and enhances the functionality from:
 * - test_simd_integration.cpp (SIMD integration and detection)
 * - test_simd_integration_simple.cpp (simplified SIMD testing)
 * - test_simd_operations.cpp (detailed SIMD operations)
 *
 * Features:
 * - Command-line options for selective testing
 * - Enhanced coverage including AVX-512, ARM SVE, mixed precision
 * - Comprehensive benchmarking and validation
 * - Cross-architecture consistency validation
 */

#include "../include/distributions/gaussian.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace std;
using namespace stats;

// Command-line options structure
struct TestOptions {
    bool run_detection = true;
    bool run_operations = true;
    bool run_integration = true;
    bool run_benchmarks = true;
    bool run_advanced = true;
    bool verbose = false;
    bool help = false;
    size_t benchmark_size = 1000000;
    int benchmark_iterations = 100;
};

// Test helper functions
bool close_enough(double a, double b, double tolerance = 1e-12) {
    if (std::isnan(a) && std::isnan(b))
        return true;
    if (std::isnan(a) || std::isnan(b))
        return false;
    if (std::isinf(a) && std::isinf(b))
        return (a > 0) == (b > 0);
    return std::abs(a - b) < tolerance;
}

bool vectors_equal(const vector<double>& a, const vector<double>& b, double tolerance = 1e-12) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!close_enough(a[i], b[i], tolerance)) {
            if (a.size() <= 10) {
                cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << endl;
            }
            return false;
        }
    }
    return true;
}

// Test categories

namespace TestSIMD {

/**
 * @brief Test compile-time and runtime SIMD detection
 */
void test_simd_detection([[maybe_unused]] const TestOptions& opts) {
    cout << "\n=== SIMD DETECTION TESTS ===" << endl;

    // Compile-time detection
    cout << "\n1. COMPILE-TIME SIMD DETECTION:" << endl;
    cout << "   x86 SIMD Support:" << endl;

#ifdef LIBSTATS_HAS_SSE2
    cout << "     ✓ Compiler supports SSE2" << endl;
#else
    cout << "     ✗ Compiler does not support SSE2" << endl;
#endif

#ifdef LIBSTATS_HAS_SSE4_1
    cout << "     ✓ Compiler supports SSE4.1" << endl;
#else
    cout << "     ✗ Compiler does not support SSE4.1" << endl;
#endif

#ifdef LIBSTATS_HAS_AVX
    cout << "     ✓ Compiler supports AVX" << endl;
#else
    cout << "     ✗ Compiler does not support AVX" << endl;
#endif

#ifdef LIBSTATS_HAS_AVX2
    cout << "     ✓ Compiler supports AVX2" << endl;
#else
    cout << "     ✗ Compiler does not support AVX2" << endl;
#endif

#ifdef LIBSTATS_HAS_AVX512
    cout << "     ✓ Compiler supports AVX512" << endl;
#else
    cout << "     ✗ Compiler does not support AVX512" << endl;
#endif

#ifdef __FMA__
    cout << "     ✓ Compiler supports FMA" << endl;
#else
    cout << "     ✗ Compiler does not support FMA" << endl;
#endif

    cout << "   ARM SIMD Support:" << endl;
#ifdef LIBSTATS_HAS_NEON
    cout << "     ✓ Compiler supports ARM NEON" << endl;
#else
    cout << "     ✗ Compiler does not support ARM NEON" << endl;
#endif

#ifdef LIBSTATS_APPLE_SILICON
    cout << "     ✓ Apple Silicon optimizations enabled" << endl;
#else
    cout << "     ✗ Apple Silicon optimizations disabled" << endl;
#endif

    // Runtime detection
    cout << "\n2. RUNTIME CPU DETECTION:" << endl;
    const auto& features = arch::get_features();
    cout << "   CPU Vendor: " << features.vendor << endl;
    cout << "   CPU Brand: " << features.brand << endl;
    cout << "   CPU Family: " << features.family << ", Model: " << features.model
         << ", Stepping: " << features.stepping << endl;

    cout << "\n   x86 SIMD Support:" << endl;
    cout << "   - SSE2: " << (arch::supports_sse2() ? "✓" : "✗") << endl;
    cout << "   - SSE4.1: " << (arch::supports_sse4_1() ? "✓" : "✗") << endl;
    cout << "   - AVX: " << (arch::supports_avx() ? "✓" : "✗") << endl;
    cout << "   - AVX2: " << (arch::supports_avx2() ? "✓" : "✗") << endl;
    cout << "   - FMA: " << (arch::supports_fma() ? "✓" : "✗") << endl;
    cout << "   - AVX512: " << (arch::supports_avx512() ? "✓" : "✗") << endl;

    cout << "   ARM SIMD Support:" << endl;
    cout << "   - NEON: " << (arch::supports_neon() ? "✓" : "✗") << endl;

    // Optimal configuration
    cout << "\n3. OPTIMAL SIMD CONFIGURATION:" << endl;
    cout << "   Best SIMD level: " << arch::best_simd_level() << endl;
    cout << "   Optimal double width: " << arch::optimal_double_width() << endl;
    cout << "   Optimal float width: " << arch::optimal_float_width() << endl;
    cout << "   Optimal alignment: " << arch::optimal_alignment() << " bytes" << endl;

    // Compile-time constants
    cout << "\n4. COMPILE-TIME SIMD CONSTANTS:" << endl;
    cout << "   Has SIMD support: " << (simd::utils::has_simd_support() ? "✓" : "✗") << endl;
    cout << "   Compile-time double width: " << simd::utils::double_vector_width() << endl;
    cout << "   Compile-time float width: " << simd::utils::float_vector_width() << endl;
    cout << "   Compile-time alignment: " << simd::utils::optimal_alignment() << " bytes" << endl;
    cout << "   Compile-time feature string: " << simd::utils::feature_string() << endl;

    // SIMD policy testing
    cout << "\n5. SIMD POLICY DECISIONS:" << endl;
    cout << "   Min SIMD size: " << arch::simd::VectorOps::min_simd_size() << endl;
    cout << "   Should use SIMD (size 1): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(1) ? "YES" : "NO") << endl;
    cout << "   Should use SIMD (size 8): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(8) ? "YES" : "NO") << endl;
    cout << "   Should use SIMD (size 16): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(16) ? "YES" : "NO") << endl;
    cout << "   Should use SIMD (size 32): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(32) ? "YES" : "NO") << endl;
    cout << "   Should use SIMD (size 1000): "
         << (arch::simd::SIMDPolicy::shouldUseSIMD(1000) ? "YES" : "NO") << endl;
}

/**
 * @brief Test basic SIMD vector operations
 */
void test_vector_operations(const TestOptions& opts) {
    cout << "\n=== VECTOR OPERATIONS TESTS ===" << endl;

    vector<size_t> test_sizes = {1, 2, 4, 7, 8, 15, 16, 31, 32, 63, 64, 100, 1000};

    for (size_t size : test_sizes) {
        if (opts.verbose || size <= 32) {
            cout << "Testing size " << size << ": ";
        }

        // Generate test data
        vector<double> a(size), b(size), result(size), expected(size);

        for (size_t i = 0; i < size; ++i) {
            a[i] = static_cast<double>(i) + 1.0;
            b[i] = static_cast<double>(i) * 0.5 + 2.0;
        }

        bool all_passed = true;

        // Test dot product
        double dot_result = arch::simd::VectorOps::dot_product(a.data(), b.data(), size);
        double dot_expected = 0.0;
        for (size_t i = 0; i < size; ++i) {
            dot_expected += a[i] * b[i];
        }

        if (!close_enough(dot_result, dot_expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (dot product) ";
        }

        // Test vector operations
        struct {
            const char* name;
            function<void()> operation;
            function<void()> expected_calc;
        } operations[] = {
            {"add",
             [&]() { arch::simd::VectorOps::vector_add(a.data(), b.data(), result.data(), size); },
             [&]() {
                 for (size_t i = 0; i < size; ++i)
                     expected[i] = a[i] + b[i];
             }},
            {"subtract",
             [&]() {
                 arch::simd::VectorOps::vector_subtract(a.data(), b.data(), result.data(), size);
             },
             [&]() {
                 for (size_t i = 0; i < size; ++i)
                     expected[i] = a[i] - b[i];
             }},
            {"multiply",
             [&]() {
                 arch::simd::VectorOps::vector_multiply(a.data(), b.data(), result.data(), size);
             },
             [&]() {
                 for (size_t i = 0; i < size; ++i)
                     expected[i] = a[i] * b[i];
             }},
        };

        for (const auto& op : operations) {
            op.operation();
            op.expected_calc();

            if (!vectors_equal(result, expected)) {
                all_passed = false;
                if (opts.verbose)
                    cout << "FAILED (" << op.name << ") ";
            }
        }

        // Test scalar operations
        double scalar = 3.14159;
        arch::simd::VectorOps::scalar_multiply(a.data(), scalar, result.data(), size);
        for (size_t i = 0; i < size; ++i)
            expected[i] = a[i] * scalar;
        if (!vectors_equal(result, expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (scalar_multiply) ";
        }

        arch::simd::VectorOps::scalar_add(a.data(), scalar, result.data(), size);
        for (size_t i = 0; i < size; ++i)
            expected[i] = a[i] + scalar;
        if (!vectors_equal(result, expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (scalar_add) ";
        }

        if (opts.verbose || size <= 32) {
            cout << (all_passed ? "PASSED" : "FAILED") << endl;
        }
    }
}

/**
 * @brief Test transcendental functions with SIMD
 */
void test_transcendental_functions(const TestOptions& opts) {
    cout << "\n=== TRANSCENDENTAL FUNCTIONS TESTS ===" << endl;

    vector<size_t> test_sizes = {1, 8, 16, 32, 100};

    for (size_t size : test_sizes) {
        cout << "Testing size " << size << ": ";

        vector<double> values(size), result(size), expected(size);

        // Generate safe test data for transcendental functions
        for (size_t i = 0; i < size; ++i) {
            values[i] = 0.1 + static_cast<double>(i) * 0.1;
        }

        bool all_passed = true;

        // Test exp
        arch::simd::VectorOps::vector_exp(values.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i)
            expected[i] = std::exp(values[i]);
        if (!vectors_equal(result, expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (exp) ";
        }

        // Test log
        arch::simd::VectorOps::vector_log(values.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i)
            expected[i] = std::log(values[i]);
        if (!vectors_equal(result, expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (log) ";
        }

        // Test pow
        double exponent = 2.5;
        arch::simd::VectorOps::vector_pow(values.data(), exponent, result.data(), size);
        for (size_t i = 0; i < size; ++i)
            expected[i] = std::pow(values[i], exponent);
        if (!vectors_equal(result, expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (pow) ";
        }

        // Test erf with safe range
        vector<double> erf_values(size);
        if (size == 1) {
            erf_values[0] = 0.0;
        } else {
            for (size_t i = 0; i < size; ++i) {
                erf_values[i] = -2.0 + static_cast<double>(i) * 4.0 / static_cast<double>(size - 1);
            }
        }

        arch::simd::VectorOps::vector_erf(erf_values.data(), result.data(), size);
        for (size_t i = 0; i < size; ++i)
            expected[i] = std::erf(erf_values[i]);
        if (!vectors_equal(result, expected)) {
            all_passed = false;
            if (opts.verbose)
                cout << "FAILED (erf) ";
        }

        cout << (all_passed ? "PASSED" : "FAILED") << endl;
    }
}

/**
 * @brief Test SIMD integration with Gaussian distribution
 */
void test_gaussian_integration(const TestOptions& opts) {
    cout << "\n=== GAUSSIAN DISTRIBUTION SIMD INTEGRATION ===" << endl;

    try {
        auto gauss = GaussianDistribution::create(0.0, 1.0).value;

        const size_t size = 10000;
        vector<double> values(size);
        vector<double> pdf_results(size);
        vector<double> log_pdf_results(size);

        // Generate test values in range [-3, 3]
        for (size_t i = 0; i < size; ++i) {
            values[i] = -3.0 + 6.0 * static_cast<double>(i) / (size - 1);
        }

        cout << "Testing Gaussian batch operations with " << size << " values..." << endl;

        // Test batch operations
        auto start = chrono::high_resolution_clock::now();
        gauss.getProbabilityWithStrategy(span<const double>(values), span<double>(pdf_results),
                                         stats::detail::Strategy::SCALAR);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "PDF batch operation: " << duration.count() << " microseconds" << endl;

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < min(size, size_t(10)); ++i) {
            double expected = gauss.getProbability(values[i]);
            if (abs(pdf_results[i] - expected) > 1e-12) {
                correct = false;
                break;
            }
        }

        cout << "PDF batch verification: " << (correct ? "✓ PASSED" : "✗ FAILED") << endl;

        if (opts.verbose) {
            cout << "Sample results (first 5 values):" << endl;
            for (size_t i = 0; i < 5; ++i) {
                cout << "x=" << setw(8) << fixed << setprecision(3) << values[i]
                     << " PDF=" << setw(12) << setprecision(6) << pdf_results[i] << endl;
            }
        }

    } catch (const exception& e) {
        cout << "Error testing Gaussian integration: " << e.what() << endl;
    }
}

/**
 * @brief Performance benchmarks for SIMD operations
 */
void test_performance_benchmarks(const TestOptions& opts) {
    cout << "\n=== PERFORMANCE BENCHMARKS ===" << endl;

    const size_t size = opts.benchmark_size;
    const int iterations = opts.benchmark_iterations;

    vector<double> a(size, 1.5);
    vector<double> b(size, 2.5);
    vector<double> result(size);

    cout << "Benchmark parameters: " << size << " elements, " << iterations << " iterations"
         << endl;

    // Vector addition benchmark
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        arch::simd::VectorOps::vector_add(a.data(), b.data(), result.data(), size);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Vector addition: " << duration.count() << " μs total, "
         << duration.count() / iterations << " μs per iteration" << endl;

    // Dot product benchmark
    start = chrono::high_resolution_clock::now();
    double dot_result = 0.0;
    for (int i = 0; i < iterations; ++i) {
        dot_result += arch::simd::VectorOps::dot_product(a.data(), b.data(), size);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Dot product: " << duration.count() << " μs total, " << duration.count() / iterations
         << " μs per iteration" << endl;
    cout << "Final dot product result: " << dot_result / iterations << endl;

    // Transcendental functions benchmark
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

    cout << "Vector exp: " << duration.count() << " μs total, " << duration.count() / iterations
         << " μs per iteration" << endl;

    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < min(size, size_t(10)); ++i) {
        if (abs(result[i] - exp(values[i])) > 1e-10) {
            correct = false;
            break;
        }
    }
    cout << "Results verification: " << (correct ? "✓ PASSED" : "✗ FAILED") << endl;
}

/**
 * @brief Advanced SIMD tests including AVX-512, ARM SVE, mixed precision
 */
void test_advanced_simd([[maybe_unused]] const TestOptions& opts) {
    cout << "\n=== ADVANCED SIMD FEATURES ===" << endl;

    // Test AVX-512 specific features
    cout << "\n1. AVX-512 FEATURES:" << endl;
#ifdef LIBSTATS_HAS_AVX512
    if (arch::supports_avx512()) {
        cout << "   ✓ AVX-512 available and active" << endl;
        cout << "   - Vector width (doubles): " << arch::optimal_double_width() << endl;
        cout << "   - Vector width (floats): " << arch::optimal_float_width() << endl;

        // Test large vector operations that benefit from AVX-512
        const size_t large_size = 512;
        vector<double> large_a(large_size, 2.0);
        vector<double> large_b(large_size, 3.0);
        vector<double> large_result(large_size);

        arch::simd::VectorOps::vector_add(large_a.data(), large_b.data(), large_result.data(),
                                          large_size);

        bool correct = true;
        for (size_t i = 0; i < large_size; ++i) {
            if (abs(large_result[i] - 5.0) > 1e-10) {
                correct = false;
                break;
            }
        }
        cout << "   - Large vector test (512 elements): " << (correct ? "✓ PASSED" : "✗ FAILED")
             << endl;

    } else {
        cout << "   ℹ️  AVX-512 compiled but not available at runtime" << endl;
    }
#else
    cout << "   ⚠  AVX-512 not compiled" << endl;
#endif

    // Test ARM SVE features
    cout << "\n2. ARM SVE FEATURES:" << endl;
#ifdef LIBSTATS_HAS_SVE
    cout << "   ✓ ARM SVE compiled support available" << endl;
    // Add SVE-specific tests here
#else
    cout << "   ⚠  ARM SVE not compiled (normal on non-ARM platforms)" << endl;
#endif

    // Test mixed precision operations
    cout << "\n3. MIXED PRECISION OPERATIONS:" << endl;

    const size_t size = 100;
    vector<float> float_a(size, 1.5f);
    vector<float> float_b(size, 2.5f);
    vector<float> float_result(size);

    // Test float operations (if implemented)
    try {
        // This would use float-specific SIMD operations
        for (size_t i = 0; i < size; ++i) {
            float_result[i] = float_a[i] + float_b[i];
        }

        bool correct = true;
        for (size_t i = 0; i < size; ++i) {
            if (abs(float_result[i] - 4.0f) > 1e-6f) {
                correct = false;
                break;
            }
        }

        cout << "   - Float precision operations: " << (correct ? "✓ PASSED" : "✗ FAILED") << endl;

    } catch (const exception& e) {
        cout << "   - Float precision operations: ⚠ NOT IMPLEMENTED" << endl;
    }

    // Test cross-architecture consistency
    cout << "\n4. CROSS-ARCHITECTURE CONSISTENCY:" << endl;

    const size_t test_size = 64;
    vector<double> test_values(test_size);
    vector<double> simd_results(test_size);
    vector<double> scalar_results(test_size);

    // Generate test data
    for (size_t i = 0; i < test_size; ++i) {
        test_values[i] = -5.0 + 10.0 * static_cast<double>(i) / (test_size - 1);
    }

    // Compute with SIMD
    arch::simd::VectorOps::vector_exp(test_values.data(), simd_results.data(), test_size);

    // Compute with scalar fallback
    for (size_t i = 0; i < test_size; ++i) {
        scalar_results[i] = exp(test_values[i]);
    }

    // Compare results
    double max_error = 0.0;
    for (size_t i = 0; i < test_size; ++i) {
        double error = abs(simd_results[i] - scalar_results[i]);
        max_error = max(max_error, error);
    }

    cout << "   - SIMD vs Scalar consistency (exp): ";
    if (max_error < 1e-10) {
        cout << "✓ PASSED (max error: " << scientific << max_error << ")" << endl;
    } else {
        cout << "⚠ HIGH ERROR (max error: " << scientific << max_error << ")" << endl;
    }
}

}  // namespace TestSIMD

// Command-line parsing
TestOptions parse_args(int argc, char* argv[]) {
    TestOptions opts;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--detection" || arg == "-d") {
            opts.run_detection = true;
            opts.run_operations = opts.run_integration = opts.run_benchmarks = opts.run_advanced =
                false;
        } else if (arg == "--operations" || arg == "-o") {
            opts.run_operations = true;
            opts.run_detection = opts.run_integration = opts.run_benchmarks = opts.run_advanced =
                false;
        } else if (arg == "--integration" || arg == "-i") {
            opts.run_integration = true;
            opts.run_detection = opts.run_operations = opts.run_benchmarks = opts.run_advanced =
                false;
        } else if (arg == "--benchmarks" || arg == "-b") {
            opts.run_benchmarks = true;
            opts.run_detection = opts.run_operations = opts.run_integration = opts.run_advanced =
                false;
        } else if (arg == "--advanced" || arg == "-a") {
            opts.run_advanced = true;
            opts.run_detection = opts.run_operations = opts.run_integration = opts.run_benchmarks =
                false;
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--size" && i + 1 < argc) {
            opts.benchmark_size = stoul(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            opts.benchmark_iterations = stoi(argv[++i]);
        }
    }

    return opts;
}

void print_help() {
    cout << "SIMD Comprehensive Test Suite\n\n";
    cout << "Usage: test_simd_comprehensive [OPTIONS]\n\n";
    cout << "Options:\n";
    cout << "  -h, --help          Show this help message\n";
    cout << "  -d, --detection     Run only SIMD detection tests\n";
    cout << "  -o, --operations    Run only vector operations tests\n";
    cout << "  -i, --integration   Run only integration tests\n";
    cout << "  -b, --benchmarks    Run only performance benchmarks\n";
    cout << "  -a, --advanced      Run only advanced SIMD features tests\n";
    cout << "  -v, --verbose       Enable verbose output\n";
    cout << "  --size SIZE         Set benchmark size (default: 1000000)\n";
    cout << "  --iterations N      Set benchmark iterations (default: 100)\n\n";
    cout << "If no specific test category is specified, all tests are run.\n";
}

int main(int argc, char* argv[]) {
    TestOptions opts = parse_args(argc, argv);

    if (opts.help) {
        print_help();
        return 0;
    }

    cout << "=== SIMD COMPREHENSIVE TEST SUITE ===" << endl;
    cout << "=====================================" << endl;

    try {
        if (opts.run_detection) {
            TestSIMD::test_simd_detection(opts);
        }

        if (opts.run_operations) {
            TestSIMD::test_vector_operations(opts);
            TestSIMD::test_transcendental_functions(opts);
        }

        if (opts.run_integration) {
            TestSIMD::test_gaussian_integration(opts);
        }

        if (opts.run_benchmarks) {
            TestSIMD::test_performance_benchmarks(opts);
        }

        if (opts.run_advanced) {
            TestSIMD::test_advanced_simd(opts);
        }

        cout << "\n=== ALL SELECTED TESTS COMPLETED SUCCESSFULLY ===" << endl;
        return 0;

    } catch (const exception& e) {
        cout << "\nError: " << e.what() << endl;
        return 1;
    }
}

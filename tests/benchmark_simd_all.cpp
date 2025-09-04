/**
 * @file benchmark_simd_all.cpp
 * @brief Comprehensive benchmark and accuracy test for ALL SIMD functions
 *
 * This test covers:
 * - All basic arithmetic operations (add, subtract, multiply, scalar ops)
 * - All transcendental functions (exp, log, pow, erf)
 * - Dot product
 * - Performance measurements
 * - Accuracy verification
 */

#include "../include/platform/simd.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct TestResult {
    std::string name;
    double scalar_time;
    double simd_time;
    double speedup;
    double max_error;
    double avg_error;
    bool passed;
};

template <typename Func>
double benchmark(Func f, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        f();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

class SIMDBenchmark {
   private:
    size_t size_;
    int iterations_;
    std::vector<double> input_a_;
    std::vector<double> input_b_;
    std::vector<double> output_scalar_;
    std::vector<double> output_simd_;
    std::vector<TestResult> results_;

    static constexpr double ACCURACY_THRESHOLD = 1e-10;       // For exact operations
    static constexpr double TRANSCENDENTAL_THRESHOLD = 1e-6;  // For approximations

   public:
    SIMDBenchmark(size_t size = 1000, int iterations = 10000)
        : size_(size),
          iterations_(iterations),
          input_a_(size),
          input_b_(size),
          output_scalar_(size),
          output_simd_(size) {
        // Initialize with varied but safe values
        for (size_t i = 0; i < size; ++i) {
            input_a_[i] = 1.0 + (i % 100) * 0.01;  // 1.0 to 1.99
            input_b_[i] = 0.5 + (i % 50) * 0.01;   // 0.5 to 0.99
        }
    }

    void runTest(const std::string& name, std::function<void()> scalar_func,
                 std::function<void()> simd_func, double accuracy_threshold = ACCURACY_THRESHOLD) {
        TestResult result;
        result.name = name;

        // Warm up
        scalar_func();
        simd_func();

        // Benchmark scalar
        result.scalar_time = benchmark(scalar_func, iterations_);

        // Benchmark SIMD
        result.simd_time = benchmark(simd_func, iterations_);

        // Calculate speedup
        result.speedup = result.scalar_time / result.simd_time;

        // Check accuracy
        scalar_func();  // Generate reference results
        simd_func();    // Generate SIMD results

        result.max_error = 0.0;
        double total_error = 0.0;
        for (size_t i = 0; i < size_; ++i) {
            double error = std::abs(output_simd_[i] - output_scalar_[i]);
            result.max_error = std::max(result.max_error, error);
            total_error += error;
        }
        result.avg_error = total_error / size_;
        result.passed = result.max_error <= accuracy_threshold;

        results_.push_back(result);
    }

    void testDotProduct() {
        TestResult result;
        result.name = "dot_product";

        // Use volatile to prevent optimization
        volatile double scalar_result = 0.0;
        volatile double simd_result = 0.0;

        // Scalar function
        auto scalar_func = [&]() {
            double temp = 0.0;
            for (size_t i = 0; i < size_; ++i) {
                temp += input_a_[i] * input_b_[i];
            }
            scalar_result = temp;  // Force write to volatile
        };

        // SIMD function
        auto simd_func = [&]() {
            simd_result =
                stats::simd::ops::VectorOps::dot_product(input_a_.data(), input_b_.data(), size_);
        };

        // Warm up
        scalar_func();
        simd_func();

        // Benchmark
        result.scalar_time = benchmark(scalar_func, iterations_);
        result.simd_time = benchmark(simd_func, iterations_);
        result.speedup = result.scalar_time / result.simd_time;

        // Check accuracy
        scalar_func();
        simd_func();
        result.max_error = std::abs(simd_result - scalar_result);
        result.avg_error = result.max_error;  // Single value
        result.passed = result.max_error <= ACCURACY_THRESHOLD * std::abs(scalar_result);

        results_.push_back(result);
    }

    void runAllTests() {
        std::cout << "SIMD Comprehensive Benchmark Test" << std::endl;
        std::cout << "==================================" << std::endl;
        std::cout << "Size: " << size_ << ", Iterations: " << iterations_ << std::endl;
        std::cout << "SIMD Level: " << stats::simd::ops::VectorOps::get_active_simd_level()
                  << std::endl;
        std::cout << std::endl;

        // 1. Basic Arithmetic Operations
        std::cout << "Testing basic arithmetic operations..." << std::endl;

        // vector_add
        runTest(
            "vector_add",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = input_a_[i] + input_b_[i];
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_add(input_a_.data(), input_b_.data(),
                                                        output_simd_.data(), size_);
            });

        // vector_subtract
        runTest(
            "vector_subtract",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = input_a_[i] - input_b_[i];
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_subtract(input_a_.data(), input_b_.data(),
                                                             output_simd_.data(), size_);
            });

        // vector_multiply
        runTest(
            "vector_multiply",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = input_a_[i] * input_b_[i];
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_multiply(input_a_.data(), input_b_.data(),
                                                             output_simd_.data(), size_);
            });

        // scalar_multiply
        const double scalar = 2.5;
        runTest(
            "scalar_multiply",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = input_a_[i] * scalar;
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::scalar_multiply(input_a_.data(), scalar,
                                                             output_simd_.data(), size_);
            });

        // scalar_add
        runTest(
            "scalar_add",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = input_a_[i] + scalar;
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::scalar_add(input_a_.data(), scalar,
                                                        output_simd_.data(), size_);
            });

        // dot_product (special case - returns single value)
        testDotProduct();

        // 2. Transcendental Functions
        std::cout << "Testing transcendental functions..." << std::endl;

        // vector_exp
        runTest(
            "vector_exp",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = std::exp(input_a_[i]);
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_exp(input_a_.data(), output_simd_.data(),
                                                        size_);
            },
            TRANSCENDENTAL_THRESHOLD);

        // vector_log
        runTest(
            "vector_log",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = std::log(input_a_[i]);
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_log(input_a_.data(), output_simd_.data(),
                                                        size_);
            },
            TRANSCENDENTAL_THRESHOLD);

        // vector_pow (scalar exponent)
        const double exponent = 2.3;
        runTest(
            "vector_pow (scalar exp)",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = std::pow(input_a_[i], exponent);
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_pow(input_a_.data(), exponent,
                                                        output_simd_.data(), size_);
            },
            TRANSCENDENTAL_THRESHOLD);

        // vector_pow_elementwise
        runTest(
            "vector_pow_elementwise",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = std::pow(input_a_[i], input_b_[i]);
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_pow_elementwise(
                    input_a_.data(), input_b_.data(), output_simd_.data(), size_);
            },
            TRANSCENDENTAL_THRESHOLD);

        // vector_erf
        runTest(
            "vector_erf",
            [&]() {
                for (size_t i = 0; i < size_; ++i) {
                    output_scalar_[i] = std::erf(input_a_[i]);
                }
            },
            [&]() {
                stats::simd::ops::VectorOps::vector_erf(input_a_.data(), output_simd_.data(),
                                                        size_);
            },
            TRANSCENDENTAL_THRESHOLD);

        // 3. Platform info functions (not benchmarked, just tested)
        std::cout << "\nTesting platform query functions..." << std::endl;
        std::cout << "  should_use_simd(100): "
                  << (stats::simd::ops::VectorOps::should_use_simd(100) ? "yes" : "no")
                  << std::endl;
        std::cout << "  min_simd_size(): " << stats::simd::ops::VectorOps::min_simd_size()
                  << std::endl;
        std::cout << "  is_simd_available(): "
                  << (stats::simd::ops::VectorOps::is_simd_available() ? "yes" : "no") << std::endl;
        std::cout << "  get_optimal_block_size(): "
                  << stats::simd::ops::VectorOps::get_optimal_block_size() << std::endl;
        std::cout << "  supports_vectorization(): "
                  << (stats::simd::ops::VectorOps::supports_vectorization() ? "yes" : "no")
                  << std::endl;
        std::cout << "  double_vector_width(): "
                  << stats::simd::ops::VectorOps::double_vector_width() << std::endl;
        std::cout << "  get_platform_optimization_info(): "
                  << stats::simd::ops::VectorOps::get_platform_optimization_info() << std::endl;
    }

    void printResults() {
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "RESULTS SUMMARY" << std::endl;
        std::cout << std::string(100, '=') << std::endl;

        // Header
        std::cout << std::left << std::setw(25) << "Function" << std::right << std::setw(12)
                  << "Scalar (µs)" << std::setw(12) << "SIMD (µs)" << std::setw(10) << "Speedup"
                  << std::setw(14) << "Max Error" << std::setw(14) << "Avg Error" << std::setw(10)
                  << "Status" << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        // Results
        for (const auto& result : results_) {
            std::cout << std::left << std::setw(25) << result.name << std::right << std::setw(12)
                      << std::fixed << std::setprecision(1) << result.scalar_time << std::setw(12)
                      << result.simd_time << std::setw(10) << std::setprecision(2) << result.speedup
                      << "x" << std::setw(14) << std::scientific << std::setprecision(2)
                      << result.max_error << std::setw(14) << result.avg_error << std::setw(10)
                      << (result.passed ? "✓ PASS" : "✗ FAIL") << std::endl;
        }

        // Summary statistics
        std::cout << std::string(100, '=') << std::endl;

        int passed = 0, failed = 0;
        double total_speedup = 0.0;
        int arithmetic_count = 0, transcendental_count = 0;
        double arithmetic_speedup = 0.0, transcendental_speedup = 0.0;

        for (const auto& result : results_) {
            if (result.passed)
                passed++;
            else
                failed++;
            total_speedup += result.speedup;

            // Categorize
            if (result.name.find("exp") != std::string::npos ||
                result.name.find("log") != std::string::npos ||
                result.name.find("pow") != std::string::npos ||
                result.name.find("erf") != std::string::npos) {
                transcendental_speedup += result.speedup;
                transcendental_count++;
            } else {
                arithmetic_speedup += result.speedup;
                arithmetic_count++;
            }
        }

        std::cout << "\nOVERALL STATISTICS:" << std::endl;
        std::cout << "  Tests passed: " << passed << "/" << results_.size() << std::endl;
        std::cout << "  Average speedup: " << std::fixed << std::setprecision(2)
                  << total_speedup / results_.size() << "x" << std::endl;
        std::cout << "  Arithmetic avg speedup: "
                  << (arithmetic_count > 0 ? arithmetic_speedup / arithmetic_count : 0.0) << "x"
                  << std::endl;
        std::cout << "  Transcendental avg speedup: "
                  << (transcendental_count > 0 ? transcendental_speedup / transcendental_count
                                               : 0.0)
                  << "x" << std::endl;

        // Warnings
        if (failed > 0) {
            std::cout << "\n⚠️  WARNING: " << failed << " test(s) failed accuracy checks!"
                      << std::endl;
            std::cout << "   Transcendental functions may need better polynomial approximations."
                      << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    // Parse command line arguments
    size_t size = 1000;
    int iterations = 10000;

    if (argc > 1)
        size = std::stoul(argv[1]);
    if (argc > 2)
        iterations = std::stoi(argv[2]);

    SIMDBenchmark benchmark(size, iterations);
    benchmark.runAllTests();
    benchmark.printResults();

    return 0;
}

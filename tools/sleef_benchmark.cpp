/**
 * @file sleef_benchmark.cpp
 * @brief Benchmark comparing libstats SIMD vs SLEEF to establish theoretical speedup limits
 *
 * This benchmark tests:
 * - Elementary functions (exp, log, pow, erf)
 * - Statistical operations (normal PDF/CDF)
 * - Various batch sizes
 * - Different SIMD instruction sets
 *
 * Goal: Establish what speedups we should expect from highly optimized SIMD
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sleef.h>
#include <string>
#include <vector>

// Include libstats SIMD operations
#include "../include/platform/simd.h"

// Check available SIMD features
#ifdef __AVX2__
    #define HAS_AVX2 1
#else
    #define HAS_AVX2 0
#endif

#ifdef __AVX__
    #define HAS_AVX 1
#else
    #define HAS_AVX 0
#endif

#ifdef __SSE2__
    #define HAS_SSE2 1
#else
    #define HAS_SSE2 0
#endif

// Platform detection for Apple Silicon
#ifdef __ARM_NEON
    #define HAS_NEON 1
    #include <arm_neon.h>
#else
    #define HAS_NEON 0
#endif

namespace {

constexpr double SQRT_2PI = 2.5066282746310005024157652848110452530069867406099383166299;
constexpr double SQRT_2 = 1.4142135623730950488016887242096980785696718753769480731767;

struct BenchmarkResult {
    std::string operation;
    size_t batch_size;
    double scalar_time_us;
    double libstats_simd_time_us;
    double sleef_time_us;
    double libstats_speedup;
    double sleef_speedup;
    double max_error_libstats;
    double max_error_sleef;
};

class SleefBenchmark {
   private:
    std::vector<BenchmarkResult> results_;
    std::mt19937 rng_{42};

    static constexpr int WARMUP_ITERATIONS = 100;
    static constexpr int TIMING_ITERATIONS = 1000;

    template <typename Func>
    double timeFunction(Func f, int iterations) {
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            f();
        }

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            f();
        }
        auto end = std::chrono::high_resolution_clock::now();

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() /
               static_cast<double>(iterations);
    }

    double computeMaxError(const std::vector<double>& reference,
                           const std::vector<double>& computed) {
        double max_error = 0.0;
        for (size_t i = 0; i < reference.size(); ++i) {
            double error = std::abs(reference[i] - computed[i]);
            // Relative error for large values, absolute for small
            if (std::abs(reference[i]) > 1e-10) {
                error /= std::abs(reference[i]);
            }
            max_error = std::max(max_error, error);
        }
        return max_error;
    }

   public:
    void runExponentialBenchmark(size_t batch_size) {
        std::cout << "\n--- Exponential (exp) - Size: " << batch_size << " ---\n";

        // Generate test data
        std::vector<double> input(batch_size);
        std::uniform_real_distribution<double> dist(-10.0, 2.0);
        for (auto& val : input) {
            val = dist(rng_);
        }

        std::vector<double> output_scalar(batch_size);
        std::vector<double> output_libstats(batch_size);
        std::vector<double> output_sleef(batch_size);

        // Scalar baseline
        auto scalar_func = [&]() {
            for (size_t i = 0; i < batch_size; ++i) {
                output_scalar[i] = std::exp(input[i]);
            }
        };

        // libstats SIMD
        auto libstats_func = [&]() {
            stats::simd::ops::VectorOps::vector_exp(input.data(), output_libstats.data(),
                                                    batch_size);
        };

        // SLEEF vectorized
        auto sleef_func = [&]() {
            size_t i = 0;

#if HAS_AVX2
            // Process 4 doubles at a time with AVX2
            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d y = Sleef_expd4_u10(x);
                _mm256_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_AVX
            // Process 4 doubles at a time with AVX
            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d y = Sleef_expd4_u10(x);
                _mm256_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_SSE2
            // Process 2 doubles at a time with SSE2
            for (; i + 1 < batch_size; i += 2) {
                __m128d x = _mm_loadu_pd(&input[i]);
                __m128d y = Sleef_expd2_u10(x);
                _mm_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_NEON
            // Process 2 doubles at a time with NEON
            for (; i + 1 < batch_size; i += 2) {
                float64x2_t x = vld1q_f64(&input[i]);
                float64x2_t y = Sleef_expd2_u10(x);
                vst1q_f64(&output_sleef[i], y);
            }
#endif
            // Handle remaining elements
            for (; i < batch_size; ++i) {
                output_sleef[i] = Sleef_exp_u10(input[i]);
            }
        };

        // Benchmark
        double scalar_time = timeFunction(scalar_func, TIMING_ITERATIONS);
        double libstats_time = timeFunction(libstats_func, TIMING_ITERATIONS);
        double sleef_time = timeFunction(sleef_func, TIMING_ITERATIONS);

        // Compute errors (run once for error calculation)
        scalar_func();
        libstats_func();
        sleef_func();

        double libstats_error = computeMaxError(output_scalar, output_libstats);
        double sleef_error = computeMaxError(output_scalar, output_sleef);

        // Record results
        BenchmarkResult result;
        result.operation = "exp";
        result.batch_size = batch_size;
        result.scalar_time_us = scalar_time;
        result.libstats_simd_time_us = libstats_time;
        result.sleef_time_us = sleef_time;
        result.libstats_speedup = scalar_time / libstats_time;
        result.sleef_speedup = scalar_time / sleef_time;
        result.max_error_libstats = libstats_error;
        result.max_error_sleef = sleef_error;

        results_.push_back(result);

        std::cout << "  Scalar:    " << std::fixed << std::setprecision(2) << scalar_time
                  << " μs (baseline)\n";
        std::cout << "  libstats:  " << libstats_time << " μs (" << result.libstats_speedup
                  << "x speedup)\n";
        std::cout << "  SLEEF:     " << sleef_time << " μs (" << result.sleef_speedup
                  << "x speedup)\n";
        std::cout << "  Max error: libstats=" << std::scientific << std::setprecision(2)
                  << libstats_error << ", SLEEF=" << sleef_error << "\n";
    }

    void runLogarithmBenchmark(size_t batch_size) {
        std::cout << "\n--- Logarithm (log) - Size: " << batch_size << " ---\n";

        // Generate test data (positive values for log)
        std::vector<double> input(batch_size);
        std::uniform_real_distribution<double> dist(0.01, 100.0);
        for (auto& val : input) {
            val = dist(rng_);
        }

        std::vector<double> output_scalar(batch_size);
        std::vector<double> output_libstats(batch_size);
        std::vector<double> output_sleef(batch_size);

        // Scalar baseline
        auto scalar_func = [&]() {
            for (size_t i = 0; i < batch_size; ++i) {
                output_scalar[i] = std::log(input[i]);
            }
        };

        // libstats SIMD
        auto libstats_func = [&]() {
            stats::simd::ops::VectorOps::vector_log(input.data(), output_libstats.data(),
                                                    batch_size);
        };

        // SLEEF vectorized
        auto sleef_func = [&]() {
            size_t i = 0;

#if HAS_AVX2
            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d y = Sleef_logd4_u10(x);
                _mm256_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_AVX
            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d y = Sleef_logd4_u10(x);
                _mm256_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_SSE2
            for (; i + 1 < batch_size; i += 2) {
                __m128d x = _mm_loadu_pd(&input[i]);
                __m128d y = Sleef_logd2_u10(x);
                _mm_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_NEON
            for (; i + 1 < batch_size; i += 2) {
                float64x2_t x = vld1q_f64(&input[i]);
                float64x2_t y = Sleef_logd2_u10(x);
                vst1q_f64(&output_sleef[i], y);
            }
#endif
            for (; i < batch_size; ++i) {
                output_sleef[i] = Sleef_log_u10(input[i]);
            }
        };

        // Benchmark
        double scalar_time = timeFunction(scalar_func, TIMING_ITERATIONS);
        double libstats_time = timeFunction(libstats_func, TIMING_ITERATIONS);
        double sleef_time = timeFunction(sleef_func, TIMING_ITERATIONS);

        // Compute errors
        scalar_func();
        libstats_func();
        sleef_func();

        double libstats_error = computeMaxError(output_scalar, output_libstats);
        double sleef_error = computeMaxError(output_scalar, output_sleef);

        // Record results
        BenchmarkResult result;
        result.operation = "log";
        result.batch_size = batch_size;
        result.scalar_time_us = scalar_time;
        result.libstats_simd_time_us = libstats_time;
        result.sleef_time_us = sleef_time;
        result.libstats_speedup = scalar_time / libstats_time;
        result.sleef_speedup = scalar_time / sleef_time;
        result.max_error_libstats = libstats_error;
        result.max_error_sleef = sleef_error;

        results_.push_back(result);

        std::cout << "  Scalar:    " << std::fixed << std::setprecision(2) << scalar_time
                  << " μs (baseline)\n";
        std::cout << "  libstats:  " << libstats_time << " μs (" << result.libstats_speedup
                  << "x speedup)\n";
        std::cout << "  SLEEF:     " << sleef_time << " μs (" << result.sleef_speedup
                  << "x speedup)\n";
        std::cout << "  Max error: libstats=" << std::scientific << std::setprecision(2)
                  << libstats_error << ", SLEEF=" << sleef_error << "\n";
    }

    void runErrorFunctionBenchmark(size_t batch_size) {
        std::cout << "\n--- Error Function (erf) - Size: " << batch_size << " ---\n";

        // Generate test data
        std::vector<double> input(batch_size);
        std::uniform_real_distribution<double> dist(-3.0, 3.0);
        for (auto& val : input) {
            val = dist(rng_);
        }

        std::vector<double> output_scalar(batch_size);
        std::vector<double> output_libstats(batch_size);
        std::vector<double> output_sleef(batch_size);

        // Scalar baseline
        auto scalar_func = [&]() {
            for (size_t i = 0; i < batch_size; ++i) {
                output_scalar[i] = std::erf(input[i]);
            }
        };

        // libstats SIMD
        auto libstats_func = [&]() {
            stats::simd::ops::VectorOps::vector_erf(input.data(), output_libstats.data(),
                                                    batch_size);
        };

        // SLEEF vectorized
        auto sleef_func = [&]() {
            size_t i = 0;

#if HAS_AVX2
            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d y = Sleef_erfd4_u10(x);
                _mm256_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_AVX
            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d y = Sleef_erfd4_u10(x);
                _mm256_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_SSE2
            for (; i + 1 < batch_size; i += 2) {
                __m128d x = _mm_loadu_pd(&input[i]);
                __m128d y = Sleef_erfd2_u10(x);
                _mm_storeu_pd(&output_sleef[i], y);
            }
#elif HAS_NEON
            for (; i + 1 < batch_size; i += 2) {
                float64x2_t x = vld1q_f64(&input[i]);
                float64x2_t y = Sleef_erfd2_u10(x);
                vst1q_f64(&output_sleef[i], y);
            }
#endif
            for (; i < batch_size; ++i) {
                output_sleef[i] = Sleef_erf_u10(input[i]);
            }
        };

        // Benchmark
        double scalar_time = timeFunction(scalar_func, TIMING_ITERATIONS);
        double libstats_time = timeFunction(libstats_func, TIMING_ITERATIONS);
        double sleef_time = timeFunction(sleef_func, TIMING_ITERATIONS);

        // Compute errors
        scalar_func();
        libstats_func();
        sleef_func();

        double libstats_error = computeMaxError(output_scalar, output_libstats);
        double sleef_error = computeMaxError(output_scalar, output_sleef);

        // Record results
        BenchmarkResult result;
        result.operation = "erf";
        result.batch_size = batch_size;
        result.scalar_time_us = scalar_time;
        result.libstats_simd_time_us = libstats_time;
        result.sleef_time_us = sleef_time;
        result.libstats_speedup = scalar_time / libstats_time;
        result.sleef_speedup = scalar_time / sleef_time;
        result.max_error_libstats = libstats_error;
        result.max_error_sleef = sleef_error;

        results_.push_back(result);

        std::cout << "  Scalar:    " << std::fixed << std::setprecision(2) << scalar_time
                  << " μs (baseline)\n";
        std::cout << "  libstats:  " << libstats_time << " μs (" << result.libstats_speedup
                  << "x speedup)\n";
        std::cout << "  SLEEF:     " << sleef_time << " μs (" << result.sleef_speedup
                  << "x speedup)\n";
        std::cout << "  Max error: libstats=" << std::scientific << std::setprecision(2)
                  << libstats_error << ", SLEEF=" << sleef_error << "\n";
    }

    void runNormalPDFBenchmark(size_t batch_size) {
        std::cout << "\n--- Normal PDF - Size: " << batch_size << " ---\n";

        // Generate test data
        std::vector<double> input(batch_size);
        std::uniform_real_distribution<double> dist(-4.0, 4.0);
        for (auto& val : input) {
            val = dist(rng_);
        }

        std::vector<double> output_scalar(batch_size);
        std::vector<double> output_sleef(batch_size);

        // Parameters for standard normal
        const double mean = 0.0;
        const double stddev = 1.0;

        // Scalar baseline - standard normal PDF
        auto scalar_func = [&]() {
            for (size_t i = 0; i < batch_size; ++i) {
                double z = (input[i] - mean) / stddev;
                output_scalar[i] = std::exp(-0.5 * z * z) / (stddev * SQRT_2PI);
            }
        };

        // SLEEF-optimized normal PDF
        auto sleef_func = [&]() {
            size_t i = 0;

#if HAS_AVX2
            __m256d mean_vec = _mm256_set1_pd(mean);
            __m256d stddev_vec = _mm256_set1_pd(stddev);
            __m256d half = _mm256_set1_pd(-0.5);
            __m256d norm = _mm256_set1_pd(1.0 / (stddev * SQRT_2PI));

            for (; i + 3 < batch_size; i += 4) {
                __m256d x = _mm256_loadu_pd(&input[i]);
                __m256d z = _mm256_div_pd(_mm256_sub_pd(x, mean_vec), stddev_vec);
                __m256d z2 = _mm256_mul_pd(z, z);
                __m256d exp_arg = _mm256_mul_pd(half, z2);
                __m256d exp_val = Sleef_expd4_u10(exp_arg);
                __m256d pdf = _mm256_mul_pd(exp_val, norm);
                _mm256_storeu_pd(&output_sleef[i], pdf);
            }
#elif HAS_SSE2
            __m128d mean_vec = _mm_set1_pd(mean);
            __m128d stddev_vec = _mm_set1_pd(stddev);
            __m128d half = _mm_set1_pd(-0.5);
            __m128d norm = _mm_set1_pd(1.0 / (stddev * SQRT_2PI));

            for (; i + 1 < batch_size; i += 2) {
                __m128d x = _mm_loadu_pd(&input[i]);
                __m128d z = _mm_div_pd(_mm_sub_pd(x, mean_vec), stddev_vec);
                __m128d z2 = _mm_mul_pd(z, z);
                __m128d exp_arg = _mm_mul_pd(half, z2);
                __m128d exp_val = Sleef_expd2_u10(exp_arg);
                __m128d pdf = _mm_mul_pd(exp_val, norm);
                _mm_storeu_pd(&output_sleef[i], pdf);
            }
#endif
            // Handle remaining elements
            for (; i < batch_size; ++i) {
                double z = (input[i] - mean) / stddev;
                output_sleef[i] = Sleef_exp_u10(-0.5 * z * z) / (stddev * SQRT_2PI);
            }
        };

        // Benchmark
        double scalar_time = timeFunction(scalar_func, TIMING_ITERATIONS);
        double sleef_time = timeFunction(sleef_func, TIMING_ITERATIONS);

        // Compute errors
        scalar_func();
        sleef_func();

        double sleef_error = computeMaxError(output_scalar, output_sleef);

        // Record results
        BenchmarkResult result;
        result.operation = "normal_pdf";
        result.batch_size = batch_size;
        result.scalar_time_us = scalar_time;
        result.libstats_simd_time_us = 0;  // Not using libstats for this
        result.sleef_time_us = sleef_time;
        result.libstats_speedup = 0;
        result.sleef_speedup = scalar_time / sleef_time;
        result.max_error_libstats = 0;
        result.max_error_sleef = sleef_error;

        results_.push_back(result);

        std::cout << "  Scalar:    " << std::fixed << std::setprecision(2) << scalar_time
                  << " μs (baseline)\n";
        std::cout << "  SLEEF:     " << sleef_time << " μs (" << result.sleef_speedup
                  << "x speedup)\n";
        std::cout << "  Max error: SLEEF=" << std::scientific << std::setprecision(2) << sleef_error
                  << "\n";
    }

    void printSummary() {
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "BENCHMARK SUMMARY - THEORETICAL SIMD SPEEDUP LIMITS\n";
        std::cout << std::string(100, '=') << "\n\n";

        std::cout << "System Information:\n";
        std::cout << "  SIMD Level: " << stats::simd::ops::VectorOps::get_active_simd_level()
                  << "\n";
        std::cout << "  AVX2: " << (HAS_AVX2 ? "YES" : "NO") << "\n";
        std::cout << "  AVX:  " << (HAS_AVX ? "YES" : "NO") << "\n";
        std::cout << "  SSE2: " << (HAS_SSE2 ? "YES" : "NO") << "\n";
        std::cout << "  NEON: " << (HAS_NEON ? "YES" : "NO") << "\n";
        std::cout << "\n";

        // Print detailed results table
        std::cout << std::left << std::setw(12) << "Operation" << std::right << std::setw(10)
                  << "Size" << std::setw(12) << "Scalar(μs)" << std::setw(12) << "libstats(μs)"
                  << std::setw(12) << "SLEEF(μs)" << std::setw(12) << "libstats-X" << std::setw(12)
                  << "SLEEF-X" << std::setw(14) << "Efficiency"
                  << "\n";
        std::cout << std::string(100, '-') << "\n";

        for (const auto& r : results_) {
            double efficiency = 0.0;
            if (r.sleef_speedup > 0 && r.libstats_speedup > 0) {
                efficiency = (r.libstats_speedup / r.sleef_speedup) * 100.0;
            }

            std::cout << std::left << std::setw(12) << r.operation << std::right << std::setw(10)
                      << r.batch_size << std::setw(12) << std::fixed << std::setprecision(1)
                      << r.scalar_time_us << std::setw(12)
                      << (r.libstats_simd_time_us > 0
                              ? std::to_string(static_cast<int>(r.libstats_simd_time_us))
                              : "N/A")
                      << std::setw(12) << std::fixed << std::setprecision(1) << r.sleef_time_us
                      << std::setw(11) << std::fixed << std::setprecision(2) << r.libstats_speedup
                      << "x" << std::setw(11) << std::fixed << std::setprecision(2)
                      << r.sleef_speedup << "x";

            if (efficiency > 0) {
                std::cout << std::setw(13) << std::fixed << std::setprecision(1) << efficiency
                          << "%";
            } else {
                std::cout << std::setw(14) << "N/A";
            }
            std::cout << "\n";
        }

        // Calculate averages by operation type
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "AVERAGE SPEEDUPS BY OPERATION:\n";
        std::cout << std::string(100, '-') << "\n";

        std::map<std::string, std::pair<double, double>> op_averages;
        std::map<std::string, int> op_counts;

        for (const auto& r : results_) {
            op_averages[r.operation].first += r.libstats_speedup;
            op_averages[r.operation].second += r.sleef_speedup;
            op_counts[r.operation]++;
        }

        for (auto& [op, speedups] : op_averages) {
            int count = op_counts[op];
            speedups.first /= count;
            speedups.second /= count;

            std::cout << "  " << std::left << std::setw(12) << op << ": ";
            if (speedups.first > 0) {
                std::cout << "libstats=" << std::fixed << std::setprecision(2) << speedups.first
                          << "x, ";
            }
            std::cout << "SLEEF=" << speedups.second << "x";

            if (speedups.first > 0) {
                double efficiency = (speedups.first / speedups.second) * 100.0;
                std::cout << " (efficiency: " << std::fixed << std::setprecision(1) << efficiency
                          << "%)";
            }
            std::cout << "\n";
        }

        // Overall conclusions
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "KEY INSIGHTS:\n";
        std::cout << std::string(100, '-') << "\n";

        // Find max speedups
        double max_sleef_speedup = 0.0;
        double avg_sleef_speedup = 0.0;
        int count = 0;

        for (const auto& r : results_) {
            max_sleef_speedup = std::max(max_sleef_speedup, r.sleef_speedup);
            avg_sleef_speedup += r.sleef_speedup;
            count++;
        }
        avg_sleef_speedup /= count;

        std::cout << "  Maximum SLEEF speedup observed: " << std::fixed << std::setprecision(2)
                  << max_sleef_speedup << "x\n";
        std::cout << "  Average SLEEF speedup: " << avg_sleef_speedup << "x\n";
        std::cout << "\n";

        std::cout << "  THEORETICAL EXPECTATIONS:\n";
        if (HAS_AVX2 || HAS_AVX) {
            std::cout << "    - AVX/AVX2 processes 4 doubles per instruction\n";
            std::cout << "    - Theoretical maximum: ~4x speedup\n";
            std::cout << "    - Practical maximum: 2.5-3.5x (due to memory bandwidth)\n";
        } else if (HAS_SSE2) {
            std::cout << "    - SSE2 processes 2 doubles per instruction\n";
            std::cout << "    - Theoretical maximum: ~2x speedup\n";
            std::cout << "    - Practical maximum: 1.5-1.8x (due to memory bandwidth)\n";
        } else if (HAS_NEON) {
            std::cout << "    - NEON processes 2 doubles per instruction\n";
            std::cout << "    - Theoretical maximum: ~2x speedup\n";
            std::cout << "    - Practical maximum: 1.5-1.8x (due to memory bandwidth)\n";
        }

        std::cout << "\n  PERFORMANCE ASSESSMENT:\n";
        if (avg_sleef_speedup > 2.5 && (HAS_AVX2 || HAS_AVX)) {
            std::cout << "    ✓ SLEEF achieving good speedups for AVX\n";
        } else if (avg_sleef_speedup > 1.5 && HAS_SSE2) {
            std::cout << "    ✓ SLEEF achieving good speedups for SSE2\n";
        }

        std::cout << "    - SLEEF represents near-optimal SIMD performance\n";
        std::cout << "    - These speedups should be our target for libstats\n";
        std::cout << "\n";
    }

    void runAllBenchmarks() {
        std::cout << "\n" << std::string(100, '=') << "\n";
        std::cout << "SLEEF vs libstats SIMD Benchmark\n";
        std::cout << "Establishing theoretical SIMD speedup limits\n";
        std::cout << std::string(100, '=') << "\n";

        // Test different batch sizes
        std::vector<size_t> batch_sizes = {100, 1000, 10000, 100000};

        for (size_t batch_size : batch_sizes) {
            std::cout << "\n### Batch Size: " << batch_size << " ###\n";

            runExponentialBenchmark(batch_size);
            runLogarithmBenchmark(batch_size);
            runErrorFunctionBenchmark(batch_size);
            runNormalPDFBenchmark(batch_size);
        }

        printSummary();
    }
};

}  // anonymous namespace

int main(int argc, char* argv[]) {
    try {
        SleefBenchmark benchmark;
        benchmark.runAllBenchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

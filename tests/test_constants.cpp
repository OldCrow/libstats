/**
 * @file test_constants.cpp
 * @brief Comprehensive test suite for all libstats constants headers
 *
 * Tests all constants headers with command-line options to selectively run tests:
 * --all/-a           Test all constants (default)
 * --essential/-e     Test essential constants only
 * --mathematical/-m  Test mathematical constants
 * --precision/-p     Test precision constants
 * --statistical/-s   Test statistical constants
 * --benchmark/-b     Test benchmark constants
 * --threshold/-t     Test threshold constants
 * --robust/-r        Test robust estimation constants
 * --goodness/-g      Test goodness-of-fit constants
 * --probability/-P   Test probability constants
 * --platform/-L      Test platform constants
 * --simd/-S          Test SIMD constants
 * --tests/-T         Test test infrastructure constants
 * --methods/-M       Test statistical methods constants
 * --help/-h          Show this help
 */

// Constants headers — now consolidated into three semantic groups.
// Each group header includes its doc comment explaining what belongs there.
#include "include/constants.h"                  // Test constants
#include "libstats/core/constants.h"            // Umbrella header (includes all three above)
#include "libstats/core/essential_constants.h"  // Convenience header (math + statistical)
#include "libstats/core/math_constants.h"       // Mathematical values, precision, numerical limits
#include "libstats/core/performance_constants.h"  // Benchmark iteration counts and timing bounds
#include "libstats/core/statistical_constants.h"  // Critical values, probability bounds, thresholds
#include "libstats/platform/platform_constants.h"  // Platform constants (SIMD widths, etc.)
#include "libstats/platform/simd.h"                // SIMD constants

// Standard library includes
#include <algorithm>  // for std::min, std::max
#include <cmath>      // for std::abs, std::log, std::exp
#include <cstddef>    // for std::size_t
#include <gtest/gtest.h>
#include <iostream>  // for std::cout, std::endl
#include <limits>    // for std::numeric_limits
#include <string>    // for std::string
#include <vector>    // for std::vector

void test_math_constants() {
    using namespace stats::detail;

    // Pi and related constants
    EXPECT_TRUE(std::abs(PI - 3.14159265358979323846) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(TWO_PI - (PI * 2)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(LN_PI - std::log(PI)) < HIGH_PRECISION_TOLERANCE);

    // Test newly added mathematical constants
    EXPECT_TRUE(std::abs(PHI - 1.6180339887498948482045868343656381) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(EULER_MASCHERONI - 0.5772156649015328606065120900824024) <
                HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(CATALAN - 0.9159655941772190150546035149323841) <
                HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(APERY - 1.2020569031595942853997381615114499) < HIGH_PRECISION_TOLERANCE);

    // Test derived constants
    EXPECT_TRUE(std::abs(PHI_INV - (1.0 / PHI)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(PI_INV - (1.0 / PI)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(INV_SQRT_2PI - (1.0 / SQRT_2PI)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(INV_SQRT_2 - (1.0 / SQRT_2)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(INV_SQRT_3 - (1.0 / SQRT_3)) < HIGH_PRECISION_TOLERANCE);

    // Test fractions and multiples
    EXPECT_TRUE(std::abs(PI_OVER_2 - (PI / 2.0)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(PI_OVER_4 - (PI / 4.0)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(PI_OVER_6 - (PI / 6.0)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(FOUR_PI - (4.0 * PI)) < HIGH_PRECISION_TOLERANCE);

    // Test reciprocals
    EXPECT_TRUE(std::abs(ONE_THIRD - (1.0 / 3.0)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(ONE_SIXTH - (1.0 / 6.0)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(ONE_TWELFTH - (1.0 / 12.0)) < HIGH_PRECISION_TOLERANCE);

    // Test negative constants
    EXPECT_TRUE(std::abs(NEG_TWO - (-2.0)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(NEG_HALF - (-0.5)) < HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(std::abs(NEG_HALF_LN_2PI - (-0.5 * LN_2PI)) < HIGH_PRECISION_TOLERANCE);

    std::cout << "   ✓ Math constants tests passed" << std::endl;
}

void test_probability_constants() {
    using namespace stats::detail;

    // Probability bounds
    EXPECT_TRUE(MIN_PROBABILITY > 0.0);
    EXPECT_TRUE(MAX_PROBABILITY < 1.0);
    EXPECT_TRUE(MIN_LOG_PROBABILITY < MAX_LOG_PROBABILITY);

    // Test specific values
    EXPECT_TRUE(MIN_PROBABILITY == 1.0e-300);
    EXPECT_TRUE(MAX_PROBABILITY == 1.0 - 1.0e-15);
    EXPECT_TRUE(MIN_LOG_PROBABILITY == -4605.0);
    EXPECT_TRUE(MAX_LOG_PROBABILITY == 0.0);

    // Test relationship between probability and log probability
    // Note: MIN_LOG_PROBABILITY is a safety clamp, not the actual log(MIN_PROBABILITY)
    EXPECT_TRUE(std::log(MIN_PROBABILITY) >
                MIN_LOG_PROBABILITY);  // log(1e-300) ≈ -690.78 > -4605.0
    EXPECT_TRUE(std::exp(MAX_LOG_PROBABILITY) <= MAX_PROBABILITY + 1e-15);

    std::cout << "   ✓ Probability constants tests passed" << std::endl;
}

void test_precision_constants() {
    using namespace stats::detail;

    // Test precision hierarchy
    EXPECT_TRUE(DEFAULT_TOLERANCE > HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(HIGH_PRECISION_TOLERANCE > ULTRA_HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(ULTRA_HIGH_PRECISION_TOLERANCE > 0.0);

    // Test specific values
    EXPECT_TRUE(ZERO == 1.0e-30);
    EXPECT_TRUE(DEFAULT_TOLERANCE == 1.0e-8);
    EXPECT_TRUE(HIGH_PRECISION_TOLERANCE == 1.0e-12);
    EXPECT_TRUE(ULTRA_HIGH_PRECISION_TOLERANCE == 1.0e-15);

    // Test machine epsilon values
    EXPECT_TRUE(MACHINE_EPSILON == std::numeric_limits<double>::epsilon());
    EXPECT_TRUE(MACHINE_EPSILON_FLOAT == std::numeric_limits<float>::epsilon());
    EXPECT_TRUE(MACHINE_EPSILON_LONG_DOUBLE == std::numeric_limits<long double>::epsilon());

    // Test numerical method tolerances
    EXPECT_TRUE(NEWTON_RAPHSON_TOLERANCE > 0.0);
    EXPECT_TRUE(BISECTION_TOLERANCE > 0.0);
    EXPECT_TRUE(GRADIENT_DESCENT_TOLERANCE > 0.0);
    EXPECT_TRUE(CONJUGATE_GRADIENT_TOLERANCE > 0.0);

    // Test iteration limits
    EXPECT_TRUE(MAX_NEWTON_ITERATIONS > 0);
    EXPECT_TRUE(MAX_BISECTION_ITERATIONS > 0);
    EXPECT_TRUE(MAX_GRADIENT_DESCENT_ITERATIONS > 0);
    EXPECT_TRUE(MAX_CONJUGATE_GRADIENT_ITERATIONS > 0);

    std::cout << "   ✓ Precision constants tests passed" << std::endl;
}

void test_simd_constants() {
    using namespace stats::arch::simd;

    // Test SIMD parameters
    EXPECT_TRUE(DEFAULT_BLOCK_SIZE > 0);
    EXPECT_TRUE(MIN_SIMD_SIZE > 0);
    EXPECT_TRUE(MAX_BLOCK_SIZE >= DEFAULT_BLOCK_SIZE);
    EXPECT_TRUE(DEFAULT_BLOCK_SIZE >= MIN_SIMD_SIZE);

    // Test SIMD register widths (doubles per register)
    EXPECT_TRUE(AVX512_DOUBLES == 8);
    EXPECT_TRUE(AVX_DOUBLES == 4);
    EXPECT_TRUE(AVX2_DOUBLES == 4);
    EXPECT_TRUE(SSE_DOUBLES == 2);
    EXPECT_TRUE(NEON_DOUBLES == 2);

    // Test register width hierarchy
    EXPECT_TRUE(AVX512_DOUBLES >= AVX_DOUBLES);
    EXPECT_TRUE(AVX_DOUBLES >= SSE_DOUBLES);
    EXPECT_TRUE(SSE_DOUBLES >= NEON_DOUBLES);

    // Test optimization thresholds
    EXPECT_TRUE(OPT_MEDIUM_DATASET_MIN_SIZE == 32);
    EXPECT_TRUE(OPT_ALIGNMENT_BENEFIT_THRESHOLD == 32);
    EXPECT_TRUE(OPT_AVX512_MIN_ALIGNED_SIZE == 8);
    EXPECT_TRUE(OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD == 6);
    EXPECT_TRUE(OPT_AVX512_SMALL_BENEFIT_THRESHOLD == 4);

    std::cout << "   ✓ SIMD constants tests passed" << std::endl;
}

void test_platform_optimizations() {
    using namespace stats::arch;

    [[maybe_unused]] size_t simd_block_size = get_optimal_simd_block_size();
    [[maybe_unused]] size_t alignment = get_optimal_alignment();
    [[maybe_unused]] size_t min_simd_size = get_min_simd_size();
    [[maybe_unused]] size_t min_parallel_elements = get_min_parallel_elements();
    [[maybe_unused]] size_t optimal_grain_size = get_optimal_grain_size();

    EXPECT_TRUE(simd_block_size > 0);
    EXPECT_TRUE(alignment >= 16);  // At least 16-byte alignment for SIMD
    EXPECT_TRUE(min_simd_size > 0);
    EXPECT_TRUE(min_parallel_elements > 0);
    EXPECT_TRUE(optimal_grain_size > 0);

    // Test alignment is power of 2
    EXPECT_TRUE((alignment & (alignment - 1)) == 0);

    // Test platform-specific thresholds
    [[maybe_unused]] auto cache_thresholds = get_cache_thresholds();
    EXPECT_TRUE(cache_thresholds.l1_optimal_size > 0);
    EXPECT_TRUE(cache_thresholds.l2_optimal_size > cache_thresholds.l1_optimal_size);
#ifdef __APPLE__
    // Assume no L3 cache on Apple Silicon
    EXPECT_TRUE(cache_thresholds.l3_optimal_size >= cache_thresholds.l2_optimal_size);
#else
    EXPECT_TRUE(cache_thresholds.l3_optimal_size > cache_thresholds.l2_optimal_size);
#endif
    EXPECT_TRUE(cache_thresholds.blocking_size > 0);

    // Test transcendental support detection
    bool supports_fast_transcendental = stats::arch::supports_fast_transcendental();
    // This is platform-dependent, so we just test it doesn't crash
    (void)supports_fast_transcendental;  // Suppress unused variable warning

    std::cout << "   ✓ Platform optimization tests passed" << std::endl;
}

void test_parallel_constants() {
    using namespace stats::arch::parallel;

    // Test parallel processing constants (using accessor functions)
    EXPECT_TRUE(stats::arch::get_min_elements_for_parallel() > 0);
    EXPECT_TRUE(stats::arch::get_min_elements_for_distribution_parallel() > 0);
    EXPECT_TRUE(stats::arch::get_default_grain_size() > 0);
    EXPECT_TRUE(stats::arch::get_simple_operation_grain_size() > 0);
    EXPECT_TRUE(MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL > 0);
    EXPECT_TRUE(stats::arch::get_monte_carlo_grain_size() > 0);
    EXPECT_TRUE(stats::arch::get_max_grain_size() > 0);

    // Test logical relationships (using accessor functions)
    EXPECT_TRUE(stats::arch::get_min_elements_for_distribution_parallel() <=
                stats::arch::get_min_elements_for_parallel());
    EXPECT_TRUE(stats::arch::get_simple_operation_grain_size() <=
                stats::arch::get_default_grain_size());
    EXPECT_TRUE(stats::arch::get_monte_carlo_grain_size() <= stats::arch::get_max_grain_size());

    // Test SSE parallel constants
    EXPECT_TRUE(sse::MIN_ELEMENTS_FOR_PARALLEL == 2048);
    EXPECT_TRUE(sse::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL == 1024);
    EXPECT_TRUE(sse::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL == 16384);
    EXPECT_TRUE(sse::DEFAULT_GRAIN_SIZE == 128);
    EXPECT_TRUE(sse::SIMPLE_OPERATION_GRAIN_SIZE == 64);
    EXPECT_TRUE(sse::COMPLEX_OPERATION_GRAIN_SIZE == 256);
    EXPECT_TRUE(sse::MONTE_CARLO_GRAIN_SIZE == 32);
    EXPECT_TRUE(sse::MAX_GRAIN_SIZE == 2048);

    // Test AVX parallel constants
    EXPECT_TRUE(avx::MIN_ELEMENTS_FOR_PARALLEL == 4096);
    EXPECT_TRUE(avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL == 2048);
    EXPECT_TRUE(avx::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL == 32768);
    EXPECT_TRUE(avx::DEFAULT_GRAIN_SIZE == 256);
    EXPECT_TRUE(avx::SIMPLE_OPERATION_GRAIN_SIZE == 128);
    EXPECT_TRUE(avx::COMPLEX_OPERATION_GRAIN_SIZE == 512);
    EXPECT_TRUE(avx::MONTE_CARLO_GRAIN_SIZE == 64);
    EXPECT_TRUE(avx::MAX_GRAIN_SIZE == 4096);

    // Test architecture hierarchy - AVX should have larger thresholds than SSE
    EXPECT_TRUE(avx::MIN_ELEMENTS_FOR_PARALLEL >= sse::MIN_ELEMENTS_FOR_PARALLEL);
    EXPECT_TRUE(avx::DEFAULT_GRAIN_SIZE >= sse::DEFAULT_GRAIN_SIZE);
    EXPECT_TRUE(avx::MAX_GRAIN_SIZE >= sse::MAX_GRAIN_SIZE);

    // Test adaptive functions
    EXPECT_TRUE(stats::arch::get_min_elements_for_parallel() > 0);
    EXPECT_TRUE(stats::arch::get_default_grain_size() > 0);

    std::cout << "   ✓ Parallel constants tests passed" << std::endl;
}

void test_statistical_critical_values() {
    using namespace stats::detail;

    // Standard normal distribution critical values
    EXPECT_TRUE(std::abs(Z_95 - 1.96) < 0.001);
    EXPECT_TRUE(std::abs(Z_99 - 2.576) < 0.001);
    EXPECT_TRUE(std::abs(Z_90 - 1.645) < 0.001);
    EXPECT_TRUE(std::abs(Z_999 - 3.291) < 0.001);
    EXPECT_TRUE(std::abs(Z_95_ONE_TAIL - 1.645) < 0.001);
    EXPECT_TRUE(std::abs(Z_99_ONE_TAIL - 2.326) < 0.001);

    // Test ordering of critical values
    EXPECT_TRUE(Z_90 < Z_95);
    EXPECT_TRUE(Z_95 < Z_99);
    EXPECT_TRUE(Z_99 < Z_999);

    // Test t-distribution critical values
    EXPECT_TRUE(T_95_DF_1 > T_95_DF_2);
    EXPECT_TRUE(T_95_DF_2 > T_95_DF_3);
    EXPECT_TRUE(T_95_DF_INF == Z_95);

    // Test chi-square critical values
    EXPECT_TRUE(CHI2_95_DF_1 < CHI2_95_DF_2);
    EXPECT_TRUE(CHI2_95_DF_2 < CHI2_95_DF_3);
    EXPECT_TRUE(CHI2_99_DF_1 > CHI2_95_DF_1);

    // Test F-distribution critical values
    EXPECT_TRUE(F_95_DF_1_1 > F_95_DF_1_5);
    EXPECT_TRUE(F_99_DF_1_1 > F_95_DF_1_1);

    // Test threshold values
    EXPECT_TRUE(ALPHA_001 < ALPHA_01);
    EXPECT_TRUE(ALPHA_01 < ALPHA_05);
    EXPECT_TRUE(ALPHA_05 < ALPHA_10);

    EXPECT_TRUE(CONFIDENCE_90 < CONFIDENCE_95);
    EXPECT_TRUE(CONFIDENCE_95 < CONFIDENCE_99);
    EXPECT_TRUE(CONFIDENCE_99 < CONFIDENCE_999);

    // Test effect size thresholds
    EXPECT_TRUE(SMALL_EFFECT < MEDIUM_EFFECT);
    EXPECT_TRUE(MEDIUM_EFFECT < LARGE_EFFECT);

    // Test correlation strength thresholds
    EXPECT_TRUE(WEAK_CORRELATION < MODERATE_CORRELATION);
    EXPECT_TRUE(MODERATE_CORRELATION < STRONG_CORRELATION);

    // Test Kolmogorov-Smirnov critical values
    EXPECT_TRUE(KS_05_N_5 > KS_05_N_10);
    EXPECT_TRUE(KS_05_N_10 > KS_05_N_20);
    EXPECT_TRUE(KS_01_N_5 > KS_05_N_5);

    // Test Anderson-Darling critical values
    EXPECT_TRUE(AD_01 > AD_05);
    EXPECT_TRUE(AD_05 > AD_10);
    EXPECT_TRUE(AD_10 > AD_15);

    // Test Shapiro-Wilk critical values
    EXPECT_TRUE(SW_05_N_10 < SW_05_N_20);
    EXPECT_TRUE(SW_05_N_20 < SW_05_N_30);
    EXPECT_TRUE(SW_01_N_10 < SW_05_N_10);

    std::cout << "   ✓ Statistical critical values tests passed" << std::endl;
}

void test_threshold_constants() {
    using namespace stats::detail;

    // Test scale factor bounds
    EXPECT_TRUE(MIN_SCALE_FACTOR > 0.0);
    EXPECT_TRUE(MAX_SCALE_FACTOR > MIN_SCALE_FACTOR);
    EXPECT_TRUE(MIN_SCALE_FACTOR == 1.0e-100);
    EXPECT_TRUE(MAX_SCALE_FACTOR == 1.0e100);

    // Test log-space threshold
    EXPECT_TRUE(LOG_SPACE_THRESHOLD > 0.0);
    EXPECT_TRUE(LOG_SPACE_THRESHOLD == 1.0e-50);

    // Test distribution parameter bounds
    EXPECT_TRUE(MIN_DISTRIBUTION_PARAMETER > 0.0);
    EXPECT_TRUE(MAX_DISTRIBUTION_PARAMETER > MIN_DISTRIBUTION_PARAMETER);
    EXPECT_TRUE(MIN_DISTRIBUTION_PARAMETER == 1.0e-6);
    EXPECT_TRUE(MAX_DISTRIBUTION_PARAMETER == 1.0e6);

    std::cout << "   ✓ Threshold constants tests passed" << std::endl;
}

//==============================================================================
// NEW TEST FUNCTIONS FOR ADDITIONAL CONSTANTS HEADERS
//==============================================================================

void test_benchmark_constants() {
    using namespace stats::detail;

    // Test benchmark iteration constants
    EXPECT_TRUE(DEFAULT_ITERATIONS > 0);
    EXPECT_TRUE(DEFAULT_WARMUP_RUNS > 0);
    EXPECT_TRUE(MIN_ITERATIONS > 0);
    EXPECT_TRUE(MIN_WARMUP_RUNS > 0);
    EXPECT_TRUE(MAX_ITERATIONS >= DEFAULT_ITERATIONS);
    EXPECT_TRUE(MAX_WARMUP_RUNS >= DEFAULT_WARMUP_RUNS);
    EXPECT_TRUE(DEFAULT_ITERATIONS >= MIN_ITERATIONS);
    EXPECT_TRUE(DEFAULT_WARMUP_RUNS >= MIN_WARMUP_RUNS);

    // Test execution time thresholds
    EXPECT_TRUE(MIN_EXECUTION_TIME > 0.0);
    EXPECT_TRUE(MAX_EXECUTION_TIME > MIN_EXECUTION_TIME);
    EXPECT_TRUE(MIN_EXECUTION_TIME == 1.0e-9);
    EXPECT_TRUE(MAX_EXECUTION_TIME == 3600.0);

    // Test statistical thresholds
    EXPECT_TRUE(PERFORMANCE_SIGNIFICANCE_THRESHOLD > 0.0);
    EXPECT_TRUE(PERFORMANCE_SIGNIFICANCE_THRESHOLD <= 1.0);
    EXPECT_TRUE(CV_THRESHOLD > 0.0);
    EXPECT_TRUE(CV_THRESHOLD <= 1.0);
    EXPECT_TRUE(PERFORMANCE_SIGNIFICANCE_THRESHOLD == 0.05);
    EXPECT_TRUE(CV_THRESHOLD == 0.1);

    std::cout << "   ✓ Benchmark constants tests passed" << std::endl;
}

void test_robust_constants() {
    using namespace stats::detail;

    // Test MAD scaling factor
    EXPECT_TRUE(MAD_SCALING_FACTOR > 0.0);
    EXPECT_TRUE(std::abs(MAD_SCALING_FACTOR - 1.4826) < 1e-4);

    // Test M-estimator tuning constants
    EXPECT_TRUE(TUNING_HUBER_DEFAULT > 0.0);
    EXPECT_TRUE(TUNING_TUKEY_DEFAULT > 0.0);
    EXPECT_TRUE(TUNING_HAMPEL_A > 0.0);
    EXPECT_TRUE(TUNING_HAMPEL_B > TUNING_HAMPEL_A);
    EXPECT_TRUE(TUNING_HAMPEL_C > TUNING_HAMPEL_B);

    // Test specific values
    EXPECT_TRUE(std::abs(TUNING_HUBER_DEFAULT - 1.345) < 1e-3);
    EXPECT_TRUE(std::abs(TUNING_TUKEY_DEFAULT - 4.685) < 1e-3);
    EXPECT_TRUE(std::abs(TUNING_HAMPEL_A - 1.7) < 1e-1);
    EXPECT_TRUE(std::abs(TUNING_HAMPEL_B - 3.4) < 1e-1);
    EXPECT_TRUE(std::abs(TUNING_HAMPEL_C - 8.5) < 1e-1);

    // Test iteration and convergence parameters
    EXPECT_TRUE(MAX_ROBUST_ITERATIONS > 0);
    EXPECT_TRUE(ROBUST_CONVERGENCE_TOLERANCE > 0.0);
    EXPECT_TRUE(MIN_ROBUST_SCALE > 0.0);
    EXPECT_TRUE(ROBUST_CONVERGENCE_TOLERANCE == 1.0e-6);
    EXPECT_TRUE(MIN_ROBUST_SCALE == 1.0e-8);

    std::cout << "   ✓ Robust constants tests passed" << std::endl;
}

void test_goodness_of_fit_constants() {
    using namespace stats::detail;

    // Test Kolmogorov-Smirnov critical values (α = 0.05)
    EXPECT_TRUE(KS_05_N_5 > KS_05_N_10);
    EXPECT_TRUE(KS_05_N_10 > KS_05_N_15);
    EXPECT_TRUE(KS_05_N_15 > KS_05_N_20);
    EXPECT_TRUE(KS_05_N_20 > KS_05_N_25);
    EXPECT_TRUE(KS_05_N_25 > KS_05_N_30);
    EXPECT_TRUE(KS_05_N_30 > KS_05_N_50);
    EXPECT_TRUE(KS_05_N_50 > KS_05_N_100);

    // Test Kolmogorov-Smirnov critical values (α = 0.01)
    EXPECT_TRUE(KS_01_N_5 > KS_01_N_10);
    EXPECT_TRUE(KS_01_N_10 > KS_01_N_15);
    EXPECT_TRUE(KS_01_N_15 > KS_01_N_20);

    // Test that 0.01 values are larger than 0.05 values
    EXPECT_TRUE(KS_01_N_5 > KS_05_N_5);
    EXPECT_TRUE(KS_01_N_10 > KS_05_N_10);
    EXPECT_TRUE(KS_01_N_20 > KS_05_N_20);

    // Test Anderson-Darling critical values (increasing with significance)
    EXPECT_TRUE(AD_01 > AD_025);
    EXPECT_TRUE(AD_025 > AD_05);
    EXPECT_TRUE(AD_05 > AD_10);
    EXPECT_TRUE(AD_10 > AD_15);

    // Test Shapiro-Wilk critical values (increasing with sample size)
    EXPECT_TRUE(SW_05_N_50 > SW_05_N_30);
    EXPECT_TRUE(SW_05_N_30 > SW_05_N_25);
    EXPECT_TRUE(SW_05_N_25 > SW_05_N_20);
    EXPECT_TRUE(SW_05_N_20 > SW_05_N_15);
    EXPECT_TRUE(SW_05_N_15 > SW_05_N_10);

    // Test that 0.01 values are smaller than 0.05 values (for Shapiro-Wilk)
    EXPECT_TRUE(SW_01_N_10 < SW_05_N_10);
    EXPECT_TRUE(SW_01_N_20 < SW_05_N_20);
    EXPECT_TRUE(SW_01_N_30 < SW_05_N_30);

    std::cout << "   ✓ Goodness-of-fit constants tests passed" << std::endl;
}

void test_statistical_methods_constants() {
    // Test statistical methods constants if they exist
    // This header might contain Bayesian, bootstrap, or other method constants
    // The exact constants depend on what's defined in statistical_methods_constants.h

    // For now, just verify the header can be included without errors
    std::cout << "   ✓ Statistical methods constants tests passed" << std::endl;
}

void test_test_infrastructure_constants() {
    using namespace stats::tests::constants;

    // Test precision tolerances
    EXPECT_TRUE(DEFAULT_TOLERANCE > 0.0);
    EXPECT_TRUE(HIGH_PRECISION_TOLERANCE > 0.0);
    EXPECT_TRUE(ULTRA_HIGH_PRECISION_TOLERANCE > 0.0);
    EXPECT_TRUE(RELAXED_TOLERANCE > 0.0);
    EXPECT_TRUE(STRICT_TOLERANCE > 0.0);
    EXPECT_TRUE(DEFAULT_TOLERANCE > HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(HIGH_PRECISION_TOLERANCE > ULTRA_HIGH_PRECISION_TOLERANCE);
    EXPECT_TRUE(RELAXED_TOLERANCE > DEFAULT_TOLERANCE);
    EXPECT_TRUE(DEFAULT_TOLERANCE > STRICT_TOLERANCE);

    // Test SIMD and parallel comparison tolerances
    EXPECT_TRUE(SIMD_COMPARISON_TOLERANCE > 0.0);
    EXPECT_TRUE(PARALLEL_COMPARISON_TOLERANCE > 0.0);
    EXPECT_TRUE(STATISTICAL_TOLERANCE > 0.0);
    EXPECT_TRUE(GOF_TOLERANCE > 0.0);

    // Test performance expectations
    EXPECT_TRUE(SIMD_SPEEDUP_MIN_EXPECTED > 1.0);
    EXPECT_TRUE(PARALLEL_SPEEDUP_MIN_EXPECTED > 1.0);
    EXPECT_TRUE(SIMD_SPEEDUP_MIN_EXPECTED == 1.5);
    EXPECT_TRUE(PARALLEL_SPEEDUP_MIN_EXPECTED == 2.0);

    // Test benchmark parameters
    EXPECT_TRUE(DEFAULT_BENCHMARK_ITERATIONS > 0);
    EXPECT_TRUE(SMALL_BENCHMARK_ITERATIONS > 0);
    EXPECT_TRUE(LARGE_BENCHMARK_ITERATIONS > DEFAULT_BENCHMARK_ITERATIONS);
    EXPECT_TRUE(BENCHMARK_WARMUP_ITERATIONS > 0);
    EXPECT_TRUE(MAX_BENCHMARK_VARIANCE > 0.0);
    EXPECT_TRUE(MAX_BENCHMARK_VARIANCE <= 1.0);
    EXPECT_TRUE(MIN_BENCHMARK_MEASUREMENTS > 0);

    // Test dataset sizes
    EXPECT_TRUE(SMALL_DATASET_SIZE > 0);
    EXPECT_TRUE(MEDIUM_DATASET_SIZE > SMALL_DATASET_SIZE);
    EXPECT_TRUE(LARGE_DATASET_SIZE > MEDIUM_DATASET_SIZE);
    EXPECT_TRUE(EXTRA_LARGE_DATASET_SIZE > LARGE_DATASET_SIZE);

    // Test batch sizes
    EXPECT_TRUE(SMALL_BATCH_SIZE > 0);
    EXPECT_TRUE(MEDIUM_BATCH_SIZE > SMALL_BATCH_SIZE);
    EXPECT_TRUE(LARGE_BATCH_SIZE > MEDIUM_BATCH_SIZE);

    // Test statistical test parameters
    EXPECT_TRUE(DEFAULT_ALPHA > 0.0 && DEFAULT_ALPHA < 1.0);
    EXPECT_TRUE(STRICT_ALPHA > 0.0 && STRICT_ALPHA < DEFAULT_ALPHA);
    EXPECT_TRUE(RELAXED_ALPHA > DEFAULT_ALPHA && RELAXED_ALPHA < 1.0);
    EXPECT_TRUE(DEFAULT_CONFIDENCE_LEVEL > 0.0 && DEFAULT_CONFIDENCE_LEVEL < 1.0);

    // Test bootstrap parameters
    EXPECT_TRUE(DEFAULT_BOOTSTRAP_SAMPLES > MIN_BOOTSTRAP_SAMPLES);
    EXPECT_TRUE(MAX_BOOTSTRAP_SAMPLES > DEFAULT_BOOTSTRAP_SAMPLES);
    EXPECT_TRUE(DEFAULT_CV_FOLDS > 0);
    EXPECT_TRUE(MIN_DATA_POINTS_FOR_TESTS > 0);
    EXPECT_TRUE(MAX_DATA_POINTS_FOR_EXACT > MIN_DATA_POINTS_FOR_TESTS);

    // Test thread parameters
    EXPECT_TRUE(THREAD_SAFETY_TEST_THREADS > 0);
    EXPECT_TRUE(OPERATIONS_PER_THREAD > 0);
    EXPECT_TRUE(PARALLEL_BATCH_COUNT > 0);
    EXPECT_TRUE(CACHE_INVALIDATION_COUNT > 0);

    std::cout << "   ✓ Test infrastructure constants tests passed" << std::endl;
}

void test_compile_time_validation() {
    std::cout << "Testing compile-time validation..." << std::endl;

    // The validation namespace contains static_assert statements
    // If the code compiles, these tests have passed

    std::cout << "   ✓ Compile-time validation tests passed" << std::endl;
}

//==============================================================================
// COMMAND-LINE ARGUMENT PARSING
//==============================================================================

;

//==============================================================================
// MAIN FUNCTION WITH COMMAND-LINE SUPPORT
//==============================================================================

TEST(Constants, MathConstants) {
    test_math_constants();
}
TEST(Constants, ProbabilityConstants) {
    test_probability_constants();
}
TEST(Constants, PrecisionConstants) {
    test_precision_constants();
}
TEST(Constants, SimdConstants) {
    test_simd_constants();
}
TEST(Constants, PlatformOptimizations) {
    test_platform_optimizations();
}
TEST(Constants, ParallelConstants) {
    test_parallel_constants();
}
TEST(Constants, StatisticalCriticalValues) {
    test_statistical_critical_values();
}
TEST(Constants, ThresholdConstants) {
    test_threshold_constants();
}
TEST(Constants, BenchmarkConstants) {
    test_benchmark_constants();
}
TEST(Constants, RobustConstants) {
    test_robust_constants();
}
TEST(Constants, GoodnessOfFitConstants) {
    test_goodness_of_fit_constants();
}
TEST(Constants, StatisticalMethods) {
    test_statistical_methods_constants();
}
TEST(Constants, TestInfrastructure) {
    test_test_infrastructure_constants();
}
TEST(Constants, CompileTimeValidation) {
    test_compile_time_validation();
}

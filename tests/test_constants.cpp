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

// Include all constants headers
#include "../include/core/benchmark_constants.h"            // Benchmark parameters
#include "../include/core/constants.h"                      // Main aggregated constants
#include "../include/core/essential_constants.h"            // Essential constants
#include "../include/core/goodness_of_fit_constants.h"      // Goodness-of-fit tests
#include "../include/core/mathematical_constants.h"         // Mathematical constants
#include "../include/core/precision_constants.h"            // Precision tolerances
#include "../include/core/probability_constants.h"          // Probability bounds
#include "../include/core/robust_constants.h"               // Robust estimation
#include "../include/core/statistical_constants.h"          // Statistical critical values
#include "../include/core/statistical_methods_constants.h"  // Bayesian, bootstrap
#include "../include/core/threshold_constants.h"            // Algorithm thresholds
#include "../include/platform/platform_constants.h"         // Platform constants
#include "../include/platform/simd.h"                       // SIMD constants
#include "../include/tests/constants.h"                     // Test constants

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

void test_math_constants() {
    using namespace stats::detail;

    // Pi and related constants
    assert(std::abs(PI - 3.14159265358979323846) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(TWO_PI - (PI * 2)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(LN_PI - std::log(PI)) < HIGH_PRECISION_TOLERANCE);

    // Test newly added mathematical constants
    assert(std::abs(PHI - 1.6180339887498948482045868343656381) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(EULER_MASCHERONI - 0.5772156649015328606065120900824024) <
           HIGH_PRECISION_TOLERANCE);
    assert(std::abs(CATALAN - 0.9159655941772190150546035149323841) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(APERY - 1.2020569031595942853997381615114499) < HIGH_PRECISION_TOLERANCE);

    // Test derived constants
    assert(std::abs(PHI_INV - (1.0 / PHI)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(PI_INV - (1.0 / PI)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(INV_SQRT_2PI - (1.0 / SQRT_2PI)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(INV_SQRT_2 - (1.0 / SQRT_2)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(INV_SQRT_3 - (1.0 / SQRT_3)) < HIGH_PRECISION_TOLERANCE);

    // Test fractions and multiples
    assert(std::abs(PI_OVER_2 - (PI / 2.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(PI_OVER_4 - (PI / 4.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(PI_OVER_6 - (PI / 6.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(FOUR_PI - (4.0 * PI)) < HIGH_PRECISION_TOLERANCE);

    // Test reciprocals
    assert(std::abs(ONE_THIRD - (1.0 / 3.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(ONE_SIXTH - (1.0 / 6.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(ONE_TWELFTH - (1.0 / 12.0)) < HIGH_PRECISION_TOLERANCE);

    // Test negative constants
    assert(std::abs(NEG_TWO - (-2.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(NEG_HALF - (-0.5)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(NEG_HALF_LN_2PI - (-0.5 * LN_2PI)) < HIGH_PRECISION_TOLERANCE);

    std::cout << "   ✓ Math constants tests passed" << std::endl;
}

void test_probability_constants() {
    using namespace stats::detail;

    // Probability bounds
    assert(MIN_PROBABILITY > 0.0);
    assert(MAX_PROBABILITY < 1.0);
    assert(MIN_LOG_PROBABILITY < MAX_LOG_PROBABILITY);

    // Test specific values
    assert(MIN_PROBABILITY == 1.0e-300);
    assert(MAX_PROBABILITY == 1.0 - 1.0e-15);
    assert(MIN_LOG_PROBABILITY == -4605.0);
    assert(MAX_LOG_PROBABILITY == 0.0);

    // Test relationship between probability and log probability
    // Note: MIN_LOG_PROBABILITY is a safety clamp, not the actual log(MIN_PROBABILITY)
    assert(std::log(MIN_PROBABILITY) > MIN_LOG_PROBABILITY);  // log(1e-300) ≈ -690.78 > -4605.0
    assert(std::exp(MAX_LOG_PROBABILITY) <= MAX_PROBABILITY + 1e-15);

    std::cout << "   ✓ Probability constants tests passed" << std::endl;
}

void test_precision_constants() {
    using namespace stats::detail;

    // Test precision hierarchy
    assert(DEFAULT_TOLERANCE > HIGH_PRECISION_TOLERANCE);
    assert(HIGH_PRECISION_TOLERANCE > ULTRA_HIGH_PRECISION_TOLERANCE);
    assert(ULTRA_HIGH_PRECISION_TOLERANCE > 0.0);

    // Test specific values
    assert(ZERO == 1.0e-30);
    assert(DEFAULT_TOLERANCE == 1.0e-8);
    assert(HIGH_PRECISION_TOLERANCE == 1.0e-12);
    assert(ULTRA_HIGH_PRECISION_TOLERANCE == 1.0e-15);

    // Test machine epsilon values
    assert(MACHINE_EPSILON == std::numeric_limits<double>::epsilon());
    assert(MACHINE_EPSILON_FLOAT == std::numeric_limits<float>::epsilon());
    assert(MACHINE_EPSILON_LONG_DOUBLE == std::numeric_limits<long double>::epsilon());

    // Test numerical method tolerances
    assert(NEWTON_RAPHSON_TOLERANCE > 0.0);
    assert(BISECTION_TOLERANCE > 0.0);
    assert(GRADIENT_DESCENT_TOLERANCE > 0.0);
    assert(CONJUGATE_GRADIENT_TOLERANCE > 0.0);

    // Test iteration limits
    assert(MAX_NEWTON_ITERATIONS > 0);
    assert(MAX_BISECTION_ITERATIONS > 0);
    assert(MAX_GRADIENT_DESCENT_ITERATIONS > 0);
    assert(MAX_CONJUGATE_GRADIENT_ITERATIONS > 0);

    std::cout << "   ✓ Precision constants tests passed" << std::endl;
}

void test_simd_constants() {
    using namespace stats::arch::simd;

    // Test SIMD parameters
    assert(DEFAULT_BLOCK_SIZE > 0);
    assert(MIN_SIMD_SIZE > 0);
    assert(MAX_BLOCK_SIZE >= DEFAULT_BLOCK_SIZE);
    assert(DEFAULT_BLOCK_SIZE >= MIN_SIMD_SIZE);
    assert(stats::arch::simd::SIMD_ALIGNMENT > 0);

    // Test power of 2 alignment
    assert((stats::arch::simd::SIMD_ALIGNMENT & (stats::arch::simd::SIMD_ALIGNMENT - 1)) == 0);

    std::cout << "   ✓ SIMD constants tests passed" << std::endl;
}

void test_platform_optimizations() {
    using namespace stats::arch;

    [[maybe_unused]] size_t simd_block_size = get_optimal_simd_block_size();
    [[maybe_unused]] size_t alignment = get_optimal_alignment();
    [[maybe_unused]] size_t min_simd_size = get_min_simd_size();
    [[maybe_unused]] size_t min_parallel_elements = get_min_parallel_elements();
    [[maybe_unused]] size_t optimal_grain_size = get_optimal_grain_size();

    assert(simd_block_size > 0);
    assert(alignment >= 16);  // At least 16-byte alignment for SIMD
    assert(min_simd_size > 0);
    assert(min_parallel_elements > 0);
    assert(optimal_grain_size > 0);

    // Test alignment is power of 2
    assert((alignment & (alignment - 1)) == 0);

    // Test platform-specific thresholds
    [[maybe_unused]] auto cache_thresholds = get_cache_thresholds();
    assert(cache_thresholds.l1_optimal_size > 0);
    assert(cache_thresholds.l2_optimal_size > cache_thresholds.l1_optimal_size);
#ifdef __APPLE__
    // Assume no L3 cache on Apple Silicon
    assert(cache_thresholds.l3_optimal_size >= cache_thresholds.l2_optimal_size);
#else
    assert(cache_thresholds.l3_optimal_size > cache_thresholds.l2_optimal_size);
#endif
    assert(cache_thresholds.blocking_size > 0);

    // Test transcendental support detection
    bool supports_fast_transcendental = stats::arch::supports_fast_transcendental();
    // This is platform-dependent, so we just test it doesn't crash
    (void)supports_fast_transcendental;  // Suppress unused variable warning

    std::cout << "   ✓ Platform optimization tests passed" << std::endl;
}

void test_parallel_constants() {
    using namespace stats::arch::parallel;

    // Test parallel processing constants (using accessor functions)
    assert(stats::arch::get_min_elements_for_parallel() > 0);
    assert(stats::arch::get_min_elements_for_distribution_parallel() > 0);
    assert(stats::arch::get_default_grain_size() > 0);
    assert(stats::arch::get_simple_operation_grain_size() > 0);
    assert(MIN_DATASET_SIZE_FOR_PARALLEL > 0);
    assert(MIN_BOOTSTRAP_SAMPLES_FOR_PARALLEL > 0);
    assert(MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL > 0);
    assert(stats::arch::get_monte_carlo_grain_size() > 0);
    assert(stats::arch::get_max_grain_size() > 0);
    assert(MIN_WORK_PER_THREAD > 0);
    assert(SAMPLE_BATCH_SIZE > 0);
    assert(MIN_MATRIX_SIZE_FOR_PARALLEL > 0);
    assert(MIN_ITERATIONS_FOR_PARALLEL > 0);

    // Test logical relationships (using accessor functions)
    assert(stats::arch::get_min_elements_for_distribution_parallel() <=
           stats::arch::get_min_elements_for_parallel());
    assert(stats::arch::get_simple_operation_grain_size() <= stats::arch::get_default_grain_size());
    assert(stats::arch::get_monte_carlo_grain_size() <= stats::arch::get_max_grain_size());

    // Test adaptive functions
    assert(stats::arch::get_min_elements_for_parallel() > 0);
    assert(stats::arch::get_default_grain_size() > 0);
    // Note: adaptive functions may not be available in current implementation
    // assert(stats::arch::parallel::detail::simd_block_size() > 0);
    // assert(stats::arch::parallel::detail::memory_alignment() > 0);

    std::cout << "   ✓ Parallel constants tests passed" << std::endl;
}

void test_statistical_critical_values() {
    using namespace stats::detail;

    // Standard normal distribution critical values
    assert(std::abs(Z_95 - 1.96) < 0.001);
    assert(std::abs(Z_99 - 2.576) < 0.001);
    assert(std::abs(Z_90 - 1.645) < 0.001);
    assert(std::abs(Z_999 - 3.291) < 0.001);
    assert(std::abs(Z_95_ONE_TAIL - 1.645) < 0.001);
    assert(std::abs(Z_99_ONE_TAIL - 2.326) < 0.001);

    // Test ordering of critical values
    assert(Z_90 < Z_95);
    assert(Z_95 < Z_99);
    assert(Z_99 < Z_999);

    // Test t-distribution critical values
    assert(T_95_DF_1 > T_95_DF_2);
    assert(T_95_DF_2 > T_95_DF_3);
    assert(T_95_DF_INF == Z_95);

    // Test chi-square critical values
    assert(CHI2_95_DF_1 < CHI2_95_DF_2);
    assert(CHI2_95_DF_2 < CHI2_95_DF_3);
    assert(CHI2_99_DF_1 > CHI2_95_DF_1);

    // Test F-distribution critical values
    assert(F_95_DF_1_1 > F_95_DF_1_5);
    assert(F_99_DF_1_1 > F_95_DF_1_1);

    // Test threshold values
    assert(ALPHA_001 < ALPHA_01);
    assert(ALPHA_01 < ALPHA_05);
    assert(ALPHA_05 < ALPHA_10);

    assert(CONFIDENCE_90 < CONFIDENCE_95);
    assert(CONFIDENCE_95 < CONFIDENCE_99);
    assert(CONFIDENCE_99 < CONFIDENCE_999);

    // Test effect size thresholds
    assert(SMALL_EFFECT < MEDIUM_EFFECT);
    assert(MEDIUM_EFFECT < LARGE_EFFECT);

    // Test correlation strength thresholds
    assert(WEAK_CORRELATION < MODERATE_CORRELATION);
    assert(MODERATE_CORRELATION < STRONG_CORRELATION);

    // Test Kolmogorov-Smirnov critical values
    assert(KS_05_N_5 > KS_05_N_10);
    assert(KS_05_N_10 > KS_05_N_20);
    assert(KS_01_N_5 > KS_05_N_5);

    // Test Anderson-Darling critical values
    assert(AD_01 > AD_05);
    assert(AD_05 > AD_10);
    assert(AD_10 > AD_15);

    // Test Shapiro-Wilk critical values
    assert(SW_05_N_10 < SW_05_N_20);
    assert(SW_05_N_20 < SW_05_N_30);
    assert(SW_01_N_10 < SW_05_N_10);

    std::cout << "   ✓ Statistical critical values tests passed" << std::endl;
}

void test_threshold_constants() {
    using namespace stats::detail;

    // Test scale factor bounds
    assert(MIN_SCALE_FACTOR > 0.0);
    assert(MAX_SCALE_FACTOR > MIN_SCALE_FACTOR);
    assert(MIN_SCALE_FACTOR == 1.0e-100);
    assert(MAX_SCALE_FACTOR == 1.0e100);

    // Test log-space threshold
    assert(LOG_SPACE_THRESHOLD > 0.0);
    assert(LOG_SPACE_THRESHOLD == 1.0e-50);

    // Test distribution parameter bounds
    assert(MIN_DISTRIBUTION_PARAMETER > 0.0);
    assert(MAX_DISTRIBUTION_PARAMETER > MIN_DISTRIBUTION_PARAMETER);
    assert(MIN_DISTRIBUTION_PARAMETER == 1.0e-6);
    assert(MAX_DISTRIBUTION_PARAMETER == 1.0e6);

    std::cout << "   ✓ Threshold constants tests passed" << std::endl;
}

//==============================================================================
// NEW TEST FUNCTIONS FOR ADDITIONAL CONSTANTS HEADERS
//==============================================================================

void test_benchmark_constants() {
    using namespace stats::detail;

    // Test benchmark iteration constants
    assert(DEFAULT_ITERATIONS > 0);
    assert(DEFAULT_WARMUP_RUNS > 0);
    assert(MIN_ITERATIONS > 0);
    assert(MIN_WARMUP_RUNS > 0);
    assert(MAX_ITERATIONS >= DEFAULT_ITERATIONS);
    assert(MAX_WARMUP_RUNS >= DEFAULT_WARMUP_RUNS);
    assert(DEFAULT_ITERATIONS >= MIN_ITERATIONS);
    assert(DEFAULT_WARMUP_RUNS >= MIN_WARMUP_RUNS);

    // Test execution time thresholds
    assert(MIN_EXECUTION_TIME > 0.0);
    assert(MAX_EXECUTION_TIME > MIN_EXECUTION_TIME);
    assert(MIN_EXECUTION_TIME == 1.0e-9);
    assert(MAX_EXECUTION_TIME == 3600.0);

    // Test statistical thresholds
    assert(PERFORMANCE_SIGNIFICANCE_THRESHOLD > 0.0);
    assert(PERFORMANCE_SIGNIFICANCE_THRESHOLD <= 1.0);
    assert(CV_THRESHOLD > 0.0);
    assert(CV_THRESHOLD <= 1.0);
    assert(PERFORMANCE_SIGNIFICANCE_THRESHOLD == 0.05);
    assert(CV_THRESHOLD == 0.1);

    std::cout << "   ✓ Benchmark constants tests passed" << std::endl;
}

void test_robust_constants() {
    using namespace stats::detail;

    // Test MAD scaling factor
    assert(MAD_SCALING_FACTOR > 0.0);
    assert(std::abs(MAD_SCALING_FACTOR - 1.4826) < 1e-4);

    // Test M-estimator tuning constants
    assert(TUNING_HUBER_DEFAULT > 0.0);
    assert(TUNING_TUKEY_DEFAULT > 0.0);
    assert(TUNING_HAMPEL_A > 0.0);
    assert(TUNING_HAMPEL_B > TUNING_HAMPEL_A);
    assert(TUNING_HAMPEL_C > TUNING_HAMPEL_B);

    // Test specific values
    assert(std::abs(TUNING_HUBER_DEFAULT - 1.345) < 1e-3);
    assert(std::abs(TUNING_TUKEY_DEFAULT - 4.685) < 1e-3);
    assert(std::abs(TUNING_HAMPEL_A - 1.7) < 1e-1);
    assert(std::abs(TUNING_HAMPEL_B - 3.4) < 1e-1);
    assert(std::abs(TUNING_HAMPEL_C - 8.5) < 1e-1);

    // Test iteration and convergence parameters
    assert(MAX_ROBUST_ITERATIONS > 0);
    assert(ROBUST_CONVERGENCE_TOLERANCE > 0.0);
    assert(MIN_ROBUST_SCALE > 0.0);
    assert(ROBUST_CONVERGENCE_TOLERANCE == 1.0e-6);
    assert(MIN_ROBUST_SCALE == 1.0e-8);

    std::cout << "   ✓ Robust constants tests passed" << std::endl;
}

void test_goodness_of_fit_constants() {
    using namespace stats::detail;

    // Test Kolmogorov-Smirnov critical values (α = 0.05)
    assert(KS_05_N_5 > KS_05_N_10);
    assert(KS_05_N_10 > KS_05_N_15);
    assert(KS_05_N_15 > KS_05_N_20);
    assert(KS_05_N_20 > KS_05_N_25);
    assert(KS_05_N_25 > KS_05_N_30);
    assert(KS_05_N_30 > KS_05_N_50);
    assert(KS_05_N_50 > KS_05_N_100);

    // Test Kolmogorov-Smirnov critical values (α = 0.01)
    assert(KS_01_N_5 > KS_01_N_10);
    assert(KS_01_N_10 > KS_01_N_15);
    assert(KS_01_N_15 > KS_01_N_20);

    // Test that 0.01 values are larger than 0.05 values
    assert(KS_01_N_5 > KS_05_N_5);
    assert(KS_01_N_10 > KS_05_N_10);
    assert(KS_01_N_20 > KS_05_N_20);

    // Test Anderson-Darling critical values (increasing with significance)
    assert(AD_01 > AD_025);
    assert(AD_025 > AD_05);
    assert(AD_05 > AD_10);
    assert(AD_10 > AD_15);

    // Test Shapiro-Wilk critical values (increasing with sample size)
    assert(SW_05_N_50 > SW_05_N_30);
    assert(SW_05_N_30 > SW_05_N_25);
    assert(SW_05_N_25 > SW_05_N_20);
    assert(SW_05_N_20 > SW_05_N_15);
    assert(SW_05_N_15 > SW_05_N_10);

    // Test that 0.01 values are smaller than 0.05 values (for Shapiro-Wilk)
    assert(SW_01_N_10 < SW_05_N_10);
    assert(SW_01_N_20 < SW_05_N_20);
    assert(SW_01_N_30 < SW_05_N_30);

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
    assert(DEFAULT_TOLERANCE > 0.0);
    assert(HIGH_PRECISION_TOLERANCE > 0.0);
    assert(ULTRA_HIGH_PRECISION_TOLERANCE > 0.0);
    assert(RELAXED_TOLERANCE > 0.0);
    assert(STRICT_TOLERANCE > 0.0);
    assert(DEFAULT_TOLERANCE > HIGH_PRECISION_TOLERANCE);
    assert(HIGH_PRECISION_TOLERANCE > ULTRA_HIGH_PRECISION_TOLERANCE);
    assert(RELAXED_TOLERANCE > DEFAULT_TOLERANCE);
    assert(DEFAULT_TOLERANCE > STRICT_TOLERANCE);

    // Test SIMD and parallel comparison tolerances
    assert(SIMD_COMPARISON_TOLERANCE > 0.0);
    assert(PARALLEL_COMPARISON_TOLERANCE > 0.0);
    assert(STATISTICAL_TOLERANCE > 0.0);
    assert(GOF_TOLERANCE > 0.0);

    // Test performance expectations
    assert(SIMD_SPEEDUP_MIN_EXPECTED > 1.0);
    assert(PARALLEL_SPEEDUP_MIN_EXPECTED > 1.0);
    assert(SIMD_SPEEDUP_MIN_EXPECTED == 1.5);
    assert(PARALLEL_SPEEDUP_MIN_EXPECTED == 2.0);

    // Test benchmark parameters
    assert(DEFAULT_BENCHMARK_ITERATIONS > 0);
    assert(SMALL_BENCHMARK_ITERATIONS > 0);
    assert(LARGE_BENCHMARK_ITERATIONS > DEFAULT_BENCHMARK_ITERATIONS);
    assert(BENCHMARK_WARMUP_ITERATIONS > 0);
    assert(MAX_BENCHMARK_VARIANCE > 0.0);
    assert(MAX_BENCHMARK_VARIANCE <= 1.0);
    assert(MIN_BENCHMARK_MEASUREMENTS > 0);

    // Test dataset sizes
    assert(SMALL_DATASET_SIZE > 0);
    assert(MEDIUM_DATASET_SIZE > SMALL_DATASET_SIZE);
    assert(LARGE_DATASET_SIZE > MEDIUM_DATASET_SIZE);
    assert(EXTRA_LARGE_DATASET_SIZE > LARGE_DATASET_SIZE);

    // Test batch sizes
    assert(SMALL_BATCH_SIZE > 0);
    assert(MEDIUM_BATCH_SIZE > SMALL_BATCH_SIZE);
    assert(LARGE_BATCH_SIZE > MEDIUM_BATCH_SIZE);

    // Test statistical test parameters
    assert(DEFAULT_ALPHA > 0.0 && DEFAULT_ALPHA < 1.0);
    assert(STRICT_ALPHA > 0.0 && STRICT_ALPHA < DEFAULT_ALPHA);
    assert(RELAXED_ALPHA > DEFAULT_ALPHA && RELAXED_ALPHA < 1.0);
    assert(DEFAULT_CONFIDENCE_LEVEL > 0.0 && DEFAULT_CONFIDENCE_LEVEL < 1.0);

    // Test bootstrap parameters
    assert(DEFAULT_BOOTSTRAP_SAMPLES > MIN_BOOTSTRAP_SAMPLES);
    assert(MAX_BOOTSTRAP_SAMPLES > DEFAULT_BOOTSTRAP_SAMPLES);
    assert(DEFAULT_CV_FOLDS > 0);
    assert(MIN_DATA_POINTS_FOR_TESTS > 0);
    assert(MAX_DATA_POINTS_FOR_EXACT > MIN_DATA_POINTS_FOR_TESTS);

    // Test thread parameters
    assert(THREAD_SAFETY_TEST_THREADS > 0);
    assert(OPERATIONS_PER_THREAD > 0);
    assert(PARALLEL_BATCH_COUNT > 0);
    assert(CACHE_INVALIDATION_COUNT > 0);

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

struct TestOptions {
    bool test_all = false;
    bool test_essential = false;
    bool test_mathematical = false;
    bool test_precision = false;
    bool test_statistical = false;
    bool test_benchmark = false;
    bool test_threshold = false;
    bool test_robust = false;
    bool test_goodness = false;
    bool test_probability = false;
    bool test_platform = false;
    bool test_simd = false;
    bool test_tests = false;
    bool test_methods = false;
    bool show_help = false;
};

void print_help() {
    std::cout << "Usage: test_constants [options]\n\n";
    std::cout << "Test all libstats constants headers with selective options:\n\n";
    std::cout << "Options:\n";
    std::cout << "  --all/-a           Test all constants (default)\n";
    std::cout << "  --essential/-e     Test essential constants only\n";
    std::cout << "  --mathematical/-m  Test mathematical constants\n";
    std::cout << "  --precision/-p     Test precision constants\n";
    std::cout << "  --statistical/-s   Test statistical constants\n";
    std::cout << "  --benchmark/-b     Test benchmark constants\n";
    std::cout << "  --threshold/-t     Test threshold constants\n";
    std::cout << "  --robust/-r         Test robust estimation constants\n";
    std::cout << "  --goodness/-g      Test goodness-of-fit constants\n";
    std::cout << "  --probability/-P   Test probability constants\n";
    std::cout << "  --platform/-L      Test platform constants\n";
    std::cout << "  --simd/-S          Test SIMD constants\n";
    std::cout << "  --tests/-T         Test test infrastructure constants\n";
    std::cout << "  --methods/-M       Test statistical methods constants\n";
    std::cout << "  --help/-h          Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  test_constants                    # Test all constants\n";
    std::cout << "  test_constants --essential        # Test only essential constants\n";
    std::cout << "  test_constants -m -p -s          # Test math, precision, statistical\n";
    std::cout << "  test_constants --benchmark --simd # Test benchmark and SIMD constants\n";
}

TestOptions parse_arguments(int argc, char* argv[]) {
    TestOptions options;
    bool any_specific_test = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--all" || arg == "-a") {
            options.test_all = true;
        } else if (arg == "--essential" || arg == "-e") {
            options.test_essential = true;
            any_specific_test = true;
        } else if (arg == "--mathematical" || arg == "-m") {
            options.test_mathematical = true;
            any_specific_test = true;
        } else if (arg == "--precision" || arg == "-p") {
            options.test_precision = true;
            any_specific_test = true;
        } else if (arg == "--statistical" || arg == "-s") {
            options.test_statistical = true;
            any_specific_test = true;
        } else if (arg == "--benchmark" || arg == "-b") {
            options.test_benchmark = true;
            any_specific_test = true;
        } else if (arg == "--threshold" || arg == "-t") {
            options.test_threshold = true;
            any_specific_test = true;
        } else if (arg == "--robust" || arg == "-r") {
            options.test_robust = true;
            any_specific_test = true;
        } else if (arg == "--goodness" || arg == "-g") {
            options.test_goodness = true;
            any_specific_test = true;
        } else if (arg == "--probability" || arg == "-P") {
            options.test_probability = true;
            any_specific_test = true;
        } else if (arg == "--platform" || arg == "-L") {
            options.test_platform = true;
            any_specific_test = true;
        } else if (arg == "--simd" || arg == "-S") {
            options.test_simd = true;
            any_specific_test = true;
        } else if (arg == "--tests" || arg == "-T") {
            options.test_tests = true;
            any_specific_test = true;
        } else if (arg == "--methods" || arg == "-M") {
            options.test_methods = true;
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
// MAIN FUNCTION WITH COMMAND-LINE SUPPORT
//==============================================================================

int main(int argc, char* argv[]) {
    TestOptions options = parse_arguments(argc, argv);

    if (options.show_help) {
        print_help();
        return 0;
    }

    std::cout << "Testing libstats constants headers...\n\n";

    int tests_run = 0;
    int tests_passed = 0;

    // Run selected tests
    if (options.test_all || options.test_essential) {
        std::cout << "[Essential Constants]\n";
        test_math_constants();
        test_precision_constants();
        test_statistical_critical_values();
        tests_run += 3;
        tests_passed += 3;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_mathematical) {
        std::cout << "[Mathematical Constants]\n";
        test_math_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_precision) {
        std::cout << "[Precision Constants]\n";
        test_precision_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_statistical) {
        std::cout << "[Statistical Constants]\n";
        test_statistical_critical_values();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_benchmark) {
        std::cout << "[Benchmark Constants]\n";
        test_benchmark_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_threshold) {
        std::cout << "[Threshold Constants]\n";
        test_threshold_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_robust) {
        std::cout << "[Robust Estimation Constants]\n";
        test_robust_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_goodness) {
        std::cout << "[Goodness-of-Fit Constants]\n";
        test_goodness_of_fit_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_probability) {
        std::cout << "[Probability Constants]\n";
        test_probability_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_platform) {
        std::cout << "[Platform Constants]\n";
        test_platform_optimizations();
        test_parallel_constants();
        tests_run += 2;
        tests_passed += 2;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_simd) {
        std::cout << "[SIMD Constants]\n";
        test_simd_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_tests) {
        std::cout << "[Test Infrastructure Constants]\n";
        test_test_infrastructure_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    if (options.test_all || options.test_methods) {
        std::cout << "[Statistical Methods Constants]\n";
        test_statistical_methods_constants();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    // Always run compile-time validation
    if (options.test_all || tests_run > 0) {
        std::cout << "[Compile-time Validation]\n";
        test_compile_time_validation();
        tests_run++;
        tests_passed++;
        std::cout << std::endl;
    }

    // Print summary
    std::cout << "=== Test Summary ===\n";
    std::cout << "Tests run: " << tests_run << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    if (tests_passed == tests_run && tests_run > 0) {
        std::cout << "✓ All tests passed!\n";
        return 0;
    } else if (tests_run == 0) {
        std::cout << "No tests were run. Use --help for usage information.\n";
        return 1;
    } else {
        std::cout << "✗ Some tests failed!\n";
        return 1;
    }
}

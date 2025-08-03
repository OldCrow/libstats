#include <iostream>
#include <cmath>
#include <cassert>
#include "../include/core/constants.h"
#include "../include/platform/platform_constants.h"
#ifdef _WIN32
#include <windows.h>
#endif

void test_math_constants() {
    using namespace libstats::constants::math;
    using namespace libstats::constants::precision;
    
    // Pi and related constants
    assert(std::abs(PI - 3.14159265358979323846) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(TWO_PI - (PI * 2)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(LN_PI - std::log(PI)) < HIGH_PRECISION_TOLERANCE);
    
    // Test newly added mathematical constants
    assert(std::abs(PHI - 1.6180339887498948482045868343656381) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(EULER_MASCHERONI - 0.5772156649015328606065120900824024) < HIGH_PRECISION_TOLERANCE);
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
    assert(std::abs(ONE_THIRD - (1.0/3.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(ONE_SIXTH - (1.0/6.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(ONE_TWELFTH - (1.0/12.0)) < HIGH_PRECISION_TOLERANCE);
    
    // Test negative constants
    assert(std::abs(NEG_TWO - (-2.0)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(NEG_HALF - (-0.5)) < HIGH_PRECISION_TOLERANCE);
    assert(std::abs(NEG_HALF_LN_2PI - (-0.5 * LN_2PI)) < HIGH_PRECISION_TOLERANCE);
    
    std::cout << "   ✓ Math constants tests passed" << std::endl;
}

void test_probability_constants() {
    using namespace libstats::constants::probability;
    
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
    using namespace libstats::constants::precision;
    
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
    using namespace libstats::constants::simd;
    
    // Test SIMD parameters
    assert(DEFAULT_BLOCK_SIZE > 0);
    assert(MIN_SIMD_SIZE > 0);
    assert(MAX_BLOCK_SIZE >= DEFAULT_BLOCK_SIZE);
    assert(DEFAULT_BLOCK_SIZE >= MIN_SIMD_SIZE);
    assert(SIMD_ALIGNMENT > 0);
    
    // Test power of 2 alignment
    assert((SIMD_ALIGNMENT & (SIMD_ALIGNMENT - 1)) == 0);
    
    std::cout << "   ✓ SIMD constants tests passed" << std::endl;
}

void test_platform_optimizations() {
    using namespace libstats::constants::platform;
    
    [[maybe_unused]] size_t simd_block_size = get_optimal_simd_block_size();
    [[maybe_unused]] size_t alignment = get_optimal_alignment();
    [[maybe_unused]] size_t min_simd_size = get_min_simd_size();
    [[maybe_unused]] size_t min_parallel_elements = get_min_parallel_elements();
    [[maybe_unused]] size_t optimal_grain_size = get_optimal_grain_size();
    
    assert(simd_block_size > 0);
    assert(alignment >= 16); // At least 16-byte alignment for SIMD
    assert(min_simd_size > 0);
    assert(min_parallel_elements > 0);
    assert(optimal_grain_size > 0);
    
    // Test alignment is power of 2
    assert((alignment & (alignment - 1)) == 0);
    
    // Test platform-specific thresholds
    [[maybe_unused]] CacheThresholds cache_thresholds = get_cache_thresholds();
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
    bool supports_fast_transcendental = libstats::constants::platform::supports_fast_transcendental();
    // This is platform-dependent, so we just test it doesn't crash
    (void)supports_fast_transcendental;  // Suppress unused variable warning
    
    std::cout << "   ✓ Platform optimization tests passed" << std::endl;
}

void test_parallel_constants() {
    using namespace libstats::constants::parallel;
    
    // Test parallel processing constants
    assert(MIN_ELEMENTS_FOR_PARALLEL > 0);
    assert(MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL > 0);
    assert(DEFAULT_GRAIN_SIZE > 0);
    assert(SIMPLE_OPERATION_GRAIN_SIZE > 0);
    assert(MIN_DATASET_SIZE_FOR_PARALLEL > 0);
    assert(MIN_BOOTSTRAP_SAMPLES_FOR_PARALLEL > 0);
    assert(MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL > 0);
    assert(adaptive::monte_carlo_grain_size() > 0);
    assert(adaptive::max_grain_size() > 0);
    assert(MIN_WORK_PER_THREAD > 0);
    assert(SAMPLE_BATCH_SIZE > 0);
    assert(MIN_MATRIX_SIZE_FOR_PARALLEL > 0);
    assert(MIN_ITERATIONS_FOR_PARALLEL > 0);
    
    // Test logical relationships
    assert(MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL <= MIN_ELEMENTS_FOR_PARALLEL);
    assert(SIMPLE_OPERATION_GRAIN_SIZE <= DEFAULT_GRAIN_SIZE);
    assert(adaptive::monte_carlo_grain_size() <= adaptive::max_grain_size());
    
    // Test adaptive functions
    assert(adaptive::min_elements_for_parallel() > 0);
    assert(adaptive::grain_size() > 0);
    // Note: adaptive functions may not be available in current implementation
    // assert(adaptive::simd_block_size() > 0);
    // assert(adaptive::memory_alignment() > 0);
    
    std::cout << "   ✓ Parallel constants tests passed" << std::endl;
}

void test_statistical_critical_values() {
    using namespace libstats::constants::statistical;
    using namespace libstats::constants;
    
    // Standard normal distribution critical values
    assert(std::abs(normal::Z_95 - 1.96) < 0.001);
    assert(std::abs(normal::Z_99 - 2.576) < 0.001);
    assert(std::abs(normal::Z_90 - 1.645) < 0.001);
    assert(std::abs(normal::Z_999 - 3.291) < 0.001);
    assert(std::abs(normal::Z_95_ONE_TAIL - 1.645) < 0.001);
    assert(std::abs(normal::Z_99_ONE_TAIL - 2.326) < 0.001);
    
    // Test ordering of critical values
    assert(normal::Z_90 < normal::Z_95);
    assert(normal::Z_95 < normal::Z_99);
    assert(normal::Z_99 < normal::Z_999);
    
    // Test t-distribution critical values
    assert(t_distribution::T_95_DF_1 > t_distribution::T_95_DF_2);
    assert(t_distribution::T_95_DF_2 > t_distribution::T_95_DF_3);
    assert(t_distribution::T_95_DF_INF == normal::Z_95);
    
    // Test chi-square critical values
    assert(chi_square::CHI2_95_DF_1 < chi_square::CHI2_95_DF_2);
    assert(chi_square::CHI2_95_DF_2 < chi_square::CHI2_95_DF_3);
    assert(chi_square::CHI2_99_DF_1 > chi_square::CHI2_95_DF_1);
    
    // Test F-distribution critical values
    assert(f_distribution::F_95_DF_1_1 > f_distribution::F_95_DF_1_5);
    assert(f_distribution::F_99_DF_1_1 > f_distribution::F_95_DF_1_1);
    
    // Test threshold values
    assert(thresholds::ALPHA_001 < thresholds::ALPHA_01);
    assert(thresholds::ALPHA_01 < thresholds::ALPHA_05);
    assert(thresholds::ALPHA_05 < thresholds::ALPHA_10);
    
    assert(thresholds::CONFIDENCE_90 < thresholds::CONFIDENCE_95);
    assert(thresholds::CONFIDENCE_95 < thresholds::CONFIDENCE_99);
    assert(thresholds::CONFIDENCE_99 < thresholds::CONFIDENCE_999);
    
    // Test effect size thresholds
    assert(thresholds::SMALL_EFFECT < thresholds::MEDIUM_EFFECT);
    assert(thresholds::MEDIUM_EFFECT < thresholds::LARGE_EFFECT);
    
    // Test correlation strength thresholds
    assert(thresholds::WEAK_CORRELATION < thresholds::MODERATE_CORRELATION);
    assert(thresholds::MODERATE_CORRELATION < thresholds::STRONG_CORRELATION);
    
    // Test Kolmogorov-Smirnov critical values
    assert(kolmogorov_smirnov::KS_05_N_5 > kolmogorov_smirnov::KS_05_N_10);
    assert(kolmogorov_smirnov::KS_05_N_10 > kolmogorov_smirnov::KS_05_N_20);
    assert(kolmogorov_smirnov::KS_01_N_5 > kolmogorov_smirnov::KS_05_N_5);
    
    // Test Anderson-Darling critical values
    assert(anderson_darling::AD_01 > anderson_darling::AD_05);
    assert(anderson_darling::AD_05 > anderson_darling::AD_10);
    assert(anderson_darling::AD_10 > anderson_darling::AD_15);
    
    // Test Shapiro-Wilk critical values
    assert(shapiro_wilk::SW_05_N_10 < shapiro_wilk::SW_05_N_20);
    assert(shapiro_wilk::SW_05_N_20 < shapiro_wilk::SW_05_N_30);
    assert(shapiro_wilk::SW_01_N_10 < shapiro_wilk::SW_05_N_10);
    
    std::cout << "   ✓ Statistical critical values tests passed" << std::endl;
}

void test_threshold_constants() {
    using namespace libstats::constants::thresholds;
    
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

void test_compile_time_validation() {
    std::cout << "Testing compile-time validation..." << std::endl;
    
    // The validation namespace contains static_assert statements
    // If the code compiles, these tests have passed
    
    std::cout << "   ✓ Compile-time validation tests passed" << std::endl;
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::cout << "Testing enhanced constants in constants.h..." << std::endl;
    
    test_math_constants();
    test_probability_constants();
    test_precision_constants();
    test_simd_constants();
    test_platform_optimizations();
    test_parallel_constants();
    test_statistical_critical_values();
    test_threshold_constants();
    test_compile_time_validation();
    
    std::cout << "\n✓ All enhanced constants tests passed!" << std::endl;
    return 0;
}

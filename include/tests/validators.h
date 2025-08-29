#pragma once

/**
 * @file tests/validators.h
 * @brief Test validation utilities for performance and correctness verification
 *
 * This header provides validation utilities specifically used by the test infrastructure,
 * including adaptive performance threshold calculation, correctness verification helpers,
 * and statistical validation functions.
 *
 * Phase 3E: Test Infrastructure Namespace
 * Part of the stats::tests:: namespace hierarchy reorganization
 */

#include "../platform/cpu_detection.h"
#include "../platform/cpu_vendor_constants.h"
#include "constants.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <map>
#include <thread>

namespace stats {
namespace tests {
namespace validators {

//==============================================================================
// Architecture-Aware Performance Validation
//==============================================================================

/**
 * @brief Get expected minimum SIMD speedup for the current architecture
 * @return Expected minimum speedup factor based on detected CPU capabilities
 */
inline double getAdaptiveSIMDExpectation() noexcept {
    // Base expectation for SIMD speedup
    double base_expectation = 1.2;  // Conservative baseline

    // Adjust based on detected architecture
    if (stats::arch::cpu::is_apple_silicon()) {
        // Apple Silicon NEON - realistic expectation based on observed performance
        return 1.3;
    } else if (stats::arch::cpu::is_intel_cpu()) {
// Intel processors typically have good SIMD performance
#if defined(__AVX512F__)
        return 2.5;  // AVX-512 can provide excellent speedups
#elif defined(__AVX2__)
        return 2.0;  // AVX2 with FMA provides good speedups
#elif defined(__AVX__)
        return 1.6;  // AVX without AVX2 more limited
#elif defined(__SSE4_1__)
        return 1.4;  // SSE4.1 provides moderate speedups
#else
        return base_expectation;
#endif
    } else if (stats::arch::cpu::is_amd_cpu()) {
// AMD Zen architecture has good but slightly different SIMD characteristics
#if defined(__AVX2__)
        return 1.8;  // Zen2+ with good AVX2 performance
#elif defined(__AVX__)
        return 1.5;  // Zen/Zen+ with moderate AVX performance
#else
        return base_expectation;
#endif
    } else if (stats::arch::cpu::is_arm_cpu()) {
        // Generic ARM with NEON
        return 1.5;  // NEON provides good but variable speedups
    }

    return base_expectation;  // Conservative fallback
}

/**
 * @brief Get expected minimum parallel speedup for the current architecture
 * @return Expected minimum speedup factor based on detected CPU and thread count
 */
inline double getAdaptiveParallelExpectation() noexcept {
    std::size_t hardware_threads = std::thread::hardware_concurrency();

    if (hardware_threads <= 1) {
        return 0.9;  // Single core - parallel may be slower due to overhead
    }

    // Base parallel efficiency varies by architecture
    double parallel_efficiency = 0.7;  // Conservative default

    if (stats::arch::cpu::is_apple_silicon()) {
        // Apple Silicon has excellent thread creation and scheduling
        parallel_efficiency = 0.85;
    } else if (stats::arch::cpu::is_intel_cpu()) {
        // Intel generally good for parallel work
        parallel_efficiency = 0.75;
    } else if (stats::arch::cpu::is_amd_cpu()) {
        // AMD Zen architecture good for parallel, but CCX topology matters
        parallel_efficiency = 0.70;
    } else if (stats::arch::cpu::is_arm_cpu()) {
        // Generic ARM variable - conservative
        parallel_efficiency = 0.65;
    }

    // Calculate expected speedup: min(theoretical_max, practical_limit)
    double theoretical_max = static_cast<double>(hardware_threads) * parallel_efficiency;
    double practical_limit = 4.0;  // Diminishing returns after 4x

    return std::min(theoretical_max, practical_limit);
}

/**
 * @brief Get architecture-aware SIMD speedup threshold for batch operations
 * @param batch_size Number of elements being processed
 * @param is_complex_distribution True for computationally intensive distributions (Poisson, Gamma)
 * @return Minimum expected speedup for test validation
 */
inline double getSIMDValidationThreshold(std::size_t batch_size,
                                         bool is_complex_distribution = false) noexcept {
    double base = getAdaptiveSIMDExpectation();

    // SIMD efficiency increases with batch size due to setup cost amortization
    if (batch_size >= 50000) {
        base *= 1.2;  // Large batches get better SIMD utilization
    } else if (batch_size >= 10000) {
        base *= 1.1;  // Medium batches get moderate boost
    } else if (batch_size < 1000) {
        base *= 0.8;  // Small batches may have SIMD overhead
    }

    // Complex distributions benefit more from SIMD due to computational intensity
    if (is_complex_distribution) {
        base *= 1.15;
    } else {
        // Simple distributions (Uniform, Discrete) may have overhead that limits speedup
        base *= 0.9;
    }

    return base;
}

/**
 * @brief Get architecture-aware parallel speedup threshold for batch operations
 * @param batch_size Number of elements being processed
 * @param is_complex_distribution True for computationally intensive distributions (Poisson, Gamma)
 * @return Minimum expected speedup for test validation
 */
inline double getParallelValidationThreshold(std::size_t batch_size,
                                             bool is_complex_distribution = false) noexcept {
    double base = getAdaptiveParallelExpectation();

    // Parallel efficiency is highly dependent on batch size due to thread overhead
    if (batch_size >= 100000) {
        // Large batches achieve close to full parallel potential
        base *= 1.0;
    } else if (batch_size >= 10000) {
        // Medium batches have some thread overhead
        base *= 0.8;
    } else if (batch_size >= 1000) {
        // Small batches have significant overhead - be very conservative
        base = std::max(0.9, base * 0.3);
    } else {
        // Very small batches may be inefficient - just expect some speedup
        base = std::max(0.8, base * 0.2);
    }

    // Complex distributions benefit more from parallelization
    if (is_complex_distribution) {
        base *= 1.15;
    } else {
        base *= 0.9;
    }

    return base;
}

//==============================================================================
// Performance Validation Helpers
//==============================================================================

/**
 * @brief Validate SIMD speedup meets architecture-aware expectations
 * @param measured_speedup The actual measured speedup
 * @param batch_size Number of elements processed
 * @param is_complex_distribution True for complex distributions
 * @return True if speedup meets adaptive expectations
 */
inline bool validateSIMDSpeedup(double measured_speedup, std::size_t batch_size,
                                bool is_complex_distribution = false) noexcept {
    double threshold = getSIMDValidationThreshold(batch_size, is_complex_distribution);
    return measured_speedup >= threshold;
}

/**
 * @brief Validate parallel speedup meets architecture-aware expectations
 * @param measured_speedup The actual measured speedup
 * @param batch_size Number of elements processed
 * @param is_complex_distribution True for complex distributions
 * @return True if speedup meets adaptive expectations
 */
inline bool validateParallelSpeedup(double measured_speedup, std::size_t batch_size,
                                    bool is_complex_distribution = false) noexcept {
    double threshold = getParallelValidationThreshold(batch_size, is_complex_distribution);
    return measured_speedup >= threshold;
}

/**
 * @brief Get relaxed validation threshold for noisy test environments
 * @param base_threshold Base speedup expectation
 * @param relaxation_factor Factor to reduce expectation (0.7 = 70% of original)
 * @return Relaxed threshold that maintains at least 1.0x speedup
 */
inline double getRelaxedThreshold(double base_threshold, double relaxation_factor = 0.7) noexcept {
    return std::max(1.0, base_threshold * relaxation_factor);
}

//==============================================================================
// Distribution-Specific Performance Classification
//==============================================================================

/**
 * @brief Determine if a distribution type is computationally complex
 * @param distribution_name Name of the distribution (case-insensitive)
 * @return True if the distribution requires complex computations
 */
inline bool isComplexDistribution(const std::string& distribution_name) noexcept {
    // Convert to lowercase for comparison
    std::string lower_name = distribution_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    // Complex distributions that benefit more from parallel processing
    return (lower_name == "poisson" || lower_name == "gamma" || lower_name == "beta" ||
            lower_name == "weibull" || lower_name == "lognormal");
}

//==============================================================================
// Legacy Compatibility Functions
//==============================================================================

/**
 * @brief Get fixed SIMD speedup expectation for backward compatibility
 * @return Fixed expectation from constants namespace
 * @deprecated Use getSIMDValidationThreshold() for architecture-aware validation
 */
inline double getFixedSIMDExpectation() noexcept {
    return constants::SIMD_SPEEDUP_MIN_EXPECTED;
}

/**
 * @brief Get fixed parallel speedup expectation for backward compatibility
 * @return Fixed expectation from constants namespace
 * @deprecated Use getParallelValidationThreshold() for architecture-aware validation
 */
inline double getFixedParallelExpectation() noexcept {
    return constants::PARALLEL_SPEEDUP_MIN_EXPECTED;
}

//==============================================================================
// Performance Validation Class Interface
//==============================================================================

/**
 * @brief Wrapper class for performance validation functions
 *
 * This class provides a convenient object-oriented interface to the validation
 * functions, useful for test infrastructure that expects a class interface.
 */
class PerformanceValidator {
   public:
    /**
     * @brief Validate SIMD performance meets architecture-aware expectations
     * @param measured_speedup The actual measured speedup
     * @param batch_size Number of elements processed
     * @param is_complex_distribution True for complex distributions
     * @return True if speedup meets adaptive expectations
     */
    static bool validateSIMD(double measured_speedup, std::size_t batch_size,
                             bool is_complex_distribution = false) {
        return validateSIMDSpeedup(measured_speedup, batch_size, is_complex_distribution);
    }

    /**
     * @brief Validate parallel performance meets architecture-aware expectations
     * @param measured_speedup The actual measured speedup
     * @param batch_size Number of elements processed
     * @param is_complex_distribution True for complex distributions
     * @return True if speedup meets adaptive expectations
     */
    static bool validateParallel(double measured_speedup, std::size_t batch_size,
                                 bool is_complex_distribution = false) {
        return validateParallelSpeedup(measured_speedup, batch_size, is_complex_distribution);
    }

    /**
     * @brief Get expected SIMD threshold for current architecture
     * @param batch_size Number of elements being processed
     * @param is_complex_distribution True for complex distributions
     * @return Expected minimum speedup threshold
     */
    static double getSIMDThreshold(std::size_t batch_size, bool is_complex_distribution = false) {
        return getSIMDValidationThreshold(batch_size, is_complex_distribution);
    }

    /**
     * @brief Get expected parallel threshold for current architecture
     * @param batch_size Number of elements being processed
     * @param is_complex_distribution True for complex distributions
     * @return Expected minimum speedup threshold
     */
    static double getParallelThreshold(std::size_t batch_size,
                                       bool is_complex_distribution = false) {
        return getParallelValidationThreshold(batch_size, is_complex_distribution);
    }

    /**
     * @brief Validate both SIMD and parallel performance together
     * @param simd_speedup Measured SIMD speedup
     * @param parallel_speedup Measured parallel speedup
     * @param batch_size Number of elements processed
     * @param distribution_name Name of distribution (for complexity detection)
     * @return True if both meet expectations
     */
    static bool validateBoth(double simd_speedup, double parallel_speedup, std::size_t batch_size,
                             const std::string& distribution_name = "") {
        bool is_complex = !distribution_name.empty() && isComplexDistribution(distribution_name);
        return validateSIMD(simd_speedup, batch_size, is_complex) &&
               validateParallel(parallel_speedup, batch_size, is_complex);
    }
};

//==============================================================================
// Numerical Accuracy Validation
//==============================================================================

/**
 * @brief Validate numerical accuracy meets expected tolerances
 * @param measured_value Computed value
 * @param expected_value Expected/reference value
 * @param tolerance Acceptable relative tolerance
 * @param absolute_tolerance Acceptable absolute tolerance (for values near zero)
 * @return True if accuracy meets expectations
 */
inline bool validateNumericalAccuracy(double measured_value, double expected_value,
                                      double tolerance = 1e-12,
                                      double absolute_tolerance = 1e-15) noexcept {
    if (std::abs(expected_value) < absolute_tolerance) {
        // For values near zero, use absolute tolerance
        return std::abs(measured_value - expected_value) <= absolute_tolerance;
    }

    // For normal values, use relative tolerance
    double relative_error = std::abs((measured_value - expected_value) / expected_value);
    return relative_error <= tolerance;
}

/**
 * @brief Validate batch operation correctness against reference implementation
 * @param batch_results Results from batch operation
 * @param reference_results Results from reference implementation
 * @param tolerance Acceptable relative tolerance
 * @return True if batch results are sufficiently accurate
 */
inline bool validateBatchAccuracy(const std::vector<double>& batch_results,
                                  const std::vector<double>& reference_results,
                                  double tolerance = 1e-12) noexcept {
    if (batch_results.size() != reference_results.size()) {
        return false;
    }

    for (std::size_t i = 0; i < batch_results.size(); ++i) {
        if (!validateNumericalAccuracy(batch_results[i], reference_results[i], tolerance)) {
            return false;
        }
    }
    return true;
}

//==============================================================================
// Memory Usage Validation
//==============================================================================

/**
 * @brief Validate memory allocation patterns for batch operations
 * @param operation_func Function to test
 * @param expected_max_allocs Maximum expected allocations
 * @param batch_size Size of batch being processed
 * @return True if memory usage is reasonable
 * @note This is a placeholder - actual implementation would require memory profiling
 */
template <typename Func>
inline bool validateMemoryUsage(Func&& operation_func, std::size_t expected_max_allocs,
                                std::size_t batch_size) noexcept {
    // TODO: Implement memory profiling when available
    // For now, return true but this is a placeholder for future memory tracking
    (void)operation_func;
    (void)expected_max_allocs;
    (void)batch_size;
    return true;
}

/**
 * @brief Validate that operations don't leak memory
 * @param operation_func Function to test repeatedly
 * @param iterations Number of iterations to run
 * @return True if no memory leaks detected
 * @note This is a placeholder - actual implementation would require memory profiling
 */
template <typename Func>
inline bool validateNoMemoryLeaks(Func&& operation_func, std::size_t iterations = 100) noexcept {
    // TODO: Implement memory leak detection when available
    // For now, just run the operation multiple times
    for (std::size_t i = 0; i < iterations; ++i) {
        operation_func();
    }
    return true;  // Placeholder
}

//==============================================================================
// Test Environment Validation
//==============================================================================

/**
 * @brief Validate test environment has sufficient resources
 * @param min_memory_mb Minimum required memory in MB
 * @param min_cpu_cores Minimum required CPU cores
 * @return True if environment meets requirements
 */
inline bool validateTestEnvironment(std::size_t min_memory_mb = 512,
                                    std::size_t min_cpu_cores = 1) noexcept {
    // Check CPU cores
    std::size_t available_cores = std::thread::hardware_concurrency();
    if (available_cores < min_cpu_cores) {
        return false;
    }

    // TODO: Check available memory when platform utilities are available
    (void)min_memory_mb;  // Suppress warning

    return true;
}

/**
 * @brief Check if current architecture supports expected SIMD instructions
 * @param required_features Vector of required SIMD features
 * @return True if all required features are supported
 */
inline bool validateSIMDSupport(const std::vector<std::string>& required_features = {}) noexcept {
    // Basic architecture detection - expand as needed
    bool has_basic_simd = false;

#if defined(__SSE2__) || defined(__ARM_NEON) || defined(__AVX__)
    has_basic_simd = true;
#endif

    // If no specific features required, just check for basic SIMD
    if (required_features.empty()) {
        return has_basic_simd;
    }

    // TODO: Implement specific feature checking when needed
    for (const auto& feature : required_features) {
        (void)feature;  // Suppress warning - placeholder for feature checking
    }

    return has_basic_simd;
}

//==============================================================================
// Performance Regression Detection
//==============================================================================

/**
 * @brief Detect performance regressions against baseline
 * @param current_time Current measured time (any unit)
 * @param baseline_time Baseline time (same unit)
 * @param regression_threshold Maximum acceptable slowdown factor (1.1 = 10% slower)
 * @return True if performance is acceptable (no significant regression)
 */
inline bool validateNoPerformanceRegression(double current_time, double baseline_time,
                                            double regression_threshold = 1.2) noexcept {
    if (baseline_time <= 0.0 || current_time <= 0.0) {
        return false;  // Invalid timing data
    }

    double slowdown_factor = current_time / baseline_time;
    return slowdown_factor <= regression_threshold;
}

/**
 * @brief Validate timing measurement stability
 * @param timing_samples Vector of timing measurements
 * @param max_coefficient_of_variation Maximum acceptable CV (0.2 = 20%)
 * @return True if timing measurements are stable
 */
inline bool validateTimingStability(const std::vector<double>& timing_samples,
                                    double max_coefficient_of_variation = 0.3) noexcept {
    if (timing_samples.size() < 3) {
        return false;  // Need at least 3 samples for stability
    }

    // Calculate mean
    double sum = 0.0;
    for (double t : timing_samples) {
        sum += t;
    }
    double mean = sum / static_cast<double>(timing_samples.size());

    if (mean <= 0.0) {
        return false;
    }

    // Calculate standard deviation
    double variance = 0.0;
    for (double t : timing_samples) {
        double diff = t - mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(timing_samples.size() - 1);
    double std_dev = std::sqrt(variance);

    // Calculate coefficient of variation
    double cv = std_dev / mean;
    return cv <= max_coefficient_of_variation;
}

//==============================================================================
// Statistical Test Correctness Validation
//==============================================================================

/**
 * @brief Validate that a statistical test has reasonable power
 * @param sample_size Size of test sample
 * @param effect_size Expected effect size
 * @param alpha Significance level
 * @param min_power Minimum acceptable statistical power
 * @return True if test has sufficient power
 */
inline bool validateStatisticalPower(std::size_t sample_size, double effect_size,
                                     double alpha = 0.05, double min_power = 0.8) noexcept {
    // Simplified power calculation for basic validation
    // TODO: Implement proper power analysis for specific tests

    // Very basic heuristic - larger samples and effect sizes increase power
    double estimated_power = std::min(
        1.0, (static_cast<double>(sample_size) * effect_size * effect_size) / (100.0 * alpha));

    return estimated_power >= min_power;
}

/**
 * @brief Validate p-value is in valid range
 * @param p_value P-value from statistical test
 * @return True if p-value is valid
 */
inline bool validatePValue(double p_value) noexcept {
    return std::isfinite(p_value) && p_value >= 0.0 && p_value <= 1.0;
}

//==============================================================================
// Error Handling Validation
//==============================================================================

/**
 * @brief Validate that a function properly handles edge cases
 * @param test_func Function to test
 * @param edge_case_inputs Vector of edge case inputs
 * @param should_throw Whether function should throw on edge cases
 * @return True if error handling behaves as expected
 */
template <typename Func, typename Input>
inline bool validateEdgeCaseHandling(Func&& test_func, const std::vector<Input>& edge_case_inputs,
                                     bool should_throw = true) noexcept {
    for (const auto& input : edge_case_inputs) {
        try {
            test_func(input);
            if (should_throw) {
                return false;  // Expected exception but none was thrown
            }
        } catch (...) {
            if (!should_throw) {
                return false;  // Unexpected exception
            }
        }
    }
    return true;
}

/**
 * @brief Validate thread safety of operations
 * @param test_func Function to test
 * @param num_threads Number of threads to use
 * @param iterations_per_thread Iterations per thread
 * @return True if no race conditions detected
 */
template <typename Func>
inline bool validateThreadSafety(Func&& test_func, std::size_t num_threads = 4,
                                 std::size_t iterations_per_thread = 100) noexcept {
    std::vector<std::thread> threads;
    std::atomic<bool> error_detected{false};

    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&test_func, &error_detected, iterations_per_thread]() {
            try {
                for (std::size_t i = 0; i < iterations_per_thread; ++i) {
                    test_func();
                }
            } catch (...) {
                error_detected = true;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return !error_detected;
}

//==============================================================================
// Comprehensive Validation Orchestrator
//==============================================================================

/**
 * @brief Comprehensive test validator that runs multiple validation checks
 */
class ComprehensiveValidator {
   public:
    /**
     * @brief Validate complete test suite health
     * @param performance_data Performance measurements
     * @param accuracy_data Accuracy measurements
     * @param environment_checks Environment validation results
     * @return Overall validation score (0.0 = fail, 1.0 = perfect)
     */
    static double validateTestSuite(
        const std::map<std::string, double>& performance_data,
        const std::map<std::string, double>& accuracy_data,
        const std::map<std::string, bool>& environment_checks) noexcept {
        double score = 0.0;

        // Weight different validation categories
        const double performance_weight = 0.4;
        const double accuracy_weight = 0.4;
        const double environment_weight = 0.2;

        // TODO: Implement comprehensive scoring based on all validation results
        (void)performance_data;
        (void)accuracy_data;
        (void)environment_checks;
        (void)performance_weight;
        (void)accuracy_weight;
        (void)environment_weight;

        return score;
    }
};

}  // namespace validators
}  // namespace tests
}  // namespace stats

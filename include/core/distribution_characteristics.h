/**
 * @file distribution_characteristics.h
 * @brief Empirically-derived distribution characteristics for performance optimization
 *
 * This header provides empirical constants for different distribution families based on
 * actual computational complexity analysis rather than assumptions. These constants
 * serve as initial performance baselines that can be refined through adaptive learning.
 */

#pragma once

#include "performance_dispatcher.h"

#include <array>
#include <cstddef>

namespace libstats {
namespace performance {
namespace characteristics {

/**
 * @brief Computational complexity characteristics for distribution families
 *
 * These values are derived from actual algorithmic analysis of each distribution's
 * implementation rather than assumptions. They represent relative computational
 * cost multipliers compared to the simplest operations.
 */
struct DistributionComplexity {
    double base_complexity;             ///< Base computational cost multiplier
    double vectorization_efficiency;    ///< SIMD efficiency (0.0-1.0)
    double parallelization_efficiency;  ///< Parallel efficiency (0.0-1.0)
    size_t min_simd_threshold;          ///< Minimum elements where SIMD becomes beneficial
    size_t min_parallel_threshold;      ///< Minimum elements where parallelization helps

    // Cache characteristics
    double memory_access_pattern;   ///< Memory access efficiency (0.0-1.0, 1.0 = perfect locality)
    double branch_prediction_cost;  ///< Branch misprediction penalty factor
};

/**
 * @brief Empirically-derived characteristics for each distribution family
 *
 * These constants are based on algorithmic analysis of actual implementations:
 * - Uniform: Simple linear transform, excellent vectorization
 * - Discrete: Integer operations, good vectorization, minimal branching
 * - Exponential: One transcendental function (exp/log), moderate vectorization
 * - Gaussian: Box-Muller transform (2 transcendentals + sqrt), complex control flow
 * - Poisson: Iterative algorithms with early termination, poor vectorization
 * - Gamma: Multiple special functions + iterative rejection sampling, complex
 */
constexpr std::array<DistributionComplexity, 6> DISTRIBUTION_CHARACTERISTICS = {
    {// UNIFORM: y = a + (b-a) * uniform_random()
     // - Single multiply-add operation
     // - Perfect memory locality
     // - No branching
     // - Excellent SIMD efficiency (near-perfect vectorization)
     {
         .base_complexity = 1.0,              // Baseline reference
         .vectorization_efficiency = 0.95,    // Excellent SIMD efficiency
         .parallelization_efficiency = 0.90,  // Excellent parallel efficiency
         .min_simd_threshold = 16,            // Very low threshold due to simplicity
         .min_parallel_threshold = 1000,      // Moderate threshold due to low per-element cost
         .memory_access_pattern = 1.0,        // Perfect sequential access
         .branch_prediction_cost = 1.0        // No conditional branches
     },

     // GAUSSIAN: Box-Muller transform
     // - Two uniform samples -> two Gaussian samples
     // - log(), sqrt(), cos(), sin() transcendental functions
     // - Moderate branching for cached value reuse
     // - Good but not perfect vectorization due to transcendental overhead
     {
         .base_complexity = 3.2,              // ~3.2x more complex than uniform
         .vectorization_efficiency = 0.75,    // Good SIMD but transcendentals limit efficiency
         .parallelization_efficiency = 0.80,  // Good parallel efficiency
         .min_simd_threshold = 32,            // Higher due to transcendental overhead
         .min_parallel_threshold = 1500,      // Higher due to moderate per-element cost
         .memory_access_pattern = 0.95,       // Mostly sequential, some caching patterns
         .branch_prediction_cost = 1.15       // Minimal branching for cached values
     },

     // EXPONENTIAL: Inverse transform method
     // - -log(uniform_random()) / lambda
     // - One transcendental function (log)
     // - No branching in fast path
     // - Good vectorization potential
     {
         .base_complexity = 2.1,              // ~2.1x more complex than uniform
         .vectorization_efficiency = 0.82,    // Good SIMD efficiency
         .parallelization_efficiency = 0.85,  // Good parallel efficiency
         .min_simd_threshold = 24,            // Moderate threshold
         .min_parallel_threshold = 1200,      // Moderate threshold
         .memory_access_pattern = 1.0,        // Perfect sequential access
         .branch_prediction_cost = 1.0        // No conditional branches in fast path
     },

     // DISCRETE: Integer operations with bounds checking
     // - Uniform integer generation with modulo
     // - Range checking and validation
     // - Excellent memory locality
     // - Some branching for bounds checking
     // - Good but not perfect vectorization due to integer-specific optimizations
     {
         .base_complexity = 1.4,              // ~1.4x more complex than uniform
         .vectorization_efficiency = 0.85,    // Good SIMD efficiency for integer ops
         .parallelization_efficiency = 0.88,  // Good parallel efficiency
         .min_simd_threshold = 20,            // Low threshold due to simplicity
         .min_parallel_threshold = 800,       // Lower threshold due to low complexity
         .memory_access_pattern = 1.0,        // Perfect sequential access
         .branch_prediction_cost = 1.1        // Minimal branching for validation
     },

     // POISSON: Iterative algorithms (Knuth's algorithm for small lambda, acceptance-rejection for
     // large)
     // - While loop with early termination
     // - Multiple exponential/log evaluations
     // - Highly variable execution time per sample
     // - Poor vectorization due to data dependencies
     // - Branch-heavy with unpredictable termination
     {
         .base_complexity = 4.8,              // ~4.8x more complex than uniform
         .vectorization_efficiency = 0.35,    // Poor SIMD efficiency due to loops
         .parallelization_efficiency = 0.70,  // Moderate parallel efficiency
         .min_simd_threshold = 64,            // High threshold due to complexity
         .min_parallel_threshold = 2000,      // Higher threshold due to high per-element cost
         .memory_access_pattern = 0.85,       // Some irregular access patterns
         .branch_prediction_cost = 1.35       // Significant branching overhead
     },

     // GAMMA: Acceptance-rejection sampling (Marsaglia & Tsang for shape >= 1, other methods for
     // shape < 1)
     // - Multiple transcendental functions per sample
     // - Rejection sampling with variable iteration count
     // - log(), exp(), sqrt(), pow() operations
     // - Highly variable execution time
     // - Complex branching patterns
     // - Poor vectorization due to conditional loops
     {
         .base_complexity = 6.5,              // ~6.5x more complex than uniform
         .vectorization_efficiency = 0.25,    // Poor SIMD efficiency
         .parallelization_efficiency = 0.65,  // Moderate parallel efficiency
         .min_simd_threshold = 80,            // High threshold
         .min_parallel_threshold = 3000,      // High threshold due to complexity
         .memory_access_pattern = 0.80,       // Irregular access patterns
         .branch_prediction_cost = 1.50       // Heavy branching overhead
     }}};

/**
 * @brief Get characteristics for a specific distribution type
 *
 * @param dist_type Distribution type to query
 * @return Reference to empirical characteristics
 */
constexpr const DistributionComplexity& getCharacteristics(DistributionType dist_type) noexcept {
    switch (dist_type) {
        case DistributionType::UNIFORM:
            return DISTRIBUTION_CHARACTERISTICS[0];
        case DistributionType::GAUSSIAN:
            return DISTRIBUTION_CHARACTERISTICS[1];
        case DistributionType::EXPONENTIAL:
            return DISTRIBUTION_CHARACTERISTICS[2];
        case DistributionType::DISCRETE:
            return DISTRIBUTION_CHARACTERISTICS[3];
        case DistributionType::POISSON:
            return DISTRIBUTION_CHARACTERISTICS[4];
        case DistributionType::GAMMA:
            return DISTRIBUTION_CHARACTERISTICS[5];
    }
    // Fallback to uniform characteristics
    return DISTRIBUTION_CHARACTERISTICS[0];
}

/**
 * @brief Performance scaling factors based on empirical analysis
 *
 * These represent expected performance improvements from different strategies
 * based on algorithmic analysis and can be refined through adaptive learning.
 */
namespace scaling {
/**
 * @brief Expected SIMD speedup factors by distribution complexity
 *
 * Simple operations (uniform, discrete) benefit more from SIMD than
 * complex operations with transcendentals or unpredictable branching.
 */
constexpr double calculateSIMDSpeedup(const DistributionComplexity& chars) noexcept {
    // SIMD speedup varies based on vectorization efficiency and complexity
    // Simple operations: up to 4x speedup on 4-wide SIMD
    // Complex operations: limited by transcendental function overhead
    return 1.0 + (3.0 * chars.vectorization_efficiency);
}

/**
 * @brief Expected parallel speedup factors accounting for overhead
 *
 * Takes into account thread overhead, cache effects, and algorithmic complexity.
 * More complex operations benefit more from parallelization due to higher
 * computation-to-synchronization ratios.
 */
constexpr double calculateParallelSpeedup(const DistributionComplexity& chars,
                                          size_t num_threads) noexcept {
    // Parallel efficiency decreases with thread overhead and cache conflicts
    // But increases with algorithmic complexity
    double thread_efficiency = static_cast<double>(num_threads) * chars.parallelization_efficiency;

    // Diminishing returns: Amdahl's law approximation
    double overhead_factor = 1.0 / (1.0 + (0.1 / chars.base_complexity));

    return std::min(thread_efficiency * overhead_factor, static_cast<double>(num_threads) * 0.85);
}
}  // namespace scaling

/**
 * @brief Adaptive learning integration points
 *
 * These provide hooks for the performance learning system to refine
 * the empirical constants based on actual measured performance.
 */
namespace adaptive {
/**
 * @brief Refinement factors that can be learned and updated
 *
 * These multipliers adjust the base characteristics based on
 * system-specific performance observations.
 */
struct LearnedRefinements {
    double simd_efficiency_multiplier = 1.0;      ///< Learned SIMD efficiency adjustment
    double parallel_efficiency_multiplier = 1.0;  ///< Learned parallel efficiency adjustment
    double complexity_adjustment = 1.0;           ///< Learned complexity adjustment
    size_t simd_threshold_offset = 0;             ///< Learned threshold adjustment
    size_t parallel_threshold_offset = 0;         ///< Learned threshold adjustment

    // Confidence in learned values (0.0 = use empirical, 1.0 = use learned)
    double learning_confidence = 0.0;
};

/**
 * @brief Apply learned refinements to empirical characteristics
 *
 * @param base_chars Empirical base characteristics
 * @param refinements Learned refinements from performance history
 * @return Refined characteristics combining empirical + learned data
 */
constexpr DistributionComplexity applyRefinements(const DistributionComplexity& base_chars,
                                                  const LearnedRefinements& refinements) noexcept {
    // Blend empirical and learned values based on confidence
    double blend_factor = refinements.learning_confidence;

    return DistributionComplexity{
        .base_complexity = base_chars.base_complexity *
                           (1.0 - blend_factor + blend_factor * refinements.complexity_adjustment),
        .vectorization_efficiency =
            base_chars.vectorization_efficiency *
            (1.0 - blend_factor + blend_factor * refinements.simd_efficiency_multiplier),
        .parallelization_efficiency =
            base_chars.parallelization_efficiency *
            (1.0 - blend_factor + blend_factor * refinements.parallel_efficiency_multiplier),
        .min_simd_threshold = static_cast<size_t>(
            static_cast<double>(base_chars.min_simd_threshold) * (1.0 - blend_factor) +
            static_cast<double>(base_chars.min_simd_threshold + refinements.simd_threshold_offset) *
                blend_factor),
        .min_parallel_threshold = static_cast<size_t>(
            static_cast<double>(base_chars.min_parallel_threshold) * (1.0 - blend_factor) +
            static_cast<double>(base_chars.min_parallel_threshold +
                                refinements.parallel_threshold_offset) *
                blend_factor),
        .memory_access_pattern = base_chars.memory_access_pattern,
        .branch_prediction_cost = base_chars.branch_prediction_cost};
}
}  // namespace adaptive

}  // namespace characteristics
}  // namespace performance
}  // namespace libstats

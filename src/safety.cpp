#include "../include/core/safety.h"

#include "../include/core/mathematical_constants.h"
#include "../include/core/precision_constants.h"
#include "../include/core/threshold_constants.h"

#include <algorithm>
#include <cmath>

/**
 * @file safety.cpp
 * @brief Implementation of vectorized safety functions with SIMD optimization
 *
 * This file implements the vectorized versions of safety functions declared in safety.h.
 * These functions are designed to work efficiently with arrays of data using SIMD
 * when beneficial, while maintaining consistency with their scalar counterparts.
 *
 * ## Implementation Strategy
 *
 * Each vectorized function follows this pattern:
 * 1. Size validation: Ensure input and output arrays match
 * 2. Threshold check: Use SIMD only for arrays >= VECTORIZED_THRESHOLD (32 elements)
 * 3. SIMD processing: Process data in SIMD-width chunks when beneficial
 * 4. Scalar fallback: Use inline scalar functions for remainder and small arrays
 *
 * This ensures optimal performance across different array sizes while maintaining
 * behavioral consistency with the inline scalar functions in safety.h.
 */

namespace stats {
namespace detail {

//==============================================================================
// VECTORIZED SAFETY FUNCTIONS IMPLEMENTATION
//==============================================================================

// Minimum array size threshold for vectorized operations
constexpr std::size_t VECTORIZED_THRESHOLD = 32;

bool should_use_vectorized_safety(std::size_t size) noexcept {
    return size >= VECTORIZED_THRESHOLD && arch::simd::SIMDPolicy::shouldUseSIMD(size);
}

std::size_t vectorized_safety_threshold() noexcept {
    return VECTORIZED_THRESHOLD;
}

void vector_safe_log(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // For now, use single-pass scalar processing to avoid double-loop overhead
    // Future optimization: implement SIMD-accelerated safe_log in simd_avx.cpp etc.
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = safe_log(input[i]);
    }
}

void vector_safe_exp(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // For now, use single-pass scalar processing to avoid double-loop overhead
    // Future optimization: implement SIMD-accelerated safe_exp in simd_avx.cpp etc.
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = safe_exp(input[i]);
    }
}

void vector_safe_sqrt(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // For now, use single-pass scalar processing
    // Future optimization: implement SIMD-accelerated safe_sqrt in simd_avx.cpp etc.
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = safe_sqrt(input[i]);
    }
}

void vector_clamp_probability(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = clamp_probability(input[i]);
    }
}

void vector_clamp_log_probability(std::span<const double> input,
                                  std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = clamp_log_probability(input[i]);
    }
}

}  // namespace detail
}  // namespace stats

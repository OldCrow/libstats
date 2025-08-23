#include "../include/core/safety.h"

#include "../include/core/constants.h"

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
namespace safety {

//==============================================================================
// VECTORIZED SAFETY FUNCTIONS IMPLEMENTATION
//==============================================================================

// Minimum array size threshold for vectorized operations
constexpr std::size_t VECTORIZED_THRESHOLD = 32;

bool should_use_vectorized_safety(std::size_t size) noexcept {
    return size >= VECTORIZED_THRESHOLD && simd::SIMDPolicy::shouldUseSIMD(size);
}

std::size_t vectorized_safety_threshold() noexcept {
    return VECTORIZED_THRESHOLD;
}

void vector_safe_log(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // Use SIMD for larger arrays
    if (should_use_vectorized_safety(size)) {
        // Process in SIMD-sized chunks
        const std::size_t simd_size = simd::double_vector_width();
        std::size_t i = 0;

        // SIMD processing loop
        for (; i + simd_size <= size; i += simd_size) {
            // Load input chunk
            alignas(simd::optimal_alignment()) double chunk_input[simd_size];
            alignas(simd::optimal_alignment()) double chunk_output[simd_size];

            std::copy(input.data() + i, input.data() + i + simd_size, chunk_input);

            // Apply per-element safe_log logic with SIMD
            for (std::size_t j = 0; j < simd_size; ++j) {
                double value = chunk_input[j];
                if (value <= constants::math::ZERO_DOUBLE || std::isnan(value)) {
                    chunk_output[j] = constants::probability::MIN_LOG_PROBABILITY;
                } else if (std::isinf(value)) {
                    chunk_output[j] = std::numeric_limits<double>::max();
                } else {
                    chunk_output[j] = std::log(value);
                }
            }

            // Store output chunk
            std::copy(chunk_output, chunk_output + simd_size, output.data() + i);
        }

        // Handle remaining elements
        for (; i < size; ++i) {
            output[i] = safe_log(input[i]);
        }
    } else {
        // Scalar fallback for small arrays
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = safe_log(input[i]);
        }
    }
}

void vector_safe_exp(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // Use SIMD for larger arrays
    if (should_use_vectorized_safety(size)) {
        // Process in SIMD-sized chunks
        const std::size_t simd_size = simd::double_vector_width();
        std::size_t i = 0;

        // SIMD processing loop
        for (; i + simd_size <= size; i += simd_size) {
            // Load input chunk
            alignas(simd::optimal_alignment()) double chunk_input[simd_size];
            alignas(simd::optimal_alignment()) double chunk_output[simd_size];

            std::copy(input.data() + i, input.data() + i + simd_size, chunk_input);

            // Apply per-element safe_exp logic with SIMD
            for (std::size_t j = 0; j < simd_size; ++j) {
                double value = chunk_input[j];
                if (std::isnan(value)) {
                    chunk_output[j] = constants::math::ZERO_DOUBLE;
                } else if (value < constants::probability::MIN_LOG_PROBABILITY) {
                    chunk_output[j] = constants::probability::MIN_PROBABILITY;
                } else if (value > 700.0) {
                    chunk_output[j] = std::numeric_limits<double>::max();
                } else {
                    chunk_output[j] = std::exp(value);
                }
            }

            // Store output chunk
            std::copy(chunk_output, chunk_output + simd_size, output.data() + i);
        }

        // Handle remaining elements
        for (; i < size; ++i) {
            output[i] = safe_exp(input[i]);
        }
    } else {
        // Scalar fallback for small arrays
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = safe_exp(input[i]);
        }
    }
}

void vector_safe_sqrt(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // Use SIMD for larger arrays
    if (should_use_vectorized_safety(size)) {
        // Process in SIMD-sized chunks
        const std::size_t simd_size = simd::double_vector_width();
        std::size_t i = 0;

        // SIMD processing loop
        for (; i + simd_size <= size; i += simd_size) {
            // Load input chunk
            alignas(simd::optimal_alignment()) double chunk_input[simd_size];
            alignas(simd::optimal_alignment()) double chunk_output[simd_size];

            std::copy(input.data() + i, input.data() + i + simd_size, chunk_input);

            // Apply per-element safe_sqrt logic with SIMD
            for (std::size_t j = 0; j < simd_size; ++j) {
                double value = chunk_input[j];
                if (std::isnan(value) || value < constants::math::ZERO_DOUBLE) {
                    chunk_output[j] = constants::math::ZERO_DOUBLE;
                } else if (std::isinf(value)) {
                    chunk_output[j] = std::numeric_limits<double>::max();
                } else {
                    chunk_output[j] = std::sqrt(value);
                }
            }

            // Store output chunk
            std::copy(chunk_output, chunk_output + simd_size, output.data() + i);
        }

        // Handle remaining elements
        for (; i < size; ++i) {
            output[i] = safe_sqrt(input[i]);
        }
    } else {
        // Scalar fallback for small arrays
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = safe_sqrt(input[i]);
        }
    }
}

void vector_clamp_probability(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // Use SIMD for larger arrays
    if (should_use_vectorized_safety(size)) {
        // Process in SIMD-sized chunks
        const std::size_t simd_size = simd::double_vector_width();
        std::size_t i = 0;

        // SIMD processing loop
        for (; i + simd_size <= size; i += simd_size) {
            // Load input chunk
            alignas(simd::optimal_alignment()) double chunk_input[simd_size];
            alignas(simd::optimal_alignment()) double chunk_output[simd_size];

            std::copy(input.data() + i, input.data() + i + simd_size, chunk_input);

            // Apply per-element clamp_probability logic with SIMD
            for (std::size_t j = 0; j < simd_size; ++j) {
                double prob = chunk_input[j];
                if (std::isnan(prob)) {
                    chunk_output[j] = constants::probability::MIN_PROBABILITY;
                } else if (prob <= constants::math::ZERO_DOUBLE) {
                    chunk_output[j] = constants::probability::MIN_PROBABILITY;
                } else if (prob >= constants::math::ONE) {
                    chunk_output[j] = constants::probability::MAX_PROBABILITY;
                } else {
                    chunk_output[j] = prob;
                }
            }

            // Store output chunk
            std::copy(chunk_output, chunk_output + simd_size, output.data() + i);
        }

        // Handle remaining elements
        for (; i < size; ++i) {
            output[i] = clamp_probability(input[i]);
        }
    } else {
        // Scalar fallback for small arrays
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = clamp_probability(input[i]);
        }
    }
}

void vector_clamp_log_probability(std::span<const double> input,
                                  std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;  // Size mismatch - fail silently in noexcept function
    }

    const std::size_t size = input.size();

    // Use SIMD for larger arrays
    if (should_use_vectorized_safety(size)) {
        // Process in SIMD-sized chunks
        const std::size_t simd_size = simd::double_vector_width();
        std::size_t i = 0;

        // SIMD processing loop
        for (; i + simd_size <= size; i += simd_size) {
            // Load input chunk
            alignas(simd::optimal_alignment()) double chunk_input[simd_size];
            alignas(simd::optimal_alignment()) double chunk_output[simd_size];

            std::copy(input.data() + i, input.data() + i + simd_size, chunk_input);

            // Apply per-element clamp_log_probability logic with SIMD
            for (std::size_t j = 0; j < simd_size; ++j) {
                double log_prob = chunk_input[j];
                if (std::isnan(log_prob)) {
                    chunk_output[j] = constants::probability::MIN_LOG_PROBABILITY;
                } else if (log_prob > constants::math::ZERO_DOUBLE) {
                    chunk_output[j] = constants::probability::MAX_LOG_PROBABILITY;
                } else if (log_prob < constants::probability::MIN_LOG_PROBABILITY) {
                    chunk_output[j] = constants::probability::MIN_LOG_PROBABILITY;
                } else {
                    chunk_output[j] = log_prob;
                }
            }

            // Store output chunk
            std::copy(chunk_output, chunk_output + simd_size, output.data() + i);
        }

        // Handle remaining elements
        for (; i < size; ++i) {
            output[i] = clamp_log_probability(input[i]);
        }
    } else {
        // Scalar fallback for small arrays
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = clamp_log_probability(input[i]);
        }
    }
}

}  // namespace safety
}  // namespace stats

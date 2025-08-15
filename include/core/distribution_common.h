#pragma once

/**
 * @file core/distribution_common.h
 * @brief Common includes and using declarations for all distribution implementations
 *
 * This header consolidates the standard library and core project headers that are
 * commonly needed by all distribution classes. Distribution headers should include
 * this instead of duplicating these common includes.
 */

// Standard library includes commonly needed by all distributions
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Modern C++20 standard headers for advanced distributions
#include <concepts>
#include <ranges>
#include <version>

// Core libstats headers needed by all distributions
#include "distribution_base.h"
#include "distribution_interface.h"
#include "essential_constants.h"
#include "error_handling.h"

// Performance and platform headers commonly used
#include "performance_dispatcher.h"

// Utility using declarations to avoid repetition
using std::shared_lock;
using std::unique_lock;
using std::shared_mutex;

/**
 * @brief Common validation helper for distribution parameters
 * @tparam T Parameter type
 * @param value Parameter value to validate
 * @param name Parameter name for error messages
 * @param min_val Minimum allowed value (inclusive)
 * @param max_val Maximum allowed value (inclusive)
 * @throws std::invalid_argument if value is outside valid range
 */
template<typename T>
inline void validateParameter(T value, const std::string& name, T min_val, T max_val) {
    if (value < min_val || value > max_val || !std::isfinite(value)) {
        throw std::invalid_argument("Parameter " + name + " must be finite and in range [" +
                                  std::to_string(min_val) + ", " + std::to_string(max_val) + "]");
    }
}

/**
 * @brief Common validation helper for positive parameters
 * @tparam T Parameter type
 * @param value Parameter value to validate
 * @param name Parameter name for error messages
 * @throws std::invalid_argument if value is not positive and finite
 */
template<typename T>
inline void validatePositiveParameter(T value, const std::string& name) {
    if (value <= T(0) || !std::isfinite(value)) {
        throw std::invalid_argument("Parameter " + name + " must be positive and finite");
    }
}

/**
 * @brief Common validation helper for non-negative parameters
 * @tparam T Parameter type
 * @param value Parameter value to validate
 * @param name Parameter name for error messages
 * @throws std::invalid_argument if value is negative or not finite
 */
template<typename T>
inline void validateNonNegativeParameter(T value, const std::string& name) {
    if (value < T(0) || !std::isfinite(value)) {
        throw std::invalid_argument("Parameter " + name + " must be non-negative and finite");
    }
}

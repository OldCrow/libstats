#pragma once

/**
 * @file common/distribution_common.h
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

// Core libstats headers needed by all distributions
#include "libstats/core/distribution_base.h"
#include "libstats/core/distribution_interface.h"
#include "libstats/core/error_handling.h"
#include "libstats/core/essential_constants.h"

// Performance and platform headers commonly used
#include "libstats/core/performance_dispatcher.h"

namespace stats::detail {

/**
 * @brief Common validation helper for distribution parameters.
 * @note Moved to `stats::detail` in v2.0.0; previously at global scope.
 */
template <typename T>
inline void validateParameter(T value, const std::string& name, T min_val, T max_val) {
    if (value < min_val || value > max_val || !std::isfinite(value)) {
        throw std::invalid_argument("Parameter " + name + " must be finite and in range [" +
                                    std::to_string(min_val) + ", " + std::to_string(max_val) + "]");
    }
}

/** @brief Common validation helper for positive parameters. */
template <typename T>
inline void validatePositiveParameter(T value, const std::string& name) {
    if (value <= T(0) || !std::isfinite(value)) {
        throw std::invalid_argument("Parameter " + name + " must be positive and finite");
    }
}

/** @brief Common validation helper for non-negative parameters. */
template <typename T>
inline void validateNonNegativeParameter(T value, const std::string& name) {
    if (value < T(0) || !std::isfinite(value)) {
        throw std::invalid_argument("Parameter " + name + " must be non-negative and finite");
    }
}

}  // namespace stats::detail

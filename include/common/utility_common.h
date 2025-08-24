#pragma once

/**
 * @file common/utility_common.h
 * @brief Common dependencies for utility and statistical helper functions
 *
 * This header provides shared standard library includes and forward declarations
 * for utility functions that don't directly implement distribution interfaces.
 *
 * This header is used by:
 * - math_utils.h (mathematical utilities and special functions)
 * - statistical_utilities.h (statistical convenience wrappers)
 * - validation.h (goodness-of-fit testing)
 * - safety.h (numerical safety utilities)
 *
 * Individual headers include their specific mathematical/platform dependencies.
 *
 * Design Principle: This consolidates only the most common standard library
 * includes while preserving the focused nature of individual utility headers.
 */

// Standard library includes common to utility functions
#include <algorithm>  // Common algorithms
#include <cassert>    // Assertions for safety checks
#include <cmath>      // Mathematical functions
#include <concepts>   // For C++20 concepts in math_utils
#include <functional>
#include <span>
#include <stdexcept>  // Exception types
#include <string>
#include <vector>

// Core constants - but only essential ones to avoid heavy dependencies
#include "../core/essential_constants.h"

namespace stats {

// Forward declarations for utility headers
class DistributionBase;

// C++20 concepts for type safety - moved to top level for broader access
template <typename T>
concept FloatingPoint = std::floating_point<T> && requires(T t) {
    std::isfinite(t);
    std::isnan(t);
    std::isinf(t);
};

template <typename F, typename T>
concept MathFunction =
    std::invocable<F, T> && std::convertible_to<std::invoke_result_t<F, T>, double>;

namespace safety {
// Forward declarations for safety utilities
enum class RecoveryStrategy;
class ConvergenceDetector;
}  // namespace safety

namespace detail {  // validation utilities
// Forward declarations for validation utilities
struct KSTestResult;
struct ADTestResult;
struct ChiSquaredResult;
struct ModelDiagnostics;
}  // namespace detail

}  // namespace stats

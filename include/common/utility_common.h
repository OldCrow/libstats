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
#if __has_include(<concepts>)
    #include <concepts>  // For C++20 concepts in math_utils
    #define LIBSTATS_HAS_STD_CONCEPTS_HEADER 1
#else
    #define LIBSTATS_HAS_STD_CONCEPTS_HEADER 0
#endif

// Catalina-specific language compatibility:
// AppleClang 12 on Catalina rejects partial concept argument syntax like
// `template <MathFunction<double> F>`, while newer AppleClang releases accept it.
#if defined(__APPLE__) && defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) &&                \
    (__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ <= 101500)
    #define LIBSTATS_NEEDS_CATALINA_CONCEPT_SYNTAX_FALLBACK 1
#else
    #define LIBSTATS_NEEDS_CATALINA_CONCEPT_SYNTAX_FALLBACK 0
#endif
#include <functional>
#include <span>
#include <stdexcept>  // Exception types
#include <string>
#include <type_traits>
#include <vector>

// Core constants - but only essential ones to avoid heavy dependencies
#include "libstats/core/essential_constants.h"

namespace stats {

// Forward declarations for utility headers
class DistributionBase;

// C++20 concepts for type safety - moved to top level for broader access
#if LIBSTATS_HAS_STD_CONCEPTS_HEADER
template <typename T>
concept FloatingPoint = std::floating_point<T>;

template <typename F, typename T>
concept MathFunction =
    std::invocable<F, T> && std::convertible_to<std::invoke_result_t<F, T>, double>;
#else
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

template <typename F, typename T>
concept MathFunction = std::is_invocable_r_v<double, F, T>;
#endif

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

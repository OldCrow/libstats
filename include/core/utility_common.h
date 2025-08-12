#pragma once

/**
 * @file core/utility_common.h
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
#include <vector>
#include <string>
#include <span>
#include <functional>
#include <concepts>  // For C++20 concepts in math_utils
#include <algorithm> // Common algorithms
#include <cmath>     // Mathematical functions
#include <cassert>   // Assertions for safety checks
#include <stdexcept> // Exception types

// Core constants - but only essential ones to avoid heavy dependencies
#include "essential_constants.h"

namespace libstats {

// Forward declarations for utility headers
class DistributionBase;

namespace math {
    // Forward declarations for mathematical utilities
    template<typename T>
    concept FloatingPoint = std::floating_point<T> && requires(T t) {
        std::isfinite(t);
        std::isnan(t); 
        std::isinf(t);
    };
    
    template<typename F, typename T>
    concept MathFunction = std::invocable<F, T> && 
                           std::convertible_to<std::invoke_result_t<F, T>, double>;
}

namespace safety {
    // Forward declarations for safety utilities
    enum class RecoveryStrategy;
    class ConvergenceDetector;
}

namespace validation {
    // Forward declarations for validation utilities
    struct KSTestResult;
    struct ADTestResult;
    struct ChiSquaredResult;
    struct ModelDiagnostics;
}

} // namespace libstats

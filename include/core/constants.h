#pragma once

/**
 * @file core/constants.h
 * @brief Convenience header that includes all libstats constants
 * 
 * This umbrella header provides access to all constants in libstats by including
 * the focused constant headers. Use this for convenience when you need access to
 * multiple categories of constants, or include specific headers for better
 * compilation times and clearer dependencies.
 * 
 * For more focused includes, consider using:
 * - precision_constants.h - Numerical precision and tolerance values
 * - mathematical_constants.h - Fundamental mathematical constants (π, e, etc.)
 * - statistical_constants.h - Statistical critical values and test parameters
 * - probability_constants.h - Probability bounds and safety limits
 * - threshold_constants.h - Algorithmic and statistical thresholds
 * - benchmark_constants.h - Performance testing and algorithm parameters
 * - robust_constants.h - Robust estimation parameters
 * - statistical_methods_constants.h - Bayesian, bootstrap, cross-validation
 * - goodness_of_fit_constants.h - Critical values for goodness-of-fit tests
 */

// Include all focused constants headers
#include "precision_constants.h"
#include "mathematical_constants.h"
#include "statistical_constants.h"
#include "probability_constants.h"
#include "threshold_constants.h"
#include "benchmark_constants.h"
#include "robust_constants.h"
#include "statistical_methods_constants.h"
#include "goodness_of_fit_constants.h"

// Platform-specific constants are available separately in:
// #include "../platform/platform_constants.h"  // SIMD, parallel processing, etc.

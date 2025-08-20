#pragma once

/**
 * @file core/essential_constants.h
 * @brief Essential constants commonly needed across all libstats distributions
 *
 * This header includes only the most frequently used constants that are needed
 * by virtually every distribution implementation. For specialized constants,
 * include the specific headers directly:
 *
 * - threshold_constants.h     - Algorithm thresholds and limits
 * - benchmark_constants.h     - Performance testing parameters
 * - robust_constants.h        - Robust estimation parameters
 * - statistical_methods_constants.h - Bayesian, bootstrap constants
 * - goodness_of_fit_constants.h - Critical values for tests
 * - probability_constants.h   - Probability bounds and limits
 */

// The three most commonly used constants headers
#include "mathematical_constants.h"  // π, e, sqrt(2π), common mathematical values
#include "precision_constants.h"     // Tolerances, epsilons, convergence criteria
#include "statistical_constants.h"   // Critical values, statistical parameters

// Note: Platform constants available separately via platform headers
// Note: Specialized constants available via their specific headers as needed

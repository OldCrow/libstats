#pragma once

/**
 * @file core/constants.h
 * @brief Convenience umbrella header for all libstats constants
 *
 * Includes all three constant groups. Use this when you need broad access
 * to constants and compilation time is not a concern. For more focused
 * dependencies, include the specific headers directly:
 *
 *   math_constants.h         — mathematical values, precision, numerical limits
 *   statistical_constants.h  — critical values, probability bounds, thresholds
 *   performance_constants.h  — benchmark iteration counts and timing bounds
 *
 * Platform constants (SIMD widths, parallel thresholds) are separate:
 *   ../platform/platform_constants.h
 */

#include "math_constants.h"
#include "performance_constants.h"
#include "statistical_constants.h"

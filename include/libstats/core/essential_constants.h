#pragma once

/**
 * @file core/essential_constants.h
 * @brief Convenience header including the two most commonly needed constant groups
 *
 * Most distribution implementations and algorithms need both mathematical
 * constants and statistical domain constants. This header provides them
 * in a single include.
 *
 * For performance testing constants (benchmark tools only), also include
 * performance_constants.h.
 */

#include "math_constants.h"         // Mathematical values, precision, numerical limits
#include "statistical_constants.h"  // Critical values, probability bounds, thresholds

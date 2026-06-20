#pragma once

/**
 * @file stats/analysis/analysis.h
 * @brief Umbrella include for the stats::analysis namespace.
 *
 * Includes all generic analysis utilities. Distribution-specific headers
 * (gaussian_analysis.h, exponential_analysis.h, …) must be included
 * explicitly when needed.
 */

#include "bootstrap.h"
#include "cross_validation.h"
#include "goodness_of_fit.h"
#include "information_criteria.h"

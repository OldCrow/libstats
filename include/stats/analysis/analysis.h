#pragma once

/**
 * @file stats/analysis/analysis.h
 * @brief Umbrella include for the stats::analysis namespace.
 *
 * Includes all generic analysis utilities. Distribution-specific headers
 * (gaussian_analysis.h, exponential_analysis.h, …) must be included
 * explicitly when needed.
 */

#include "libstats/stats/analysis/bootstrap.h"
#include "libstats/stats/analysis/cross_validation.h"
#include "libstats/stats/analysis/goodness_of_fit.h"
#include "libstats/stats/analysis/information_criteria.h"
#include "libstats/stats/analysis/statistical_utilities.h"

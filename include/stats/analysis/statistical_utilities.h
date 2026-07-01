#pragma once

/**
 * @file stats/analysis/statistical_utilities.h
 * @brief General-purpose statistical utility functions (stats::analysis:: public API).
 *
 * These functions were previously only in stats::detail:: with no public callers.
 * Promoted to stats::analysis:: in v2.0.0 (D3 resolution, API rationalization).
 *
 * Migration: there is no old public form to migrate from — this is new public API.
 */

#include <array>
#include <span>
#include <vector>

namespace stats::analysis {

/**
 * @brief Compute the empirical CDF for a data sample.
 *
 * Sorts the data internally, then assigns each datum the plotting position
 * (i+1)/n, where i is its rank (0-based) and n is the sample size. Equivalent
 * to Hazen's midpoint formula shifted by half a step; compatible with standard
 * Kolmogorov-Smirnov comparisons.
 *
 * @param data  Sample values (need not be sorted; any finite doubles accepted).
 * @return      Vector of length data.size() containing the CDF values in
 *              increasing order, corresponding to the sorted sample.
 *
 * @note Returns an empty vector for empty input.
 */
[[nodiscard]] std::vector<double> empirical_cdf(std::span<const double> data);

/**
 * @brief Compute quantile values from a data sample using linear interpolation.
 *
 * Sorts the data internally and applies linear interpolation between adjacent
 * order statistics (equivalent to R's quantile type 7 / numpy's 'linear').
 *
 * @param data       Sample values (need not be sorted).
 * @param quantiles  Quantile levels, each in [0, 1].
 * @return           Vector of quantile values, one per entry of `quantiles`.
 *
 * @throws std::invalid_argument if `data` is empty, or if any quantile level
 *         is outside [0, 1].
 */
[[nodiscard]] std::vector<double> calculate_quantiles(std::span<const double> data,
                                                      std::span<const double> quantiles);

/**
 * @brief Compute the first four sample moments of a data set.
 *
 * @param data  Sample values (all must be finite).
 * @return      Array {mean, variance, skewness, excess_kurtosis}.
 *              - variance: Bessel-corrected (n-1 denominator).
 *              - skewness / excess_kurtosis: NaN when variance is zero
 *                (e.g., constant data).
 *
 * @throws std::invalid_argument if `data` is empty or contains non-finite values.
 */
[[nodiscard]] std::array<double, 4> sample_moments(std::span<const double> data);

/**
 * @brief Check whether a data vector is valid for statistical fitting.
 *
 * Returns true if and only if every element is finite (no NaN or infinity).
 * Does not check for sufficient sample size; use data.size() for that.
 *
 * @param data  Data to validate.
 * @return      true if all values are finite, false otherwise.
 */
[[nodiscard]] bool validate_fitting_data(std::span<const double> data) noexcept;

}  // namespace stats::analysis

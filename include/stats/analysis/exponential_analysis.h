#pragma once

/**
 * @file stats/analysis/exponential_analysis.h
 * @brief Exponential-specific statistical analysis functions.
 *
 * Tests and inference for exponential data, extracted in v2.0.0.
 * Migration:
 *   ExponentialDistribution::coefficientOfVariationTest(data, alpha)
 *   → stats::analysis::exponential::coefficientOfVariationTest(data, alpha)
 */

#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace stats::analysis::exponential {

/**
 * @brief Exact chi-squared confidence interval for the exponential rate λ.
 *
 * Given MLE λ̂ = n/Σxᵢ, the exact CI uses the fact that 2nλ/λ̂ ~ χ²(2n):
 *   lower = χ²(α/2,  2n) / (2·Σxᵢ)
 *   upper = χ²(1-α/2, 2n) / (2·Σxᵢ)
 *
 * @param data             Positive-valued observations.
 * @param confidence_level CI level, e.g. 0.95.
 * @return {lower, upper} for λ.
 */
[[nodiscard]] std::pair<double, double>
confidenceIntervalRate(const std::vector<double>& data,
                       double confidence_level = 0.95);

/**
 * @brief Exponentiality test using the coefficient of variation.
 *
 * Tests H₀: data is exponential (CV = 1) using an asymptotic normal
 * approximation: z = |CV − 1| / (1/√n).
 *
 * @param data  Positive-valued observations (n ≥ 2).
 * @param alpha Significance level (default 0.05).
 * @return {cv_statistic, p_value, reject_null}
 */
[[nodiscard]] std::tuple<double, double, bool>
coefficientOfVariationTest(const std::vector<double>& data,
                           double alpha = 0.05);

}  // namespace stats::analysis::exponential

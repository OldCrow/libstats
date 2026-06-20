#pragma once

/**
 * @file stats/analysis/gamma_analysis.h
 * @brief Gamma-specific statistical analysis functions.
 *
 * Extracted in v2.0.0.
 * Migration:
 *   GammaDistribution::normalApproximationTest(data, alpha)
 *   → stats::analysis::gamma::normalApproximationTest(data, alpha)
 */

#include <tuple>
#include <vector>

namespace stats::analysis::gamma {

/**
 * @brief Test whether a Gamma(α,β) normal approximation is valid.
 *
 * For large α, Gamma(α,β) ≈ N(α/β, α/β²).  This test checks whether the
 * sample mean falls within the normal approximation's confidence band for α̂.
 * Returns the band bounds and whether the approximation is valid at the given
 * significance level.
 *
 * @param data               Positive observations.
 * @param significance_level Significance level (default 0.05).
 * @return {lower_bound, upper_bound, approximation_is_valid}
 */
[[nodiscard]] std::tuple<double, double, bool>
normalApproximationTest(const std::vector<double>& data,
                        double significance_level = 0.05);

}  // namespace stats::analysis::gamma

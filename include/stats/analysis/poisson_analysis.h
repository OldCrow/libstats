#pragma once

/**
 * @file stats/analysis/poisson_analysis.h
 * @brief Poisson-specific statistical analysis functions.
 *
 * These functions test Poisson-specific properties (equidispersion, rate
 * stability, excess zeros) and provide exact inference for the rate parameter.
 * They replace the static methods previously on PoissonDistribution.
 *
 * Extracted in v2.0.0. Migration:
 *   PoissonDistribution::chiSquareGoodnessOfFit(data, dist, alpha)
 *   → stats::analysis::poisson::chiSquareGoodnessOfFit(data, dist, alpha)
 */

#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace stats {
class PoissonDistribution;  // forward declaration
}

namespace stats::analysis::poisson {

// ---------------------------------------------------------------------------
// Exact rate inference
// ---------------------------------------------------------------------------

/**
 * @brief Exact chi-squared confidence interval for the Poisson rate λ.
 *
 * Uses the relationship between the Poisson and chi-squared distributions:
 * given total count T = Σxᵢ from n observations,
 *   lower = χ²(α/2,  2T)   / (2n)
 *   upper = χ²(1-α/2, 2(T+1)) / (2n)
 *
 * @param data           Non-negative count observations.
 * @param confidence_level CI level, e.g. 0.95.
 * @return {lower, upper} for λ.
 */
[[nodiscard]] std::pair<double, double>
confidenceIntervalRate(const std::vector<double>& data,
                       double confidence_level = 0.95);

// ---------------------------------------------------------------------------
// Dispersion tests
// ---------------------------------------------------------------------------

/**
 * @brief Overdispersion test (dispersion index / variance-mean ratio).
 *
 * Tests H₀: data is Poisson (variance = mean) against H₁: variance > mean.
 * Test statistic: C = (n−1)·S²/x̄ ~ χ²(n−1) under H₀.
 *
 * @param data              Non-negative count observations.
 * @param significance_level Significance level (default 0.05).
 * @return {dispersion_index, p_value, is_overdispersed}
 */
[[nodiscard]] std::tuple<double, double, bool>
overdispersionTest(const std::vector<double>& data,
                   double significance_level = 0.05);

/**
 * @brief Test for excess zeros relative to a Poisson model.
 *
 * Tests whether observed zero count exceeds what a Poisson(λ̂) would predict,
 * using a z-test on the count of zeros.
 *
 * @param data               Non-negative count observations.
 * @param significance_level Significance level (default 0.05).
 * @return {z_statistic, p_value, has_excess_zeros}
 */
[[nodiscard]] std::tuple<double, double, bool>
excessZerosTest(const std::vector<double>& data,
                double significance_level = 0.05);

/**
 * @brief Rate stability test for time-ordered count data.
 *
 * Tests H₀: Poisson rate λ is constant over time (no trend).
 * Fits a linear regression to the counts and tests the slope for zero.
 *
 * @param data               Chronologically ordered count observations.
 * @param significance_level Significance level (default 0.05).
 * @return {t_statistic, p_value, rate_is_stable}
 */
[[nodiscard]] std::tuple<double, double, bool>
rateStabilityTest(const std::vector<double>& data,
                  double significance_level = 0.05);

// ---------------------------------------------------------------------------
// Goodness-of-fit
// ---------------------------------------------------------------------------

/**
 * @brief Chi-square goodness-of-fit test for a Poisson distribution.
 *
 * Groups rare events to maintain expected cell frequencies ≥ 5, then
 * computes the chi-square statistic with one degree of freedom subtracted
 * for the estimated λ.
 *
 * @param data               Non-negative count observations.
 * @param distribution       Hypothesised Poisson distribution.
 * @param significance_level Significance level (default 0.05).
 * @return {chi_square_statistic, p_value, reject_null}
 */
[[nodiscard]] std::tuple<double, double, bool>
chiSquareGoodnessOfFit(const std::vector<double>& data,
                       const stats::PoissonDistribution& distribution,
                       double significance_level = 0.05);

}  // namespace stats::analysis::poisson

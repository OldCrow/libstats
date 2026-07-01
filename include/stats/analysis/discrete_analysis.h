#pragma once

/**
 * @file stats/analysis/discrete_analysis.h
 * @brief Statistical tests specific to discrete uniform distributions.
 *
 * Extracted in v2.0.0 as part of the per-distribution analysis layer.
 *
 * Two tests are provided:
 *  - runsTest         — Wald-Wolfowitz runs test for independence/randomness
 *  - frequencyTest    — chi-square uniformity test over a known integer range
 *
 * Include this header explicitly when needed.  It is not part of the
 * analysis.h umbrella because it is distribution-specific.
 */

#include <tuple>
#include <vector>

namespace stats::analysis::discrete {

/**
 * @brief Wald-Wolfowitz runs test for independence/randomness.
 *
 * Tests H₀: the sequence is randomly ordered (no serial structure).
 * Each element is classified as above or below the sample median;
 * values equal to the median are excluded.  The run count is
 * approximately normally distributed under H₀.
 *
 * @param data  Integer or real-valued sequence (≥ 8 elements).
 * @param alpha Significance level (default 0.05).
 * @return {z_statistic, two_tailed_p_value, reject_null}
 * @throws std::invalid_argument if n < 8, too many median ties, or
 *         alpha ∉ (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> runsTest(const std::vector<double>& data,
                                                        double alpha = 0.05);

/**
 * @brief Chi-square uniformity test over a known integer range.
 *
 * Tests H₀: data is uniformly distributed over {lo, …, hi}.
 * Convenience wrapper around the generic stats::analysis::chiSquaredGoodnessOfFit
 * that constructs the reference DiscreteDistribution(lo, hi) internally and
 * filters data to the expected support before testing.
 *
 * Because lo and hi are supplied by the caller (not estimated from data),
 * df = bins − 1 and estimated_params = 0.
 *
 * @param data  Observed data (values outside [lo, hi] are dropped).
 * @param lo    Lower bound (inclusive).
 * @param hi    Upper bound (inclusive); must be > lo.
 * @param alpha Significance level (default 0.05).
 * @return {chi2_statistic, p_value, reject_null}
 * @throws std::invalid_argument if lo ≥ hi, fewer than 5 values fall within
 *         [lo, hi], or alpha ∉ (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> frequencyTest(const std::vector<double>& data,
                                                             int lo, int hi, double alpha = 0.05);

}  // namespace stats::analysis::discrete

#pragma once

/**
 * @file stats/analysis/binomial_analysis.h
 * @brief Binomial-specific statistical analysis functions.
 *
 * Exact and asymptotic inference for proportions, added in v2.0.0.
 */

#include <stdexcept>
#include <tuple>
#include <utility>

namespace stats::analysis::binomial {

/**
 * @brief Exact Clopper-Pearson confidence interval for a proportion.
 *
 * Inverts the exact binomial test to produce a conservative CI.
 *   lower = Beta(α/2,   k,   n-k+1)
 *   upper = Beta(1-α/2, k+1, n-k  )
 * where Beta(p, a, b) is the p-th quantile of the Beta(a,b) distribution.
 *
 * @param k                Number of successes (0 ≤ k ≤ n).
 * @param n                Number of trials (n ≥ 1).
 * @param confidence_level CI level, e.g. 0.95.
 * @return {lower, upper} for the proportion p.
 */
[[nodiscard]] std::pair<double, double> clopperPearsonCI(int k, int n,
                                                         double confidence_level = 0.95);

/**
 * @brief One-proportion z-test.
 *
 * Tests H₀: p = p₀ against H₁: p ≠ p₀ using the asymptotic normal
 * approximation: z = (p̂ − p₀) / √(p₀(1−p₀)/n).
 *
 * @param k    Number of successes.
 * @param n    Number of trials.
 * @param p0   Hypothesised proportion (0 < p₀ < 1).
 * @param alpha Significance level (default 0.05).
 * @return {z_statistic, p_value, reject_null}
 */
[[nodiscard]] std::tuple<double, double, bool> proportionZTest(int k, int n, double p0,
                                                               double alpha = 0.05);

/**
 * @brief Two-proportion z-test (pooled).
 *
 * Tests H₀: p₁ = p₂ using the pooled standard error:
 *   z = (p̂₁ − p̂₂) / √(p̂·(1−p̂)·(1/n₁ + 1/n₂))  where p̂ = (k₁+k₂)/(n₁+n₂).
 *
 * @param k1, n1 Successes and trials in group 1.
 * @param k2, n2 Successes and trials in group 2.
 * @param alpha  Significance level (default 0.05).
 * @return {z_statistic, p_value, reject_null}
 */
[[nodiscard]] std::tuple<double, double, bool> twoProportionZTest(int k1, int n1, int k2, int n2,
                                                                  double alpha = 0.05);

}  // namespace stats::analysis::binomial

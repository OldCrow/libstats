#pragma once

/**
 * @file stats/analysis/goodness_of_fit.h
 * @brief Generic goodness-of-fit tests for any libstats distribution.
 *
 * Three tests, each constrained by distribution category:
 *   - kolmogorovSmirnovTest    — ContinuousDistribution only (MC-12)
 *   - andersonDarlingTest      — ContinuousDistribution only (MC-12)
 *   - chiSquaredGoodnessOfFit  — DiscreteDistribution only  (5E, v2.0.0)
 *   - likelihoodRatioTest      — AnyDistribution
 *
 * Extracted in v2.0.0. Migration:
 *   GaussianDistribution::kolmogorovSmirnovTest(data, dist, alpha)
 *   → stats::analysis::kolmogorovSmirnovTest(data, dist, alpha)
 *
 *   DiscreteDistribution::chiSquaredGoodnessOfFitTest(data, dist, alpha)
 *   → stats::analysis::chiSquaredGoodnessOfFit(data, dist, alpha)
 */

#include "libstats/core/distribution_concepts.h"
#include "libstats/core/math_utils.h"  // detail::calculate_ks_statistic, calculate_ad_statistic
#include "libstats/core/statistical_constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace stats::analysis {

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov test
// ---------------------------------------------------------------------------

/**
 * @brief Kolmogorov-Smirnov goodness-of-fit test.
 *
 * @tparam D Continuous distribution satisfying stats::concepts::ContinuousDistribution.
 *           Applying this test to a discrete distribution is a compile-time error.
 * @param data  Observed data vector (unsorted; sorted internally).
 * @param dist  Distribution to test against.
 * @param alpha Significance level (default 0.05).
 * @return {ks_statistic, p_value, reject_null}
 *
 * p-value uses the asymptotic Kolmogorov approximation: 2·exp(−2·n·D²).
 */
template <concepts::ContinuousDistribution D>
[[nodiscard]] std::tuple<double, double, bool> kolmogorovSmirnovTest(
    const std::vector<double>& data, const D& dist, double alpha = 0.05) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    // D inherits DistributionBase; implicit upcast to const DistributionBase&
    const double ks_stat = detail::calculate_ks_statistic(data, dist);

    const double n = static_cast<double>(data.size());
    // Asymptotic Kolmogorov approximation (distribution-agnostic)
    const double p_value =
        std::min(1.0, std::max(0.0, 2.0 * std::exp(-2.0 * n * ks_stat * ks_stat)));

    return {ks_stat, p_value, p_value < alpha};
}

// ---------------------------------------------------------------------------
// Anderson-Darling test
// ---------------------------------------------------------------------------

/**
 * @brief Anderson-Darling goodness-of-fit test.
 *
 * @tparam D Continuous distribution satisfying stats::concepts::ContinuousDistribution.
 *           Applying this test to a discrete distribution is a compile-time error.
 * @param data  Observed data vector.
 * @param dist  Distribution to test against.
 * @param alpha Significance level (default 0.05).
 * @return {ad_statistic, p_value, reject_null}
 *
 * p-value: single-segment exponential approximation calibrated to the 5% critical
 * value of the distribution-agnostic AD asymptotic distribution (Stephens 1974).
 * For distribution-specific critical values (e.g. Gaussian AD with Lilliefors
 * correction) use the per-distribution analysis header instead.
 *
 * **Continuity guarantee (MC-6)**: a single formula is used for all A < 13,
 * eliminating the jump discontinuity that existed between the two-segment
 * approximation in earlier versions.
 */
template <concepts::ContinuousDistribution D>
[[nodiscard]] std::tuple<double, double, bool> andersonDarlingTest(const std::vector<double>& data,
                                                                   const D& dist,
                                                                   double alpha = 0.05) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    const double ad_stat = detail::calculate_ad_statistic(data, dist);

    // Single-formula asymptotic approximation: p ≈ 4.48·exp(−1.8·A).
    // Calibrated to the 5% critical value (A ≈ 2.49 → p ≈ 0.05) of the
    // distribution-agnostic AD test (Stephens 1974, Table 4.2). The constant
    // 1.5 shifts the intercept so that the formula naturally saturates at 1.0
    // (via the min-clamp) for small A. A single segment is used throughout
    // [0, 13) to guarantee strict monotone decrease (MC-6).
    const double p_value = std::min(1.0, std::max(0.0, std::exp(-1.8 * ad_stat + 1.5)));

    return {ad_stat, p_value, p_value < alpha};
}

// ---------------------------------------------------------------------------
// Likelihood ratio test
// ---------------------------------------------------------------------------

/**
 * @brief Likelihood ratio test comparing two nested models.
 *
 * Tests H₀: restricted model is adequate against H₁: unrestricted model.
 * The statistic −2(ℓ_r − ℓ_u) is asymptotically χ²(df) where df = Δparameters.
 *
 * @tparam D Any distribution satisfying stats::concepts::AnyDistribution.
 * @param data               Observed data vector.
 * @param restricted         Restricted (null) model.
 * @param unrestricted       Unrestricted (alternative) model.
 * @param alpha              Significance level (default 0.05).
 * @param df                 Degrees of freedom (number of constraints; df > 0).
 *                           Must be specified explicitly (MC-11):
 *                           - Nested models: df = k_unrestricted − k_restricted.
 *                           - Same distribution, different parameters: df = number
 *                             of fixed parameters tested (e.g. 1 for a single mean).
 *                           A zero or negative value causes an exception.
 * @return {lr_statistic, p_value, reject_null}
 *
 * **Migration note (v2.0.0, MC-11):** `df` is now an explicit required parameter.
 * Previous versions silently inferred df, which was ambiguous when both models
 * had the same number of parameters. Callers must supply df.
 */
template <concepts::AnyDistribution D>
[[nodiscard]] std::tuple<double, double, bool> likelihoodRatioTest(const std::vector<double>& data,
                                                                   const D& restricted,
                                                                   const D& unrestricted, int df,
                                                                   double alpha = 0.05) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");
    if (df <= 0)
        throw std::invalid_argument(
            "Degrees of freedom must be positive. "
            "Supply df = k_unrestricted - k_restricted for nested models, or the "
            "number of tested constraints for same-distribution comparisons.");

    double ll_r = 0.0, ll_u = 0.0;
    for (double x : data) {
        ll_r += restricted.getLogProbability(x);
        ll_u += unrestricted.getLogProbability(x);
    }

    const double lr_stat = 2.0 * (ll_u - ll_r);
    // ANA-4: lr_stat <= 0 is a valid result — the restricted model fits at least as
    // well as the unrestricted model (models are equivalent). Return p = 1, no
    // rejection rather than throwing. Callers testing for strict improvement should
    // check lr_stat > 0 themselves.
    if (lr_stat <= 0.0)
        return {0.0, 1.0, false};

    const double p_value = 1.0 - detail::chi_squared_cdf(lr_stat, df);
    return {lr_stat, p_value, p_value < alpha};
}

// ---------------------------------------------------------------------------
// Chi-squared goodness-of-fit test (discrete distributions only)
// ---------------------------------------------------------------------------

/**
 * @brief Pearson chi-squared goodness-of-fit test for discrete distributions.
 *
 * @tparam D Discrete distribution satisfying stats::concepts::DiscreteDistribution.
 *           Applying this test to a continuous distribution is a compile-time error.
 * @param data             Observed data (values rounded to nearest integer).
 * @param dist             Fitted discrete distribution.
 * @param alpha            Significance level (default 0.05).
 * @param estimated_params Number of parameters estimated from @p data (default 0).
 *                         Pass `dist.getNumParameters()` when the distribution was
 *                         fitted to the same data being tested; omitting this makes
 *                         the test too liberal (df is over-counted).
 * @return {chi2_statistic, p_value, reject_null}
 *
 * Bins observed counts over [min(data), max(data)], computes expected
 * frequencies from the distribution PMF, and merges cells with expected
 * count < 1 before computing the Pearson statistic with
 * df = merged_bins - 1 - estimated_params.
 *
 * Added in v2.0.0 (5E). Migration:
 *   DiscreteDistribution::chiSquaredGoodnessOfFitTest(data, dist, alpha)
 *   → stats::analysis::chiSquaredGoodnessOfFit(data, dist, alpha)
 */
template <concepts::DiscreteDistribution D>
[[nodiscard]] std::tuple<double, double, bool> chiSquaredGoodnessOfFit(
    const std::vector<double>& data, const D& dist, double alpha = 0.05, int estimated_params = 0) {
    if (data.size() < 5)
        throw std::invalid_argument(
            "At least 5 observations required for chi-square goodness-of-fit");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    const std::size_t n = data.size();
    const auto [it_min, it_max] = std::minmax_element(data.begin(), data.end());
    const int lo = static_cast<int>(std::round(*it_min));
    const int hi = static_cast<int>(std::round(*it_max));
    const int nbins = hi - lo + 1;

    std::vector<double> observed(static_cast<std::size_t>(nbins), 0.0);
    for (double v : data) {
        const int k = static_cast<int>(std::round(v));
        if (k >= lo && k <= hi)
            observed[static_cast<std::size_t>(k - lo)] += 1.0;
    }

    std::vector<double> expected(static_cast<std::size_t>(nbins));
    for (int k = lo; k <= hi; ++k) {
        const double p_k = dist.getProbability(static_cast<double>(k));
        expected[static_cast<std::size_t>(k - lo)] = static_cast<double>(n) * std::max(0.0, p_k);
    }

    // ANA-3: merge cells with expected count < 5 (Cochran 1954 guideline), not < 1.
    // Using < 1 retained too many bins, inflated df, and produced over-liberal p-values.
    // The Poisson-specific chiSquareGoodnessOfFit in poisson_analysis.cpp correctly uses 5.
    std::vector<double> obs_m, exp_m;
    double oa = 0.0, ea = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(nbins); ++i) {
        oa += observed[i];
        ea += expected[i];
        if (ea >= 5.0 || i == static_cast<std::size_t>(nbins) - 1) {
            obs_m.push_back(oa);
            exp_m.push_back(ea);
            oa = ea = 0.0;
        }
    }

    double chi2 = 0.0;
    for (std::size_t i = 0; i < exp_m.size(); ++i) {
        if (exp_m[i] > 0.0) {
            const double d = obs_m[i] - exp_m[i];
            chi2 += d * d / exp_m[i];
        }
    }

    const int df = static_cast<int>(exp_m.size()) - 1 - estimated_params;
    if (df <= 0)
        throw std::invalid_argument(
            "Insufficient distinct values after cell merging; collect more data");

    const double p_value = 1.0 - detail::chi_squared_cdf(chi2, static_cast<double>(df));
    return {chi2, p_value, p_value < alpha};
}

}  // namespace stats::analysis

#pragma once

/**
 * @file stats/analysis/goodness_of_fit.h
 * @brief Generic goodness-of-fit tests for any libstats distribution.
 *
 * Functions here accept any type D satisfying stats::concepts::AnyDistribution.
 * They replace the identical static methods previously duplicated across 16
 * distribution classes.
 *
 * Extracted in v2.0.0. Migration:
 *   GaussianDistribution::kolmogorovSmirnovTest(data, dist, alpha)
 *   → stats::analysis::kolmogorovSmirnovTest(data, dist, alpha)
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../core/distribution_concepts.h"
#include "../../core/math_utils.h"         // detail::calculate_ks_statistic, calculate_ad_statistic
#include "../../core/statistical_constants.h"

namespace stats::analysis {

// ---------------------------------------------------------------------------
// Kolmogorov-Smirnov test
// ---------------------------------------------------------------------------

/**
 * @brief Kolmogorov-Smirnov goodness-of-fit test.
 *
 * @tparam D Any distribution satisfying stats::concepts::AnyDistribution.
 * @param data  Observed data vector (unsorted; sorted internally).
 * @param dist  Distribution to test against.
 * @param alpha Significance level (default 0.05).
 * @return {ks_statistic, p_value, reject_null}
 *
 * p-value uses the asymptotic approximation: 2·exp(−2·n·D²).
 */
template <concepts::AnyDistribution D>
[[nodiscard]] std::tuple<double, double, bool>
kolmogorovSmirnovTest(const std::vector<double>& data,
                      const D& dist,
                      double alpha = 0.05) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    // D inherits DistributionBase; implicit upcast to const DistributionBase&
    const double ks_stat = detail::calculate_ks_statistic(data, dist);

    const double n = static_cast<double>(data.size());
    // Asymptotic Kolmogorov approximation (distribution-agnostic)
    const double p_value = std::min(1.0, std::max(0.0,
        2.0 * std::exp(-2.0 * n * ks_stat * ks_stat)));

    return {ks_stat, p_value, p_value < alpha};
}

// ---------------------------------------------------------------------------
// Anderson-Darling test
// ---------------------------------------------------------------------------

/**
 * @brief Anderson-Darling goodness-of-fit test.
 *
 * @tparam D Any distribution satisfying stats::concepts::AnyDistribution.
 * @param data  Observed data vector.
 * @param dist  Distribution to test against.
 * @param alpha Significance level (default 0.05).
 * @return {ad_statistic, p_value, reject_null}
 *
 * p-value uses a general exponential approximation. For distribution-specific
 * critical values (e.g. Gaussian AD with Lilliefors correction) use the
 * per-distribution analysis header instead.
 */
template <concepts::AnyDistribution D>
[[nodiscard]] std::tuple<double, double, bool>
andersonDarlingTest(const std::vector<double>& data,
                    const D& dist,
                    double alpha = 0.05) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    const double ad_stat = detail::calculate_ad_statistic(data, dist);

    // General exponential p-value approximation (distribution-agnostic)
    double p_value;
    if (ad_stat >= 13.0) {
        p_value = 0.0;
    } else if (ad_stat >= 6.0) {
        p_value = std::exp(-1.28 * ad_stat);
    } else {
        p_value = std::exp(-1.8 * ad_stat + 1.5);
    }
    p_value = std::min(1.0, std::max(0.0, p_value));

    return {ad_stat, p_value, p_value < alpha};
}

// ---------------------------------------------------------------------------
// Likelihood ratio test
// ---------------------------------------------------------------------------

/**
 * @brief Likelihood ratio test comparing two nested models.
 *
 * Tests H₀: restricted model is adequate against H₁: unrestricted model.
 * The statistic −2(ℓ_r − ℓ_u) is asymptotically χ²(df) where
 * df = Δparameters.
 *
 * @tparam D Any distribution satisfying stats::concepts::AnyDistribution.
 * @param data               Observed data vector.
 * @param restricted         Restricted (null) model.
 * @param unrestricted       Unrestricted (alternative) model.
 * @param alpha              Significance level (default 0.05).
 * @return {lr_statistic, p_value, reject_null}
 */
template <concepts::AnyDistribution D>
[[nodiscard]] std::tuple<double, double, bool>
likelihoodRatioTest(const std::vector<double>& data,
                    const D& restricted,
                    const D& unrestricted,
                    double alpha = 0.05) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    double ll_r = 0.0, ll_u = 0.0;
    for (double x : data) {
        ll_r += restricted.getLogProbability(x);
        ll_u += unrestricted.getLogProbability(x);
    }

    const double lr_stat = 2.0 * (ll_u - ll_r);
    // A non-positive LR stat means the restricted model fits at least as well,
    // which violates the nested-model assumption (or both models are identical).
    if (lr_stat <= 0)
        throw std::invalid_argument(
            "LR statistic is non-positive: restricted model fits at least as well as "
            "unrestricted. Check model ordering or whether models are identical.");

    const int k_r = restricted.getNumParameters();
    const int k_u = unrestricted.getNumParameters();
    // Same-type comparison (both models have k parameters): treat as a joint
    // hypothesis test on all k parameters → df = k_r.
    // Different-type comparison (nested models): df = |k_u - k_r|.
    const int df = (k_u != k_r) ? std::abs(k_u - k_r) : k_r;
    if (df <= 0)
        throw std::invalid_argument("Degrees of freedom must be positive");

    const double p_value = 1.0 - detail::chi_squared_cdf(lr_stat, df);
    return {lr_stat, p_value, p_value < alpha};
}

}  // namespace stats::analysis

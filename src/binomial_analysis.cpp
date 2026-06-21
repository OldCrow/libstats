#include "libstats/stats/analysis/binomial_analysis.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)

#include "libstats/core/math_utils.h"
#include "libstats/core/statistical_constants.h"

#include <cmath>
#include <stdexcept>

namespace stats::analysis::binomial {

std::pair<double, double>
clopperPearsonCI(int k, int n, double confidence_level) {
    if (n < 1)
        throw std::invalid_argument("n must be at least 1");
    if (k < 0 || k > n)
        throw std::invalid_argument("k must satisfy 0 <= k <= n");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be in (0, 1)");

    const double alpha = 1.0 - confidence_level;

    // Lower bound: Beta(α/2, k, n-k+1) — 0 when k=0
    const double lower = (k == 0) ? 0.0
        : detail::inverse_beta_i(alpha / 2.0,
                                  static_cast<double>(k),
                                  static_cast<double>(n - k + 1));

    // Upper bound: Beta(1-α/2, k+1, n-k) — 1 when k=n
    const double upper = (k == n) ? 1.0
        : detail::inverse_beta_i(1.0 - alpha / 2.0,
                                  static_cast<double>(k + 1),
                                  static_cast<double>(n - k));

    return {lower, upper};
}

std::tuple<double, double, bool>
proportionZTest(int k, int n, double p0, double alpha) {
    if (n < 1)
        throw std::invalid_argument("n must be at least 1");
    if (k < 0 || k > n)
        throw std::invalid_argument("k must satisfy 0 <= k <= n");
    if (p0 <= 0.0 || p0 >= 1.0)
        throw std::invalid_argument("p0 must be in (0, 1)");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("alpha must be in (0, 1)");

    const double p_hat = static_cast<double>(k) / static_cast<double>(n);
    const double se    = std::sqrt(p0 * (1.0 - p0) / static_cast<double>(n));
    const double z     = (p_hat - p0) / se;
    const double p_val = 2.0 * (1.0 - detail::normal_cdf(std::abs(z)));

    return {z, p_val, p_val < alpha};
}

std::tuple<double, double, bool>
twoProportionZTest(int k1, int n1, int k2, int n2, double alpha) {
    if (n1 < 1 || n2 < 1)
        throw std::invalid_argument("n1 and n2 must each be at least 1");
    if (k1 < 0 || k1 > n1 || k2 < 0 || k2 > n2)
        throw std::invalid_argument("k values must satisfy 0 <= k <= n");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("alpha must be in (0, 1)");

    const double p1  = static_cast<double>(k1) / static_cast<double>(n1);
    const double p2  = static_cast<double>(k2) / static_cast<double>(n2);
    const double p   = static_cast<double>(k1 + k2) / static_cast<double>(n1 + n2);
    const double se  = std::sqrt(p * (1.0 - p) * (1.0 / n1 + 1.0 / n2));

    if (se <= 0.0)
        throw std::runtime_error("Standard error is zero (perfect separation or degenerate data)");

    const double z     = (p1 - p2) / se;
    const double p_val = 2.0 * (1.0 - detail::normal_cdf(std::abs(z)));

    return {z, p_val, p_val < alpha};
}

}  // namespace stats::analysis::binomial

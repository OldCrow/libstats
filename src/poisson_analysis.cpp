#include "libstats/stats/analysis/poisson_analysis.h"

#include "libstats/core/math_utils.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/distributions/poisson.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace stats::analysis::poisson {

// ---------------------------------------------------------------------------
// Exact rate inference
// ---------------------------------------------------------------------------

std::pair<double, double>
confidenceIntervalRate(const std::vector<double>& data, double confidence_level) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be in (0, 1)");
    for (double x : data)
        if (x < 0.0 || !std::isfinite(x))
            throw std::invalid_argument("Poisson data must be non-negative and finite");

    const double alpha = 1.0 - confidence_level;
    const double n = static_cast<double>(data.size());
    const double total = std::accumulate(data.begin(), data.end(), 0.0);

    // Exact Poisson CI via the chi-squared relationship (Garwood, 1936):
    //   lower = χ²(α/2,  2T)   / (2n)  where T = Σxᵢ
    //   upper = χ²(1-α/2, 2(T+1)) / (2n)
    const double lower = (total > 0.0)
        ? detail::inverse_chi_squared_cdf(alpha / 2.0, 2.0 * total) / (2.0 * n)
        : 0.0;
    const double upper =
        detail::inverse_chi_squared_cdf(1.0 - alpha / 2.0, 2.0 * (total + 1.0)) / (2.0 * n);

    return {lower, upper};
}

// ---------------------------------------------------------------------------
// Dispersion tests
// ---------------------------------------------------------------------------

std::tuple<double, double, bool>
overdispersionTest(const std::vector<double>& data, double significance_level) {
    if (data.size() < 2)
        throw std::invalid_argument("At least 2 data points required for overdispersion test");
    if (significance_level <= 0.0 || significance_level >= 1.0)
        throw std::invalid_argument("Significance level must be in (0, 1)");
    for (double x : data)
        if (x < 0.0 || !std::isfinite(x))
            throw std::invalid_argument("Poisson data must be non-negative and finite");

    const std::size_t n = data.size();
    const double nd = static_cast<double>(n);
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / nd;

    if (mean <= 0.0)
        throw std::invalid_argument("Sample mean must be positive for overdispersion test");

    double var = 0.0;
    for (double x : data) var += (x - mean) * (x - mean);
    var /= (nd - 1.0);

    // Dispersion index: D = (n-1)*S²/x̄  ~ χ²(n-1) under H₀: Poisson
    const double dispersion_index = (nd - 1.0) * var / mean;

    // One-sided p-value: P(χ²(n-1) > D)
    const double p_value = 1.0 - detail::chi_squared_cdf(dispersion_index,
                                                           static_cast<int>(n - 1));

    return {var / mean, p_value, p_value < significance_level};
}

std::tuple<double, double, bool>
excessZerosTest(const std::vector<double>& data, double significance_level) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (significance_level <= 0.0 || significance_level >= 1.0)
        throw std::invalid_argument("Significance level must be in (0, 1)");
    for (double x : data)
        if (x < 0.0 || !std::isfinite(x))
            throw std::invalid_argument("Poisson data must be non-negative and finite");

    const std::size_t n = data.size();
    const std::size_t observed_zeros =
        static_cast<std::size_t>(std::count(data.begin(), data.end(), 0.0));
    const double lambda_hat =
        std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);

    const double exp_neg_lambda = std::exp(-lambda_hat);
    const double expected_zeros = static_cast<double>(n) * exp_neg_lambda;
    const double variance_zeros =
        static_cast<double>(n) * exp_neg_lambda * (1.0 - exp_neg_lambda);

    if (variance_zeros <= 0.0)
        throw std::runtime_error("Variance of zero count is non-positive");

    const double z = (static_cast<double>(observed_zeros) - expected_zeros) /
                     std::sqrt(variance_zeros);
    const double p_value = 2.0 * (1.0 - detail::normal_cdf(std::abs(z)));

    return {z, p_value, p_value < significance_level};
}

std::tuple<double, double, bool>
rateStabilityTest(const std::vector<double>& data, double significance_level) {
    if (data.size() < 3)
        throw std::invalid_argument("At least 3 data points required for rate stability test");
    if (significance_level <= 0.0 || significance_level >= 1.0)
        throw std::invalid_argument("Significance level must be in (0, 1)");
    for (double x : data)
        if (x < 0.0 || !std::isfinite(x))
            throw std::invalid_argument("Poisson data must be non-negative and finite");

    const std::size_t n = data.size();
    const double nd = static_cast<double>(n);

    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i + 1);
        sum_x  += x;
        sum_y  += data[i];
        sum_xx += x * x;
        sum_xy += x * data[i];
    }

    const double mean_x = sum_x / nd;
    const double mean_y = sum_y / nd;
    const double denom  = sum_xx - nd * mean_x * mean_x;

    if (std::abs(denom) < detail::DEFAULT_TOLERANCE)
        throw std::runtime_error("Cannot perform regression: denominator too small");

    const double slope     = (sum_xy - nd * mean_x * mean_y) / denom;
    const double intercept = mean_y - slope * mean_x;

    double rss = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double res = data[i] - (intercept + slope * static_cast<double>(i + 1));
        rss += res * res;
    }

    const double mse      = rss / (nd - 2.0);
    const double se_slope = std::sqrt(mse / denom);
    const double t_stat   = slope / se_slope;
    const double p_value  = 2.0 * (1.0 - detail::t_cdf(std::abs(t_stat),
                                                          static_cast<double>(n - 2)));

    return {t_stat, p_value, p_value >= significance_level};
}

// ---------------------------------------------------------------------------
// Goodness-of-fit
// ---------------------------------------------------------------------------

std::tuple<double, double, bool>
chiSquareGoodnessOfFit(const std::vector<double>& data,
                       const stats::PoissonDistribution& distribution,
                       double significance_level) {
    if (data.size() < 5)
        throw std::invalid_argument("At least 5 data points required for chi-square test");
    if (significance_level <= 0.0 || significance_level >= 1.0)
        throw std::invalid_argument("Significance level must be in (0, 1)");
    for (double x : data)
        if (x < 0.0 || !std::isfinite(x))
            throw std::invalid_argument("Poisson data must be non-negative and finite");

    const std::size_t n = data.size();

    // Frequency table
    std::map<int, int> obs_freq;
    int max_val = 0;
    for (double x : data) {
        const int k = static_cast<int>(std::round(x));
        obs_freq[k]++;
        max_val = std::max(max_val, k);
    }

    // Group to keep expected frequency ≥ 5
    std::vector<int> grouped_obs;
    std::vector<double> expected_freq;
    int group_obs = 0, group_start = 0;

    for (int k = 0; k <= max_val + 5; ++k) {
        group_obs += obs_freq[k];
        const double exp_k = static_cast<double>(n) * distribution.getProbabilityExact(k);

        if (exp_k >= 5.0 || (group_obs > 0 && k >= max_val)) {
            grouped_obs.push_back(group_obs);
            double grp_exp = 0.0;
            for (int j = group_start; j <= k; ++j)
                grp_exp += static_cast<double>(n) * distribution.getProbabilityExact(j);
            expected_freq.push_back(grp_exp);

            group_start = k + 1;
            group_obs   = 0;

            if (k >= max_val && grp_exp < 1e-10)
                break;
        }
    }

    double chi2 = 0.0;
    for (std::size_t i = 0; i < grouped_obs.size(); ++i)
        if (expected_freq[i] > 0.0)
            chi2 += std::pow(grouped_obs[i] - expected_freq[i], 2.0) / expected_freq[i];

    const int df = static_cast<int>(grouped_obs.size()) - 1 - 1;  // -1 for estimated λ
    if (df <= 0)
        throw std::runtime_error("Insufficient degrees of freedom for chi-square test");

    const double p_value = 1.0 - detail::chi_squared_cdf(chi2, df);
    return {chi2, p_value, p_value < significance_level};
}

}  // namespace stats::analysis::poisson

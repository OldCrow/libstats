#include "libstats/stats/analysis/discrete_analysis.h"

#include "libstats/common/distribution_impl_common.h"
#include "libstats/core/math_utils.h"
#include "libstats/distributions/discrete.h"
#include "libstats/stats/analysis/goodness_of_fit.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace stats::analysis::discrete {

std::tuple<double, double, bool> runsTest(const std::vector<double>& data, double alpha) {
    if (data.size() < 8)
        throw std::invalid_argument("Runs test requires at least 8 data points");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be between 0 and 1");

    const std::size_t n = data.size();

    // Compute sample median.
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    const double median = (n % 2 == 0) ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];

    // Build binary sequence: +1 above median, -1 below; skip ties at median.
    std::vector<int> signs;
    signs.reserve(n);
    for (double x : data) {
        if (x > median)
            signs.push_back(1);
        else if (x < median)
            signs.push_back(-1);
    }

    const std::size_t m = signs.size();
    if (m < 4)
        throw std::invalid_argument(
            "Too many ties at the median: fewer than 4 non-median values remain");

    // Count runs (consecutive sequences of the same sign).
    std::size_t runs = 1;
    for (std::size_t i = 1; i < m; ++i) {
        if (signs[i] != signs[i - 1])
            ++runs;
    }

    const double n1 = static_cast<double>(std::count(signs.begin(), signs.end(), 1));
    const double n2 = static_cast<double>(std::count(signs.begin(), signs.end(), -1));
    const double nm = n1 + n2;

    // Expected runs and variance under H₀ (Wald-Wolfowitz large-sample formula).
    const double mu_r = 2.0 * n1 * n2 / nm + 1.0;
    const double var_r =
        (nm <= 1.0) ? 1.0 : 2.0 * n1 * n2 * (2.0 * n1 * n2 - nm) / (nm * nm * (nm - 1.0));

    if (var_r <= 0.0)
        throw std::invalid_argument(
            "Zero variance: all values have the same sign relative to the median");

    const double z = (static_cast<double>(runs) - mu_r) / std::sqrt(var_r);
    const double p_value = 2.0 * (1.0 - detail::normal_cdf(std::abs(z)));

    return {z, p_value, p_value < alpha};
}

std::tuple<double, double, bool> frequencyTest(const std::vector<double>& data, int lo, int hi,
                                               double alpha) {
    if (lo >= hi)
        throw std::invalid_argument("hi must be strictly greater than lo");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be between 0 and 1");

    auto dist_result = DiscreteDistribution::create(lo, hi);
    if (!dist_result.isOk())
        throw std::invalid_argument("Invalid range for DiscreteDistribution: " +
                                    dist_result.message);

    // Retain only values within the expected support.
    std::vector<double> filtered;
    filtered.reserve(data.size());
    for (double v : data) {
        const int k = static_cast<int>(std::round(v));
        if (k >= lo && k <= hi)
            filtered.push_back(v);
    }

    if (filtered.size() < 5)
        throw std::invalid_argument("Fewer than 5 data values fall within [lo, hi]");

    // lo and hi are given (not estimated from data) so estimated_params = 0.
    return chiSquaredGoodnessOfFit(filtered, dist_result.value, alpha, 0);
}

}  // namespace stats::analysis::discrete

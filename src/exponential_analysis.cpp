#include "libstats/stats/analysis/exponential_analysis.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)

#include "libstats/core/math_utils.h"
#include "libstats/core/statistical_constants.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace stats::analysis::exponential {

std::pair<double, double>
confidenceIntervalRate(const std::vector<double>& data, double confidence_level) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be in (0, 1)");

    const double sum = std::accumulate(data.begin(), data.end(), 0.0);
    if (sum <= 0.0)
        throw std::invalid_argument("All data values must be positive");

    const double alpha = 1.0 - confidence_level;
    const double df    = 2.0 * static_cast<double>(data.size());

    // Exact CI: 2nλ/λ̂ = 2Σxᵢλ ~ χ²(2n) under the model
    const double lower = detail::inverse_chi_squared_cdf(alpha / 2.0,       df) / (2.0 * sum);
    const double upper = detail::inverse_chi_squared_cdf(1.0 - alpha / 2.0, df) / (2.0 * sum);
    return {lower, upper};
}

std::tuple<double, double, bool>
coefficientOfVariationTest(const std::vector<double>& data, double alpha) {
    if (data.size() < 2)
        throw std::invalid_argument("At least 2 data points required");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be in (0, 1)");

    const std::size_t n = data.size();
    const double sum    = std::accumulate(data.begin(), data.end(), 0.0);
    const double mean   = sum / static_cast<double>(n);
    if (mean <= 0.0)
        throw std::invalid_argument("All data values must be positive");

    double ssd = 0.0;
    for (double x : data) ssd += (x - mean) * (x - mean);
    const double cv = std::sqrt(ssd / static_cast<double>(n - 1)) / mean;

    // Asymptotic: |CV - 1| / (1/√n) ~ N(0,1)
    const double se = 1.0 / std::sqrt(static_cast<double>(n));
    const double z  = std::abs(cv - 1.0) / se;
    const double p  = 2.0 * (1.0 - detail::normal_cdf(z));

    return {std::abs(cv - 1.0), p, p < alpha};
}

}  // namespace stats::analysis::exponential

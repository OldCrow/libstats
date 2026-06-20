#pragma once

/**
 * @file stats/analysis/information_criteria.h
 * @brief AIC, BIC, and AICc for any libstats distribution.
 *
 * Extracted in v2.0.0. Migration:
 *   GaussianDistribution::computeInformationCriteria(data, dist)
 *   → stats::analysis::informationCriteria(data, dist)
 */

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../core/distribution_concepts.h"

namespace stats::analysis {

/**
 * @brief Compute AIC, BIC, and AICc for a fitted distribution.
 *
 * @tparam D Any distribution satisfying stats::concepts::AnyDistribution.
 * @param data Distribution must have been fitted to this data.
 * @param dist Fitted distribution instance.
 * @return {AIC, BIC, AICc, log_likelihood}
 *
 * AIC  = 2k − 2ℓ
 * BIC  = k·ln(n) − 2ℓ
 * AICc = AIC + 2k(k+1)/(n−k−1)  (undefined → +∞ when n−k−1 ≤ 0)
 */
template <concepts::AnyDistribution D>
[[nodiscard]] std::tuple<double, double, double, double>
informationCriteria(const std::vector<double>& data, const D& dist) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    const double n = static_cast<double>(data.size());
    const int k = dist.getNumParameters();

    double log_likelihood = 0.0;
    for (double val : data)
        log_likelihood += dist.getLogProbability(val);

    const double aic  = 2.0 * k - 2.0 * log_likelihood;
    const double bic  = std::log(n) * k - 2.0 * log_likelihood;
    const double aicc = (n - k - 1.0 > 0.0)
        ? aic + (2.0 * k * (k + 1.0)) / (n - k - 1.0)
        : std::numeric_limits<double>::infinity();

    return {aic, bic, aicc, log_likelihood};
}

}  // namespace stats::analysis

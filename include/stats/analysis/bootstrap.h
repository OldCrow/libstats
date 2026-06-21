#pragma once

/**
 * @file stats/analysis/bootstrap.h
 * @brief Parametric bootstrap confidence intervals for any libstats distribution.
 *
 * Requires D to be default-constructible and to implement fit().
 *
 * Extracted in v2.0.0. Migration:
 *   GaussianDistribution::bootstrapParameterConfidenceIntervals(data, level, n_boot, seed)
 *   → stats::analysis::bootstrapMeanCI<GaussianDistribution>(data, level, n_boot, seed)
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "libstats/core/distribution_concepts.h"

namespace stats::analysis {

/**
 * @brief Parametric bootstrap confidence interval for the mean of a fitted distribution.
 *
 * Generates @p n_bootstrap bootstrap samples, fits D to each, and returns
 * the percentile CI for the mean.
 *
 * @tparam D Default-constructible distribution satisfying AnyDistribution.
 * @param data             Original data.
 * @param confidence_level CI level, e.g. 0.95.
 * @param n_bootstrap      Number of bootstrap replicates (default 1000).
 * @param random_seed      RNG seed (default 42).
 * @return {lower_bound, upper_bound} for the mean.
 */
template <concepts::FittableDistribution D>
[[nodiscard]] std::pair<double, double>
bootstrapMeanCI(const std::vector<double>& data,
                double confidence_level = 0.95,
                int n_bootstrap = 1000,
                unsigned int random_seed = 42) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be in (0, 1)");
    if (n_bootstrap <= 0)
        throw std::invalid_argument("Number of bootstrap samples must be positive");

    const std::size_t n = data.size();
    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<std::size_t> idx_dist(0, n - 1);

    std::vector<double> bootstrap_means;
    bootstrap_means.reserve(static_cast<std::size_t>(n_bootstrap));

    for (int b = 0; b < n_bootstrap; ++b) {
        std::vector<double> sample;
        sample.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            sample.push_back(data[idx_dist(rng)]);

        D fitted;
        fitted.fit(sample);
        bootstrap_means.push_back(fitted.getMean());
    }

    std::sort(bootstrap_means.begin(), bootstrap_means.end());

    const double alpha_half = (1.0 - confidence_level) / 2.0;
    const std::size_t lower_idx =
        static_cast<std::size_t>(alpha_half * (n_bootstrap - 1));
    const std::size_t upper_idx =
        static_cast<std::size_t>((1.0 - alpha_half) * (n_bootstrap - 1));

    return {bootstrap_means[lower_idx], bootstrap_means[upper_idx]};
}

/**
 * @brief Parametric bootstrap CI for all estimated parameters via getMean() and getVariance().
 *
 * Returns CIs for the mean and variance of the fitted distribution.
 *
 * @tparam D Default-constructible distribution satisfying AnyDistribution.
 * @return {{mean_lower, mean_upper}, {variance_lower, variance_upper}}
 */
template <concepts::FittableDistribution D>
[[nodiscard]] std::pair<std::pair<double, double>, std::pair<double, double>>
bootstrapMeanVarianceCI(const std::vector<double>& data,
                        double confidence_level = 0.95,
                        int n_bootstrap = 1000,
                        unsigned int random_seed = 42) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be in (0, 1)");
    if (n_bootstrap <= 0)
        throw std::invalid_argument("Number of bootstrap samples must be positive");

    const std::size_t n = data.size();
    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<std::size_t> idx_dist(0, n - 1);

    std::vector<double> means, variances;
    means.reserve(static_cast<std::size_t>(n_bootstrap));
    variances.reserve(static_cast<std::size_t>(n_bootstrap));

    for (int b = 0; b < n_bootstrap; ++b) {
        std::vector<double> sample;
        sample.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            sample.push_back(data[idx_dist(rng)]);

        D fitted;
        fitted.fit(sample);
        means.push_back(fitted.getMean());
        variances.push_back(fitted.getVariance());
    }

    std::sort(means.begin(), means.end());
    std::sort(variances.begin(), variances.end());

    const double alpha_half = (1.0 - confidence_level) / 2.0;
    const std::size_t lo = static_cast<std::size_t>(alpha_half * (n_bootstrap - 1));
    const std::size_t hi = static_cast<std::size_t>((1.0 - alpha_half) * (n_bootstrap - 1));

    return {{means[lo], means[hi]}, {variances[lo], variances[hi]}};
}

}  // namespace stats::analysis

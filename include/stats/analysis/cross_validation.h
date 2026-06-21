#pragma once

/**
 * @file stats/analysis/cross_validation.h
 * @brief K-fold and leave-one-out cross-validation for any libstats distribution.
 *
 * Both functions return only log-likelihood values. MAE and RMSE were removed
 * in v2.0.0 (AR-3): they were computed as |x_i − μ̂| (mean-prediction errors)
 * rather than distributional fit metrics, making them misleading in a CV context.
 * Log-likelihood under the fitted model is the correct distribution-fit statistic.
 *
 * Extracted in v2.0.0. Migration:
 *   GaussianDistribution::kFoldCrossValidation(data, k, seed)
 *   → stats::analysis::kFoldCrossValidation<GaussianDistribution>(data, k, seed)
 *
 *   Old: vector<tuple<double,double,double>> {MAE, RMSE, log_likelihood}
 *   New: vector<double>                     {log_likelihood per fold}
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "libstats/core/distribution_concepts.h"

namespace stats::analysis {

/**
 * @brief K-fold cross-validation.
 *
 * Shuffles data, splits into k folds, fits D to each training set, and
 * evaluates on the held-out fold using the fitted model's mean as the
 * point prediction.
 *
 * @tparam D Default-constructible distribution satisfying FittableDistribution.
 * @param data        Data vector.
 * @param k           Number of folds (k ≥ 2, k ≤ data.size()).
 * @param random_seed Seed for fold shuffle reproducibility.
 * @return Vector of k fold log-likelihoods: sum of log P(x_i | θ̂) over each
 *         held-out fold under the model fitted to the training fold.
 */
template <concepts::FittableDistribution D>
[[nodiscard]] std::vector<double>
kFoldCrossValidation(const std::vector<double>& data,
                     int k,
                     unsigned int random_seed = 42) {
    if (static_cast<std::size_t>(k) > data.size())
        throw std::invalid_argument("Data size must be at least k for k-fold CV");
    if (k <= 1)
        throw std::invalid_argument("Number of folds k must be greater than 1");

    const std::size_t n = data.size();
    const std::size_t fold_size = n / static_cast<std::size_t>(k);

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<double> results;
    results.reserve(static_cast<std::size_t>(k));

    for (int fold = 0; fold < k; ++fold) {
        const std::size_t start_idx = static_cast<std::size_t>(fold) * fold_size;
        const std::size_t end_idx =
            (fold == k - 1) ? n : (static_cast<std::size_t>(fold) + 1) * fold_size;

        std::vector<double> training, validation;
        training.reserve(n - (end_idx - start_idx));
        validation.reserve(end_idx - start_idx);

        for (std::size_t i = 0; i < n; ++i) {
            if (i >= start_idx && i < end_idx)
                validation.push_back(data[indices[i]]);
            else
                training.push_back(data[indices[i]]);
        }

        D fitted;
        fitted.fit(training);

        double fold_ll = 0.0;
        for (double val : validation)
            fold_ll += fitted.getLogProbability(val);

        results.push_back(fold_ll);
    }

    return results;
}

/**
 * @brief Leave-one-out cross-validation (LOOCV).
 *
 * @tparam D Default-constructible distribution satisfying FittableDistribution.
 * @param data Data vector (at least 3 points required).
 * @return Total log-likelihood: sum of log P(x_i | θ̂_{-i}) over all leave-one-out
 *         fits, where θ̂_{-i} is the parameter estimate with x_i held out.
 */
template <concepts::FittableDistribution D>
[[nodiscard]] double
leaveOneOutCrossValidation(const std::vector<double>& data) {
    if (data.size() < 3)
        throw std::invalid_argument("At least 3 data points required for LOOCV");

    const std::size_t n = data.size();
    double total_log_likelihood = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> training;
        training.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) training.push_back(data[j]);

        D fitted;
        fitted.fit(training);
        total_log_likelihood += fitted.getLogProbability(data[i]);
    }

    return total_log_likelihood;
}

}  // namespace stats::analysis

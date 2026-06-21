#pragma once

/**
 * @file stats/analysis/cross_validation.h
 * @brief K-fold and leave-one-out cross-validation for any libstats distribution.
 *
 * Requires D to be default-constructible (the default parameters represent a
 * valid starting distribution) and to implement fit(const std::vector<double>&).
 * All standard libstats distributions satisfy this contract.
 *
 * Extracted in v2.0.0. Migration:
 *   GaussianDistribution::kFoldCrossValidation(data, k, seed)
 *   → stats::analysis::kFoldCrossValidation<GaussianDistribution>(data, k, seed)
 */

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "../../core/distribution_concepts.h"

namespace stats::analysis {

/**
 * @brief K-fold cross-validation.
 *
 * Shuffles data, splits into k folds, fits D to each training set, and
 * evaluates on the held-out fold using the fitted model's mean as the
 * point prediction.
 *
 * @tparam D Default-constructible distribution satisfying AnyDistribution.
 * @param data        Data vector.
 * @param k           Number of folds (k ≥ 2, k ≤ data.size()).
 * @param random_seed Seed for fold shuffle reproducibility.
 * @return Vector of k tuples: {MAE, RMSE, fold_log_likelihood}.
 */
template <concepts::FittableDistribution D>
[[nodiscard]] std::vector<std::tuple<double, double, double>>
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

    std::vector<std::tuple<double, double, double>> results;
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

        const double predicted_mean = fitted.getMean();
        double log_likelihood = 0.0;
        std::vector<double> errors;
        errors.reserve(validation.size());

        for (double val : validation) {
            errors.push_back(std::abs(val - predicted_mean));
            log_likelihood += fitted.getLogProbability(val);
        }

        const double mae = std::accumulate(errors.begin(), errors.end(), 0.0)
                         / static_cast<double>(errors.size());
        double mse = 0.0;
        for (double e : errors) mse += e * e;
        mse /= static_cast<double>(errors.size());

        results.emplace_back(mae, std::sqrt(mse), log_likelihood);
    }

    return results;
}

/**
 * @brief Leave-one-out cross-validation (LOOCV).
 *
 * @tparam D Default-constructible distribution satisfying AnyDistribution.
 * @param data Data vector (at least 3 points required).
 * @return {mean_absolute_error, root_mean_squared_error, total_log_likelihood}
 */
template <concepts::FittableDistribution D>
[[nodiscard]] std::tuple<double, double, double>
leaveOneOutCrossValidation(const std::vector<double>& data) {
    if (data.size() < 3)
        throw std::invalid_argument("At least 3 data points required for LOOCV");

    const std::size_t n = data.size();
    std::vector<double> abs_errors, sq_errors;
    abs_errors.reserve(n);
    sq_errors.reserve(n);
    double total_log_likelihood = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> training;
        training.reserve(n - 1);
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) training.push_back(data[j]);

        D fitted;
        fitted.fit(training);

        const double pred = fitted.getMean();
        const double diff = data[i] - pred;
        abs_errors.push_back(std::abs(diff));
        sq_errors.push_back(diff * diff);
        total_log_likelihood += fitted.getLogProbability(data[i]);
    }

    const double mae = std::accumulate(abs_errors.begin(), abs_errors.end(), 0.0)
                     / static_cast<double>(n);
    const double mse = std::accumulate(sq_errors.begin(), sq_errors.end(), 0.0)
                     / static_cast<double>(n);

    return {mae, std::sqrt(mse), total_log_likelihood};
}

}  // namespace stats::analysis

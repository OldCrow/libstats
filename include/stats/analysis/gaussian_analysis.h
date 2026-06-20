#pragma once

/**
 * @file stats/analysis/gaussian_analysis.h
 * @brief Gaussian-specific statistical analysis: normality tests, T-tests,
 *        Bayesian inference, robust estimation, and confidence intervals.
 *
 * Generic tests (K-S, AD, k-fold CV, information criteria) are in the
 * sibling headers goodness_of_fit.h, cross_validation.h, etc.
 *
 * Extracted in v2.0.0 from GaussianDistribution static methods.
 * Migration: GaussianDistribution::shapiroWilkTest(data, α)
 *          → stats::analysis::gaussian::shapiroWilkTest(data, α)
 */

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace stats {
class GaussianDistribution;  // forward declaration
}

namespace stats::analysis::gaussian {

// ---------------------------------------------------------------------------
// Normality tests (Gaussian-specific)
// ---------------------------------------------------------------------------

/** @brief Shapiro-Wilk normality test (3 – 5000 data points). */
[[nodiscard]] std::tuple<double, double, bool>
shapiroWilkTest(const std::vector<double>& data, double alpha = 0.05);

/** @brief Jarque-Bera normality test based on skewness and excess kurtosis. */
[[nodiscard]] std::tuple<double, double, bool>
jarqueBeraTest(const std::vector<double>& data, double alpha = 0.05);

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

/**
 * @brief Confidence interval for the mean (z or t depending on sample size /
 *        population_variance_known flag).
 */
[[nodiscard]] std::pair<double, double>
confidenceIntervalMean(const std::vector<double>& data,
                       double confidence_level = 0.95,
                       bool population_variance_known = false);

/** @brief Chi-squared confidence interval for the variance. */
[[nodiscard]] std::pair<double, double>
confidenceIntervalVariance(const std::vector<double>& data,
                           double confidence_level = 0.95);

// ---------------------------------------------------------------------------
// T-tests
// ---------------------------------------------------------------------------

/** @brief One-sample t-test against a hypothesized mean. */
[[nodiscard]] std::tuple<double, double, bool>
oneSampleTTest(const std::vector<double>& data,
               double hypothesized_mean,
               double alpha = 0.05);

/** @brief Two-sample t-test (pooled or Welch). */
[[nodiscard]] std::tuple<double, double, bool>
twoSampleTTest(const std::vector<double>& data1,
               const std::vector<double>& data2,
               bool equal_variances = false,
               double alpha = 0.05);

/** @brief Paired t-test. */
[[nodiscard]] std::tuple<double, double, bool>
pairedTTest(const std::vector<double>& data1,
            const std::vector<double>& data2,
            double alpha = 0.05);

// ---------------------------------------------------------------------------
// Bayesian inference (Normal-Inverse-Gamma conjugate)
// ---------------------------------------------------------------------------

/**
 * @brief Update a Normal-Inverse-Gamma prior with observed data.
 * @return {posterior_mean, posterior_precision, posterior_shape, posterior_rate}
 */
[[nodiscard]] std::tuple<double, double, double, double>
bayesianEstimation(const std::vector<double>& data,
                   double prior_mean = 0.0,
                   double prior_precision = 1.0,
                   double prior_shape = 1.0,
                   double prior_rate = 1.0);

/** @brief Credible interval for the mean given a NIG prior. */
[[nodiscard]] std::pair<double, double>
bayesianCredibleInterval(const std::vector<double>& data,
                         double credibility_level = 0.95,
                         double prior_mean = 0.0,
                         double prior_precision = 1.0,
                         double prior_shape = 1.0,
                         double prior_rate = 1.0);

// ---------------------------------------------------------------------------
// Robust estimation
// ---------------------------------------------------------------------------

/**
 * @brief Iterative M-estimator for location and scale.
 * @param estimator_type "huber" | "tukey" | "hampel"
 * @return {robust_location, robust_scale}
 */
[[nodiscard]] std::pair<double, double>
robustEstimation(const std::vector<double>& data,
                 const std::string& estimator_type = "huber",
                 double tuning_constant = 1.345);

// ---------------------------------------------------------------------------
// Alternative estimators
// ---------------------------------------------------------------------------

/** @brief Method-of-moments estimates for μ and σ. */
[[nodiscard]] std::pair<double, double>
methodOfMomentsEstimation(const std::vector<double>& data);

/** @brief L-moments estimates for μ and σ (robust to outliers). */
[[nodiscard]] std::pair<double, double>
lMomentsEstimation(const std::vector<double>& data);

/** @brief Sample moments up to 6th order (raw or central). */
[[nodiscard]] std::vector<double>
calculateHigherMoments(const std::vector<double>& data, bool center_on_mean = true);

}  // namespace stats::analysis::gaussian

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

/**
 * @brief Shapiro-Wilk normality test (3 – 5000 data points).
 *
 * Uses Blom (1958) approximate coefficients via expected order statistics.
 * The p-value uses n·(1−W) ∼ χ²(1) approximation, which is adequate for
 * large samples and exploratory analysis but over-rejects for n < 50.
 *
 * @param data  Sample data (3 ≤ n ≤ 5000).
 * @param alpha Significance level (default 0.05).
 * @return {W_statistic, p_value, reject_normality}
 * @throws std::invalid_argument if n < 3, n > 5000, or alpha not in (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> shapiroWilkTest(const std::vector<double>& data,
                                                               double alpha = 0.05);

/**
 * @brief Jarque-Bera normality test: JB = n·(S²/6 + K²/24).
 *
 * S is sample skewness, K is excess kurtosis. Under normality, JB ~ χ²(2)
 * asymptotically. Sensitive to both skewness and kurtosis deviations.
 *
 * @param data  Sample data (n ≥ 8).
 * @param alpha Significance level (default 0.05).
 * @return {jb_statistic, p_value, reject_normality}
 * @throws std::invalid_argument if n < 8 or alpha not in (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> jarqueBeraTest(const std::vector<double>& data,
                                                              double alpha = 0.05);

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

/**
 * @brief Confidence interval for the mean (z or t depending on sample size /
 *        population_variance_known flag).
 */
[[nodiscard]] std::pair<double, double> confidenceIntervalMean(
    const std::vector<double>& data, double confidence_level = 0.95,
    bool population_variance_known = false);

/** @brief Chi-squared confidence interval for the variance. */
[[nodiscard]] std::pair<double, double> confidenceIntervalVariance(const std::vector<double>& data,
                                                                   double confidence_level = 0.95);

// ---------------------------------------------------------------------------
// T-tests
// ---------------------------------------------------------------------------

/**
 * @brief One-sample t-test: H₀: μ = hypothesized_mean.
 *
 * @param data             Sample data (non-empty).
 * @param hypothesized_mean Null-hypothesis mean value.
 * @param alpha            Significance level (default 0.05).
 * @return {t_statistic, two_tailed_p_value, reject_null}
 * @throws std::invalid_argument if data is empty or alpha not in (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> oneSampleTTest(const std::vector<double>& data,
                                                              double hypothesized_mean,
                                                              double alpha = 0.05);

/**
 * @brief Two-sample t-test: H₀: μ₁ = μ₂ (Welch or pooled).
 *
 * @param data1           First sample (non-empty).
 * @param data2           Second sample (non-empty).
 * @param equal_variances Use pooled-variance t-test when true; Welch (default).
 * @param alpha           Significance level (default 0.05).
 * @return {t_statistic, two_tailed_p_value, reject_null}
 * @throws std::invalid_argument if either sample is empty or alpha not in (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> twoSampleTTest(const std::vector<double>& data1,
                                                              const std::vector<double>& data2,
                                                              bool equal_variances = false,
                                                              double alpha = 0.05);

/**
 * @brief Paired t-test: H₀: mean(data1ᵢ − data2ᵢ) = 0.
 *
 * @param data1 First sample.
 * @param data2 Second sample (same length as data1).
 * @param alpha Significance level (default 0.05).
 * @return {t_statistic, two_tailed_p_value, reject_null}
 * @throws std::invalid_argument if sizes differ or alpha not in (0,1).
 */
[[nodiscard]] std::tuple<double, double, bool> pairedTTest(const std::vector<double>& data1,
                                                           const std::vector<double>& data2,
                                                           double alpha = 0.05);

// ---------------------------------------------------------------------------
// Bayesian inference (Normal-Inverse-Gamma conjugate)
// ---------------------------------------------------------------------------

/**
 * @brief Update a Normal-Inverse-Gamma prior with observed data.
 * @return {posterior_mean, posterior_precision, posterior_shape, posterior_rate}
 */
[[nodiscard]] std::tuple<double, double, double, double> bayesianEstimation(
    const std::vector<double>& data, double prior_mean = 0.0, double prior_precision = 1.0,
    double prior_shape = 1.0, double prior_rate = 1.0);

/** @brief Credible interval for the mean given a NIG prior. */
[[nodiscard]] std::pair<double, double> bayesianCredibleInterval(
    const std::vector<double>& data, double credibility_level = 0.95, double prior_mean = 0.0,
    double prior_precision = 1.0, double prior_shape = 1.0, double prior_rate = 1.0);

// ---------------------------------------------------------------------------
// Robust estimation
// ---------------------------------------------------------------------------

/**
 * @brief Iterative M-estimator for location and scale.
 *
 * @param data           Sample data.
 * @param estimator_type Influence function: "huber" (default), "tukey", or "hampel".
 * @param tuning_constant Robustness tuning constant (default 1.345, calibrated for
 *                        95% Gaussian efficiency with Huber).
 * @return {robust_location, robust_scale}
 *
 * @note The returned scale is a raw weighted-RMS estimate. It does not include
 *       a Fisher consistency correction, so it converges to a biased (low) estimate
 *       of σ under Gaussian data — approximately 27–35% below the true σ depending
 *       on the tuning constant. Do not use the scale for inference without applying
 *       the appropriate consistency factor for the chosen estimator.
 *       A corrected version is planned for v2.1.
 */
[[nodiscard]] std::pair<double, double> robustEstimation(
    const std::vector<double>& data, const std::string& estimator_type = "huber",
    double tuning_constant = 1.345);

// ---------------------------------------------------------------------------
// Alternative estimators
// ---------------------------------------------------------------------------

/** @brief Method-of-moments estimates for μ and σ. */
[[nodiscard]] std::pair<double, double> methodOfMomentsEstimation(const std::vector<double>& data);

/** @brief L-moments estimates for μ and σ (robust to outliers). */
[[nodiscard]] std::pair<double, double> lMomentsEstimation(const std::vector<double>& data);

/** @brief Sample moments up to 6th order (raw or central). */
[[nodiscard]] std::vector<double> calculateHigherMoments(const std::vector<double>& data,
                                                         bool center_on_mean = true);

}  // namespace stats::analysis::gaussian

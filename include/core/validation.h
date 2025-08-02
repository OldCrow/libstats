#pragma once

/**
 * @file validation.h
 * @brief Statistical validation and goodness-of-fit testing utilities for libstats
 * 
 * This module provides comprehensive statistical validation tools including:
 * - Kolmogorov-Smirnov goodness-of-fit tests
 * - Anderson-Darling goodness-of-fit tests
 * - Chi-squared goodness-of-fit tests for discrete distributions
 * - Model selection criteria (AIC/BIC)
 * - Residual analysis and diagnostics
 * 
 * IMPORTANT NOTE ON CONSTANTS POLICY:
 * Statistical validation functions have unique mathematical requirements that are
 * distinct from the core statistical distribution functions. The critical values,
 * significance levels, and mathematical constants used in these validation routines
 * are specialized for hypothesis testing and are not part of the general statistical
 * constants framework. Therefore, these validation-specific constants are defined
 * locally within validation.h and validation.cpp files where they are used, along
 * with detailed documentation of their mathematical purpose and source.
 * 
 * This approach ensures:
 * - Clear separation between core statistical constants and validation-specific values
 * - Self-contained validation module with all necessary mathematical constants
 * - Proper documentation of critical values and their statistical significance
 * - Maintainability by keeping validation constants close to their usage
 * 
 * For core statistical constants (e.g., π, e, probability bounds), refer to constants.h
 * For validation-specific constants (e.g., critical values, significance levels), 
 * see the implementation details in this file and validation.cpp
 */

#include <vector>
#include <string>

namespace libstats {

// Forward declaration
class DistributionBase;

namespace validation {

/**
 * @brief Kolmogorov-Smirnov test result
 */
struct KSTestResult {
    double statistic;         ///< KS test statistic
    double p_value;           ///< P-value
    bool reject_null;         ///< True if null hypothesis rejected
    std::string interpretation; ///< Human-readable result
};

/**
 * @brief Anderson-Darling test result
 */
struct ADTestResult {
    double statistic;         ///< AD test statistic
    double p_value;           ///< P-value
    bool reject_null;         ///< True if null hypothesis rejected
    std::string interpretation; ///< Human-readable result
};

/**
 * @brief Chi-squared goodness-of-fit test result
 */
struct ChiSquaredResult {
    double statistic;         ///< Chi-squared statistic
    double p_value;           ///< P-value
    int degrees_of_freedom;   ///< Degrees of freedom
    bool reject_null;         ///< True if null hypothesis rejected
    std::string interpretation; ///< Human-readable result
};

/**
 * @brief Model information criteria
 */
struct ModelDiagnostics {
    double log_likelihood;    ///< Log-likelihood
    double aic;              ///< Akaike Information Criterion
    double bic;              ///< Bayesian Information Criterion
    int num_parameters;      ///< Number of parameters
    size_t sample_size;      ///< Sample size
};

/**
 * @brief Perform Kolmogorov-Smirnov goodness-of-fit test
 * @param data Observed data
 * @param distribution Theoretical distribution
 * @return KS test result
 */
KSTestResult kolmogorov_smirnov_test(const std::vector<double>& data,
                                   const DistributionBase& distribution);

/**
 * @brief Perform Anderson-Darling goodness-of-fit test  
 * @param data Observed data
 * @param distribution Theoretical distribution
 * @return AD test result
 */
ADTestResult anderson_darling_test(const std::vector<double>& data,
                                 const DistributionBase& distribution);

/**
 * @brief Perform Chi-squared goodness-of-fit test for discrete distributions
 * @param observed_counts Observed frequencies
 * @param expected_probabilities Expected probabilities
 * @return Chi-squared test result
 */
ChiSquaredResult chi_squared_goodness_of_fit(const std::vector<int>& observed_counts,
                                            const std::vector<double>& expected_probabilities);

/**
 * @brief Calculate model diagnostics (AIC/BIC)
 * @param distribution Fitted distribution
 * @param data Data used for fitting
 * @return Model diagnostics
 */
ModelDiagnostics calculate_model_diagnostics(const DistributionBase& distribution,
                                           const std::vector<double>& data);

/**
 * @brief Calculate standardized residuals
 * @param data Observed data
 * @param distribution Fitted distribution
 * @return Vector of standardized residuals
 */
std::vector<double> calculate_residuals(const std::vector<double>& data,
                                      const DistributionBase& distribution);

/**
 * @brief Bootstrap-based goodness-of-fit test result
 */
struct BootstrapTestResult {
    double observed_statistic;    ///< Original test statistic
    double bootstrap_p_value;     ///< Bootstrap p-value
    std::vector<double> bootstrap_statistics; ///< Bootstrap statistics
    bool reject_null;            ///< True if null hypothesis rejected
    std::string interpretation;  ///< Human-readable result
    size_t num_bootstrap_samples; ///< Number of bootstrap samples used
};

/**
 * @brief Bootstrap-based Kolmogorov-Smirnov test
 * 
 * Uses bootstrap resampling to estimate the p-value distribution,
 * providing more robust results especially for small samples or
 * when the theoretical distribution is uncertain.
 * 
 * @param data Observed data
 * @param distribution Theoretical distribution
 * @param num_bootstrap Number of bootstrap samples (default: 1000)
 * @param alpha Significance level (default: 0.05)
 * @return Bootstrap test result
 */
BootstrapTestResult bootstrap_kolmogorov_smirnov_test(
    const std::vector<double>& data,
    const DistributionBase& distribution,
    size_t num_bootstrap = 1000,
    double alpha = 0.05);

/**
 * @brief Bootstrap-based Anderson-Darling test
 * 
 * Uses bootstrap resampling to estimate the p-value distribution,
 * providing more accurate results for non-normal distributions
 * and small sample sizes.
 * 
 * @param data Observed data
 * @param distribution Theoretical distribution
 * @param num_bootstrap Number of bootstrap samples (default: 1000)
 * @param alpha Significance level (default: 0.05)
 * @return Bootstrap test result
 */
BootstrapTestResult bootstrap_anderson_darling_test(
    const std::vector<double>& data,
    const DistributionBase& distribution,
    size_t num_bootstrap = 1000,
    double alpha = 0.05);

/**
 * @brief Parametric bootstrap test for distribution parameters
 * 
 * Tests whether the observed data could have come from the specified
 * distribution by generating bootstrap samples from the fitted
 * distribution and comparing parameter estimates.
 * 
 * @param data Observed data
 * @param distribution Fitted distribution
 * @param num_bootstrap Number of bootstrap samples (default: 1000)
 * @param alpha Significance level (default: 0.05)
 * @return Bootstrap test result
 */
BootstrapTestResult bootstrap_parameter_test(
    const std::vector<double>& data,
    const DistributionBase& distribution,
    size_t num_bootstrap = 1000,
    double alpha = 0.05);

/**
 * @brief Non-parametric bootstrap confidence interval
 * 
 * Computes confidence intervals for distribution parameters using
 * non-parametric bootstrap resampling.
 * 
 * @param data Observed data
 * @param distribution Distribution type to fit
 * @param confidence_level Confidence level (default: 0.95)
 * @param num_bootstrap Number of bootstrap samples (default: 1000)
 * @return Vector of parameter confidence intervals
 */
struct ConfidenceInterval {
    double lower_bound;
    double upper_bound;
    double point_estimate;
    double confidence_level;
};

std::vector<ConfidenceInterval> bootstrap_confidence_intervals(
    const std::vector<double>& data,
    const DistributionBase& distribution,
    double confidence_level = 0.95,
    size_t num_bootstrap = 1000);

} // namespace validation
} // namespace libstats

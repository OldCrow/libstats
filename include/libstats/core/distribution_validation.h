#pragma once

/**
 * @file core/distribution_validation.h
 * @brief ValidationResult and FitResults structures used by DistributionBase.
 *
 * The following dead framework has been removed in v2.0.0 (Step 3D):
 *   - DistributionValidator abstract class (no inheritors)
 *   - ExtendedValidationError enum (no callers)
 *   - extendedValidationErrorToString() (no callers)
 *   - 9 detail:: utility functions: validateFittingData, isDataFinite,
 *     hasSufficientVariance, calculateDataStatistics, kolmogorovSmirnovTest,
 *     andersonDarlingTest, calculateStandardizedResiduals,
 *     generateValidationRecommendations, isAdequateSampleSize,
 *     getRecommendedSampleSize.
 *   These duplicated machinery already in stats::analysis:: goodness_of_fit.h
 *   and had zero callers across the codebase.
 *
 * ValidationResult and FitResults are load-bearing: returned by
 * DistributionBase::validate() and fitWithDiagnostics() respectively.
 */

#include <algorithm>
#include <string>
#include <vector>

namespace stats {

// =============================================================================
// VALIDATION RESULT STRUCTURES
// =============================================================================

/**
 * @brief Validation result structure for goodness-of-fit tests
 */
struct ValidationResult {
    double ks_statistic;          ///< Kolmogorov-Smirnov test statistic
    double ks_p_value;            ///< KS test p-value
    double ad_statistic;          ///< Anderson-Darling test statistic
    double ad_p_value;            ///< AD test p-value
    bool distribution_adequate;   ///< Overall assessment
    std::string recommendations;  ///< Improvement suggestions

    /**
     * @brief Check if distribution passes validation at given significance level
     * @param alpha Significance level (default 0.05)
     * @return true if both KS and AD tests pass
     */
    bool isValid(double alpha = 0.05) const noexcept {
        return ks_p_value > alpha && ad_p_value > alpha;
    }

    /**
     * @brief Get the worst (minimum) p-value from all tests
     * @return Minimum p-value across all performed tests
     */
    double getWorstPValue() const noexcept { return std::min(ks_p_value, ad_p_value); }
};

/**
 * @brief Enhanced fitting result structure with comprehensive diagnostics
 */
struct FitResults {
    double log_likelihood;          ///< Log-likelihood of fitted parameters
    double aic;                     ///< Akaike Information Criterion
    double bic;                     ///< Bayesian Information Criterion
    ValidationResult validation;    ///< Goodness-of-fit assessment
    std::vector<double> residuals;  ///< Standardized residuals
    bool fit_successful;            ///< Whether fitting converged
    std::string fit_diagnostics;    ///< Detailed fitting information

    /**
     * @brief Check if fit is considered successful and valid
     * @param alpha Significance level for validation tests
     * @return true if fit converged and passes validation
     */
    bool isGoodFit(double alpha = 0.05) const noexcept {
        return fit_successful && validation.isValid(alpha);
    }

    /**
     * @brief Calculate information criterion difference vs. saturated model
     * @param use_bic If true, use BIC; otherwise use AIC
     * @return IC value (lower is better)
     * @note Useful for model comparison
     */
    double getInformationCriterion(bool use_bic = false) const noexcept {
        return use_bic ? bic : aic;
    }
};

}  // namespace stats

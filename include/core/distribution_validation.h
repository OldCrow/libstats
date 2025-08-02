#pragma once

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace libstats {

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
    double getWorstPValue() const noexcept {
        return std::min(ks_p_value, ad_p_value);
    }
};

/**
 * @brief Enhanced fitting result structure with comprehensive diagnostics
 */
struct FitResults {
    double log_likelihood;        ///< Log-likelihood of fitted parameters
    double aic;                   ///< Akaike Information Criterion
    double bic;                   ///< Bayesian Information Criterion
    ValidationResult validation;  ///< Goodness-of-fit assessment
    std::vector<double> residuals; ///< Standardized residuals
    bool fit_successful;          ///< Whether fitting converged
    std::string fit_diagnostics;  ///< Detailed fitting information
    
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

// =============================================================================
// VALIDATION ERROR TYPES
// =============================================================================

/**
 * @brief Extended enumeration of validation error types
 * @note Extends the ValidationError from error_handling.h
 */
enum class ExtendedValidationError {
    NONE = 0,                    ///< No validation errors
    EMPTY_DATA,                  ///< Data vector is empty
    INSUFFICIENT_DATA,           ///< Not enough data points for reliable validation
    INVALID_VALUES,              ///< Data contains NaN, infinity, or other invalid values
    DUPLICATE_VALUES,            ///< All data values are identical (no variance)
    PARAMETER_OUT_OF_BOUNDS,     ///< Distribution parameters are outside valid range
    NUMERICAL_INSTABILITY,       ///< Numerical computation failed
    FITTING_CONVERGENCE_FAILED,  ///< Parameter estimation did not converge
    VALIDATION_TEST_FAILED       ///< Statistical validation tests failed
};

/**
 * @brief Convert validation error enum to descriptive string
 * @param error Validation error type
 * @return Human-readable error description
 */
inline std::string extendedValidationErrorToString(ExtendedValidationError error) {
    switch (error) {
        case ExtendedValidationError::NONE:
            return "No validation errors";
        case ExtendedValidationError::EMPTY_DATA:
            return "Data vector is empty";
        case ExtendedValidationError::INSUFFICIENT_DATA:
            return "Insufficient data points for reliable validation";
        case ExtendedValidationError::INVALID_VALUES:
            return "Data contains invalid values (NaN, infinity, etc.)";
        case ExtendedValidationError::DUPLICATE_VALUES:
            return "All data values are identical (no variance)";
        case ExtendedValidationError::PARAMETER_OUT_OF_BOUNDS:
            return "Distribution parameters are outside valid range";
        case ExtendedValidationError::NUMERICAL_INSTABILITY:
            return "Numerical computation failed";
        case ExtendedValidationError::FITTING_CONVERGENCE_FAILED:
            return "Parameter estimation did not converge";
        case ExtendedValidationError::VALIDATION_TEST_FAILED:
            return "Statistical validation tests failed";
        default:
            return "Unknown validation error";
    }
}

// =============================================================================
// DISTRIBUTION VALIDATION INTERFACE
// =============================================================================

/**
 * @brief Interface for distribution validation and diagnostics
 * 
 * Provides standardized methods for validating distribution fits,
 * performing goodness-of-fit tests, and generating diagnostic information.
 */
class DistributionValidator {
public:
    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     */
    virtual ~DistributionValidator() = default;
    
    /**
     * @brief Validate distribution fit against data
     * @param data Vector of observations to validate against
     * @return Validation results with test statistics and p-values
     * @note Base implementation performs KS and AD tests
     */
    virtual ValidationResult validate(const std::vector<double>& data) const = 0;
    
    /**
     * @brief Fit distribution with comprehensive diagnostics
     * @param data Vector of observations
     * @return Detailed fitting results and validation
     * @note Base implementation calls fit() and validates; override for efficiency
     */
    virtual FitResults fitWithDiagnostics(const std::vector<double>& data) = 0;
    
    /**
     * @brief Check if two distributions are approximately equal
     * @param other Distribution to compare
     * @param tolerance Numerical tolerance for comparison
     * @return true if distributions are approximately equal
     */
    virtual bool isApproximatelyEqual(const DistributionValidator& other, 
                                    double tolerance = 1e-10) const = 0;
};

// =============================================================================
// DATA VALIDATION UTILITIES
// =============================================================================

/**
 * @brief Comprehensive data validation utilities
 */
namespace validation {

/**
 * @brief Validate data for distribution fitting
 * @param data Data to validate
 * @return Extended validation error type (NONE if valid)
 */
ExtendedValidationError validateFittingData(const std::vector<double>& data) noexcept;

/**
 * @brief Check if data contains only finite values
 * @param data Data to check
 * @return true if all values are finite
 */
bool isDataFinite(const std::vector<double>& data) noexcept;

/**
 * @brief Check if data has sufficient variance for fitting
 * @param data Data to check
 * @param min_variance Minimum required variance (default: 1e-12)
 * @return true if data has sufficient variance
 */
bool hasSufficientVariance(const std::vector<double>& data, 
                          double min_variance = 1e-12) noexcept;

/**
 * @brief Calculate basic data statistics for validation
 * @param data Input data
 * @return Vector with [mean, variance, skewness, kurtosis, min, max]
 */
std::vector<double> calculateDataStatistics(const std::vector<double>& data);

/**
 * @brief Perform Kolmogorov-Smirnov goodness-of-fit test
 * @param data Observed data (will be sorted internally)
 * @param cdf_func Theoretical CDF function
 * @return Vector with [test_statistic, p_value]
 */
std::vector<double> kolmogorovSmirnovTest(std::vector<double> data,
                                        std::function<double(double)> cdf_func);

/**
 * @brief Perform Anderson-Darling goodness-of-fit test
 * @param data Observed data (will be sorted internally)
 * @param cdf_func Theoretical CDF function
 * @return Vector with [test_statistic, p_value]
 */
std::vector<double> andersonDarlingTest(std::vector<double> data,
                                      std::function<double(double)> cdf_func);

/**
 * @brief Calculate standardized residuals for fitted distribution
 * @param data Original data
 * @param fitted_cdf CDF of fitted distribution
 * @return Vector of standardized residuals
 */
std::vector<double> calculateStandardizedResiduals(const std::vector<double>& data,
                                                  std::function<double(double)> fitted_cdf);

/**
 * @brief Generate recommendations based on validation results
 * @param result Validation result to analyze
 * @param data_size Size of original dataset
 * @return String with specific recommendations for improvement
 */
std::string generateValidationRecommendations(const ValidationResult& result,
                                             size_t data_size);

/**
 * @brief Check if sample size is adequate for distribution fitting
 * @param data_size Size of the dataset
 * @param num_parameters Number of distribution parameters
 * @return true if sample size is adequate
 */
bool isAdequateSampleSize(size_t data_size, int num_parameters) noexcept;

/**
 * @brief Calculate minimum sample size recommendation
 * @param num_parameters Number of distribution parameters
 * @param confidence_level Desired confidence level (default: 0.95)
 * @return Recommended minimum sample size
 */
size_t getRecommendedSampleSize(int num_parameters, 
                               double confidence_level = 0.95) noexcept;

} // namespace validation

} // namespace libstats

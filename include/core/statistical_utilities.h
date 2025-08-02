#pragma once

/**
 * @file statistical_utilities.h
 * @brief Statistical utility classes that provide convenient interfaces to distribution methods
 * 
 * This header provides utility classes that wrap the static statistical methods
 * implemented in the distribution classes, offering a consistent interface
 * for common statistical operations like goodness-of-fit tests, cross-validation,
 * information criteria, and bootstrap methods.
 */

#include <vector>
#include <tuple>
#include <pair>
#include "../distributions/discrete.h"
#include "../distributions/exponential.h"
#include "../distributions/gamma.h"
#include "../distributions/gaussian.h"
#include "../distributions/poisson.h"
#include "../distributions/uniform.h"

namespace libstats {

/**
 * @brief Goodness-of-fit test utilities
 * 
 * Provides convenient interfaces to various goodness-of-fit tests
 * implemented across different distribution types.
 */
class GoodnessOfFit {
public:
    /**
     * @brief Perform chi-squared goodness-of-fit test
     * 
     * @param distribution The distribution to test against
     * @param data Sample data to test
     * @param alpha Significance level (default: 0.05)
     * @return true if data appears to follow the distribution
     */
    template<typename DistributionType>
    static bool performChiSquareTest(const DistributionType& distribution, 
                                   const std::vector<double>& data, 
                                   double alpha = 0.05) {
        auto [statistic, p_value, reject_null] = DistributionType::chiSquaredGoodnessOfFitTest(data, distribution, alpha);
        return !reject_null; // Return true if we don't reject null (good fit)
    }
    
    /**
     * @brief Perform Kolmogorov-Smirnov goodness-of-fit test
     * 
     * @param distribution The distribution to test against
     * @param data Sample data to test
     * @param alpha Significance level (default: 0.05)
     * @return true if data appears to follow the distribution
     */
    template<typename DistributionType>
    static bool performKolmogorovSmirnovTest(const DistributionType& distribution, 
                                           const std::vector<double>& data, 
                                           double alpha = 0.05) {
        auto [statistic, p_value, reject_null] = DistributionType::kolmogorovSmirnovTest(data, distribution, alpha);
        return !reject_null; // Return true if we don't reject null (good fit)
    }
};

/**
 * @brief Cross-validation utilities
 * 
 * Provides convenient interfaces to cross-validation methods
 * for model validation and selection.
 */
class CrossValidation {
public:
    /**
     * @brief Perform k-fold cross-validation
     * 
     * @param distribution The distribution type to validate
     * @param data Sample data for validation
     * @param k Number of folds (default: 5)
     * @param random_seed Random seed (default: 42)
     * @return Average validation score (lower is better)
     */
    template<typename DistributionType>
    static double performKFoldValidation(const DistributionType& distribution, 
                                       const std::vector<double>& data, 
                                       int k = 5, 
                                       unsigned int random_seed = 42) {
        auto results = DistributionType::kFoldCrossValidation(data, k, random_seed);
        
        // Calculate average mean absolute error across all folds
        double total_mae = 0.0;
        for (const auto& [mae, stderr, loglik] : results) {
            total_mae += mae;
        }
        return total_mae / results.size();
    }
    
    /**
     * @brief Perform leave-one-out cross-validation
     * 
     * @param distribution The distribution type to validate
     * @param data Sample data for validation
     * @return Validation score (lower is better)
     */
    template<typename DistributionType>
    static double performLeaveOneOutValidation(const DistributionType& distribution, 
                                             const std::vector<double>& data) {
        auto [mae, rmse, loglik] = DistributionType::leaveOneOutCrossValidation(data);
        return mae; // Return mean absolute error
    }
};

/**
 * @brief Information criteria utilities
 * 
 * Provides convenient interfaces to information criteria
 * for model selection and comparison.
 */
class InformationCriteria {
public:
    /**
     * @brief Calculate Akaike Information Criterion (AIC)
     * 
     * @param distribution The fitted distribution
     * @param data Sample data used for fitting
     * @return AIC value (lower is better)
     */
    template<typename DistributionType>
    static double calculateAIC(const DistributionType& distribution, 
                             const std::vector<double>& data) {
        auto [aic, bic, aicc, loglik] = DistributionType::computeInformationCriteria(data, distribution);
        return aic;
    }
    
    /**
     * @brief Calculate Bayesian Information Criterion (BIC)
     * 
     * @param distribution The fitted distribution
     * @param data Sample data used for fitting
     * @return BIC value (lower is better)
     */
    template<typename DistributionType>
    static double calculateBIC(const DistributionType& distribution, 
                             const std::vector<double>& data) {
        auto [aic, bic, aicc, loglik] = DistributionType::computeInformationCriteria(data, distribution);
        return bic;
    }
    
    /**
     * @brief Calculate corrected Akaike Information Criterion (AICc)
     * 
     * @param distribution The fitted distribution
     * @param data Sample data used for fitting
     * @return AICc value (lower is better)
     */
    template<typename DistributionType>
    static double calculateAICc(const DistributionType& distribution, 
                              const std::vector<double>& data) {
        auto [aic, bic, aicc, loglik] = DistributionType::computeInformationCriteria(data, distribution);
        return aicc;
    }
};

/**
 * @brief Bootstrap utilities
 * 
 * Provides convenient interfaces to bootstrap methods
 * for parameter estimation and confidence intervals.
 */
class Bootstrap {
public:
    /**
     * @brief Bootstrap confidence interval result
     */
    struct ConfidenceIntervalResult {
        bool is_valid;
        std::vector<std::pair<double, double>> intervals;
        
        bool isValid() const { return is_valid; }
        const std::vector<std::pair<double, double>>& getIntervals() const { return intervals; }
    };
    
    /**
     * @brief Parameter confidence interval structure
     */
    struct ParameterInterval {
        double lower_bound;
        double upper_bound;
        std::string parameter_name;
    };
    
    /**
     * @brief Calculate bootstrap parameter confidence intervals
     * 
     * @param distribution The distribution type to bootstrap
     * @param data Sample data for bootstrap resampling
     * @param num_bootstrap Number of bootstrap samples (default: 1000)
     * @param confidence_level Confidence level (default: 0.95)
     * @param random_seed Random seed (default: 42)
     * @return Confidence interval result
     */
    template<typename DistributionType>
    static ConfidenceIntervalResult calculateParameterConfidenceIntervals(
        const DistributionType& distribution, 
        const std::vector<double>& data, 
        int num_bootstrap = 1000,
        double confidence_level = 0.95,
        unsigned int random_seed = 42) {
        
        ConfidenceIntervalResult result;
        
        try {
            if constexpr (std::is_same_v<DistributionType, DiscreteDistribution>) {
                auto [lower_ci, upper_ci] = DistributionType::bootstrapParameterConfidenceIntervals(
                    data, confidence_level, num_bootstrap, random_seed);
                result.intervals.push_back(lower_ci);
                result.intervals.push_back(upper_ci);
                result.is_valid = true;
            } else if constexpr (std::is_same_v<DistributionType, GaussianDistribution>) {
                auto [mean_ci, std_ci] = DistributionType::bootstrapParameterConfidenceIntervals(
                    data, confidence_level, num_bootstrap, random_seed);
                result.intervals.push_back(mean_ci);
                result.intervals.push_back(std_ci);
                result.is_valid = true;
            } else if constexpr (std::is_same_v<DistributionType, ExponentialDistribution>) {
                auto rate_ci = DistributionType::bootstrapParameterConfidenceInterval(
                    data, confidence_level, num_bootstrap, random_seed);
                result.intervals.push_back(rate_ci);
                result.is_valid = true;
            } else {
                // Unsupported distribution type
                result.is_valid = false;
            }
        } catch (...) {
            result.is_valid = false;
        }
        
        return result;
    }
};

} // namespace libstats

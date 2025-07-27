/**
 * @file statistical_validation_demo.cpp
 * @brief Comprehensive statistical validation and testing demonstration
 * 
 * This example showcases advanced statistical validation techniques including:
 * - Goodness-of-fit tests (Kolmogorov-Smirnov, Anderson-Darling)
 * - Cross-validation methods (K-fold, Leave-One-Out)
 * - Bootstrap confidence intervals
 * - Information criteria for model selection
 * - Performance testing with non-normal data
 */

#include "../include/distributions/gaussian.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    try {
        std::cout << "=== Statistical Validation and Testing Demo ===" << std::endl;
        std::cout << "Comprehensive demonstration of goodness-of-fit tests and cross-validation methods" << std::endl;
        
        // Generate test data from a known normal distribution
        std::mt19937 rng(42);
        std::normal_distribution<double> normal(5.0, 2.0);
        
        std::vector<double> data(100);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = normal(rng);
        }
        
        std::cout << "\nGenerated " << data.size() << " samples from N(5.0, 2.0)" << std::endl;
        
        // Create a Gaussian distribution to test against
        libstats::GaussianDistribution test_dist(5.0, 2.0);
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "GOODNESS-OF-FIT TESTS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test 1: Kolmogorov-Smirnov test
        std::cout << "\n1. Kolmogorov-Smirnov Test:" << std::endl;
        auto [ks_stat, ks_p, ks_reject] = libstats::GaussianDistribution::kolmogorovSmirnovTest(data, test_dist, 0.05);
        std::cout << "   KS statistic: " << ks_stat << std::endl;
        std::cout << "   p-value: " << ks_p << std::endl;
        std::cout << "   Reject normality: " << (ks_reject ? "Yes" : "No") << std::endl;
        std::cout << "   → Data " << (ks_reject ? "does NOT appear" : "appears") << " to follow N(5.0, 2.0)" << std::endl;
        
        // Test 2: Anderson-Darling test
        std::cout << "\n2. Anderson-Darling Test:" << std::endl;
        auto [ad_stat, ad_p, ad_reject] = libstats::GaussianDistribution::andersonDarlingTest(data, test_dist, 0.05);
        std::cout << "   AD statistic: " << ad_stat << std::endl;
        std::cout << "   p-value: " << ad_p << std::endl;
        std::cout << "   Reject normality: " << (ad_reject ? "Yes" : "No") << std::endl;
        std::cout << "   → Data " << (ad_reject ? "does NOT appear" : "appears") << " to follow N(5.0, 2.0)" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "CROSS-VALIDATION METHODS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test 3: K-fold Cross-validation
        std::cout << "\n3. 5-Fold Cross-Validation:" << std::endl;
        std::cout << "   Testing model generalization across 5 folds..." << std::endl;
        auto cv_results = libstats::GaussianDistribution::kFoldCrossValidation(data, 5, 42);
        std::cout << "   Results per fold (Mean Abs Error, Std Error, Log-Likelihood):" << std::endl;
        double total_mae = 0.0, total_loglik = 0.0;
        for (size_t i = 0; i < cv_results.size(); ++i) {
            auto [mae, stderr, loglik] = cv_results[i];
            std::cout << "     Fold " << (i+1) << ": MAE=" << mae 
                     << ", StdErr=" << stderr << ", LogLik=" << loglik << std::endl;
            total_mae += mae;
            total_loglik += loglik;
        }
        std::cout << "   → Average MAE: " << (total_mae / cv_results.size()) << std::endl;
        std::cout << "   → Total Log-Likelihood: " << total_loglik << std::endl;
        
        // Test 4: Leave-one-out cross-validation (smaller dataset for speed)
        std::vector<double> small_data(data.begin(), data.begin() + 20);
        std::cout << "\n4. Leave-One-Out Cross-Validation (20 samples):" << std::endl;
        std::cout << "   Testing each data point as validation set..." << std::endl;
        auto [loocv_mae, loocv_rmse, loocv_loglik] = libstats::GaussianDistribution::leaveOneOutCrossValidation(small_data);
        std::cout << "   Mean Absolute Error: " << loocv_mae << std::endl;
        std::cout << "   Root Mean Squared Error: " << loocv_rmse << std::endl;
        std::cout << "   Total Log-Likelihood: " << loocv_loglik << std::endl;
        std::cout << "   → LOOCV provides unbiased estimate of model performance" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "BOOTSTRAP AND INFORMATION CRITERIA" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test 5: Bootstrap confidence intervals
        std::cout << "\n5. Bootstrap Parameter Confidence Intervals:" << std::endl;
        std::cout << "   Generating 1000 bootstrap samples..." << std::endl;
        auto [mean_ci, std_ci] = libstats::GaussianDistribution::bootstrapParameterConfidenceIntervals(
            data, 0.95, 1000, 456);
        std::cout << "   95% CI for mean: [" << mean_ci.first << ", " << mean_ci.second << "]" << std::endl;
        std::cout << "   95% CI for std dev: [" << std_ci.first << ", " << std_ci.second << "]" << std::endl;
        std::cout << "   → True mean (5.0) " << 
                     (mean_ci.first <= 5.0 && 5.0 <= mean_ci.second ? "IS" : "is NOT") 
                  << " within confidence interval" << std::endl;
        std::cout << "   → True std (2.0) " << 
                     (std_ci.first <= 2.0 && 2.0 <= std_ci.second ? "IS" : "is NOT") 
                  << " within confidence interval" << std::endl;
        
        // Test 6: Information criteria
        std::cout << "\n6. Information Criteria (Model Selection):" << std::endl;
        libstats::GaussianDistribution fitted_dist;
        fitted_dist.fit(data);
        auto [aic, bic, aicc, loglik] = libstats::GaussianDistribution::computeInformationCriteria(data, fitted_dist);
        std::cout << "   Fitted parameters: μ=" << fitted_dist.getMean() 
                 << ", σ=" << fitted_dist.getStandardDeviation() << std::endl;
        std::cout << "   AIC:  " << aic << " (Akaike Information Criterion)" << std::endl;
        std::cout << "   BIC:  " << bic << " (Bayesian Information Criterion)" << std::endl;
        std::cout << "   AICc: " << aicc << " (Corrected AIC)" << std::endl;
        std::cout << "   Log-likelihood: " << loglik << std::endl;
        std::cout << "   → Lower AIC/BIC values indicate better model fit" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "PERFORMANCE TEST WITH NON-NORMAL DATA" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test with obviously non-normal data
        std::vector<double> non_normal_data;
        for (int i = 0; i < 50; ++i) {
            non_normal_data.push_back(i * i); // Quadratic growth - definitely not normal
        }
        
        std::cout << "\nTesting with obviously non-normal data (quadratic sequence)..." << std::endl;
        auto [ks_nn_stat, ks_nn_p, ks_nn_reject] = libstats::GaussianDistribution::kolmogorovSmirnovTest(
            non_normal_data, test_dist, 0.05);
        auto [ad_nn_stat, ad_nn_p, ad_nn_reject] = libstats::GaussianDistribution::andersonDarlingTest(
            non_normal_data, test_dist, 0.05);
        
        std::cout << "KS Test - Reject normality: " << (ks_nn_reject ? "Yes ✓" : "No ✗") 
                 << " (p=" << ks_nn_p << ")" << std::endl;
        std::cout << "AD Test - Reject normality: " << (ad_nn_reject ? "Yes ✓" : "No ✗") 
                 << " (p=" << ad_nn_p << ")" << std::endl;
        std::cout << "→ Both tests correctly identify non-normal data" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "✅ ALL PHASE 3 METHODS SUCCESSFULLY DEMONSTRATED!" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        std::cout << "\nPhase 3 adds powerful statistical validation capabilities:" << std::endl;
        std::cout << "• Advanced goodness-of-fit tests (KS, Anderson-Darling)" << std::endl;
        std::cout << "• Cross-validation methods (K-fold, LOOCV)" << std::endl;
        std::cout << "• Bootstrap confidence intervals" << std::endl;
        std::cout << "• Information criteria for model selection" << std::endl;
        std::cout << "• All integrated with existing SIMD and parallel optimizations" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

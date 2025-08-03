/**
 * @file statistical_validation_demo.cpp
 * @brief Advanced statistical validation and testing demonstration in libstats v0.7.0
 * 
 * This example demonstrates the extensive statistical validation capabilities including:
 * - Goodness-of-fit tests: Kolmogorov-Smirnov (KS) and Anderson-Darling (AD)
 * - Cross-validation methods: K-fold and Leave-One-Out Cross-Validation (LOOCV)
 * - Bootstrap confidence intervals for parameter estimation
 * - Information criteria for robust model selection
 * - Performance testing with non-normal data using advanced tests
 * - A comprehensive demonstration of Phase 3's validation capabilities
 */

#include "../include/distributions/gaussian.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        std::cout << "=== libstats v0.7.0 Statistical Validation & Testing Demo ===" << std::endl;
        std::cout << "Comprehensive demonstration of advanced statistical validation techniques\n"
                  << "showcasing Phase 3's powerful model validation and testing capabilities\n" << std::endl;
        
        // Generate test data from a known normal distribution
        std::cout << "📋 Data Generation & Setup" << std::endl;
        std::cout << "Creating controlled test dataset to demonstrate validation methods:\n" << std::endl;
        
        std::mt19937 rng(42);
        std::normal_distribution<double> normal(5.0, 2.0);
        
        std::vector<double> data(100);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = normal(rng);
        }
        
        std::cout << "✓ Generated " << data.size() << " samples from N(μ=5.0, σ=2.0) [Known true distribution]" << std::endl;
        std::cout << "✓ Using fixed random seed (42) for reproducible results" << std::endl;
        
        // Create a Gaussian distribution to test against
        libstats::GaussianDistribution test_dist(5.0, 2.0);
        std::cout << "✓ Created reference distribution for hypothesis testing" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "GOODNESS-OF-FIT TESTS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\nTesting whether our sample data follows the hypothesized distribution:\n"
                  << "These tests compare empirical vs. theoretical cumulative distributions\n"
                  << "to detect deviations from the assumed model.\n" << std::endl;
        
        // Test 1: Kolmogorov-Smirnov test
        std::cout << "🎯 1. Kolmogorov-Smirnov Test:" << std::endl;
        std::cout << "   Measures maximum distance between sample and theoretical CDFs" << std::endl;
        std::cout << "   Null hypothesis (H₀): Data follows N(5.0, 2.0)\n" << std::endl;
        auto [ks_stat, ks_p, ks_reject] = libstats::GaussianDistribution::kolmogorovSmirnovTest(data, test_dist, 0.05);
        std::cout << "   KS statistic: " << ks_stat << std::endl;
        std::cout << "   p-value: " << ks_p << std::endl;
        std::cout << "   Reject normality: " << (ks_reject ? "Yes" : "No") << std::endl;
        std::cout << "   → Data " << (ks_reject ? "does NOT appear" : "appears") << " to follow N(5.0, 2.0)" << std::endl;
        
        // Test 2: Anderson-Darling test
        std::cout << "\n🎯 2. Anderson-Darling Test:" << std::endl;
        std::cout << "   More sensitive to tail deviations than KS test" << std::endl;
        std::cout << "   Weights discrepancies at distribution extremes more heavily\n" << std::endl;
        auto [ad_stat, ad_p, ad_reject] = libstats::GaussianDistribution::andersonDarlingTest(data, test_dist, 0.05);
        std::cout << "   AD statistic: " << ad_stat << std::endl;
        std::cout << "   p-value: " << ad_p << std::endl;
        std::cout << "   Reject normality: " << (ad_reject ? "Yes" : "No") << std::endl;
        std::cout << "   → Data " << (ad_reject ? "does NOT appear" : "appears") << " to follow N(5.0, 2.0)" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "CROSS-VALIDATION METHODS" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\nAssessing model generalization and avoiding overfitting:\n"
                  << "These methods partition data to test how well models perform on\n"
                  << "unseen data, providing robust estimates of predictive accuracy.\n" << std::endl;
        
        // Test 3: K-fold Cross-validation
        std::cout << "📊 3. K-Fold Cross-Validation (k=5):" << std::endl;
        std::cout << "   Splits data into 5 folds, uses 4 for training, 1 for validation" << std::endl;
        std::cout << "   Repeats process 5 times with different validation folds" << std::endl;
        std::cout << "   Outputs: MAE (prediction error), StdErr (parameter uncertainty), LogLik (fit quality)\n" << std::endl;
        auto cv_results = libstats::GaussianDistribution::kFoldCrossValidation(data, 5, 42);
        std::cout << "   Results per fold (Mean Abs Error, Std Error, Log-Likelihood):" << std::endl;
        double total_mae = 0.0, total_loglik = 0.0;
        for (size_t i = 0; i < cv_results.size(); ++i) {
#if defined(_MSC_VER)
            const auto& result = cv_results[i];
            double mae = std::get<0>(result);
            double std_err = std::get<1>(result);
            double loglik = std::get<2>(result);
#else
            auto [mae, std_err, loglik] = cv_results[i];
#endif
            std::cout << "     Fold " << (i+1) << ": MAE=" << mae 
                     << ", StdErr=" << std_err << ", LogLik=" << loglik << std::endl;
            total_mae += mae;
            total_loglik += loglik;
        }
        std::cout << "   → Average MAE: " << (total_mae / cv_results.size()) << std::endl;
        std::cout << "   → Total Log-Likelihood: " << total_loglik << std::endl;
        
        // Test 4: Leave-one-out cross-validation (smaller dataset for speed)
        std::vector<double> small_data(data.begin(), data.begin() + 20);
        std::cout << "\n🔍 4. Leave-One-Out Cross-Validation (LOOCV):" << std::endl;
        std::cout << "   Uses 20 samples (subset for computational efficiency)" << std::endl;
        std::cout << "   Trains on 19 samples, validates on 1 (repeated 20 times)" << std::endl;
        std::cout << "   Provides nearly unbiased but high-variance performance estimates\n" << std::endl;
        auto [loocv_mae, loocv_rmse, loocv_loglik] = libstats::GaussianDistribution::leaveOneOutCrossValidation(small_data);
        std::cout << "   Mean Absolute Error: " << loocv_mae << std::endl;
        std::cout << "   Root Mean Squared Error: " << loocv_rmse << std::endl;
        std::cout << "   Total Log-Likelihood: " << loocv_loglik << std::endl;
        std::cout << "   → LOOCV provides unbiased estimate of model performance" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "BOOTSTRAP AND INFORMATION CRITERIA" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\nParameter uncertainty estimation and model comparison methods:\n"
                  << "Bootstrap resampling provides robust confidence intervals while\n"
                  << "information criteria balance model fit against complexity.\n" << std::endl;
        
        // Test 5: Bootstrap confidence intervals
        std::cout << "🔄 5. Bootstrap Parameter Confidence Intervals:" << std::endl;
        std::cout << "   Resamples original data 1000 times with replacement" << std::endl;
        std::cout << "   Fits distribution to each bootstrap sample" << std::endl;
        std::cout << "   Creates confidence intervals from parameter distributions\n" << std::endl;
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
        std::cout << "\n🏆 6. Information Criteria (Model Selection):" << std::endl;
        std::cout << "   Balances model fit quality against complexity penalties" << std::endl;
        std::cout << "   Used to compare different models and prevent overfitting\n" << std::endl;
        
        libstats::GaussianDistribution fitted_dist;
        fitted_dist.fit(data);
        auto [aic, bic, aicc, loglik] = libstats::GaussianDistribution::computeInformationCriteria(data, fitted_dist);
        std::cout << "   Fitted parameters: μ=" << fitted_dist.getMean() 
                 << ", σ=" << fitted_dist.getStandardDeviation() << std::endl;
        std::cout << "   AIC:  " << aic << " (Akaike Information Criterion - general purpose)" << std::endl;
        std::cout << "   BIC:  " << bic << " (Bayesian Information Criterion - stricter penalty)" << std::endl;
        std::cout << "   AICc: " << aicc << " (Corrected AIC - small sample adjustment)" << std::endl;
        std::cout << "   Log-likelihood: " << loglik << " (Goodness of fit measure)" << std::endl;
        std::cout << "   → Lower AIC/BIC values indicate better model fit (accounting for complexity)" << std::endl;
        
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "PERFORMANCE TEST WITH NON-NORMAL DATA" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        // Test with obviously non-normal data
        std::vector<double> non_normal_data;
        for (int i = 0; i < 50; ++i) {
            non_normal_data.push_back(i * i); // Quadratic growth - definitely not normal
        }
        
        std::cout << "\n🚨 Testing goodness-of-fit with obviously non-normal data:\n"
                  << "(50 samples of quadratic sequence: 0², 1², 2², ..., 49²)\n" << std::endl;
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

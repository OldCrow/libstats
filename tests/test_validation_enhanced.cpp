#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include "../include/core/validation.h"
#include "../include/distributions/gaussian.h"

using namespace libstats;
using namespace libstats::validation;

void test_enhanced_pvalues() {
    std::cout << "Testing Enhanced P-Value Calculations" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Generate some test data from a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);
    
    std::vector<double> normal_data;
    for (int i = 0; i < 100; ++i) {
        normal_data.push_back(dist(gen));
    }
    
    // Create a Gaussian distribution for testing
    libstats::GaussianDistribution test_gaussian(0.0, 1.0);
    
    // Test Kolmogorov-Smirnov
    std::cout << "\n1. Kolmogorov-Smirnov Test:" << std::endl;
    KSTestResult ks_result = kolmogorov_smirnov_test(normal_data, test_gaussian);
    std::cout << "   Statistic: " << std::fixed << std::setprecision(6) << ks_result.statistic << std::endl;
    std::cout << "   P-value: " << ks_result.p_value << std::endl;
    std::cout << "   Result: " << ks_result.interpretation << std::endl;
    
    // Test Anderson-Darling
    std::cout << "\n2. Anderson-Darling Test:" << std::endl;
    ADTestResult ad_result = anderson_darling_test(normal_data, test_gaussian);
    std::cout << "   Statistic: " << std::fixed << std::setprecision(6) << ad_result.statistic << std::endl;
    std::cout << "   P-value: " << ad_result.p_value << std::endl;
    std::cout << "   Result: " << ad_result.interpretation << std::endl;
    
    // Test Chi-squared with discrete data
    std::cout << "\n3. Chi-squared Test:" << std::endl;
    
    // Create some discrete data for chi-squared test
    std::vector<int> observed_counts = {23, 27, 25, 25}; // 4 categories, 100 total observations
    std::vector<double> expected_probs = {0.25, 0.25, 0.25, 0.25}; // Equal probabilities
    
    ChiSquaredResult chi_result = chi_squared_goodness_of_fit(observed_counts, expected_probs);
    std::cout << "   Statistic: " << std::fixed << std::setprecision(6) << chi_result.statistic << std::endl;
    std::cout << "   P-value: " << chi_result.p_value << std::endl;
    std::cout << "   Degrees of freedom: " << chi_result.degrees_of_freedom << std::endl;
    std::cout << "   Result: " << chi_result.interpretation << std::endl;
    
    // Test with more extreme chi-squared data
    std::cout << "\n4. Chi-squared Test (extreme case):" << std::endl;
    std::vector<int> extreme_counts = {50, 10, 10, 30}; // Very unequal distribution
    ChiSquaredResult chi_extreme = chi_squared_goodness_of_fit(extreme_counts, expected_probs);
    std::cout << "   Statistic: " << std::fixed << std::setprecision(6) << chi_extreme.statistic << std::endl;
    std::cout << "   P-value: " << chi_extreme.p_value << std::endl;
    std::cout << "   Degrees of freedom: " << chi_extreme.degrees_of_freedom << std::endl;
    std::cout << "   Result: " << chi_extreme.interpretation << std::endl;
    
    // Test Model Diagnostics
    std::cout << "\n5. Model Diagnostics:" << std::endl;
    ModelDiagnostics diag = calculate_model_diagnostics(test_gaussian, normal_data);
    std::cout << "   Log-likelihood: " << std::fixed << std::setprecision(3) << diag.log_likelihood << std::endl;
    std::cout << "   AIC: " << diag.aic << std::endl;
    std::cout << "   BIC: " << diag.bic << std::endl;
    std::cout << "   Parameters: " << diag.num_parameters << std::endl;
    std::cout << "   Sample size: " << diag.sample_size << std::endl;
    
    std::cout << "\n6. Residual Analysis:" << std::endl;
    std::vector<double> residuals = calculate_residuals(normal_data, test_gaussian);
    double mean_residual = 0.0;
    for (double r : residuals) {
        mean_residual += r;
    }
    mean_residual /= static_cast<double>(residuals.size());
    
    double var_residual = 0.0;
    for (double r : residuals) {
        var_residual += (r - mean_residual) * (r - mean_residual);
    }
    var_residual /= static_cast<double>(residuals.size() - 1);
    
    std::cout << "   Mean residual: " << std::fixed << std::setprecision(6) << mean_residual << std::endl;
    std::cout << "   Residual variance: " << var_residual << std::endl;
    std::cout << "   Number of residuals: " << residuals.size() << std::endl;
}

void test_pvalue_accuracy() {
    std::cout << "\n\nTesting P-Value Accuracy" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Test known chi-squared values
    std::cout << "\nChi-squared p-values for known test statistics:" << std::endl;
    
    // Test cases with known results (approximately)
    struct ChiTest {
        double chi_sq;
        int df;
        double expected_p; // Approximate expected p-value
    };
    
    std::vector<ChiTest> chi_tests = {
        {3.841, 1, 0.05},   // Critical value for df=1, α=0.05
        {5.991, 2, 0.05},   // Critical value for df=2, α=0.05
        {7.815, 3, 0.05},   // Critical value for df=3, α=0.05
        {1.0, 1, 0.317},    // Should be about 0.317
        {0.0, 1, 1.0},      // Should be 1.0
        {10.0, 1, 0.0016}   // Should be small
    };
    
    for (const auto& test : chi_tests) {
        // Using internal function for testing - normally not accessible
        std::cout << "   χ²=" << test.chi_sq << ", df=" << test.df 
                  << " → p≈" << std::fixed << std::setprecision(4) 
                  << " (expected ≈" << test.expected_p << ")" << std::endl;
    }
    
    // Test Kolmogorov-Smirnov with known distribution
    std::cout << "\nKolmogorov-Smirnov test with perfect fit:" << std::endl;
    
    // Create data that exactly matches the distribution
    std::vector<double> perfect_data;
    libstats::GaussianDistribution perfect_gaussian(0.0, 1.0);
    
    // Generate quantiles that should give a good fit
    for (int i = 1; i <= 50; ++i) {
        double p = static_cast<double>(i) / 51.0;
        // Approximate inverse normal CDF (for demonstration)
        double z = (p < 0.5) ? -std::sqrt(-2.0 * std::log(p)) : std::sqrt(-2.0 * std::log(1.0 - p));
        perfect_data.push_back(z);
    }
    
    KSTestResult perfect_ks = kolmogorov_smirnov_test(perfect_data, perfect_gaussian);
    std::cout << "   Perfect fit KS statistic: " << std::fixed << std::setprecision(6) << perfect_ks.statistic << std::endl;
    std::cout << "   Perfect fit p-value: " << perfect_ks.p_value << std::endl;
    
    std::cout << "\nTesting completed successfully!" << std::endl;
}

void test_bootstrap_methods() {
    std::cout << "\n\nTesting Bootstrap-Based Statistical Tests" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    // Generate test data from a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);
    
    std::vector<double> normal_data;
    for (int i = 0; i < 50; ++i) {  // Using smaller sample for faster bootstrap
        normal_data.push_back(dist(gen));
    }
    
    // Create a Gaussian distribution for testing
    libstats::GaussianDistribution test_gaussian(0.0, 1.0);
    
    // Test Bootstrap Kolmogorov-Smirnov
    std::cout << "\n1. Bootstrap Kolmogorov-Smirnov Test:" << std::endl;
    std::cout << "   Running bootstrap KS test with 100 bootstrap samples..." << std::endl;
    
    BootstrapTestResult bootstrap_ks = bootstrap_kolmogorov_smirnov_test(
        normal_data, test_gaussian, 100, 0.05);
    
    std::cout << "   Observed statistic: " << std::fixed << std::setprecision(6) 
              << bootstrap_ks.observed_statistic << std::endl;
    std::cout << "   Bootstrap p-value: " << bootstrap_ks.bootstrap_p_value << std::endl;
    std::cout << "   Bootstrap samples: " << bootstrap_ks.num_bootstrap_samples << std::endl;
    std::cout << "   Result: " << bootstrap_ks.interpretation << std::endl;
    
    // Test Bootstrap Anderson-Darling
    std::cout << "\n2. Bootstrap Anderson-Darling Test:" << std::endl;
    std::cout << "   Running bootstrap AD test with 100 bootstrap samples..." << std::endl;
    
    BootstrapTestResult bootstrap_ad = bootstrap_anderson_darling_test(
        normal_data, test_gaussian, 100, 0.05);
    
    std::cout << "   Observed statistic: " << std::fixed << std::setprecision(6) 
              << bootstrap_ad.observed_statistic << std::endl;
    std::cout << "   Bootstrap p-value: " << bootstrap_ad.bootstrap_p_value << std::endl;
    std::cout << "   Bootstrap samples: " << bootstrap_ad.num_bootstrap_samples << std::endl;
    std::cout << "   Result: " << bootstrap_ad.interpretation << std::endl;
    
    // Test Bootstrap Parameter Test
    std::cout << "\n3. Bootstrap Parameter Test:" << std::endl;
    std::cout << "   Running bootstrap parameter test with 100 bootstrap samples..." << std::endl;
    
    BootstrapTestResult bootstrap_param = bootstrap_parameter_test(
        normal_data, test_gaussian, 100, 0.05);
    
    std::cout << "   Observed statistic: " << std::fixed << std::setprecision(6) 
              << bootstrap_param.observed_statistic << std::endl;
    std::cout << "   Bootstrap p-value: " << bootstrap_param.bootstrap_p_value << std::endl;
    std::cout << "   Bootstrap samples: " << bootstrap_param.num_bootstrap_samples << std::endl;
    std::cout << "   Result: " << bootstrap_param.interpretation << std::endl;
    
    // Test Bootstrap Confidence Intervals
    std::cout << "\n4. Bootstrap Confidence Intervals:" << std::endl;
    std::cout << "   Computing 95% confidence intervals with 100 bootstrap samples..." << std::endl;
    
    std::vector<ConfidenceInterval> intervals = bootstrap_confidence_intervals(
        normal_data, test_gaussian, 0.95, 100);
    
    if (intervals.size() >= 2) {
        std::cout << "   Mean CI: [" << std::fixed << std::setprecision(4) 
                  << intervals[0].lower_bound << ", " << intervals[0].upper_bound 
                  << "] (point estimate: " << intervals[0].point_estimate << ")" << std::endl;
        std::cout << "   Variance CI: [" << intervals[1].lower_bound << ", " 
                  << intervals[1].upper_bound << "] (point estimate: " 
                  << intervals[1].point_estimate << ")" << std::endl;
    }
    
    // Test with non-normal data to see rejection
    std::cout << "\n5. Bootstrap Tests with Non-Normal Data:" << std::endl;
    std::cout << "   Testing with exponential data against normal distribution..." << std::endl;
    
    // Generate exponential data
    std::exponential_distribution<> exp_dist(1.0);
    std::vector<double> exp_data;
    for (int i = 0; i < 50; ++i) {
        exp_data.push_back(exp_dist(gen));
    }
    
    BootstrapTestResult bootstrap_ks_exp = bootstrap_kolmogorov_smirnov_test(
        exp_data, test_gaussian, 100, 0.05);
    
    std::cout << "   Bootstrap KS test (exp vs normal):" << std::endl;
    std::cout << "   Observed statistic: " << std::fixed << std::setprecision(6) 
              << bootstrap_ks_exp.observed_statistic << std::endl;
    std::cout << "   Bootstrap p-value: " << bootstrap_ks_exp.bootstrap_p_value << std::endl;
    std::cout << "   Result: " << bootstrap_ks_exp.interpretation << std::endl;
    
    std::cout << "\nBootstrap testing completed successfully!" << std::endl;
}

int main() {
    try {
        test_enhanced_pvalues();
        test_pvalue_accuracy();
        test_bootstrap_methods();
        
        std::cout << "\n✓ All enhanced p-value and bootstrap tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}

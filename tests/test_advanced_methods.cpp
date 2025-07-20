#include "../include/gaussian.h"
#include <iostream>
#include <vector>
#include <random>

int main() {
    try {
        std::cout << "=== Testing Advanced Gaussian Statistical Methods ===" << std::endl;
        
        // Generate some test data from a known normal distribution
        std::mt19937 rng(42);
        std::normal_distribution<double> normal(5.0, 2.0);
        
        std::vector<double> data(100);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = normal(rng);
        }
        
        std::cout << "Generated " << data.size() << " samples from N(5.0, 2.0)" << std::endl;
        
        // Test confidence interval for mean
        std::cout << "\n1. Testing Confidence Interval for Mean:" << std::endl;
        auto [ci_lower, ci_upper] = libstats::GaussianDistribution::confidenceIntervalMean(data, 0.95);
        std::cout << "   95% CI for mean: [" << ci_lower << ", " << ci_upper << "]" << std::endl;
        
        // Test one-sample t-test
        std::cout << "\n2. Testing One-Sample t-test:" << std::endl;
        auto [t_stat, p_value, reject_null] = libstats::GaussianDistribution::oneSampleTTest(data, 5.0, 0.05);
        std::cout << "   H0: μ = 5.0, α = 0.05" << std::endl;
        std::cout << "   t-statistic: " << t_stat << std::endl;
        std::cout << "   p-value: " << p_value << std::endl;
        std::cout << "   Reject H0: " << (reject_null ? "Yes" : "No") << std::endl;
        
        // Test method of moments estimation
        std::cout << "\n3. Testing Method of Moments Estimation:" << std::endl;
        auto [mom_mean, mom_std] = libstats::GaussianDistribution::methodOfMomentsEstimation(data);
        std::cout << "   Estimated mean: " << mom_mean << " (expected: ~5.0)" << std::endl;
        std::cout << "   Estimated std: " << mom_std << " (expected: ~2.0)" << std::endl;
        
        // Test Jarque-Bera normality test
        std::cout << "\n4. Testing Jarque-Bera Normality Test:" << std::endl;
        auto [jb_stat, jb_p_value, jb_reject] = libstats::GaussianDistribution::jarqueBeraTest(data, 0.05);
        std::cout << "   JB statistic: " << jb_stat << std::endl;
        std::cout << "   p-value: " << jb_p_value << std::endl;
        std::cout << "   Reject normality: " << (jb_reject ? "Yes" : "No") << std::endl;
        
        // Test robust estimation
        std::cout << "\n5. Testing Robust Estimation (Huber):" << std::endl;
        auto [robust_loc, robust_scale] = libstats::GaussianDistribution::robustEstimation(data, "huber", 1.345);
        std::cout << "   Robust location: " << robust_loc << " (expected: ~5.0)" << std::endl;
        std::cout << "   Robust scale: " << robust_scale << " (expected: ~2.0)" << std::endl;
        
        std::cout << "\n=== All Advanced Methods Test Successfully! ===" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

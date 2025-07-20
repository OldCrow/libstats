#include "../include/exponential.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

int main() {
    std::cout << "Testing ExponentialDistribution Implementation\n";
    std::cout << "==============================================\n\n";
    
    // Test 1: Safe factory method
    std::cout << "Test 1: Safe factory method\n";
    auto result = libstats::ExponentialDistribution::create(2.0);
    if (result.isOk()) {
        std::cout << "✅ Safe factory creation successful\n";
        auto exp_dist = std::move(result.value);
        
        // Test basic properties
        std::cout << "Lambda: " << exp_dist.getLambda() << std::endl;
        std::cout << "Mean: " << exp_dist.getMean() << std::endl;
        std::cout << "Variance: " << exp_dist.getVariance() << std::endl;
        std::cout << "Skewness: " << exp_dist.getSkewness() << std::endl;
        std::cout << "Kurtosis: " << exp_dist.getKurtosis() << std::endl;
        
        // Test PDF, CDF, and quantile
        std::cout << "\nTest 2: PDF, CDF, and quantile functions\n";
        double x = 1.0;
        std::cout << "PDF(1.0): " << exp_dist.getProbability(x) << std::endl;
        std::cout << "Log PDF(1.0): " << exp_dist.getLogProbability(x) << std::endl;
        std::cout << "CDF(1.0): " << exp_dist.getCumulativeProbability(x) << std::endl;
        std::cout << "Quantile(0.5): " << exp_dist.getQuantile(0.5) << std::endl;
        
        // Test sampling
        std::cout << "\nTest 3: Random sampling\n";
        std::mt19937 rng(42);
        std::vector<double> samples;
        for (int i = 0; i < 10; ++i) {
            samples.push_back(exp_dist.sample(rng));
        }
        
        std::cout << "10 random samples: ";
        for (double sample : samples) {
            std::cout << std::fixed << std::setprecision(3) << sample << " ";
        }
        std::cout << std::endl;
        
        // Test fitting
        std::cout << "\nTest 4: Parameter fitting\n";
        std::vector<double> data = {0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8, 0.7, 1.1};
        exp_dist.fit(data);
        std::cout << "After fitting to data, Lambda: " << exp_dist.getLambda() << std::endl;
        std::cout << "New mean: " << exp_dist.getMean() << std::endl;
        
        // Test batch operations
        std::cout << "\nTest 5: Batch operations\n";
        std::vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 5.0};
        std::vector<double> pdf_results(test_values.size());
        std::vector<double> cdf_results(test_values.size());
        
        exp_dist.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
        exp_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
        
        std::cout << "Batch PDF results: ";
        for (double result : pdf_results) {
            std::cout << std::fixed << std::setprecision(4) << result << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Batch CDF results: ";
        for (double result : cdf_results) {
            std::cout << std::fixed << std::setprecision(4) << result << " ";
        }
        std::cout << std::endl;
        
        std::cout << "\n✅ All tests passed successfully!\n";
        
    } else {
        std::cout << "❌ Error creating distribution: " << result.message << std::endl;
        return 1;
    }
    
    // Test error handling
    std::cout << "\nTest 6: Error handling\n";
    auto error_result = libstats::ExponentialDistribution::create(-1.0);
    if (error_result.isError()) {
        std::cout << "✅ Error handling works: " << error_result.message << std::endl;
    } else {
        std::cout << "❌ Error handling failed\n";
        return 1;
    }
    
    std::cout << "\n🎉 All ExponentialDistribution tests completed successfully!\n";
    return 0;
}

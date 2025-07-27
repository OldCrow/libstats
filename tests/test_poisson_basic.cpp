#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>

// Include the Poisson distribution
#include "../include/distributions/poisson.h"
#include "basic_test_template.h"

using namespace std;
using namespace libstats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Poisson");
    
    try {
        // Test 1: Safe factory method
        StandardizedBasicTest::printTestStart(1, "Safe factory method");
        auto result = PoissonDistribution::create(3.0);
        if (result.isOk()) {
            StandardizedBasicTest::printTestSuccess("Safe factory creation successful");
            auto poisson_dist = std::move(result.value);
            
            // Test basic properties
            StandardizedBasicTest::printProperty("Lambda", poisson_dist.getLambda());
            StandardizedBasicTest::printProperty("Mean", poisson_dist.getMean());
            StandardizedBasicTest::printProperty("Variance", poisson_dist.getVariance());
            StandardizedBasicTest::printProperty("Skewness", poisson_dist.getSkewness());
            StandardizedBasicTest::printProperty("Kurtosis", poisson_dist.getKurtosis());
            StandardizedBasicTest::printProperty("Mode", poisson_dist.getMode());
            StandardizedBasicTest::printNewline();
            
            // Test PMF, CDF, and quantile
            StandardizedBasicTest::printTestStart(2, "PMF, CDF, and quantile functions");
            int k = 3;
            StandardizedBasicTest::printProperty("PMF(3)", poisson_dist.getProbability(k));
            StandardizedBasicTest::printProperty("Log PMF(3)", poisson_dist.getLogProbability(k));
            StandardizedBasicTest::printProperty("CDF(3)", poisson_dist.getCumulativeProbability(k));
            StandardizedBasicTest::printProperty("Quantile(0.5)", poisson_dist.getQuantile(0.5));
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test exact integer methods
            StandardizedBasicTest::printTestStart(3, "Exact integer methods");
            StandardizedBasicTest::printProperty("PMF_exact(3)", poisson_dist.getProbabilityExact(3));
            StandardizedBasicTest::printProperty("Log PMF_exact(3)", poisson_dist.getLogProbabilityExact(3));
            StandardizedBasicTest::printProperty("CDF_exact(3)", poisson_dist.getCumulativeProbabilityExact(3));
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test sampling
            StandardizedBasicTest::printTestStart(4, "Random sampling");
            mt19937 rng(42);
            vector<double> samples;
            for (int i = 0; i < 10; ++i) {
                samples.push_back(poisson_dist.sample(rng));
            }
            StandardizedBasicTest::printSamples(samples, "10 random samples", 0);
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test integer sampling
            StandardizedBasicTest::printTestStart(5, "Integer sampling");
            mt19937 rng2(42);
            auto int_samples = poisson_dist.sampleIntegers(rng2, 10);
            StandardizedBasicTest::printIntegerSamples(int_samples, "10 integer samples");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test fitting
            StandardizedBasicTest::printTestStart(6, "Parameter fitting");
            vector<double> data = StandardizedBasicTest::generatePoissonTestData();
            poisson_dist.fit(data);
            StandardizedBasicTest::printProperty("After fitting - Lambda", poisson_dist.getLambda());
            StandardizedBasicTest::printProperty("After fitting - Mean", poisson_dist.getMean());
            StandardizedBasicTest::printProperty("After fitting - Variance", poisson_dist.getVariance());
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test batch operations
            StandardizedBasicTest::printTestStart(7, "Batch operations");
            vector<double> test_values = {0, 1, 2, 3, 4, 5};
            vector<double> pmf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            poisson_dist.getProbabilityBatch(test_values.data(), pmf_results.data(), test_values.size());
            poisson_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
            
            StandardizedBasicTest::printBatchResults(pmf_results, "Batch PMF results");
            StandardizedBasicTest::printBatchResults(cdf_results, "Batch CDF results");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test large batch for SIMD validation
            StandardizedBasicTest::printTestStart(8, "Large batch SIMD validation");
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 3.0);  // All threes
            vector<double> large_output(large_size);
            
            poisson_dist.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
            StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[999], "PMF at 3");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test normal approximation check
            StandardizedBasicTest::printTestStart(9, "Normal approximation capability");
            cout << "Can use normal approximation (λ=3): " << (poisson_dist.canUseNormalApproximation() ? "YES" : "NO") << endl;
            
            // Test with large lambda
            auto large_lambda_result = PoissonDistribution::create(25.0);
            if (large_lambda_result.isOk()) {
                auto large_dist = std::move(large_lambda_result.value);
                cout << "Can use normal approximation (λ=25): " << (large_dist.canUseNormalApproximation() ? "YES" : "NO") << endl;
            }
            StandardizedBasicTest::printTestSuccess();
            
        } else {
            StandardizedBasicTest::printTestError("Error creating distribution: " + result.message);
            return 1;
        }
        
        StandardizedBasicTest::printNewline();
        
        // Test error handling
        StandardizedBasicTest::printTestStart(10, "Error handling");
        auto error_result = PoissonDistribution::create(-1.0);
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Negative lambda error handling works: " + error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Negative lambda error handling failed");
            return 1;
        }
        
        // Test zero lambda error
        auto zero_result = PoissonDistribution::create(0.0);
        if (zero_result.isError()) {
            cout << "Zero lambda error handling works: " << zero_result.message << endl;
        } else {
            StandardizedBasicTest::printTestError("Zero lambda error handling failed");
            return 1;
        }
        
        StandardizedBasicTest::printCompletionMessage("Poisson");
        
        StandardizedBasicTest::printSummaryHeader();
        StandardizedBasicTest::printSummaryItem("Safe factory creation and error handling");
        StandardizedBasicTest::printSummaryItem("All distribution properties (lambda, mean, variance, skewness, kurtosis, mode)");
        StandardizedBasicTest::printSummaryItem("PMF, Log PMF, CDF, and quantile functions");
        StandardizedBasicTest::printSummaryItem("Exact integer methods for discrete distribution");
        StandardizedBasicTest::printSummaryItem("Random sampling (Knuth's algorithm for small λ, Atkinson for large λ)");
        StandardizedBasicTest::printSummaryItem("Integer sampling convenience method");
        StandardizedBasicTest::printSummaryItem("Parameter fitting (MLE)");
        StandardizedBasicTest::printSummaryItem("Batch operations with performance optimization");
        StandardizedBasicTest::printSummaryItem("Large batch SIMD validation");
        StandardizedBasicTest::printSummaryItem("Normal approximation capability check");
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>

// Include the Exponential distribution
#include "../include/distributions/exponential.h"
#include "basic_test_template.h"

using namespace std;
using namespace libstats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Exponential");
    
    try {
        // Test 1: Safe factory method
        StandardizedBasicTest::printTestStart(1, "Safe factory method");
        auto result = ExponentialDistribution::create(2.0);
        if (result.isOk()) {
            StandardizedBasicTest::printTestSuccess("Safe factory creation successful");
            auto exp_dist = std::move(result.value);
            
            // Test basic properties
            StandardizedBasicTest::printProperty("Lambda", exp_dist.getLambda());
            StandardizedBasicTest::printProperty("Mean", exp_dist.getMean());
            StandardizedBasicTest::printProperty("Variance", exp_dist.getVariance());
            StandardizedBasicTest::printProperty("Skewness", exp_dist.getSkewness());
            StandardizedBasicTest::printProperty("Kurtosis", exp_dist.getKurtosis());
            StandardizedBasicTest::printNewline();
            
            // Test PDF, CDF, and quantile
            StandardizedBasicTest::printTestStart(2, "PDF, CDF, and quantile functions");
            double x = 1.0;
            StandardizedBasicTest::printProperty("PDF(1.0)", exp_dist.getProbability(x));
            StandardizedBasicTest::printProperty("Log PDF(1.0)", exp_dist.getLogProbability(x));
            StandardizedBasicTest::printProperty("CDF(1.0)", exp_dist.getCumulativeProbability(x));
            StandardizedBasicTest::printProperty("Quantile(0.5)", exp_dist.getQuantile(0.5));
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test sampling
            StandardizedBasicTest::printTestStart(3, "Random sampling");
            mt19937 rng(42);
            vector<double> samples;
            for (int i = 0; i < 10; ++i) {
                samples.push_back(exp_dist.sample(rng));
            }
            StandardizedBasicTest::printSamples(samples, "10 random samples");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test fitting
            StandardizedBasicTest::printTestStart(4, "Parameter fitting");
            vector<double> data = StandardizedBasicTest::generateExponentialTestData();
            exp_dist.fit(data);
            StandardizedBasicTest::printProperty("After fitting - Lambda", exp_dist.getLambda());
            StandardizedBasicTest::printProperty("After fitting - Mean", exp_dist.getMean());
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test batch operations
            StandardizedBasicTest::printTestStart(5, "Batch operations");
            vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 5.0};
            vector<double> pdf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            exp_dist.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
            exp_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
            
            StandardizedBasicTest::printBatchResults(pdf_results, "Batch PDF results");
            StandardizedBasicTest::printBatchResults(cdf_results, "Batch CDF results");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test large batch for SIMD validation
            StandardizedBasicTest::printTestStart(6, "Large batch SIMD validation");
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 1.0);  // All ones
            vector<double> large_output(large_size);
            
            exp_dist.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
            StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[999], "PDF at 1.0");
            StandardizedBasicTest::printTestSuccess();
            
        } else {
            StandardizedBasicTest::printTestError("Error creating distribution: " + result.message);
            return 1;
        }
        
        StandardizedBasicTest::printNewline();
        
        // Test error handling
        StandardizedBasicTest::printTestStart(7, "Error handling");
        auto error_result = ExponentialDistribution::create(-1.0);
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Error handling failed");
            return 1;
        }
        
        StandardizedBasicTest::printCompletionMessage("Exponential");
        
        StandardizedBasicTest::printSummaryHeader();
        StandardizedBasicTest::printSummaryItem("Safe factory creation and error handling");
        StandardizedBasicTest::printSummaryItem("All distribution properties (lambda, mean, variance, skewness, kurtosis)");
        StandardizedBasicTest::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        StandardizedBasicTest::printSummaryItem("Random sampling (inverse transform method)");
        StandardizedBasicTest::printSummaryItem("Parameter fitting (MLE)");
        StandardizedBasicTest::printSummaryItem("Batch operations with SIMD optimization");
        StandardizedBasicTest::printSummaryItem("Large batch SIMD validation");
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

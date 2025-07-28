#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>

// Include the Uniform distribution
#include "../include/distributions/uniform.h"
#include "basic_test_template.h"

using namespace std;
using namespace libstats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Uniform");
    
    try {
        // Test 1: Safe factory method
        StandardizedBasicTest::printTestStart(1, "Safe factory method");
        auto result = UniformDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            StandardizedBasicTest::printTestSuccess("Safe factory creation successful");
            auto uniform_dist = std::move(result.value);
            
            // Test basic properties
            StandardizedBasicTest::printProperty("Lower Bound", uniform_dist.getLowerBound());
            StandardizedBasicTest::printProperty("Upper Bound", uniform_dist.getUpperBound());
            StandardizedBasicTest::printProperty("Mean", uniform_dist.getMean());
            StandardizedBasicTest::printProperty("Variance", uniform_dist.getVariance());
            StandardizedBasicTest::printProperty("Standard Deviation", sqrt(uniform_dist.getVariance()));
            StandardizedBasicTest::printProperty("Skewness", uniform_dist.getSkewness());
            StandardizedBasicTest::printProperty("Kurtosis", uniform_dist.getKurtosis());
            StandardizedBasicTest::printNewline();
            
            // Test PDF, CDF, and quantile
            StandardizedBasicTest::printTestStart(2, "PDF, CDF, and quantile functions");
            double x = 0.5;
            StandardizedBasicTest::printProperty("PDF(0.5)", uniform_dist.getProbability(x));
            StandardizedBasicTest::printProperty("Log PDF(0.5)", uniform_dist.getLogProbability(x));
            StandardizedBasicTest::printProperty("CDF(0.5)", uniform_dist.getCumulativeProbability(x));
            StandardizedBasicTest::printProperty("Quantile(0.5)", uniform_dist.getQuantile(0.5));
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test sampling
            StandardizedBasicTest::printTestStart(3, "Random sampling");
            mt19937 rng(42);
            vector<double> samples;
            for (int i = 0; i < 10; ++i) {
                samples.push_back(uniform_dist.sample(rng));
            }
            StandardizedBasicTest::printSamples(samples, "10 random samples");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test fitting
            StandardizedBasicTest::printTestStart(4, "Parameter fitting");
            vector<double> data = StandardizedBasicTest::generateUniformTestData();
            uniform_dist.fit(data);
            StandardizedBasicTest::printProperty("After fitting - Lower", uniform_dist.getLowerBound());
            StandardizedBasicTest::printProperty("After fitting - Upper", uniform_dist.getUpperBound());
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test batch operations
            StandardizedBasicTest::printTestStart(5, "Batch operations");
            vector<double> test_values = {-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5};
            vector<double> pdf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            uniform_dist.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
            uniform_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
            
            StandardizedBasicTest::printBatchResults(pdf_results, "Batch PDF results");
            StandardizedBasicTest::printBatchResults(cdf_results, "Batch CDF results");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test large batch for SIMD validation
            StandardizedBasicTest::printTestStart(6, "Large batch SIMD validation");
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 0.5);  // All 0.5
            vector<double> large_output(large_size);
            
            uniform_dist.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
            StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[999], "PDF at 0.5");
            StandardizedBasicTest::printTestSuccess();
            
        } else {
            StandardizedBasicTest::printTestError("Error creating distribution: " + result.message);
            return 1;
        }
        
        StandardizedBasicTest::printNewline();
        
        // Test error handling
        StandardizedBasicTest::printTestStart(7, "Error handling");
        auto error_result = UniformDistribution::create(5.0, 2.0);
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Error handling failed");
            return 1;
        }
        
        StandardizedBasicTest::printCompletionMessage("Uniform");
        
        StandardizedBasicTest::printSummaryHeader();
        StandardizedBasicTest::printSummaryItem("Safe factory creation and error handling");
        StandardizedBasicTest::printSummaryItem("All distribution properties (bounds, mean, variance, skewness, kurtosis)");
        StandardizedBasicTest::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        StandardizedBasicTest::printSummaryItem("Random sampling (direct transform method)");
        StandardizedBasicTest::printSummaryItem("Parameter fitting (method of moments)");
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

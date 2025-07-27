#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>

// Include the Discrete distribution
#include "../include/distributions/discrete.h"
#include "basic_test_template.h"

using namespace std;
using namespace libstats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Discrete");
    
    try {
        // Test 1: Safe factory method
        StandardizedBasicTest::printTestStart(1, "Safe factory method");
        auto result = DiscreteDistribution::create(1, 6);  // Standard die
        if (result.isOk()) {
            StandardizedBasicTest::printTestSuccess("Safe factory creation successful");
            auto dice_dist = std::move(result.value);
            
            // Test basic properties
            StandardizedBasicTest::printPropertyInt("Lower bound", dice_dist.getLowerBound());
            StandardizedBasicTest::printPropertyInt("Upper bound", dice_dist.getUpperBound());
            StandardizedBasicTest::printPropertyInt("Range", dice_dist.getRange());
            StandardizedBasicTest::printProperty("Mean", dice_dist.getMean());
            StandardizedBasicTest::printProperty("Variance", dice_dist.getVariance());
            StandardizedBasicTest::printProperty("Skewness", dice_dist.getSkewness());
            StandardizedBasicTest::printProperty("Kurtosis", dice_dist.getKurtosis());
            StandardizedBasicTest::printProperty("Single outcome probability", dice_dist.getSingleOutcomeProbability());
            StandardizedBasicTest::printNewline();
            
            // Test PMF, CDF, and quantile
            StandardizedBasicTest::printTestStart(2, "PMF, CDF, and quantile functions");
            double x = 3.0;
            StandardizedBasicTest::printProperty("PMF(3.0)", dice_dist.getProbability(x));
            StandardizedBasicTest::printProperty("Log PMF(3.0)", dice_dist.getLogProbability(x));
            StandardizedBasicTest::printProperty("CDF(3.0)", dice_dist.getCumulativeProbability(x));
            StandardizedBasicTest::printProperty("Quantile(0.5)", dice_dist.getQuantile(0.5));
            // Test PMF for values outside support
            cout << "PMF(0.0): " << dice_dist.getProbability(0.0) << " (outside support)" << endl;
            cout << "PMF(7.0): " << dice_dist.getProbability(7.0) << " (outside support)" << endl;
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test sampling
            StandardizedBasicTest::printTestStart(3, "Random sampling");
            mt19937 rng(42);
            vector<double> samples;
            for (int i = 0; i < 10; ++i) {
                samples.push_back(dice_dist.sample(rng));
            }
            StandardizedBasicTest::printSamples(samples, "10 random samples", 0);
            
            // Test batch sampling
            auto batch_samples = dice_dist.sample(rng, 5);
            vector<double> batch_samples_double(batch_samples.begin(), batch_samples.end());
            StandardizedBasicTest::printSamples(batch_samples_double, "Batch sampling (5 samples)", 0);
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test fitting
            StandardizedBasicTest::printTestStart(4, "Parameter fitting");
            vector<double> data = StandardizedBasicTest::generateDiscreteTestData();
            dice_dist.fit(data);
            StandardizedBasicTest::printPropertyInt("After fitting - Lower bound", dice_dist.getLowerBound());
            StandardizedBasicTest::printPropertyInt("After fitting - Upper bound", dice_dist.getUpperBound());
            StandardizedBasicTest::printProperty("After fitting - Mean", dice_dist.getMean());
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test batch operations
            StandardizedBasicTest::printTestStart(5, "Batch operations");
            vector<double> test_values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            vector<double> pmf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            dice_dist.getProbabilityBatch(test_values.data(), pmf_results.data(), test_values.size());
            dice_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
            
            StandardizedBasicTest::printBatchResults(pmf_results, "Batch PMF results");
            StandardizedBasicTest::printBatchResults(cdf_results, "Batch CDF results");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test large batch for SIMD validation
            StandardizedBasicTest::printTestStart(6, "Large batch SIMD validation");
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 3.0);  // All 3s
            vector<double> large_output(large_size);
            
            dice_dist.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
            StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[999], "PMF at 3");
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test discrete-specific utility methods
            StandardizedBasicTest::printTestStart(7, "Discrete-specific utility methods");
            cout << "Is 3.0 in support: " << (dice_dist.isInSupport(3.0) ? "YES" : "NO") << endl;
            cout << "Is 0.0 in support: " << (dice_dist.isInSupport(0.0) ? "YES" : "NO") << endl;
            cout << "Is 7.0 in support: " << (dice_dist.isInSupport(7.0) ? "YES" : "NO") << endl;
            
            // Get all outcomes (only for small ranges)
            auto outcomes = dice_dist.getAllOutcomes();
            cout << "All possible outcomes: ";
            for (int outcome : outcomes) {
                cout << outcome << " ";
            }
            cout << endl;
            
            // Test integer sampling
            auto int_samples = dice_dist.sampleIntegers(rng, 5);
            StandardizedBasicTest::printIntegerSamples(int_samples, "Integer samples");
            StandardizedBasicTest::printTestSuccess();
            
        } else {
            StandardizedBasicTest::printTestError("Error creating distribution: " + result.message);
            return 1;
        }
        
        StandardizedBasicTest::printNewline();
        
        // Test error handling
        StandardizedBasicTest::printTestStart(8, "Error handling");
        auto error_result = DiscreteDistribution::create(5, 3);  // Invalid: upper < lower
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Error handling failed");
            return 1;
        }
        
        // Test binary distribution (special case)
        StandardizedBasicTest::printTestStart(9, "Binary distribution special case");
        auto binary_result = DiscreteDistribution::create(0, 1);
        if (binary_result.isOk()) {
            auto binary_dist = std::move(binary_result.value);
            StandardizedBasicTest::printProperty("Binary distribution mean", binary_dist.getMean());
            StandardizedBasicTest::printProperty("Binary distribution variance", binary_dist.getVariance());
            StandardizedBasicTest::printProperty("Binary PMF(0)", binary_dist.getProbability(0.0));
            StandardizedBasicTest::printProperty("Binary PMF(1)", binary_dist.getProbability(1.0));
            StandardizedBasicTest::printTestSuccess();
        }
        
        StandardizedBasicTest::printCompletionMessage("Discrete");
        
        StandardizedBasicTest::printSummaryHeader();
        StandardizedBasicTest::printSummaryItem("Safe factory creation and error handling");
        StandardizedBasicTest::printSummaryItem("All distribution properties (bounds, mean, variance, skewness, kurtosis)");
        StandardizedBasicTest::printSummaryItem("PMF, Log PMF, CDF, and quantile functions");
        StandardizedBasicTest::printSummaryItem("Random sampling (single and batch)");
        StandardizedBasicTest::printSummaryItem("Parameter fitting (range estimation)");
        StandardizedBasicTest::printSummaryItem("Batch operations with SIMD optimization");
        StandardizedBasicTest::printSummaryItem("Large batch SIMD validation");
        StandardizedBasicTest::printSummaryItem("Discrete-specific utility methods");
        StandardizedBasicTest::printSummaryItem("Special case handling (binary distribution)");
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

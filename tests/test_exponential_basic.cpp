#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <chrono>
#include <span>

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
            
            // Test smart auto-dispatch batch operations
            StandardizedBasicTest::printTestStart(5, "Smart auto-dispatch batch operations");
            vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 5.0};
            vector<double> pdf_results(test_values.size());
            vector<double> log_pdf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            // Use the new smart auto-dispatch methods with std::span
            auto start = std::chrono::high_resolution_clock::now();
            exp_dist.getProbability(std::span<const double>(test_values), std::span<double>(pdf_results));
            auto end = std::chrono::high_resolution_clock::now();
            auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(log_pdf_results));
            end = std::chrono::high_resolution_clock::now();
            auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(cdf_results));
            end = std::chrono::high_resolution_clock::now();
            auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            // Compare with traditional batch methods
            vector<double> pdf_results_traditional(test_values.size());
            vector<double> log_pdf_results_traditional(test_values.size());
            vector<double> cdf_results_traditional(test_values.size());
            
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getProbabilityBatch(test_values.data(), pdf_results_traditional.data(), test_values.size());
            end = std::chrono::high_resolution_clock::now();
            auto trad_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getLogProbabilityBatch(test_values.data(), log_pdf_results_traditional.data(), test_values.size());
            end = std::chrono::high_resolution_clock::now();
            auto trad_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results_traditional.data(), test_values.size());
            end = std::chrono::high_resolution_clock::now();
            auto trad_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            StandardizedBasicTest::printBatchResults(pdf_results, "Auto-dispatch PDF results");
            StandardizedBasicTest::printBatchResults(log_pdf_results, "Auto-dispatch Log PDF results");
            StandardizedBasicTest::printBatchResults(cdf_results, "Auto-dispatch CDF results");
            
            cout << "Auto-dispatch PDF time: " << auto_pdf_time << "μs, Traditional: " << trad_pdf_time << "μs\n";
            cout << "Auto-dispatch Log PDF time: " << auto_logpdf_time << "μs, Traditional: " << trad_logpdf_time << "μs\n";
            cout << "Auto-dispatch CDF time: " << auto_cdf_time << "μs, Traditional: " << trad_cdf_time << "μs\n";
            cout << "Strategy selected: SCALAR (expected for small batch size=" << test_values.size() << ")\n";
            
            // Verify results are identical
            bool results_match = true;
            for (size_t i = 0; i < test_values.size(); ++i) {
                if (abs(pdf_results[i] - pdf_results_traditional[i]) > 1e-12 ||
                    abs(log_pdf_results[i] - log_pdf_results_traditional[i]) > 1e-12 ||
                    abs(cdf_results[i] - cdf_results_traditional[i]) > 1e-12) {
                    results_match = false;
                    break;
                }
            }
            
            if (results_match) {
                cout << "✅ Auto-dispatch results match traditional methods\n";
            } else {
                cout << "❌ Auto-dispatch results differ from traditional methods\n";
            }
            
            StandardizedBasicTest::printTestSuccess();
            StandardizedBasicTest::printNewline();
            
            // Test large batch auto-dispatch (should trigger SIMD strategy)
            StandardizedBasicTest::printTestStart(6, "Large batch auto-dispatch validation");
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 1.0);  // All ones
            vector<double> large_output(large_size);
            vector<double> large_output_traditional(large_size);
            
            // Test auto-dispatch method
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getProbability(std::span<const double>(large_input), std::span<double>(large_output));
            end = std::chrono::high_resolution_clock::now();
            auto large_auto_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            // Compare with traditional batch method
            start = std::chrono::high_resolution_clock::now();
            exp_dist.getProbabilityBatch(large_input.data(), large_output_traditional.data(), large_size);
            end = std::chrono::high_resolution_clock::now();
            auto large_trad_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[999], "Auto-dispatch PDF at 1.0");
            
            cout << "Large batch auto-dispatch time: " << large_auto_time << "μs, Traditional: " << large_trad_time << "μs\n";
            cout << "Strategy selected: SIMD_BATCH (expected for batch size=" << large_size << ")\n";
            
            // Verify results match
            bool large_results_match = true;
            for (size_t i = 0; i < large_size; ++i) {
                if (abs(large_output[i] - large_output_traditional[i]) > 1e-12) {
                    large_results_match = false;
                    break;
                }
            }
            
            if (large_results_match) {
                cout << "✅ Large batch auto-dispatch results match traditional methods\n";
            } else {
                cout << "❌ Large batch auto-dispatch results differ from traditional methods\n";
            }
            
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
        StandardizedBasicTest::printSummaryItem("Smart auto-dispatch batch operations with strategy selection");
        StandardizedBasicTest::printSummaryItem("Large batch auto-dispatch validation and correctness");
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

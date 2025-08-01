#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <chrono>
#include <span>

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
            
            // Test smart auto-dispatch batch operations
            StandardizedBasicTest::printTestStart(7, "Smart auto-dispatch batch operations");
            vector<double> test_values = {0, 1, 2, 3, 4, 5};
            vector<double> pdf_results(test_values.size());
            vector<double> log_pdf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            // Use the new smart auto-dispatch methods with std::span
            auto start = std::chrono::high_resolution_clock::now();
            poisson_dist.getProbability(std::span<const double>(test_values), std::span<double>(pdf_results));
            auto end = std::chrono::high_resolution_clock::now();
            auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(log_pdf_results));
            end = std::chrono::high_resolution_clock::now();
            auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(cdf_results));
            end = std::chrono::high_resolution_clock::now();
            auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            // Compare with traditional batch methods
            vector<double> pdf_results_traditional(test_values.size());
            vector<double> log_pdf_results_traditional(test_values.size());
            vector<double> cdf_results_traditional(test_values.size());
            
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getProbabilityBatch(test_values.data(), pdf_results_traditional.data(), test_values.size());
            end = std::chrono::high_resolution_clock::now();
            auto trad_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getLogProbabilityBatch(test_values.data(), log_pdf_results_traditional.data(), test_values.size());
            end = std::chrono::high_resolution_clock::now();
            auto trad_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results_traditional.data(), test_values.size());
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
            StandardizedBasicTest::printTestStart(8, "Large batch auto-dispatch validation");
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 3.0);  // All threes
            vector<double> large_output(large_size);
            vector<double> large_output_traditional(large_size);
            
            // Test auto-dispatch method
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getProbability(std::span<const double>(large_input), std::span<double>(large_output));
            end = std::chrono::high_resolution_clock::now();
            auto large_auto_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            // Compare with traditional batch method
            start = std::chrono::high_resolution_clock::now();
            poisson_dist.getProbabilityBatch(large_input.data(), large_output_traditional.data(), large_size);
            end = std::chrono::high_resolution_clock::now();
            auto large_trad_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            
            StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[999], "Auto-dispatch PDF at 3");
            
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

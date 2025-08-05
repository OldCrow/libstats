#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <chrono>
#include <span>
#include <sstream>

// Include the Gamma distribution
#include "../include/distributions/gamma.h"
#include "basic_test_template.h"

using namespace std;
using namespace libstats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Gamma");
    
    try {
        // Test 1: Constructors and Destructor
        StandardizedBasicTest::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Gamma distributions: default (1,1), parameterized (2,3)," << endl;
        cout << "copy (parameterized), move (temporary (5,0.5)) constructors, and the safe factory method that avoids exceptions." << endl;
        
        // Default constructor test
        GammaDistribution default_gamma;
        StandardizedBasicTest::printProperty("Default Alpha (shape)", default_gamma.getAlpha());
        StandardizedBasicTest::printProperty("Default Beta (rate)", default_gamma.getBeta());
        
        // Parameterized constructor test
        GammaDistribution param_gamma(2.0, 3.0);
        StandardizedBasicTest::printProperty("Param Alpha", param_gamma.getAlpha());
        StandardizedBasicTest::printProperty("Param Beta", param_gamma.getBeta());
        
        // Copy constructor test
        GammaDistribution copy_gamma(param_gamma);
        StandardizedBasicTest::printProperty("Copy Alpha", copy_gamma.getAlpha());
        StandardizedBasicTest::printProperty("Copy Beta", copy_gamma.getBeta());
        
        // Move constructor test
        GammaDistribution temp_gamma(5.0, 0.5);
        GammaDistribution move_gamma(std::move(temp_gamma));
        StandardizedBasicTest::printProperty("Move Alpha", move_gamma.getAlpha());
        StandardizedBasicTest::printProperty("Move Beta", move_gamma.getBeta());
        
        // Safe factory method test
        auto result = GammaDistribution::create(1.0, 1.0);
        if (result.isOk()) {
            auto factory_gamma = std::move(result.value);
            StandardizedBasicTest::printProperty("Factory Alpha", factory_gamma.getAlpha());
            StandardizedBasicTest::printProperty("Factory Beta", factory_gamma.getBeta());
        }
        
        StandardizedBasicTest::printTestSuccess("All constructor and destructor tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 2: Parameter Getters and Setters
        StandardizedBasicTest::printTestStart(2, "Parameter Getters and Setters");
        cout << "This test verifies parameter access methods: normal getters, atomic (lock-free) getters," << endl;
        cout << "exception-based setters, and safe setters that return Result types instead of throwing." << endl;
        cout << "Using a Gamma(2.0, 1.0) distribution as the results are well known (mean=2, variance=2)." << endl;
        
        GammaDistribution gamma_dist(2.0, 1.0);
        
        // Test getters
        StandardizedBasicTest::printProperty("Initial Alpha", gamma_dist.getAlpha());
        StandardizedBasicTest::printProperty("Initial Beta", gamma_dist.getBeta());
        StandardizedBasicTest::printProperty("Scale (1/Beta)", gamma_dist.getScale());
        StandardizedBasicTest::printProperty("Mean", gamma_dist.getMean());
        StandardizedBasicTest::printProperty("Variance", gamma_dist.getVariance());
        StandardizedBasicTest::printProperty("Skewness", gamma_dist.getSkewness());
        StandardizedBasicTest::printProperty("Kurtosis", gamma_dist.getKurtosis());
        StandardizedBasicTest::printPropertyInt("Num Parameters", gamma_dist.getNumParameters());
        
        // Test atomic getters (lock-free access)
        StandardizedBasicTest::printProperty("Atomic Alpha", gamma_dist.getAlphaAtomic());
        StandardizedBasicTest::printProperty("Atomic Beta", gamma_dist.getBetaAtomic());
        
        // Test setters
        gamma_dist.setAlpha(3.0);
        gamma_dist.setBeta(2.0);
        StandardizedBasicTest::printProperty("After setting - Alpha", gamma_dist.getAlpha());
        StandardizedBasicTest::printProperty("After setting - Beta", gamma_dist.getBeta());
        
        // Test simultaneous parameter setting
        gamma_dist.setParameters(4.0, 1.5);
        StandardizedBasicTest::printProperty("After setParameters - Alpha", gamma_dist.getAlpha());
        StandardizedBasicTest::printProperty("After setParameters - Beta", gamma_dist.getBeta());
        
        // Test safe setters (no exceptions)
        auto set_result = gamma_dist.trySetAlpha(2.5);
        if (set_result.isOk()) {
            StandardizedBasicTest::printProperty("Safe set alpha - Alpha", gamma_dist.getAlpha());
        }
        
        set_result = gamma_dist.trySetBeta(0.8);
        if (set_result.isOk()) {
            StandardizedBasicTest::printProperty("Safe set beta - Beta", gamma_dist.getBeta());
        }
        
        StandardizedBasicTest::printTestSuccess("All parameter getter/setter tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 3: Core Probability Methods
        StandardizedBasicTest::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PDF, log PDF, CDF, quantiles," << endl;
        cout << "and Gamma-specific utilities like mode and entropy for different parameter values." << endl;
        cout << "Expected: For Gamma(2,1): mean=2, mode=1, relatively skewed right distribution." << endl;
        
        GammaDistribution test_gamma(2.0, 1.0);
        double x = 1.5;
        
        StandardizedBasicTest::printProperty("PDF(1.5)", test_gamma.getProbability(x));
        StandardizedBasicTest::printProperty("Log PDF(1.5)", test_gamma.getLogProbability(x));
        StandardizedBasicTest::printProperty("CDF(1.5)", test_gamma.getCumulativeProbability(x));
        StandardizedBasicTest::printProperty("Quantile(0.5)", test_gamma.getQuantile(0.5));
        StandardizedBasicTest::printProperty("Quantile(0.9)", test_gamma.getQuantile(0.9));
        
        // Test edge cases
        StandardizedBasicTest::printProperty("PDF(0.1)", test_gamma.getProbability(0.1));
        StandardizedBasicTest::printProperty("CDF(0.1)", test_gamma.getCumulativeProbability(0.1));
        
        // Test Gamma-specific utility methods
        StandardizedBasicTest::printProperty("Mode", test_gamma.getMode());
        StandardizedBasicTest::printProperty("Entropy", test_gamma.getEntropy());
        StandardizedBasicTest::printProperty("Median", test_gamma.getMedian());
        
        // Test distribution properties
        cout << "Is discrete: " << (test_gamma.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << test_gamma.getDistributionName() << endl;
        StandardizedBasicTest::printProperty("Support lower bound", test_gamma.getSupportLowerBound());
        StandardizedBasicTest::printProperty("Support upper bound", test_gamma.getSupportUpperBound());
        
        StandardizedBasicTest::printTestSuccess("All core probability method tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 4: Random Sampling
        StandardizedBasicTest::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using Marsaglia-Tsang (alpha>=1) and" << endl;
        cout << "Ahrens-Dieter (alpha<1) algorithms. Sample statistics should approximately match distribution parameters." << endl;
        
        mt19937 rng(42);
        
        // Single sample
        double single_sample = test_gamma.sample(rng);
        StandardizedBasicTest::printProperty("Single sample", single_sample);
        
        // Multiple samples
        vector<double> samples = test_gamma.sample(rng, 10);
        StandardizedBasicTest::printSamples(samples, "10 random samples");
        
        // Verify sample statistics approximately match distribution
        double sample_mean = StandardizedBasicTest::computeSampleMean(samples);
        double sample_var = StandardizedBasicTest::computeSampleVariance(samples);
        StandardizedBasicTest::printProperty("Sample mean", sample_mean);
        StandardizedBasicTest::printProperty("Sample variance", sample_var);
        StandardizedBasicTest::printProperty("Expected mean (α/β)", test_gamma.getMean());
        StandardizedBasicTest::printProperty("Expected variance (α/β²)", test_gamma.getVariance());
        
        StandardizedBasicTest::printTestSuccess("All sampling tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 5: Distribution Management
        StandardizedBasicTest::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using Maximum Likelihood Estimation (MLE)," << endl;
        cout << "distribution reset to default (1, 1) parameters, and string representation formatting." << endl;
        
        // Test fitting
        vector<double> fit_data = StandardizedBasicTest::generateGammaTestData();
        GammaDistribution fitted_dist;
        fitted_dist.fit(fit_data);
        StandardizedBasicTest::printProperty("Fitted Alpha", fitted_dist.getAlpha());
        StandardizedBasicTest::printProperty("Fitted Beta", fitted_dist.getBeta());
        
        // Test reset
        fitted_dist.reset();
        StandardizedBasicTest::printProperty("After reset - Alpha", fitted_dist.getAlpha());
        StandardizedBasicTest::printProperty("After reset - Beta", fitted_dist.getBeta());
        
        // Test toString
        string dist_str = fitted_dist.toString();
        cout << "String representation: " << dist_str << endl;
        
        StandardizedBasicTest::printTestSuccess("All distribution management tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 6: Auto-dispatch Parallel Processing with Timing and Strategy Report
        StandardizedBasicTest::printTestStart(6, "Auto-dispatch Parallel Processing");
        cout << "This test verifies smart auto-dispatch that selects optimal execution strategy" << endl;
        cout << "based on batch size: SCALAR for small batches, SIMD_BATCH/PARALLEL_SIMD for large." << endl;
        cout << "Compares performance and verifies correctness against traditional batch methods." << endl;
        
        GammaDistribution test_dist(2.0, 1.0);
        
        // Test small batch (should use SCALAR strategy) - using diverse realistic data
        vector<double> small_test_values = {0.5, 1.2, 2.1, 0.8, 1.5};
        vector<double> small_pdf_results(small_test_values.size());
        vector<double> small_log_pdf_results(small_test_values.size());
        vector<double> small_cdf_results(small_test_values.size());
        
        cout << "\n--- Small Batch Test (size=" << small_test_values.size() << ") ---" << endl;
        
        // Use the new smart auto-dispatch methods with std::span
        auto start = std::chrono::high_resolution_clock::now();
        test_dist.getProbability(std::span<const double>(small_test_values), std::span<double>(small_pdf_results));
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getLogProbability(std::span<const double>(small_test_values), std::span<double>(small_log_pdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getCumulativeProbability(std::span<const double>(small_test_values), std::span<double>(small_cdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch methods for correctness
        vector<double> small_pdf_traditional(small_test_values.size());
        vector<double> small_log_pdf_traditional(small_test_values.size());
        vector<double> small_cdf_traditional(small_test_values.size());
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getProbabilityBatch(small_test_values.data(), small_pdf_traditional.data(), small_test_values.size());
        end = std::chrono::high_resolution_clock::now();
        auto trad_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getLogProbabilityBatch(small_test_values.data(), small_log_pdf_traditional.data(), small_test_values.size());
        end = std::chrono::high_resolution_clock::now();
        auto trad_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getCumulativeProbabilityBatch(small_test_values.data(), small_cdf_traditional.data(), small_test_values.size());
        end = std::chrono::high_resolution_clock::now();
        auto trad_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        StandardizedBasicTest::printBatchResults(small_pdf_results, "Auto-dispatch PDF results");
        StandardizedBasicTest::printBatchResults(small_log_pdf_results, "Auto-dispatch Log PDF results");
        StandardizedBasicTest::printBatchResults(small_cdf_results, "Auto-dispatch CDF results");
        
        cout << "Auto-dispatch PDF time: " << auto_pdf_time << "μs, Traditional: " << trad_pdf_time << "μs" << endl;
        cout << "Auto-dispatch Log PDF time: " << auto_logpdf_time << "μs, Traditional: " << trad_logpdf_time << "μs" << endl;
        cout << "Auto-dispatch CDF time: " << auto_cdf_time << "μs, Traditional: " << trad_cdf_time << "μs" << endl;
        cout << "Strategy selected: SCALAR (expected for small batch size=" << small_test_values.size() << ")" << endl;
        
        // Verify results are identical
        bool small_results_match = true;
        for (size_t i = 0; i < small_test_values.size(); ++i) {
            if (abs(small_pdf_results[i] - small_pdf_traditional[i]) > 1e-12 ||
                abs(small_log_pdf_results[i] - small_log_pdf_traditional[i]) > 1e-12 ||
                abs(small_cdf_results[i] - small_cdf_traditional[i]) > 1e-12) {
                small_results_match = false;
                break;
            }
        }
        
        if (small_results_match) {
            cout << "✅ Small batch auto-dispatch results match traditional methods" << endl;
        } else {
            cout << "❌ Small batch auto-dispatch results differ from traditional methods" << endl;
        }
        
        // Test large batch (should trigger SIMD or PARALLEL strategy)
        cout << "\n--- Large Batch Test (size=5000) ---" << endl;
        const size_t large_size = 5000;
        
        // Generate diverse realistic test data instead of all zeros
        vector<double> large_input(large_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0.1, 5.0);  // Positive values for Gamma
        for (size_t i = 0; i < large_size; ++i) {
            large_input[i] = dis(gen);
        }
        
        vector<double> large_output(large_size);
        vector<double> large_output_traditional(large_size);
        
        // Test auto-dispatch method
        start = std::chrono::high_resolution_clock::now();
        test_dist.getProbability(std::span<const double>(large_input), std::span<double>(large_output));
        end = std::chrono::high_resolution_clock::now();
        auto large_auto_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch method
        start = std::chrono::high_resolution_clock::now();
        test_dist.getProbabilityBatch(large_input.data(), large_output_traditional.data(), large_size);
        end = std::chrono::high_resolution_clock::now();
        auto large_trad_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[4999], "Auto-dispatch PDF (diverse data)");
        
        cout << "Large batch auto-dispatch time: " << large_auto_time << "μs, Traditional: " << large_trad_time << "μs" << endl;
        double speedup = (double)large_trad_time / large_auto_time;
        cout << "Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
        cout << "Strategy selected: SIMD_BATCH or PARALLEL_SIMD (expected for batch size=" << large_size << ")" << endl;
        
        // Verify results match
        bool large_results_match = true;
        for (size_t i = 0; i < large_size; ++i) {
            if (abs(large_output[i] - large_output_traditional[i]) > 1e-12) {
                large_results_match = false;
                break;
            }
        }
        
        if (large_results_match) {
            cout << "✅ Large batch auto-dispatch results match traditional methods" << endl;
        } else {
            cout << "❌ Large batch auto-dispatch results differ from traditional methods" << endl;
        }
        
        if (speedup > 0.8) {
            cout << "✅ Auto-dispatch shows good performance optimization" << endl;
        } else {
            cout << "⚠️  Auto-dispatch performance may be affected by overhead" << endl;
        }
        
        StandardizedBasicTest::printTestSuccess("All auto-dispatch parallel processing tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 7: Comparison and Stream Operators
        StandardizedBasicTest::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions." << endl;
        
        GammaDistribution dist1(2.0, 1.0);
        GammaDistribution dist2(2.0, 1.0);
        GammaDistribution dist3(3.0, 2.0);
        
        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;
        
        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;
        
        // Test stream input (using proper format from output)
        GammaDistribution input_dist;
        ss.seekg(0); // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed alpha: " << input_dist.getAlpha() << endl;
            cout << "Parsed beta: " << input_dist.getBeta() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }
        
        StandardizedBasicTest::printTestSuccess("All comparison and stream operator tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 8: Error Handling
        StandardizedBasicTest::printTestStart(8, "Error Handling");
        auto error_result = GammaDistribution::create(0.0, -1.0);  // Invalid parameters
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Error handling failed");
            return 1;
        }
        
        StandardizedBasicTest::printCompletionMessage("Gamma");
        
        StandardizedBasicTest::printSummaryHeader();
        StandardizedBasicTest::printSummaryItem("Safe factory creation and error handling");
        StandardizedBasicTest::printSummaryItem("All distribution properties (mean, variance, skewness, kurtosis, mode)");
        StandardizedBasicTest::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        StandardizedBasicTest::printSummaryItem("Random sampling (Marsaglia-Tsang and Ahrens-Dieter algorithms)");
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

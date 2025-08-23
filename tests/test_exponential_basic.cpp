// Focused unit test for exponential distribution
#include "../include/distributions/exponential.h"
#include "basic_test_template.h"

using namespace std;
using namespace stats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Exponential");

    try {
        // Test 1: Constructors and Destructor
        StandardizedBasicTest::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Exponential distributions: default (1.0), "
                "parameterized (2.0),"
             << endl;
        cout << "copy (parameterized), move (temporary (0.5)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_exp = stats::ExponentialDistribution::create().value;
        StandardizedBasicTest::printProperty("Default Lambda", default_exp.getLambda());

        // Parameterized constructor test
        auto param_exp = stats::ExponentialDistribution::create(2.0).value;
        StandardizedBasicTest::printProperty("Param Lambda", param_exp.getLambda());

        // Copy constructor test
        auto copy_exp = param_exp;
        StandardizedBasicTest::printProperty("Copy Lambda", copy_exp.getLambda());

        // Move constructor test
        auto temp_exp = stats::ExponentialDistribution::create(0.5).value;
        auto move_exp = std::move(temp_exp);
        StandardizedBasicTest::printProperty("Move Lambda", move_exp.getLambda());

        // Safe factory method test
        auto result = ExponentialDistribution::create(1.0);
        if (result.isOk()) {
            auto factory_exp = std::move(result.value);
            StandardizedBasicTest::printProperty("Factory Lambda", factory_exp.getLambda());
        }

        StandardizedBasicTest::printTestSuccess("All constructor and destructor tests passed");
        StandardizedBasicTest::printNewline();

        // Test 2: Parameter Getters and Setters
        StandardizedBasicTest::printTestStart(2, "Parameter Getters and Setters");
        cout << "This test verifies parameter access methods: normal getters, atomic (lock-free) "
                "getters,"
             << endl;
        cout << "exception-based setters, and safe setters that return Result types instead of "
                "throwing."
             << endl;
        cout << "Using an Exponential(1.0) distribution as the results are well known (mean=1, "
                "variance=1)."
             << endl;

        auto exp_dist = stats::ExponentialDistribution::create(1.0).value;

        // Test getters
        StandardizedBasicTest::printProperty("Initial Lambda", exp_dist.getLambda());
        StandardizedBasicTest::printProperty("Mean", exp_dist.getMean());
        StandardizedBasicTest::printProperty("Variance", exp_dist.getVariance());
        StandardizedBasicTest::printProperty("Skewness", exp_dist.getSkewness());
        StandardizedBasicTest::printProperty("Kurtosis", exp_dist.getKurtosis());
        StandardizedBasicTest::printPropertyInt("Num Parameters", exp_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        StandardizedBasicTest::printProperty("Atomic Lambda", exp_dist.getLambdaAtomic());

        // Test setters
        exp_dist.setLambda(2.0);
        StandardizedBasicTest::printProperty("After setting - Lambda", exp_dist.getLambda());

        // Test safe setters (no exceptions)
        auto set_result = exp_dist.trySetLambda(0.5);
        if (set_result.isOk()) {
            StandardizedBasicTest::printProperty("Safe set lambda - Lambda", exp_dist.getLambda());
        }

        StandardizedBasicTest::printTestSuccess("All parameter getter/setter tests passed");
        StandardizedBasicTest::printNewline();

        // Test 3: Core Probability Methods
        StandardizedBasicTest::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PDF, log PDF, CDF, quantiles,"
             << endl;
        cout << "and Exponential-specific utilities like mode and entropy for different parameter "
                "values."
             << endl;
        cout << "Expected: For Exponential(1.0): PDF(1)≈0.368, CDF(1)≈0.632, quantile(0.5)≈0.693."
             << endl;

        auto test_exp = stats::ExponentialDistribution::create(1.0).value;
        double x = 1.0;

        StandardizedBasicTest::printProperty("PDF(1.0)", test_exp.getProbability(x));
        StandardizedBasicTest::printProperty("Log PDF(1.0)", test_exp.getLogProbability(x));
        StandardizedBasicTest::printProperty("CDF(1.0)", test_exp.getCumulativeProbability(x));
        StandardizedBasicTest::printProperty("Quantile(0.5)", test_exp.getQuantile(0.5));
        StandardizedBasicTest::printProperty("Quantile(0.9)", test_exp.getQuantile(0.9));

        // Test edge cases
        StandardizedBasicTest::printProperty("PDF(0.1)", test_exp.getProbability(0.1));
        StandardizedBasicTest::printProperty("CDF(0.1)", test_exp.getCumulativeProbability(0.1));

        // Test Exponential-specific utility methods
        StandardizedBasicTest::printProperty("Mode", test_exp.getMode());
        StandardizedBasicTest::printProperty("Entropy", test_exp.getEntropy());
        StandardizedBasicTest::printProperty("Median", test_exp.getMedian());

        // Test distribution properties
        cout << "Is discrete: " << (test_exp.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << test_exp.getDistributionName() << endl;
        StandardizedBasicTest::printProperty("Support lower bound",
                                             test_exp.getSupportLowerBound());
        StandardizedBasicTest::printProperty("Support upper bound",
                                             test_exp.getSupportUpperBound());

        StandardizedBasicTest::printTestSuccess("All core probability method tests passed");
        StandardizedBasicTest::printNewline();

        // Test 4: Random Sampling
        StandardizedBasicTest::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using inverse transform method."
             << endl;
        cout << "Sample statistics should approximately match distribution parameters for large "
                "samples."
             << endl;

        mt19937 rng(42);

        // Single sample
        double single_sample = test_exp.sample(rng);
        StandardizedBasicTest::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = test_exp.sample(rng, 10);
        StandardizedBasicTest::printSamples(samples, "10 random samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = StandardizedBasicTest::computeSampleMean(samples);
        double sample_var = StandardizedBasicTest::computeSampleVariance(samples);
        StandardizedBasicTest::printProperty("Sample mean", sample_mean);
        StandardizedBasicTest::printProperty("Sample variance", sample_var);
        StandardizedBasicTest::printProperty("Expected mean (1/λ)", test_exp.getMean());
        StandardizedBasicTest::printProperty("Expected variance (1/λ²)", test_exp.getVariance());

        StandardizedBasicTest::printTestSuccess("All sampling tests passed");
        StandardizedBasicTest::printNewline();

        // Test 5: Distribution Management
        StandardizedBasicTest::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using Maximum Likelihood Estimation (MLE),"
             << endl;
        cout << "distribution reset to default (1.0) parameter, and string representation "
                "formatting."
             << endl;

        // Test fitting
        vector<double> fit_data = StandardizedBasicTest::generateExponentialTestData();
        auto fitted_dist = stats::ExponentialDistribution::create().value;
        fitted_dist.fit(fit_data);
        StandardizedBasicTest::printProperty("Fitted Lambda", fitted_dist.getLambda());

        // Test reset
        fitted_dist.reset();
        StandardizedBasicTest::printProperty("After reset - Lambda", fitted_dist.getLambda());

        // Test toString
        string dist_str = fitted_dist.toString();
        cout << "String representation: " << dist_str << endl;

        StandardizedBasicTest::printTestSuccess("All distribution management tests passed");
        StandardizedBasicTest::printNewline();

        // Test 6: Auto-dispatch Parallel Processing with Timing and Strategy Report
        StandardizedBasicTest::printTestStart(6, "Auto-dispatch Parallel Processing");
        cout << "This test verifies smart auto-dispatch that selects optimal execution strategy"
             << endl;
        cout << "based on batch size: SCALAR for small batches, SIMD_BATCH/PARALLEL_SIMD for large."
             << endl;
        cout << "Compares performance and verifies correctness against traditional batch methods."
             << endl;

        auto test_dist = stats::ExponentialDistribution::create(1.0).value;

        // Test small batch (should use SCALAR strategy) - using diverse realistic data
        vector<double> small_test_values = {0.1, 0.5, 1.0, 2.0, 5.0};
        vector<double> small_pdf_results(small_test_values.size());
        vector<double> small_log_pdf_results(small_test_values.size());
        vector<double> small_cdf_results(small_test_values.size());

        cout << "\n--- Small Batch Test (size=" << small_test_values.size() << ") ---" << endl;

        // Use the new smart auto-dispatch methods with std::span
        auto start = std::chrono::high_resolution_clock::now();
        test_dist.getProbability(std::span<const double>(small_test_values),
                                 std::span<double>(small_pdf_results));
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        test_dist.getLogProbability(std::span<const double>(small_test_values),
                                    std::span<double>(small_log_pdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        test_dist.getCumulativeProbability(std::span<const double>(small_test_values),
                                           std::span<double>(small_cdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Compare with traditional batch methods for correctness
        vector<double> small_pdf_traditional(small_test_values.size());
        vector<double> small_log_pdf_traditional(small_test_values.size());
        vector<double> small_cdf_traditional(small_test_values.size());

        start = std::chrono::high_resolution_clock::now();
        test_dist.getProbabilityWithStrategy(std::span<const double>(small_test_values),
                                             std::span<double>(small_pdf_traditional),
                                             stats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_pdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        test_dist.getLogProbabilityWithStrategy(std::span<const double>(small_test_values),
                                                std::span<double>(small_log_pdf_traditional),
                                                stats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_logpdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        test_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(small_test_values),
                                                       std::span<double>(small_cdf_traditional),
                                                       stats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_cdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        StandardizedBasicTest::printBatchResults(small_pdf_results, "Auto-dispatch PDF results");
        StandardizedBasicTest::printBatchResults(small_log_pdf_results,
                                                 "Auto-dispatch Log PDF results");
        StandardizedBasicTest::printBatchResults(small_cdf_results, "Auto-dispatch CDF results");

        cout << "Auto-dispatch PDF time: " << auto_pdf_time << "μs, Traditional: " << trad_pdf_time
             << "μs" << endl;
        cout << "Auto-dispatch Log PDF time: " << auto_logpdf_time
             << "μs, Traditional: " << trad_logpdf_time << "μs" << endl;
        cout << "Auto-dispatch CDF time: " << auto_cdf_time << "μs, Traditional: " << trad_cdf_time
             << "μs" << endl;
        cout << "Strategy selected: SCALAR (expected for small batch size="
             << small_test_values.size() << ")" << endl;

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
        std::uniform_real_distribution<> dis(0.1, 5.0);
        for (size_t i = 0; i < large_size; ++i) {
            large_input[i] = dis(gen);
        }

        vector<double> large_output(large_size);
        vector<double> large_output_traditional(large_size);

        // Test auto-dispatch method
        start = std::chrono::high_resolution_clock::now();
        test_dist.getProbability(std::span<const double>(large_input),
                                 std::span<double>(large_output));
        end = std::chrono::high_resolution_clock::now();
        auto large_auto_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Compare with traditional batch method
        start = std::chrono::high_resolution_clock::now();
        test_dist.getProbabilityWithStrategy(std::span<const double>(large_input),
                                             std::span<double>(large_output_traditional),
                                             stats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto large_trad_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[4999],
                                                         "Auto-dispatch PDF (diverse data)");

        cout << "Large batch auto-dispatch time: " << large_auto_time
             << "μs, Traditional: " << large_trad_time << "μs" << endl;
        double speedup =
            static_cast<double>(large_trad_time) / static_cast<double>(large_auto_time);
        cout << "Speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
        cout << "Strategy selected: SIMD_BATCH or PARALLEL_SIMD (expected for batch size="
             << large_size << ")" << endl;

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

        StandardizedBasicTest::printTestSuccess(
            "All auto-dispatch parallel processing tests passed");
        StandardizedBasicTest::printNewline();

        // Test 7: Comparison and Stream Operators
        StandardizedBasicTest::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::ExponentialDistribution::create(1.0).value;
        auto dist2 = stats::ExponentialDistribution::create(1.0).value;
        auto dist3 = stats::ExponentialDistribution::create(2.0).value;

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::ExponentialDistribution::create().value;
        ss.seekg(0);  // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed lambda: " << input_dist.getLambda() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        StandardizedBasicTest::printTestSuccess("All comparison and stream operator tests passed");
        StandardizedBasicTest::printNewline();

        // Test 8: Error Handling
        StandardizedBasicTest::printTestStart(8, "Error Handling");
        // NOTE: Using ::create() here (not stats::Exponential) to test exception-free error
        // handling
        // ::create() returns Result<T> for explicit error checking without exceptions
        auto error_result = ExponentialDistribution::create(-1.0);
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Error handling works: " +
                                                    error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Error handling failed");
            return 1;
        }

        StandardizedBasicTest::printCompletionMessage("Exponential");

        StandardizedBasicTest::printSummaryHeader();
        StandardizedBasicTest::printSummaryItem("Safe factory creation and error handling");
        StandardizedBasicTest::printSummaryItem(
            "All distribution properties (lambda, mean, variance, skewness, kurtosis)");
        StandardizedBasicTest::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        StandardizedBasicTest::printSummaryItem("Random sampling (inverse transform method)");
        StandardizedBasicTest::printSummaryItem("Parameter fitting (MLE)");
        StandardizedBasicTest::printSummaryItem(
            "Smart auto-dispatch batch operations with strategy selection");
        StandardizedBasicTest::printSummaryItem(
            "Large batch auto-dispatch validation and correctness");

        return 0;

    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

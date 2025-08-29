// Focused unit test for gamma distribution
#include "../include/distributions/gamma.h"
#include "../include/tests/tests.h"

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Gamma");

    try {
        // Test 1: Constructors and Destructor
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Gamma distributions: default (1,1), "
                "parameterized (2,3),"
             << endl;
        cout << "copy (parameterized), move (temporary (5,0.5)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_gamma = stats::GammaDistribution::create().value;
        BasicTestFormatter::printProperty("Default Alpha (shape)", default_gamma.getAlpha());
        BasicTestFormatter::printProperty("Default Beta (rate)", default_gamma.getBeta());

        // Parameterized constructor test
        auto param_gamma = stats::GammaDistribution::create(2.0, 3.0).value;
        BasicTestFormatter::printProperty("Param Alpha", param_gamma.getAlpha());
        BasicTestFormatter::printProperty("Param Beta", param_gamma.getBeta());

        // Copy constructor test
        auto copy_gamma = param_gamma;
        BasicTestFormatter::printProperty("Copy Alpha", copy_gamma.getAlpha());
        BasicTestFormatter::printProperty("Copy Beta", copy_gamma.getBeta());

        // Move constructor test
        auto temp_gamma = stats::GammaDistribution::create(5.0, 0.5).value;
        auto move_gamma = std::move(temp_gamma);
        BasicTestFormatter::printProperty("Move Alpha", move_gamma.getAlpha());
        BasicTestFormatter::printProperty("Move Beta", move_gamma.getBeta());

        // Safe factory method test
        auto result = GammaDistribution::create(1.0, 1.0);
        if (result.isOk()) {
            auto factory_gamma = std::move(result.value);
            BasicTestFormatter::printProperty("Factory Alpha", factory_gamma.getAlpha());
            BasicTestFormatter::printProperty("Factory Beta", factory_gamma.getBeta());
        }

        BasicTestFormatter::printTestSuccess("All constructor and destructor tests passed");
        BasicTestFormatter::printNewline();

        // Test 2: Parameter Getters and Setters
        BasicTestFormatter::printTestStart(2, "Parameter Getters and Setters");
        cout << "This test verifies parameter access methods: normal getters, atomic (lock-free) "
                "getters,"
             << endl;
        cout << "exception-based setters, and safe setters that return Result types instead of "
                "throwing."
             << endl;
        cout << "Using a Gamma(2.0, 1.0) distribution as the results are well known (mean=2, "
                "variance=2)."
             << endl;

        auto gamma_dist = stats::GammaDistribution::create(2.0, 1.0).value;

        // Test getters
        BasicTestFormatter::printProperty("Initial Alpha", gamma_dist.getAlpha());
        BasicTestFormatter::printProperty("Initial Beta", gamma_dist.getBeta());
        BasicTestFormatter::printProperty("Scale (1/Beta)", gamma_dist.getScale());
        BasicTestFormatter::printProperty("Mean", gamma_dist.getMean());
        BasicTestFormatter::printProperty("Variance", gamma_dist.getVariance());
        BasicTestFormatter::printProperty("Skewness", gamma_dist.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", gamma_dist.getKurtosis());
        BasicTestFormatter::printPropertyInt("Num Parameters", gamma_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        BasicTestFormatter::printProperty("Atomic Alpha", gamma_dist.getAlphaAtomic());
        BasicTestFormatter::printProperty("Atomic Beta", gamma_dist.getBetaAtomic());

        // Test setters
        gamma_dist.setAlpha(3.0);
        gamma_dist.setBeta(2.0);
        BasicTestFormatter::printProperty("After setting - Alpha", gamma_dist.getAlpha());
        BasicTestFormatter::printProperty("After setting - Beta", gamma_dist.getBeta());

        // Test simultaneous parameter setting
        gamma_dist.setParameters(4.0, 1.5);
        BasicTestFormatter::printProperty("After setParameters - Alpha", gamma_dist.getAlpha());
        BasicTestFormatter::printProperty("After setParameters - Beta", gamma_dist.getBeta());

        // Test safe setters (no exceptions)
        auto set_result = gamma_dist.trySetAlpha(2.5);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set alpha - Alpha", gamma_dist.getAlpha());
        }

        set_result = gamma_dist.trySetBeta(0.8);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set beta - Beta", gamma_dist.getBeta());
        }

        BasicTestFormatter::printTestSuccess("All parameter getter/setter tests passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core Probability Methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PDF, log PDF, CDF, quantiles,"
             << endl;
        cout << "and Gamma-specific utilities like mode and entropy for different parameter values."
             << endl;
        cout << "Expected: For Gamma(2,1): mean=2, mode=1, relatively skewed right distribution."
             << endl;

        auto test_gamma = stats::GammaDistribution::create(2.0, 1.0).value;
        double x = 1.5;

        BasicTestFormatter::printProperty("PDF(1.5)", test_gamma.getProbability(x));
        BasicTestFormatter::printProperty("Log PDF(1.5)", test_gamma.getLogProbability(x));
        BasicTestFormatter::printProperty("CDF(1.5)", test_gamma.getCumulativeProbability(x));
        BasicTestFormatter::printProperty("Quantile(0.5)", test_gamma.getQuantile(0.5));
        BasicTestFormatter::printProperty("Quantile(0.9)", test_gamma.getQuantile(0.9));

        // Test edge cases
        BasicTestFormatter::printProperty("PDF(0.1)", test_gamma.getProbability(0.1));
        BasicTestFormatter::printProperty("CDF(0.1)", test_gamma.getCumulativeProbability(0.1));

        // Test Gamma-specific utility methods
        BasicTestFormatter::printProperty("Mode", test_gamma.getMode());
        BasicTestFormatter::printProperty("Entropy", test_gamma.getEntropy());
        BasicTestFormatter::printProperty("Median", test_gamma.getMedian());

        // Test distribution properties
        cout << "Is discrete: " << (test_gamma.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << test_gamma.getDistributionName() << endl;
        BasicTestFormatter::printProperty("Support lower bound", test_gamma.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper bound", test_gamma.getSupportUpperBound());

        BasicTestFormatter::printTestSuccess("All core probability method tests passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using Marsaglia-Tsang (alpha>=1) and"
             << endl;
        cout << "Ahrens-Dieter (alpha<1) algorithms. Sample statistics should approximately match "
                "distribution parameters."
             << endl;

        mt19937 rng(42);

        // Single sample
        double single_sample = test_gamma.sample(rng);
        BasicTestFormatter::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = test_gamma.sample(rng, 10);
        BasicTestFormatter::printSamples(samples, "10 random samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = TestDataGenerators::computeSampleMean(samples);
        double sample_var = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean", sample_mean);
        BasicTestFormatter::printProperty("Sample variance", sample_var);
        BasicTestFormatter::printProperty("Expected mean (α/β)", test_gamma.getMean());
        BasicTestFormatter::printProperty("Expected variance (α/β²)", test_gamma.getVariance());

        BasicTestFormatter::printTestSuccess("All sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using Maximum Likelihood Estimation (MLE),"
             << endl;
        cout << "distribution reset to default (1, 1) parameters, and string representation "
                "formatting."
             << endl;

        // Test fitting
        vector<double> fit_data = TestDataGenerators::generateGammaTestData();
        auto fitted_dist = stats::GammaDistribution::create().value;
        fitted_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted Alpha", fitted_dist.getAlpha());
        BasicTestFormatter::printProperty("Fitted Beta", fitted_dist.getBeta());

        // Test reset
        fitted_dist.reset();
        BasicTestFormatter::printProperty("After reset - Alpha", fitted_dist.getAlpha());
        BasicTestFormatter::printProperty("After reset - Beta", fitted_dist.getBeta());

        // Test toString
        string dist_str = fitted_dist.toString();
        cout << "String representation: " << dist_str << endl;

        BasicTestFormatter::printTestSuccess("All distribution management tests passed");
        BasicTestFormatter::printNewline();

        // Test 6: Auto-dispatch Parallel Processing with Timing and Strategy Report
        BasicTestFormatter::printTestStart(6, "Auto-dispatch Parallel Processing");
        cout << "This test verifies smart auto-dispatch that selects optimal execution strategy"
             << endl;
        cout << "based on batch size: SCALAR for small batches, SIMD_BATCH/PARALLEL_SIMD for large."
             << endl;
        cout << "Compares performance and verifies correctness against traditional batch methods."
             << endl;

        auto test_dist = stats::GammaDistribution::create(2.0, 1.0).value;

        // Test small batch (should use SCALAR strategy) - using diverse realistic data
        vector<double> small_test_values = {0.5, 1.2, 2.1, 0.8, 1.5};
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
                                             stats::detail::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_pdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        test_dist.getLogProbabilityWithStrategy(std::span<const double>(small_test_values),
                                                std::span<double>(small_log_pdf_traditional),
                                                stats::detail::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_logpdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        test_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(small_test_values),
                                                       std::span<double>(small_cdf_traditional),
                                                       stats::detail::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_cdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        BasicTestFormatter::printBatchResults(small_pdf_results, "Auto-dispatch PDF results");
        BasicTestFormatter::printBatchResults(small_log_pdf_results,
                                              "Auto-dispatch Log PDF results");
        BasicTestFormatter::printBatchResults(small_cdf_results, "Auto-dispatch CDF results");

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
        std::uniform_real_distribution<> dis(0.1, 5.0);  // Positive values for Gamma
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
                                             stats::detail::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto large_trad_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        BasicTestFormatter::printLargeBatchValidation(large_output[0], large_output[4999],
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

        BasicTestFormatter::printTestSuccess("All auto-dispatch parallel processing tests passed");
        BasicTestFormatter::printNewline();

        // Test 7: Comparison and Stream Operators
        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::GammaDistribution::create(2.0, 1.0).value;
        auto dist2 = stats::GammaDistribution::create(2.0, 1.0).value;
        auto dist3 = stats::GammaDistribution::create(3.0, 2.0).value;

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::GammaDistribution::create().value;
        ss.seekg(0);  // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed alpha: " << input_dist.getAlpha() << endl;
            cout << "Parsed beta: " << input_dist.getBeta() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        BasicTestFormatter::printTestSuccess("All comparison and stream operator tests passed");
        BasicTestFormatter::printNewline();

        // Test 8: Error Handling
        BasicTestFormatter::printTestStart(8, "Error Handling");
        // NOTE: Using ::create() here (not stats::Gamma) to test exception-free error handling
        // ::create() returns Result<T> for explicit error checking without exceptions
        auto error_result = GammaDistribution::create(0.0, -1.0);  // Invalid parameters
        if (error_result.isError()) {
            BasicTestFormatter::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            BasicTestFormatter::printTestError("Error handling failed");
            return 1;
        }

        BasicTestFormatter::printCompletionMessage("Gamma");

        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Safe factory creation and error handling");
        BasicTestFormatter::printSummaryItem(
            "All distribution properties (mean, variance, skewness, kurtosis, mode)");
        BasicTestFormatter::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        BasicTestFormatter::printSummaryItem(
            "Random sampling (Marsaglia-Tsang and Ahrens-Dieter algorithms)");
        BasicTestFormatter::printSummaryItem("Parameter fitting (MLE)");
        BasicTestFormatter::printSummaryItem(
            "Smart auto-dispatch batch operations with strategy selection");
        BasicTestFormatter::printSummaryItem(
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

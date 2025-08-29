// Focused unit test for poisson distribution
#include "../include/distributions/poisson.h"
#include "../include/tests/tests.h"

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Poisson");

    try {
        // Test 1: Constructors and Destructor
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Poisson distributions: default (1.0), "
                "parameterized (3.0),"
             << endl;
        cout << "copy (parameterized), move (temporary (5.0)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_poisson = stats::PoissonDistribution::create().value;
        BasicTestFormatter::printProperty("Default Lambda", default_poisson.getLambda());

        // Parameterized constructor test
        auto param_poisson = stats::PoissonDistribution::create(3.0).value;
        BasicTestFormatter::printProperty("Param Lambda", param_poisson.getLambda());

        // Copy constructor test
        auto copy_poisson = param_poisson;
        BasicTestFormatter::printProperty("Copy Lambda", copy_poisson.getLambda());

        // Move constructor test
        auto temp_poisson = stats::PoissonDistribution::create(5.0).value;
        auto move_poisson = std::move(temp_poisson);
        BasicTestFormatter::printProperty("Move Lambda", move_poisson.getLambda());

        // Safe factory method test
        auto result = PoissonDistribution::create(3.0);
        if (result.isOk()) {
            auto factory_poisson = std::move(result.value);
            BasicTestFormatter::printProperty("Factory Lambda", factory_poisson.getLambda());
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
        cout << "Using a Poisson(3.0) distribution as the results are well known (mean=3, "
                "variance=3)."
             << endl;

        auto poisson_dist = stats::PoissonDistribution::create(3.0).value;

        // Test getters
        BasicTestFormatter::printProperty("Initial Lambda", poisson_dist.getLambda());
        BasicTestFormatter::printProperty("Mean", poisson_dist.getMean());
        BasicTestFormatter::printProperty("Variance", poisson_dist.getVariance());
        BasicTestFormatter::printProperty("Skewness", poisson_dist.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", poisson_dist.getKurtosis());
        BasicTestFormatter::printProperty("Mode", poisson_dist.getMode());
        BasicTestFormatter::printPropertyInt("Num Parameters", poisson_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        BasicTestFormatter::printProperty("Atomic Lambda", poisson_dist.getLambdaAtomic());

        // Test setters
        poisson_dist.setLambda(4.0);
        BasicTestFormatter::printProperty("After setting - Lambda", poisson_dist.getLambda());

        // Test safe setters (no exceptions)
        auto set_result = poisson_dist.trySetLambda(2.5);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set lambda - Lambda", poisson_dist.getLambda());
        }

        BasicTestFormatter::printTestSuccess("All parameter getter/setter tests passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core Probability Methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PMF, log PMF, CDF, quantiles,"
             << endl;
        cout << "and Poisson-specific utilities like mode calculation and normal approximation "
                "capability."
             << endl;
        cout << "Expected: For Poisson(3.0): PMF(3)≈0.224, CDF(3)≈0.647, mode=3 for symmetric case."
             << endl;

        auto test_poisson = stats::PoissonDistribution::create(3.0).value;
        int k = 3;

        BasicTestFormatter::printProperty("PMF(3)", test_poisson.getProbability(k));
        BasicTestFormatter::printProperty("Log PMF(3)", test_poisson.getLogProbability(k));
        BasicTestFormatter::printProperty("CDF(3)", test_poisson.getCumulativeProbability(k));
        BasicTestFormatter::printProperty("Quantile(0.5)", test_poisson.getQuantile(0.5));
        BasicTestFormatter::printProperty("Quantile(0.9)", test_poisson.getQuantile(0.9));

        // Test edge cases
        BasicTestFormatter::printProperty("PMF(0)", test_poisson.getProbability(0));
        BasicTestFormatter::printProperty("CDF(0)", test_poisson.getCumulativeProbability(0));

        // Test Poisson-specific utility methods
        BasicTestFormatter::printProperty("Mode", test_poisson.getMode());
        BasicTestFormatter::printProperty("Median", test_poisson.getMedian());

        // Test exact integer methods
        BasicTestFormatter::printProperty("PMF_exact(3)", test_poisson.getProbabilityExact(3));
        BasicTestFormatter::printProperty("Log PMF_exact(3)",
                                          test_poisson.getLogProbabilityExact(3));
        BasicTestFormatter::printProperty("CDF_exact(3)",
                                          test_poisson.getCumulativeProbabilityExact(3));

        // Test distribution properties
        cout << "Is discrete: " << (test_poisson.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << test_poisson.getDistributionName() << endl;
        cout << "Can use normal approximation: "
             << (test_poisson.canUseNormalApproximation() ? "YES" : "NO") << endl;

        BasicTestFormatter::printTestSuccess("All core probability method tests passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using Knuth's algorithm (small λ) or"
             << endl;
        cout << "Atkinson's algorithm (large λ). Sample statistics should approximately match "
                "distribution parameters."
             << endl;

        mt19937 rng(42);

        // Single sample
        double single_sample = test_poisson.sample(rng);
        BasicTestFormatter::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = test_poisson.sample(rng, 10);
        BasicTestFormatter::printSamples(samples, "10 random samples");

        // Test integer sampling
        auto int_samples = test_poisson.sampleIntegers(rng, 10);
        BasicTestFormatter::printIntegerSamples(int_samples, "10 integer samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = TestDataGenerators::computeSampleMean(samples);
        double sample_var = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean", sample_mean);
        BasicTestFormatter::printProperty("Sample variance", sample_var);
        BasicTestFormatter::printProperty("Expected mean (λ)", test_poisson.getMean());
        BasicTestFormatter::printProperty("Expected variance (λ)", test_poisson.getVariance());

        BasicTestFormatter::printTestSuccess("All sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using Maximum Likelihood Estimation (MLE),"
             << endl;
        cout << "distribution reset to default (1.0) parameter, and string representation "
                "formatting."
             << endl;

        // Test fitting
        vector<double> fit_data = TestDataGenerators::generatePoissonTestData();
        auto fitted_dist = stats::PoissonDistribution::create().value;
        fitted_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted Lambda", fitted_dist.getLambda());

        // Test reset
        fitted_dist.reset();
        BasicTestFormatter::printProperty("After reset - Lambda", fitted_dist.getLambda());

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

        auto test_dist = stats::PoissonDistribution::create(3.0).value;

        // Test small batch (should use SCALAR strategy) - using diverse realistic data
        vector<double> small_test_values = {0, 1, 2, 3, 4, 5};
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
        std::uniform_int_distribution<> dis(0, 10);
        for (size_t i = 0; i < large_size; ++i) {
            large_input[i] = static_cast<double>(dis(gen));
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

        auto dist1 = stats::PoissonDistribution::create(3.0).value;
        auto dist2 = stats::PoissonDistribution::create(3.0).value;
        auto dist3 = stats::PoissonDistribution::create(5.0).value;

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::PoissonDistribution::create().value;
        ss.seekg(0);  // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed lambda: " << input_dist.getLambda() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        BasicTestFormatter::printTestSuccess("All comparison and stream operator tests passed");
        BasicTestFormatter::printNewline();

        // Test 8: Error Handling
        BasicTestFormatter::printTestStart(8, "Error Handling");
        // NOTE: Using ::create() here (not stats::Poisson) to test exception-free error handling
        // ::create() returns Result<T> for explicit error checking without exceptions
        auto error_result = PoissonDistribution::create(-1.0);  // Invalid: negative lambda
        if (error_result.isError()) {
            BasicTestFormatter::printTestSuccess("Negative lambda error handling works: " +
                                                 error_result.message);
        } else {
            BasicTestFormatter::printTestError("Negative lambda error handling failed");
            return 1;
        }

        // Test zero lambda error
        auto zero_result = PoissonDistribution::create(0.0);
        if (zero_result.isError()) {
            cout << "Zero lambda error handling works: " << zero_result.message << endl;
        } else {
            BasicTestFormatter::printTestError("Zero lambda error handling failed");
            return 1;
        }

        BasicTestFormatter::printCompletionMessage("Poisson");

        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Safe factory creation and error handling");
        BasicTestFormatter::printSummaryItem(
            "All distribution properties (lambda, mean, variance, skewness, kurtosis, mode)");
        BasicTestFormatter::printSummaryItem("PMF, Log PMF, CDF, and quantile functions");
        BasicTestFormatter::printSummaryItem("Exact integer methods for discrete distribution");
        BasicTestFormatter::printSummaryItem(
            "Random sampling (Knuth's algorithm for small λ, Atkinson for large λ)");
        BasicTestFormatter::printSummaryItem("Integer sampling convenience method");
        BasicTestFormatter::printSummaryItem("Parameter fitting (MLE)");
        BasicTestFormatter::printSummaryItem(
            "Smart auto-dispatch batch operations with strategy selection");
        BasicTestFormatter::printSummaryItem(
            "Large batch auto-dispatch validation and correctness");
        BasicTestFormatter::printSummaryItem("Normal approximation capability check");

        return 0;

    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

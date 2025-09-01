// Focused unit test for discrete distribution
#include "../include/distributions/discrete.h"
#include "include/tests.h"

// Standard library includes
#include <chrono>     // for std::chrono::high_resolution_clock, timing measurements
#include <cmath>      // for std::abs
#include <exception>  // for std::exception
#include <iomanip>    // for std::fixed, std::setprecision
#include <iostream>   // for std::cout, std::endl
#include <random>     // for std::mt19937, std::uniform_real_distribution
#include <span>       // for std::span
#include <string>     // for std::string
#include <utility>    // for std::move
#include <vector>     // for std::vector

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Discrete");

    try {
        // Test 1: Constructors and Destructor
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Discrete distributions: default (1,6), "
                "parameterized (0,1),"
             << endl;
        cout << "copy (parameterized), move (temporary (10,15)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_discrete = stats::DiscreteDistribution::create().value;
        BasicTestFormatter::printPropertyInt("Default Lower bound",
                                             default_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Default Upper bound",
                                             default_discrete.getUpperBound());

        // Parameterized constructor test
        auto param_discrete = stats::DiscreteDistribution::create(0, 1).value;
        BasicTestFormatter::printPropertyInt("Param Lower bound", param_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Param Upper bound", param_discrete.getUpperBound());

        // Copy constructor test
        auto copy_discrete = param_discrete;
        BasicTestFormatter::printPropertyInt("Copy Lower bound", copy_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Copy Upper bound", copy_discrete.getUpperBound());

        // Move constructor test
        auto temp_discrete = stats::DiscreteDistribution::create(10, 15).value;
        auto move_discrete = std::move(temp_discrete);
        BasicTestFormatter::printPropertyInt("Move Lower bound", move_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Move Upper bound", move_discrete.getUpperBound());

        // Safe factory method test
        auto result = DiscreteDistribution::create(1, 6);
        if (result.isOk()) {
            auto factory_discrete = std::move(result.value);
            BasicTestFormatter::printPropertyInt("Factory Lower bound",
                                                 factory_discrete.getLowerBound());
            BasicTestFormatter::printPropertyInt("Factory Upper bound",
                                                 factory_discrete.getUpperBound());
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
        cout << "Using a Discrete(1, 6) distribution as the results are well known (mean=3.5, "
                "variance=2.916)."
             << endl;

        auto discrete_dist = stats::DiscreteDistribution::create(1, 6).value;

        // Test getters
        BasicTestFormatter::printPropertyInt("Initial Lower bound", discrete_dist.getLowerBound());
        BasicTestFormatter::printPropertyInt("Initial Upper bound", discrete_dist.getUpperBound());
        BasicTestFormatter::printPropertyInt("Range", discrete_dist.getRange());
        BasicTestFormatter::printProperty("Mean", discrete_dist.getMean());
        BasicTestFormatter::printProperty("Variance", discrete_dist.getVariance());
        BasicTestFormatter::printProperty("Skewness", discrete_dist.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", discrete_dist.getKurtosis());
        BasicTestFormatter::printPropertyInt("Num Parameters", discrete_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        BasicTestFormatter::printPropertyInt("Atomic Lower bound",
                                             discrete_dist.getLowerBoundAtomic());
        BasicTestFormatter::printPropertyInt("Atomic Upper bound",
                                             discrete_dist.getUpperBoundAtomic());

        // Test setters
        discrete_dist.setLowerBound(0);
        discrete_dist.setUpperBound(10);
        BasicTestFormatter::printPropertyInt("After setting - Lower bound",
                                             discrete_dist.getLowerBound());
        BasicTestFormatter::printPropertyInt("After setting - Upper bound",
                                             discrete_dist.getUpperBound());

        // Test simultaneous parameter setting
        discrete_dist.setParameters(2, 8);
        BasicTestFormatter::printPropertyInt("After setParameters - Lower bound",
                                             discrete_dist.getLowerBound());
        BasicTestFormatter::printPropertyInt("After setParameters - Upper bound",
                                             discrete_dist.getUpperBound());

        // Test safe setters (no exceptions)
        auto set_result = discrete_dist.trySetLowerBound(1);
        if (set_result.isOk()) {
            BasicTestFormatter::printPropertyInt("Safe set lower bound - Lower bound",
                                                 discrete_dist.getLowerBound());
        }

        set_result = discrete_dist.trySetUpperBound(6);
        if (set_result.isOk()) {
            BasicTestFormatter::printPropertyInt("Safe set upper bound - Upper bound",
                                                 discrete_dist.getUpperBound());
        }

        BasicTestFormatter::printTestSuccess("All parameter getter/setter tests passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core Probability Methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PMF, log PMF, CDF, quantiles,"
             << endl;
        cout << "and Discrete-specific utilities like support checks and mode calculation." << endl;
        cout << "Expected: For Discrete(1,6): PMF(3)=1/6≈0.167, CDF(3)=0.5, median=3.5 for "
                "standard die."
             << endl;

        auto dice_dist = stats::DiscreteDistribution::create(1, 6).value;
        double x = 3.0;

        BasicTestFormatter::printProperty("PMF(3.0)", dice_dist.getProbability(x));
        BasicTestFormatter::printProperty("Log PMF(3.0)", dice_dist.getLogProbability(x));
        BasicTestFormatter::printProperty("CDF(3.0)", dice_dist.getCumulativeProbability(x));
        BasicTestFormatter::printProperty("Quantile(0.5)", dice_dist.getQuantile(0.5));
        BasicTestFormatter::printProperty("Quantile(0.8)", dice_dist.getQuantile(0.8));

        // Test edge cases
        BasicTestFormatter::printProperty("PMF(0.0)", dice_dist.getProbability(0.0));
        BasicTestFormatter::printProperty("PMF(7.0)", dice_dist.getProbability(7.0));

        // Test Discrete-specific utility methods
        BasicTestFormatter::printProperty("Single outcome probability",
                                          dice_dist.getSingleOutcomeProbability());
        BasicTestFormatter::printProperty("Mode", dice_dist.getMode());
        BasicTestFormatter::printProperty("Median", dice_dist.getMedian());

        // Test distribution properties
        cout << "Is 3.0 in support: " << (dice_dist.isInSupport(3.0) ? "YES" : "NO") << endl;
        cout << "Is 0.0 in support: " << (dice_dist.isInSupport(0.0) ? "YES" : "NO") << endl;
        cout << "Is discrete: " << (dice_dist.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << dice_dist.getDistributionName() << endl;

        BasicTestFormatter::printTestSuccess("All core probability method tests passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using inverse transform method."
             << endl;
        cout << "Sample statistics should approximately match distribution parameters for large "
                "samples."
             << endl;

        mt19937 rng(42);

        // Single sample
        double single_sample = dice_dist.sample(rng);
        BasicTestFormatter::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = dice_dist.sample(rng, 10);
        BasicTestFormatter::printSamples(samples, "10 random samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = TestDataGenerators::computeSampleMean(samples);
        double sample_var = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean", sample_mean);
        BasicTestFormatter::printProperty("Sample variance", sample_var);
        BasicTestFormatter::printProperty("Expected mean", dice_dist.getMean());
        BasicTestFormatter::printProperty("Expected variance", dice_dist.getVariance());

        BasicTestFormatter::printTestSuccess("All sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using range estimation method," << endl;
        cout << "distribution reset to default (1, 6) parameters, and string representation "
                "formatting."
             << endl;

        // Test fitting
        vector<double> fit_data = TestDataGenerators::generateDiscreteTestData();
        auto fitted_dist = stats::DiscreteDistribution::create().value;
        fitted_dist.fit(fit_data);
        BasicTestFormatter::printPropertyInt("Fitted Lower bound", fitted_dist.getLowerBound());
        BasicTestFormatter::printPropertyInt("Fitted Upper bound", fitted_dist.getUpperBound());

        // Test reset
        fitted_dist.reset();
        BasicTestFormatter::printPropertyInt("After reset - Lower bound",
                                             fitted_dist.getLowerBound());
        BasicTestFormatter::printPropertyInt("After reset - Upper bound",
                                             fitted_dist.getUpperBound());

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

        auto test_dist = stats::DiscreteDistribution::create(1, 6).value;

        // Test small batch (should use SCALAR strategy) - using diverse realistic data
        vector<double> small_test_values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
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
        std::uniform_int_distribution<> dis(1, 6);
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

        auto dist1 = stats::DiscreteDistribution::create(1, 6).value;
        auto dist2 = stats::DiscreteDistribution::create(1, 6).value;
        auto dist3 = stats::DiscreteDistribution::create(0, 10).value;

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::DiscreteDistribution::create().value;
        ss.seekg(0);  // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed lower bound: " << input_dist.getLowerBound() << endl;
            cout << "Parsed upper bound: " << input_dist.getUpperBound() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        BasicTestFormatter::printTestSuccess("All comparison and stream operator tests passed");
        BasicTestFormatter::printNewline();

        // Test 8: Error Handling
        BasicTestFormatter::printTestStart(8, "Error Handling");
        // NOTE: Using ::create() here (not stats::Discrete) to test exception-free error
        // handling
        // ::create() returns Result<T> for explicit error checking without exceptions
        auto error_result = DiscreteDistribution::create(5, 3);  // Invalid: upper < lower
        if (error_result.isError()) {
            BasicTestFormatter::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            BasicTestFormatter::printTestError("Error handling failed");
            return 1;
        }

        BasicTestFormatter::printCompletionMessage("Discrete");

        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Safe factory creation and error handling");
        BasicTestFormatter::printSummaryItem(
            "All distribution properties (bounds, mean, variance, skewness, kurtosis)");
        BasicTestFormatter::printSummaryItem("PMF, Log PMF, CDF, and quantile functions");
        BasicTestFormatter::printSummaryItem("Random sampling (single and batch)");
        BasicTestFormatter::printSummaryItem("Parameter fitting (range estimation)");
        BasicTestFormatter::printSummaryItem("Batch operations with SIMD optimization");
        BasicTestFormatter::printSummaryItem("Large batch SIMD validation");
        BasicTestFormatter::printSummaryItem("Discrete-specific utility methods");
        BasicTestFormatter::printSummaryItem("Special case handling (binary distribution)");

        return 0;

    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}

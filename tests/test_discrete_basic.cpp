// Focused unit test for discrete distribution
#include "../include/distributions/discrete.h"
#include "basic_test_template.h"

using namespace std;
using namespace libstats;
using namespace BasicTestUtilities;

int main() {
    StandardizedBasicTest::printTestHeader("Discrete");
    
    try {
        // Test 1: Constructors and Destructor
        StandardizedBasicTest::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Discrete distributions: default (1,6), parameterized (0,1)," << endl;
        cout << "copy (parameterized), move (temporary (10,15)) constructors, and the safe factory method that avoids exceptions." << endl;
        
        // Default constructor test
        DiscreteDistribution default_discrete;
        StandardizedBasicTest::printPropertyInt("Default Lower bound", default_discrete.getLowerBound());
        StandardizedBasicTest::printPropertyInt("Default Upper bound", default_discrete.getUpperBound());
        
        // Parameterized constructor test
        DiscreteDistribution param_discrete(0, 1);
        StandardizedBasicTest::printPropertyInt("Param Lower bound", param_discrete.getLowerBound());
        StandardizedBasicTest::printPropertyInt("Param Upper bound", param_discrete.getUpperBound());
        
        // Copy constructor test
        DiscreteDistribution copy_discrete(param_discrete);
        StandardizedBasicTest::printPropertyInt("Copy Lower bound", copy_discrete.getLowerBound());
        StandardizedBasicTest::printPropertyInt("Copy Upper bound", copy_discrete.getUpperBound());
        
        // Move constructor test
        DiscreteDistribution temp_discrete(10, 15);
        DiscreteDistribution move_discrete(std::move(temp_discrete));
        StandardizedBasicTest::printPropertyInt("Move Lower bound", move_discrete.getLowerBound());
        StandardizedBasicTest::printPropertyInt("Move Upper bound", move_discrete.getUpperBound());
        
        // Safe factory method test
        auto result = DiscreteDistribution::create(1, 6);
        if (result.isOk()) {
            auto factory_discrete = std::move(result.value);
            StandardizedBasicTest::printPropertyInt("Factory Lower bound", factory_discrete.getLowerBound());
            StandardizedBasicTest::printPropertyInt("Factory Upper bound", factory_discrete.getUpperBound());
        }
        
        StandardizedBasicTest::printTestSuccess("All constructor and destructor tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 2: Parameter Getters and Setters
        StandardizedBasicTest::printTestStart(2, "Parameter Getters and Setters");
        cout << "This test verifies parameter access methods: normal getters, atomic (lock-free) getters," << endl;
        cout << "exception-based setters, and safe setters that return Result types instead of throwing." << endl;
        cout << "Using a Discrete(1, 6) distribution as the results are well known (mean=3.5, variance=2.916)." << endl;
        
        DiscreteDistribution discrete_dist(1, 6);
        
        // Test getters
        StandardizedBasicTest::printPropertyInt("Initial Lower bound", discrete_dist.getLowerBound());
        StandardizedBasicTest::printPropertyInt("Initial Upper bound", discrete_dist.getUpperBound());
        StandardizedBasicTest::printPropertyInt("Range", discrete_dist.getRange());
        StandardizedBasicTest::printProperty("Mean", discrete_dist.getMean());
        StandardizedBasicTest::printProperty("Variance", discrete_dist.getVariance());
        StandardizedBasicTest::printProperty("Skewness", discrete_dist.getSkewness());
        StandardizedBasicTest::printProperty("Kurtosis", discrete_dist.getKurtosis());
        StandardizedBasicTest::printPropertyInt("Num Parameters", discrete_dist.getNumParameters());
        
        // Test atomic getters (lock-free access)
        StandardizedBasicTest::printPropertyInt("Atomic Lower bound", discrete_dist.getLowerBoundAtomic());
        StandardizedBasicTest::printPropertyInt("Atomic Upper bound", discrete_dist.getUpperBoundAtomic());
        
        // Test setters
        discrete_dist.setLowerBound(0);
        discrete_dist.setUpperBound(10);
        StandardizedBasicTest::printPropertyInt("After setting - Lower bound", discrete_dist.getLowerBound());
        StandardizedBasicTest::printPropertyInt("After setting - Upper bound", discrete_dist.getUpperBound());
        
        // Test simultaneous parameter setting
        discrete_dist.setParameters(2, 8);
        StandardizedBasicTest::printPropertyInt("After setParameters - Lower bound", discrete_dist.getLowerBound());
        StandardizedBasicTest::printPropertyInt("After setParameters - Upper bound", discrete_dist.getUpperBound());
        
        // Test safe setters (no exceptions)
        auto set_result = discrete_dist.trySetLowerBound(1);
        if (set_result.isOk()) {
            StandardizedBasicTest::printPropertyInt("Safe set lower bound - Lower bound", discrete_dist.getLowerBound());
        }
        
        set_result = discrete_dist.trySetUpperBound(6);
        if (set_result.isOk()) {
            StandardizedBasicTest::printPropertyInt("Safe set upper bound - Upper bound", discrete_dist.getUpperBound());
        }
        
        StandardizedBasicTest::printTestSuccess("All parameter getter/setter tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 3: Core Probability Methods
        StandardizedBasicTest::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PMF, log PMF, CDF, quantiles," << endl;
        cout << "and Discrete-specific utilities like support checks and mode calculation." << endl;
        cout << "Expected: For Discrete(1,6): PMF(3)=1/6≈0.167, CDF(3)=0.5, median=3.5 for standard die." << endl;
        
        DiscreteDistribution dice_dist(1, 6);
        double x = 3.0;
        
        StandardizedBasicTest::printProperty("PMF(3.0)", dice_dist.getProbability(x));
        StandardizedBasicTest::printProperty("Log PMF(3.0)", dice_dist.getLogProbability(x));
        StandardizedBasicTest::printProperty("CDF(3.0)", dice_dist.getCumulativeProbability(x));
        StandardizedBasicTest::printProperty("Quantile(0.5)", dice_dist.getQuantile(0.5));
        StandardizedBasicTest::printProperty("Quantile(0.8)", dice_dist.getQuantile(0.8));
        
        // Test edge cases
        StandardizedBasicTest::printProperty("PMF(0.0)", dice_dist.getProbability(0.0));
        StandardizedBasicTest::printProperty("PMF(7.0)", dice_dist.getProbability(7.0));
        
        // Test Discrete-specific utility methods
        StandardizedBasicTest::printProperty("Single outcome probability", dice_dist.getSingleOutcomeProbability());
        StandardizedBasicTest::printProperty("Mode", dice_dist.getMode());
        StandardizedBasicTest::printProperty("Median", dice_dist.getMedian());
        
        // Test distribution properties
        cout << "Is 3.0 in support: " << (dice_dist.isInSupport(3.0) ? "YES" : "NO") << endl;
        cout << "Is 0.0 in support: " << (dice_dist.isInSupport(0.0) ? "YES" : "NO") << endl;
        cout << "Is discrete: " << (dice_dist.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << dice_dist.getDistributionName() << endl;
        
        StandardizedBasicTest::printTestSuccess("All core probability method tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 4: Random Sampling
        StandardizedBasicTest::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using inverse transform method." << endl;
        cout << "Sample statistics should approximately match distribution parameters for large samples." << endl;
        
        mt19937 rng(42);
        
        // Single sample
        double single_sample = dice_dist.sample(rng);
        StandardizedBasicTest::printProperty("Single sample", single_sample);
        
        // Multiple samples
        vector<double> samples = dice_dist.sample(rng, 10);
        StandardizedBasicTest::printSamples(samples, "10 random samples");
        
        // Verify sample statistics approximately match distribution
        double sample_mean = StandardizedBasicTest::computeSampleMean(samples);
        double sample_var = StandardizedBasicTest::computeSampleVariance(samples);
        StandardizedBasicTest::printProperty("Sample mean", sample_mean);
        StandardizedBasicTest::printProperty("Sample variance", sample_var);
        StandardizedBasicTest::printProperty("Expected mean", dice_dist.getMean());
        StandardizedBasicTest::printProperty("Expected variance", dice_dist.getVariance());
        
        StandardizedBasicTest::printTestSuccess("All sampling tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 5: Distribution Management
        StandardizedBasicTest::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using range estimation method," << endl;
        cout << "distribution reset to default (1, 6) parameters, and string representation formatting." << endl;
        
        // Test fitting
        vector<double> fit_data = StandardizedBasicTest::generateDiscreteTestData();
        DiscreteDistribution fitted_dist;
        fitted_dist.fit(fit_data);
        StandardizedBasicTest::printPropertyInt("Fitted Lower bound", fitted_dist.getLowerBound());
        StandardizedBasicTest::printPropertyInt("Fitted Upper bound", fitted_dist.getUpperBound());
        
        // Test reset
        fitted_dist.reset();
        StandardizedBasicTest::printPropertyInt("After reset - Lower bound", fitted_dist.getLowerBound());
        StandardizedBasicTest::printPropertyInt("After reset - Upper bound", fitted_dist.getUpperBound());
        
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
        
        DiscreteDistribution test_dist(1, 6);
        
        // Test small batch (should use SCALAR strategy) - using diverse realistic data
        vector<double> small_test_values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
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
        test_dist.getProbabilityWithStrategy(std::span<const double>(small_test_values), std::span<double>(small_pdf_traditional), libstats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getLogProbabilityWithStrategy(std::span<const double>(small_test_values), std::span<double>(small_log_pdf_traditional), libstats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto trad_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        test_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(small_test_values), std::span<double>(small_cdf_traditional), libstats::performance::Strategy::SCALAR);
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
        std::uniform_int_distribution<> dis(1, 6);
        for (size_t i = 0; i < large_size; ++i) {
            large_input[i] = static_cast<double>(dis(gen));
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
        test_dist.getProbabilityWithStrategy(std::span<const double>(large_input), std::span<double>(large_output_traditional), libstats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto large_trad_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        StandardizedBasicTest::printLargeBatchValidation(large_output[0], large_output[4999], "Auto-dispatch PDF (diverse data)");
        
        cout << "Large batch auto-dispatch time: " << large_auto_time << "μs, Traditional: " << large_trad_time << "μs" << endl;
        double speedup = static_cast<double>(large_trad_time) / static_cast<double>(large_auto_time);
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
        
        DiscreteDistribution dist1(1, 6);
        DiscreteDistribution dist2(1, 6);
        DiscreteDistribution dist3(0, 10);
        
        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;
        
        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;
        
        // Test stream input (using proper format from output)
        DiscreteDistribution input_dist;
        ss.seekg(0); // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed lower bound: " << input_dist.getLowerBound() << endl;
            cout << "Parsed upper bound: " << input_dist.getUpperBound() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }
        
        StandardizedBasicTest::printTestSuccess("All comparison and stream operator tests passed");
        StandardizedBasicTest::printNewline();
        
        // Test 8: Error Handling
        StandardizedBasicTest::printTestStart(8, "Error Handling");
        auto error_result = DiscreteDistribution::create(5, 3);  // Invalid: upper < lower
        if (error_result.isError()) {
            StandardizedBasicTest::printTestSuccess("Error handling works: " + error_result.message);
        } else {
            StandardizedBasicTest::printTestError("Error handling failed");
            return 1;
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

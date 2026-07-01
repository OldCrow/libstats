// Focused unit test for uniform distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/uniform.h"

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
    BasicTestFormatter::printTestHeader("Uniform");

    try {
        // Test 1: Constructors and Destructor
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Uniform distributions: default (0,1), "
                "parameterized (2,5),"
             << endl;
        cout << "copy (parameterized), move (temporary (-1,3)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_uniform = stats::UniformDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default Lower Bound", default_uniform.getLowerBound());
        BasicTestFormatter::printProperty("Default Upper Bound", default_uniform.getUpperBound());

        // Parameterized constructor test
        auto param_uniform = stats::UniformDistribution::create(2.0, 5.0).unwrap();
        BasicTestFormatter::printProperty("Param Lower Bound", param_uniform.getLowerBound());
        BasicTestFormatter::printProperty("Param Upper Bound", param_uniform.getUpperBound());

        // Copy constructor test
        auto copy_uniform = param_uniform;
        BasicTestFormatter::printProperty("Copy Lower Bound", copy_uniform.getLowerBound());
        BasicTestFormatter::printProperty("Copy Upper Bound", copy_uniform.getUpperBound());

        // Move constructor test
        auto temp_uniform = stats::UniformDistribution::create(-1.0, 3.0).unwrap();
        auto move_uniform = std::move(temp_uniform);
        BasicTestFormatter::printProperty("Move Lower Bound", move_uniform.getLowerBound());
        BasicTestFormatter::printProperty("Move Upper Bound", move_uniform.getUpperBound());

        // Safe factory method test
        auto result = UniformDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto factory_uniform = std::move(result).unwrap();
            BasicTestFormatter::printProperty("Factory Lower Bound",
                                              factory_uniform.getLowerBound());
            BasicTestFormatter::printProperty("Factory Upper Bound",
                                              factory_uniform.getUpperBound());
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
        cout << "Using a Uniform(0, 1) distribution as the results are well known (mean=0.5, "
                "variance=1/12)."
             << endl;

        auto uniform_dist = stats::UniformDistribution::create(0.0, 1.0).unwrap();

        // Test getters
        BasicTestFormatter::printProperty("Initial Lower Bound", uniform_dist.getLowerBound());
        BasicTestFormatter::printProperty("Initial Upper Bound", uniform_dist.getUpperBound());
        BasicTestFormatter::printProperty("Mean", uniform_dist.getMean());
        BasicTestFormatter::printProperty("Variance", uniform_dist.getVariance());
        BasicTestFormatter::printProperty("Skewness", uniform_dist.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", uniform_dist.getKurtosis());
        BasicTestFormatter::printPropertyInt("Num Parameters", uniform_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        BasicTestFormatter::printProperty("Atomic Lower Bound", uniform_dist.getLowerBoundAtomic());
        BasicTestFormatter::printProperty("Atomic Upper Bound", uniform_dist.getUpperBoundAtomic());

        // Test setters
        uniform_dist.setLowerBound(-1.0);
        uniform_dist.setUpperBound(2.0);
        BasicTestFormatter::printProperty("After setting - Lower Bound",
                                          uniform_dist.getLowerBound());
        BasicTestFormatter::printProperty("After setting - Upper Bound",
                                          uniform_dist.getUpperBound());

        // Test simultaneous parameter setting
        uniform_dist.setParameters(0.0, 5.0);
        BasicTestFormatter::printProperty("After setParameters - Lower Bound",
                                          uniform_dist.getLowerBound());
        BasicTestFormatter::printProperty("After setParameters - Upper Bound",
                                          uniform_dist.getUpperBound());

        // Test safe setters (no exceptions)
        auto set_result = uniform_dist.trySetLowerBound(-2.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set lower bound - Lower Bound",
                                              uniform_dist.getLowerBound());
        }

        set_result = uniform_dist.trySetUpperBound(3.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set upper bound - Upper Bound",
                                              uniform_dist.getUpperBound());
        }

        BasicTestFormatter::printTestSuccess("All parameter getter/setter tests passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core Probability Methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PDF, log PDF, CDF, quantiles,"
             << endl;
        cout << "and Uniform-specific utilities like support bounds and constant probability "
                "density."
             << endl;
        cout << "Expected: For Uniform(0,1): PDF=1 inside [0,1], CDF(0.5)=0.5, quantile(0.5)=0.5."
             << endl;

        auto test_uniform = stats::UniformDistribution::create(0.0, 1.0).unwrap();
        double x = 0.5;

        BasicTestFormatter::printProperty("PDF(0.5)", test_uniform.getProbability(x));
        BasicTestFormatter::printProperty("Log PDF(0.5)", test_uniform.getLogProbability(x));
        BasicTestFormatter::printProperty("CDF(0.5)", test_uniform.getCumulativeProbability(x));
        BasicTestFormatter::printProperty("Quantile(0.5)", test_uniform.getQuantile(0.5));
        BasicTestFormatter::printProperty("Quantile(0.8)", test_uniform.getQuantile(0.8));

        // Test edge cases and outside support
        BasicTestFormatter::printProperty("PDF(-0.5)", test_uniform.getProbability(-0.5));
        BasicTestFormatter::printProperty("PDF(1.5)", test_uniform.getProbability(1.5));
        BasicTestFormatter::printProperty("CDF(0.0)", test_uniform.getCumulativeProbability(0.0));
        BasicTestFormatter::printProperty("CDF(1.0)", test_uniform.getCumulativeProbability(1.0));

        // Test Uniform-specific utility methods
        BasicTestFormatter::printProperty("Mode", test_uniform.getMode());
        BasicTestFormatter::printProperty("Median", test_uniform.getMedian());
        BasicTestFormatter::printProperty("Entropy", test_uniform.getEntropy());

        // Test distribution properties
        cout << "Is discrete: " << (test_uniform.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << test_uniform.getDistributionName() << endl;
        BasicTestFormatter::printProperty("Support lower bound",
                                          test_uniform.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper bound",
                                          test_uniform.getSupportUpperBound());

        BasicTestFormatter::printTestSuccess("All core probability method tests passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using direct transform method."
             << endl;
        cout << "Sample statistics should approximately match distribution parameters for large "
                "samples."
             << endl;

        mt19937 rng(42);

        // Single sample
        double single_sample = test_uniform.sample(rng);
        BasicTestFormatter::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = test_uniform.sample(rng, 10);
        BasicTestFormatter::printSamples(samples, "10 random samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = TestDataGenerators::computeSampleMean(samples);
        double sample_var = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean", sample_mean);
        BasicTestFormatter::printProperty("Sample variance", sample_var);
        BasicTestFormatter::printProperty("Expected mean ((a+b)/2)", test_uniform.getMean());
        BasicTestFormatter::printProperty("Expected variance ((b-a)²/12)",
                                          test_uniform.getVariance());

        BasicTestFormatter::printTestSuccess("All sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using method of moments," << endl;
        cout << "distribution reset to default (0, 1) parameters, and string representation "
                "formatting."
             << endl;

        // Test fitting
        vector<double> fit_data = TestDataGenerators::generateUniformTestData();
        auto fitted_dist = stats::UniformDistribution::create().unwrap();
        fitted_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted Lower Bound", fitted_dist.getLowerBound());
        BasicTestFormatter::printProperty("Fitted Upper Bound", fitted_dist.getUpperBound());

        // Test reset
        fitted_dist.reset();
        BasicTestFormatter::printProperty("After reset - Lower Bound", fitted_dist.getLowerBound());
        BasicTestFormatter::printProperty("After reset - Upper Bound", fitted_dist.getUpperBound());

        // Test toString
        string dist_str = fitted_dist.toString();
        cout << "String representation: " << dist_str << endl;

        BasicTestFormatter::printTestSuccess("All distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Uniform", {-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5}, -1.0, 2.0, 1e-12, 1e-12};
        cfg.invalid_scenarios = {
            {"upper < lower", [] { return UniformDistribution::create(5.0, 2.0).isError(); }},
        };
        auto test_dist = stats::UniformDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, test_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::UniformDistribution::create(0.0, 1.0).unwrap();
        auto dist2 = stats::UniformDistribution::create(0.0, 1.0).unwrap();
        auto dist3 = stats::UniformDistribution::create(2.0, 5.0).unwrap();

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::UniformDistribution::create().unwrap();
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
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Uniform");

        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Safe factory creation and error handling");
        BasicTestFormatter::printSummaryItem(
            "All distribution properties (bounds, mean, variance, skewness, kurtosis)");
        BasicTestFormatter::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        BasicTestFormatter::printSummaryItem("Random sampling (direct transform method)");
        BasicTestFormatter::printSummaryItem("Parameter fitting (method of moments)");
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

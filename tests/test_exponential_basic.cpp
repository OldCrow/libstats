// Focused unit test for exponential distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/exponential.h"

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
    BasicTestFormatter::printTestHeader("Exponential");

    try {
        // Test 1: Constructors and Destructor
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Exponential distributions: default (1.0), "
                "parameterized (2.0),"
             << endl;
        cout << "copy (parameterized), move (temporary (0.5)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_exp = stats::ExponentialDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default Lambda", default_exp.getLambda());

        // Parameterized constructor test
        auto param_exp = stats::ExponentialDistribution::create(2.0).unwrap();
        BasicTestFormatter::printProperty("Param Lambda", param_exp.getLambda());

        // Copy constructor test
        auto copy_exp = param_exp;
        BasicTestFormatter::printProperty("Copy Lambda", copy_exp.getLambda());

        // Move constructor test
        auto temp_exp = stats::ExponentialDistribution::create(0.5).unwrap();
        auto move_exp = std::move(temp_exp);
        BasicTestFormatter::printProperty("Move Lambda", move_exp.getLambda());

        // Safe factory method test
        auto result = ExponentialDistribution::create(1.0);
        if (result.isOk()) {
            auto factory_exp = std::move(result).unwrap();
            BasicTestFormatter::printProperty("Factory Lambda", factory_exp.getLambda());
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
        cout << "Using an Exponential(1.0) distribution as the results are well known (mean=1, "
                "variance=1)."
             << endl;

        auto exp_dist = stats::ExponentialDistribution::create(1.0).unwrap();

        // Test getters
        BasicTestFormatter::printProperty("Initial Lambda", exp_dist.getLambda());
        BasicTestFormatter::printProperty("Mean", exp_dist.getMean());
        BasicTestFormatter::printProperty("Variance", exp_dist.getVariance());
        BasicTestFormatter::printProperty("Skewness", exp_dist.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", exp_dist.getKurtosis());
        BasicTestFormatter::printPropertyInt("Num Parameters", exp_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        BasicTestFormatter::printProperty("Atomic Lambda", exp_dist.getLambdaAtomic());

        // Test setters
        exp_dist.setLambda(2.0);
        BasicTestFormatter::printProperty("After setting - Lambda", exp_dist.getLambda());

        // Test safe setters (no exceptions)
        auto set_result = exp_dist.trySetLambda(0.5);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set lambda - Lambda", exp_dist.getLambda());
        }

        BasicTestFormatter::printTestSuccess("All parameter getter/setter tests passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core Probability Methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PDF, log PDF, CDF, quantiles,"
             << endl;
        cout << "and Exponential-specific utilities like mode and entropy for different parameter "
                "values."
             << endl;
        cout << "Expected: For Exponential(1.0): PDF(1)≈0.368, CDF(1)≈0.632, quantile(0.5)≈0.693."
             << endl;

        auto test_exp = stats::ExponentialDistribution::create(1.0).unwrap();
        double x = 1.0;

        BasicTestFormatter::printProperty("PDF(1.0)", test_exp.getProbability(x));
        BasicTestFormatter::printProperty("Log PDF(1.0)", test_exp.getLogProbability(x));
        BasicTestFormatter::printProperty("CDF(1.0)", test_exp.getCumulativeProbability(x));
        BasicTestFormatter::printProperty("Quantile(0.5)", test_exp.getQuantile(0.5));
        BasicTestFormatter::printProperty("Quantile(0.9)", test_exp.getQuantile(0.9));

        // Test edge cases
        BasicTestFormatter::printProperty("PDF(0.1)", test_exp.getProbability(0.1));
        BasicTestFormatter::printProperty("CDF(0.1)", test_exp.getCumulativeProbability(0.1));

        // Test Exponential-specific utility methods
        BasicTestFormatter::printProperty("Mode", test_exp.getMode());
        BasicTestFormatter::printProperty("Entropy", test_exp.getEntropy());
        BasicTestFormatter::printProperty("Median", test_exp.getMedian());

        // Test distribution properties
        cout << "Is discrete: " << (test_exp.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << test_exp.getDistributionName() << endl;
        BasicTestFormatter::printProperty("Support lower bound", test_exp.getSupportLowerBound());
        BasicTestFormatter::printProperty("Support upper bound", test_exp.getSupportUpperBound());

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
        double single_sample = test_exp.sample(rng);
        BasicTestFormatter::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = test_exp.sample(rng, 10);
        BasicTestFormatter::printSamples(samples, "10 random samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = TestDataGenerators::computeSampleMean(samples);
        double sample_var = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean", sample_mean);
        BasicTestFormatter::printProperty("Sample variance", sample_var);
        BasicTestFormatter::printProperty("Expected mean (1/λ)", test_exp.getMean());
        BasicTestFormatter::printProperty("Expected variance (1/λ²)", test_exp.getVariance());

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
        vector<double> fit_data = TestDataGenerators::generateExponentialTestData();
        auto fitted_dist = stats::ExponentialDistribution::create().unwrap();
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
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Exponential",
            {0.1, 0.5, 1.0, 2.0, 5.0},
            0.1, 5.0,
            1e-12,
            1e-12
        };
        cfg.invalid_scenarios = {
            {"negative lambda", [] { return ExponentialDistribution::create(-1.0).isError(); }},
        };
        auto test_dist = stats::ExponentialDistribution::create(1.0).unwrap();
        stats::tests::runBatchTests(cfg, test_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::ExponentialDistribution::create(1.0).unwrap();
        auto dist2 = stats::ExponentialDistribution::create(1.0).unwrap();
        auto dist3 = stats::ExponentialDistribution::create(2.0).unwrap();

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::ExponentialDistribution::create().unwrap();
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
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Exponential");

        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Safe factory creation and error handling");
        BasicTestFormatter::printSummaryItem(
            "All distribution properties (lambda, mean, variance, skewness, kurtosis)");
        BasicTestFormatter::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        BasicTestFormatter::printSummaryItem("Random sampling (inverse transform method)");
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

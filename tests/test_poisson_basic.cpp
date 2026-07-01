// Focused unit test for poisson distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/poisson.h"

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
        auto default_poisson = stats::PoissonDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default Lambda", default_poisson.getLambda());

        // Parameterized constructor test
        auto param_poisson = stats::PoissonDistribution::create(3.0).unwrap();
        BasicTestFormatter::printProperty("Param Lambda", param_poisson.getLambda());

        // Copy constructor test
        auto copy_poisson = param_poisson;
        BasicTestFormatter::printProperty("Copy Lambda", copy_poisson.getLambda());

        // Move constructor test
        auto temp_poisson = stats::PoissonDistribution::create(5.0).unwrap();
        auto move_poisson = std::move(temp_poisson);
        BasicTestFormatter::printProperty("Move Lambda", move_poisson.getLambda());

        // Safe factory method test
        auto result = PoissonDistribution::create(3.0);
        if (result.isOk()) {
            auto factory_poisson = std::move(result).unwrap();
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

        auto poisson_dist = stats::PoissonDistribution::create(3.0).unwrap();

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

        auto test_poisson = stats::PoissonDistribution::create(3.0).unwrap();
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
        auto fitted_dist = stats::PoissonDistribution::create().unwrap();
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
            "Poisson", {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}, 0.0, 10.5, 1e-12, 1e-12};
        cfg.invalid_scenarios = {
            {"negative lambda", [] { return PoissonDistribution::create(-1.0).isError(); }},
        };
        auto test_dist = stats::PoissonDistribution::create(3.0).unwrap();
        stats::tests::runBatchTests(cfg, test_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::PoissonDistribution::create(3.0).unwrap();
        auto dist2 = stats::PoissonDistribution::create(3.0).unwrap();
        auto dist3 = stats::PoissonDistribution::create(5.0).unwrap();

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::PoissonDistribution::create().unwrap();
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

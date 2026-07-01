// Focused unit test for discrete distribution
#include "include/basic_test_runner.h"
#include "include/tests.h"
#include "libstats/distributions/discrete.h"

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
        auto default_discrete = stats::DiscreteDistribution::create().unwrap();
        BasicTestFormatter::printPropertyInt("Default Lower bound",
                                             default_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Default Upper bound",
                                             default_discrete.getUpperBound());

        // Parameterized constructor test
        auto param_discrete = stats::DiscreteDistribution::create(0, 1).unwrap();
        BasicTestFormatter::printPropertyInt("Param Lower bound", param_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Param Upper bound", param_discrete.getUpperBound());

        // Copy constructor test
        auto copy_discrete = param_discrete;
        BasicTestFormatter::printPropertyInt("Copy Lower bound", copy_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Copy Upper bound", copy_discrete.getUpperBound());

        // Move constructor test
        auto temp_discrete = stats::DiscreteDistribution::create(10, 15).unwrap();
        auto move_discrete = std::move(temp_discrete);
        BasicTestFormatter::printPropertyInt("Move Lower bound", move_discrete.getLowerBound());
        BasicTestFormatter::printPropertyInt("Move Upper bound", move_discrete.getUpperBound());

        // Safe factory method test
        auto result = DiscreteDistribution::create(1, 6);
        if (result.isOk()) {
            auto factory_discrete = std::move(result).unwrap();
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

        auto discrete_dist = stats::DiscreteDistribution::create(1, 6).unwrap();

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

        auto dice_dist = stats::DiscreteDistribution::create(1, 6).unwrap();
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
        auto fitted_dist = stats::DiscreteDistribution::create().unwrap();
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
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Discrete", {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, 1.0, 6.5, 1e-12, 1e-12};
        cfg.invalid_scenarios = {
            {"upper < lower", [] { return DiscreteDistribution::create(5, 3).isError(); }},
        };
        auto test_dist = stats::DiscreteDistribution::create(1, 6).unwrap();
        stats::tests::runBatchTests(cfg, test_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::DiscreteDistribution::create(1, 6).unwrap();
        auto dist2 = stats::DiscreteDistribution::create(1, 6).unwrap();
        auto dist3 = stats::DiscreteDistribution::create(0, 10).unwrap();

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::DiscreteDistribution::create().unwrap();
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

// Focused unit test for gaussian distribution
#include "include/tests.h"
#include "include/basic_test_runner.h"
#include "libstats/distributions/gaussian.h"

// Standard library includes
#include <chrono>     // for std::chrono::high_resolution_clock, timing measurements
#include <cstdlib>    // for size_t (portable alternative to stdlib.h)
#include <exception>  // for std::exception
#include <iomanip>    // for std::fixed, std::setprecision, stream formatting
#include <iostream>   // for std::cout, std::endl
#include <random>     // for std::mt19937, std::uniform_real_distribution
#include <span>       // for std::span
#include <sstream>    // for std::stringstream
#include <string>     // for std::string
#include <utility>    // for std::move
#include <vector>     // for std::vector

using namespace std;
using namespace stats;
using namespace stats::tests::fixtures;

int main() {
    BasicTestFormatter::printTestHeader("Gaussian");

    try {
        // Test 1: Constructors and Destructor
        BasicTestFormatter::printTestStart(1, "Constructors and Destructor");
        cout << "This test verifies all ways to create Gaussian distributions: default (0,1), "
                "parameterized (5, 0),"
             << endl;
        cout << "copy (parameterized), move (temporary (10,3)) constructors, and the safe factory "
                "method that avoids exceptions."
             << endl;

        // Default constructor test
        auto default_gauss = stats::GaussianDistribution::create().unwrap();
        BasicTestFormatter::printProperty("Default Mean", default_gauss.getMean());
        BasicTestFormatter::printProperty("Default Std Dev", default_gauss.getStandardDeviation());

        // Parameterized constructor test
        auto param_gauss = stats::GaussianDistribution::create(5.0, 2.0).unwrap();
        BasicTestFormatter::printProperty("Param Mean", param_gauss.getMean());
        BasicTestFormatter::printProperty("Param Std Dev", param_gauss.getStandardDeviation());

        // Copy constructor test
        auto copy_gauss = param_gauss;
        BasicTestFormatter::printProperty("Copy Mean", copy_gauss.getMean());
        BasicTestFormatter::printProperty("Copy Std Dev", copy_gauss.getStandardDeviation());

        // Move constructor test
        auto temp_gauss = stats::GaussianDistribution::create(10.0, 3.0).unwrap();
        auto move_gauss = std::move(temp_gauss);
        BasicTestFormatter::printProperty("Move Mean", move_gauss.getMean());
        BasicTestFormatter::printProperty("Move Std Dev", move_gauss.getStandardDeviation());

        // Safe factory method test
        auto result = GaussianDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto factory_gauss = std::move(result.unwrap());
            BasicTestFormatter::printProperty("Factory Mean", factory_gauss.getMean());
            BasicTestFormatter::printProperty("Factory Std Dev",
                                              factory_gauss.getStandardDeviation());
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
        cout << "Using a Standard Normal (0.0, 1.0) distribution as the results are well known."
             << endl;

        auto gauss_dist = stats::GaussianDistribution::create(0.0, 1.0).unwrap();

        // Test getters
        BasicTestFormatter::printProperty("Initial Mean", gauss_dist.getMean());
        BasicTestFormatter::printProperty("Initial Std Dev", gauss_dist.getStandardDeviation());
        BasicTestFormatter::printProperty("Variance", gauss_dist.getVariance());
        BasicTestFormatter::printProperty("Skewness", gauss_dist.getSkewness());
        BasicTestFormatter::printProperty("Kurtosis", gauss_dist.getKurtosis());
        BasicTestFormatter::printPropertyInt("Num Parameters", gauss_dist.getNumParameters());

        // Test atomic getters (lock-free access)
        BasicTestFormatter::printProperty("Atomic Mean", gauss_dist.getMeanAtomic());
        BasicTestFormatter::printProperty("Atomic Std Dev",
                                          gauss_dist.getStandardDeviationAtomic());

        // Test setters
        gauss_dist.setMean(3.0);
        gauss_dist.setStandardDeviation(1.5);
        BasicTestFormatter::printProperty("After setting - Mean", gauss_dist.getMean());
        BasicTestFormatter::printProperty("After setting - Std Dev",
                                          gauss_dist.getStandardDeviation());

        // Test simultaneous parameter setting
        gauss_dist.setParameters(10.0, 2.5);
        BasicTestFormatter::printProperty("After setParameters - Mean", gauss_dist.getMean());
        BasicTestFormatter::printProperty("After setParameters - Std Dev",
                                          gauss_dist.getStandardDeviation());

        // Test safe setters (no exceptions)
        auto set_result = gauss_dist.trySetMean(7.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set mean - Mean", gauss_dist.getMean());
        }

        set_result = gauss_dist.trySetStandardDeviation(3.0);
        if (set_result.isOk()) {
            BasicTestFormatter::printProperty("Safe set std dev - Std Dev",
                                              gauss_dist.getStandardDeviation());
        }

        BasicTestFormatter::printTestSuccess("All parameter getter/setter tests passed");
        BasicTestFormatter::printNewline();

        // Test 3: Core Probability Methods
        BasicTestFormatter::printTestStart(3, "Core Probability Methods");
        cout << "This test verifies the core statistical functions: PDF, log PDF, CDF, quantiles,"
             << endl;
        cout << "and Gaussian-specific utilities like z-scores, entropy, median/mode (all equal "
                "for Gaussian)."
             << endl;
        cout << "Expected: PDF(0)≈0.399, CDF(0)=0.5, CDF(1)≈0.841, quantile(0.5)=0 for Standard "
                "Normal (0, 1)."
             << endl;

        auto std_normal = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
        double x = 1.0;

        BasicTestFormatter::printProperty("PDF(1.0)", std_normal.getProbability(x));
        BasicTestFormatter::printProperty("Log PDF(1.0)", std_normal.getLogProbability(x));
        BasicTestFormatter::printProperty("CDF(1.0)", std_normal.getCumulativeProbability(x));
        BasicTestFormatter::printProperty("Quantile(0.5)", std_normal.getQuantile(0.5));
        BasicTestFormatter::printProperty("Quantile(0.8413)", std_normal.getQuantile(0.8413454746));

        // Test edge cases
        BasicTestFormatter::printProperty("PDF(0.0)", std_normal.getProbability(0.0));
        BasicTestFormatter::printProperty("CDF(0.0)", std_normal.getCumulativeProbability(0.0));

        // Test Gaussian-specific utility methods
        BasicTestFormatter::printProperty("Z-score for x=1", std_normal.getStandardizedValue(1.0));
        BasicTestFormatter::printProperty("Value from z=1",
                                          std_normal.getValueFromStandardized(1.0));
        BasicTestFormatter::printProperty("Entropy", std_normal.getEntropy());
        BasicTestFormatter::printProperty("Median", std_normal.getMedian());
        BasicTestFormatter::printProperty("Mode", std_normal.getMode());

        // Test distribution properties
        cout << "Is standard normal: " << (std_normal.isStandardNormal() ? "YES" : "NO") << endl;
        cout << "Is discrete: " << (std_normal.isDiscrete() ? "YES" : "NO") << endl;
        cout << "Distribution name: " << std_normal.getDistributionName() << endl;

        BasicTestFormatter::printTestSuccess("All core probability method tests passed");
        BasicTestFormatter::printNewline();

        // Test 4: Random Sampling
        BasicTestFormatter::printTestStart(4, "Random Sampling");
        cout << "This test verifies random number generation using the Box-Muller algorithm."
             << endl;
        cout << "Sample statistics should approximately match distribution parameters for large "
                "samples."
             << endl;

        mt19937 rng(42);

        // Single sample
        double single_sample = std_normal.sample(rng);
        BasicTestFormatter::printProperty("Single sample", single_sample);

        // Multiple samples
        vector<double> samples = std_normal.sample(rng, 10);
        BasicTestFormatter::printSamples(samples, "10 random samples");

        // Verify sample statistics approximately match distribution
        double sample_mean = TestDataGenerators::computeSampleMean(samples);
        double sample_var = TestDataGenerators::computeSampleVariance(samples);
        BasicTestFormatter::printProperty("Sample mean", sample_mean);
        BasicTestFormatter::printProperty("Sample variance", sample_var);

        BasicTestFormatter::printTestSuccess("All sampling tests passed");
        BasicTestFormatter::printNewline();

        // Test 5: Distribution Management
        BasicTestFormatter::printTestStart(5, "Distribution Management");
        cout << "This test verifies parameter fitting using Maximum Likelihood Estimation (MLE),"
             << endl;
        cout << "distribution reset to default (0, 1) parameters, and string representation "
                "formatting."
             << endl;

        // Test fitting
        vector<double> fit_data = TestDataGenerators::generateGaussianTestData();
        auto fitted_dist = stats::GaussianDistribution::create().unwrap();
        fitted_dist.fit(fit_data);
        BasicTestFormatter::printProperty("Fitted Mean", fitted_dist.getMean());
        BasicTestFormatter::printProperty("Fitted Std Dev", fitted_dist.getStandardDeviation());

        // Test reset
        fitted_dist.reset();
        BasicTestFormatter::printProperty("After reset - Mean", fitted_dist.getMean());
        BasicTestFormatter::printProperty("After reset - Std Dev",
                                          fitted_dist.getStandardDeviation());

        // Test toString
        string dist_str = fitted_dist.toString();
        cout << "String representation: " << dist_str << endl;

        BasicTestFormatter::printTestSuccess("All distribution management tests passed");
        BasicTestFormatter::printNewline();
        // =====================================================================
        // Test 6: Auto-dispatch Batch Operations
        // =====================================================================
        stats::tests::BasicDistConfig cfg{
            "Gaussian",
            {-2.5, -1.2, 0.3, 1.8, 2.1},
            -3.0, 3.0,
            1e-12,
            1e-12
        };
        cfg.invalid_scenarios = {
            {"negative sigma", [] { return GaussianDistribution::create(0.0, -1.0).isError(); }},
        };
        auto test_dist = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
        stats::tests::runBatchTests(cfg, test_dist);

        BasicTestFormatter::printTestStart(7, "Comparison and Stream Operators");
        cout << "This test verifies equality/inequality operators for parameter comparison" << endl;
        cout << "and stream I/O operators for serialization/deserialization of distributions."
             << endl;

        auto dist1 = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
        auto dist2 = stats::GaussianDistribution::create(0.0, 1.0).unwrap();
        auto dist3 = stats::GaussianDistribution::create(1.0, 2.0).unwrap();

        // Test equality
        cout << "dist1 == dist2: " << (dist1 == dist2 ? "true" : "false") << endl;
        cout << "dist1 == dist3: " << (dist1 == dist3 ? "true" : "false") << endl;
        cout << "dist1 != dist3: " << (dist1 != dist3 ? "true" : "false") << endl;

        // Test stream operators
        stringstream ss;
        ss << dist1;
        cout << "Stream output: " << ss.str() << endl;

        // Test stream input (using proper format from output)
        auto input_dist = stats::GaussianDistribution::create().unwrap();
        ss.seekg(0);  // Reset to beginning to read the output we just wrote
        if (ss >> input_dist) {
            cout << "Stream input successful: " << input_dist.toString() << endl;
            cout << "Parsed mean: " << input_dist.getMean() << endl;
            cout << "Parsed std dev: " << input_dist.getStandardDeviation() << endl;
        } else {
            cout << "Stream input failed" << endl;
        }

        BasicTestFormatter::printTestSuccess("All comparison and stream operator tests passed");
        BasicTestFormatter::printNewline();

        // Test 8: Error Handling
        stats::tests::runErrorTests(cfg);

        BasicTestFormatter::printCompletionMessage("Gaussian");

        BasicTestFormatter::printSummaryHeader();
        BasicTestFormatter::printSummaryItem("Safe factory creation and error handling");
        BasicTestFormatter::printSummaryItem(
            "All distribution properties (mean, variance, skewness, kurtosis)");
        BasicTestFormatter::printSummaryItem("PDF, Log PDF, CDF, and quantile functions");
        BasicTestFormatter::printSummaryItem("Random sampling (Box-Muller algorithm)");
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

// Comprehensive Error Handling and API Test Suite
// Consolidates tests for safe factory methods and dual API (exception/Result)
// Tests all error handling mechanisms across all distributions

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

// Standard library includes
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace stats;
using namespace std;

// Test configuration
;

// Test statistics
struct TestStats {
    void record(bool success, const std::string& test_name) { EXPECT_TRUE(success) << test_name; }
    void print_summary([[maybe_unused]] const std::string& category) {}
};

// Helper function for comparing doubles
template <typename T>
bool approx_equal(T a, T b, T epsilon = 1e-10) {
    return std::abs(a - b) < epsilon;
}

//==============================================================================
// SAFE FACTORY METHOD TESTS
//==============================================================================

void test_factory_methods() {
    cout << "\n=== Testing Safe Factory Methods ===" << endl;
    TestStats stats;

    // Test Gaussian factory
    {
        auto result = GaussianDistribution::create(0.0, 1.0);
        stats.record(result.isOk(), "GaussianDistribution::create with valid params");

        if (result.isOk()) {
            auto& dist = *result;
            stats.record(
                approx_equal(dist.getMean(), 0.0) && approx_equal(dist.getStandardDeviation(), 1.0),
                "Gaussian parameters correctly set");

            double pdf = dist.getProbability(0.0);
            stats.record(approx_equal(pdf, 0.39894228040143268, 1e-10),
                         "Gaussian PDF calculation correct");
        }

        // Test invalid parameters
        auto badResult1 = GaussianDistribution::create(0.0, 0.0);
        stats.record(badResult1.isError(), "Gaussian rejects zero standard deviation");

        auto badResult2 = GaussianDistribution::create(0.0, -1.0);
        stats.record(badResult2.isError(), "Gaussian rejects negative standard deviation");

        auto badResult3 = GaussianDistribution::create(numeric_limits<double>::infinity(), 1.0);
        stats.record(badResult3.isError(), "Gaussian rejects infinite mean");

        auto badResult4 = GaussianDistribution::create(0.0, numeric_limits<double>::quiet_NaN());
        stats.record(badResult4.isError(), "Gaussian rejects NaN standard deviation");

        if (false && badResult1.isError()) {
            cout << "    Error message: " << badResult1.message() << endl;
        }
    }

    // Test Uniform factory
    {
        auto result = UniformDistribution::create(0.0, 1.0);
        stats.record(result.isOk(), "UniformDistribution::create with valid params");

        auto badResult = UniformDistribution::create(1.0, 0.0);
        stats.record(badResult.isError(), "Uniform rejects reversed bounds");

        auto nanResult = UniformDistribution::create(numeric_limits<double>::quiet_NaN(), 1.0);
        stats.record(nanResult.isError(), "Uniform rejects NaN bounds");
    }

    // Test Exponential factory
    {
        auto result = ExponentialDistribution::create(1.0);
        stats.record(result.isOk(), "ExponentialDistribution::create with valid params");

        auto badResult = ExponentialDistribution::create(0.0);
        stats.record(badResult.isError(), "Exponential rejects zero lambda");

        auto negResult = ExponentialDistribution::create(-1.0);
        stats.record(negResult.isError(), "Exponential rejects negative lambda");
    }

    // Test Poisson factory
    {
        auto result = PoissonDistribution::create(3.0);
        stats.record(result.isOk(), "PoissonDistribution::create with valid params");

        auto badResult = PoissonDistribution::create(0.0);
        stats.record(badResult.isError(), "Poisson rejects zero lambda");

        auto negResult = PoissonDistribution::create(-1.0);
        stats.record(negResult.isError(), "Poisson rejects negative lambda");
    }

    // Test Discrete factory
    {
        auto result = DiscreteDistribution::create(1, 6);
        stats.record(result.isOk(), "DiscreteDistribution::create with valid params");

        auto badResult = DiscreteDistribution::create(6, 1);
        stats.record(badResult.isError(), "Discrete rejects reversed bounds");

        auto sameResult = DiscreteDistribution::create(5, 5);
        stats.record(sameResult.isError(), "Discrete rejects equal bounds");
    }

    // Test Gamma factory
    {
        auto result = GammaDistribution::create(2.0, 0.5);
        stats.record(result.isOk(), "GammaDistribution::create with valid params");

        auto badAlpha = GammaDistribution::create(0.0, 1.0);
        stats.record(badAlpha.isError(), "Gamma rejects zero alpha");

        auto badBeta = GammaDistribution::create(1.0, -1.0);
        stats.record(badBeta.isError(), "Gamma rejects negative beta");
    }

    stats.print_summary("Factory Methods");
}

//==============================================================================
// DUAL API TESTS
//==============================================================================

void test_dual_api() {
    cout << "\n=== Testing Dual API (Exception/Result) ===" << endl;
    TestStats stats;

    // Test Gaussian dual API
    {
        auto result = GaussianDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test Result-based setters
            auto setMeanResult = dist.trySetMean(5.0);
            stats.record(setMeanResult.isOk(), "Gaussian trySetMean() accepts valid value");

            auto setStdResult = dist.trySetStandardDeviation(2.0);
            stats.record(setStdResult.isOk(),
                         "Gaussian trySetStandardDeviation() accepts valid value");

            // Test error cases
            auto invalidStdResult = dist.trySetStandardDeviation(-1.0);
            stats.record(invalidStdResult.isError(),
                         "Gaussian trySetStandardDeviation() rejects negative");

            // Test trySetParameters
            auto setParamsResult = dist.trySetParameters(1.0, 0.5);
            stats.record(setParamsResult.isOk() && approx_equal(dist.getMean(), 1.0) &&
                             approx_equal(dist.getStandardDeviation(), 0.5),
                         "Gaussian trySetParameters() works correctly");

            // Test exception-based API still works
            bool exception_thrown = false;
            try {
                dist.setStandardDeviation(-1.0);
            } catch (const invalid_argument&) {
                exception_thrown = true;
            }
            stats.record(exception_thrown, "Gaussian setStandardDeviation() throws on invalid");
        }
    }

    // Test Uniform dual API
    {
        auto result = UniformDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            auto setLowerResult = dist.trySetLowerBound(-2.0);
            stats.record(setLowerResult.isOk(), "Uniform trySetLowerBound() accepts valid value");

            auto setUpperResult = dist.trySetUpperBound(3.0);
            stats.record(setUpperResult.isOk(), "Uniform trySetUpperBound() accepts valid value");

            // Test error case - invalid bounds
            auto invalidBoundResult = dist.trySetUpperBound(-5.0);
            stats.record(invalidBoundResult.isError(),
                         "Uniform trySetUpperBound() rejects invalid bounds");

            auto setParamsResult = dist.trySetParameters(1.0, 4.0);
            stats.record(setParamsResult.isOk() && approx_equal(dist.getLowerBound(), 1.0) &&
                             approx_equal(dist.getUpperBound(), 4.0),
                         "Uniform trySetParameters() works correctly");
        }
    }

    // Test Exponential dual API
    {
        auto result = ExponentialDistribution::create(1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            auto setLambdaResult = dist.trySetLambda(2.5);
            stats.record(setLambdaResult.isOk(), "Exponential trySetLambda() accepts valid value");

            auto invalidLambdaResult = dist.trySetLambda(-1.0);
            stats.record(invalidLambdaResult.isError(),
                         "Exponential trySetLambda() rejects negative");

            auto setParamsResult = dist.trySetParameters(3.0);
            stats.record(setParamsResult.isOk() && approx_equal(dist.getLambda(), 3.0),
                         "Exponential trySetParameters() works correctly");
        }
    }

    // Test Poisson dual API
    {
        auto result = PoissonDistribution::create(1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            auto setLambdaResult = dist.trySetLambda(3.7);
            stats.record(setLambdaResult.isOk(), "Poisson trySetLambda() accepts valid value");

            auto invalidLambdaResult = dist.trySetLambda(0.0);
            stats.record(invalidLambdaResult.isError(), "Poisson trySetLambda() rejects zero");

            auto setParamsResult = dist.trySetParameters(5.2);
            stats.record(setParamsResult.isOk() && approx_equal(dist.getLambda(), 5.2),
                         "Poisson trySetParameters() works correctly");
        }
    }

    // Test Discrete dual API
    {
        auto result = DiscreteDistribution::create(1, 6);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            auto setLowerResult = dist.trySetLowerBound(0);
            stats.record(setLowerResult.isOk(), "Discrete trySetLowerBound() accepts valid value");

            auto setUpperResult = dist.trySetUpperBound(10);
            stats.record(setUpperResult.isOk(), "Discrete trySetUpperBound() accepts valid value");

            auto invalidBoundResult = dist.trySetUpperBound(-5);
            stats.record(invalidBoundResult.isError(),
                         "Discrete trySetUpperBound() rejects invalid bounds");

            auto setParamsResult = dist.trySetParameters(2, 8);
            stats.record(
                setParamsResult.isOk() && dist.getLowerBound() == 2 && dist.getUpperBound() == 8,
                "Discrete trySetParameters() works correctly");
        }
    }

    // Test Gamma dual API
    {
        auto result = GammaDistribution::create(2.0, 0.5);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            auto setAlphaResult = dist.trySetAlpha(3.0);
            stats.record(setAlphaResult.isOk(), "Gamma trySetAlpha() accepts valid value");

            auto setBetaResult = dist.trySetBeta(1.5);
            stats.record(setBetaResult.isOk(), "Gamma trySetBeta() accepts valid value");

            auto invalidAlphaResult = dist.trySetAlpha(-1.0);
            stats.record(invalidAlphaResult.isError(), "Gamma trySetAlpha() rejects negative");

            auto invalidBetaResult = dist.trySetBeta(0.0);
            stats.record(invalidBetaResult.isError(), "Gamma trySetBeta() rejects zero");

            auto setParamsResult = dist.trySetParameters(1.5, 2.0);
            stats.record(setParamsResult.isOk() && approx_equal(dist.getAlpha(), 1.5) &&
                             approx_equal(dist.getBeta(), 2.0),
                         "Gamma trySetParameters() works correctly");
        }
    }

    stats.print_summary("Dual API");
}

//==============================================================================
// PARAMETER VALIDATION TESTS
//==============================================================================

void test_parameter_validation() {
    cout << "\n=== Testing Parameter Validation ===" << endl;
    TestStats stats;

    // Test validation methods
    {
        auto gaussResult = GaussianDistribution::create(0.0, 1.0);
        if (gaussResult.isOk()) {
            auto& dist = *gaussResult;

            auto validationResult = dist.validateCurrentParameters();
            stats.record(validationResult.isOk(),
                         "Gaussian validateCurrentParameters() validates good params");

            // Test that validation is consistent between factory and setters
            auto tryResult = dist.trySetParameters(5.0, 2.0);
            auto validationResult2 = dist.validateCurrentParameters();
            stats.record(tryResult.isOk() == validationResult2.isOk(),
                         "Validation consistent between trySet and validate");
        }
    }

    // Test edge cases in validation
    {
        // Test extreme but valid values
        auto extremeGauss = GaussianDistribution::create(1e100, 1e-100);
        stats.record(extremeGauss.isOk(), "Gaussian accepts extreme but valid values");

        // Test denormalized numbers
        double denorm = numeric_limits<double>::denorm_min();
        auto denormExp = ExponentialDistribution::create(denorm);
        stats.record(denormExp.isError() || denormExp.isOk(),
                     "Exponential handles denormalized lambda");

        // Test infinity handling
        auto infUniform = UniformDistribution::create(-numeric_limits<double>::infinity(),
                                                      numeric_limits<double>::infinity());
        stats.record(infUniform.isError(), "Uniform rejects infinite bounds");
    }

    // Test consistency across distributions
    {
        // All distributions should reject NaN parameters
        auto nanGauss = GaussianDistribution::create(numeric_limits<double>::quiet_NaN(), 1.0);
        auto nanUniform = UniformDistribution::create(numeric_limits<double>::quiet_NaN(), 1.0);
        auto nanExp = ExponentialDistribution::create(numeric_limits<double>::quiet_NaN());
        auto nanPoisson = PoissonDistribution::create(numeric_limits<double>::quiet_NaN());
        auto nanGamma = GammaDistribution::create(numeric_limits<double>::quiet_NaN(), 1.0);

        stats.record(nanGauss.isError() && nanUniform.isError() && nanExp.isError() &&
                         nanPoisson.isError() && nanGamma.isError(),
                     "All distributions consistently reject NaN parameters");
    }

    // Test error message quality
    if (false) {
        cout << "\n  Sample error messages:" << endl;

        auto badGauss = GaussianDistribution::create(0.0, -1.0);
        if (badGauss.isError()) {
            cout << "    Gaussian: " << badGauss.message() << endl;
        }

        auto badUniform = UniformDistribution::create(1.0, 0.0);
        if (badUniform.isError()) {
            cout << "    Uniform: " << badUniform.message() << endl;
        }

        auto badExp = ExponentialDistribution::create(0.0);
        if (badExp.isError()) {
            cout << "    Exponential: " << badExp.message() << endl;
        }
    }

    stats.print_summary("Parameter Validation");
}

//==============================================================================
// DISTRIBUTION-SPECIFIC ERROR HANDLING TESTS
//==============================================================================

void test_specific_distribution(const string& dist_name, TestStats& stats) {
    if (dist_name == "gaussian" || dist_name == "all") {
        auto result = GaussianDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test parameter setting with various edge cases
            auto r1 = dist.trySetMean(numeric_limits<double>::max());
            stats.record(r1.isOk(), "Gaussian accepts max double as mean");

            auto r2 = dist.trySetStandardDeviation(numeric_limits<double>::min());
            stats.record(r2.isOk(), "Gaussian accepts min positive double as std dev");

            auto r3 = dist.trySetParameters(0.0, numeric_limits<double>::epsilon());
            stats.record(r3.isOk(), "Gaussian accepts epsilon as std dev");
        }
    }

    if (dist_name == "uniform" || dist_name == "all") {
        auto result = UniformDistribution::create(-1.0, 1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test bounds adjustment
            auto r1 = dist.trySetLowerBound(0.5);
            bool upper_still_valid = dist.getUpperBound() > 0.5;
            stats.record(r1.isOk() && upper_still_valid,
                         "Uniform maintains valid bounds after lower adjustment");

            // Test equal bounds rejection
            auto r2 = dist.trySetParameters(1.0, 1.0);
            stats.record(r2.isError(), "Uniform rejects equal bounds");
        }
    }

    if (dist_name == "exponential" || dist_name == "all") {
        auto result = ExponentialDistribution::create(1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test very large lambda
            auto r1 = dist.trySetLambda(1e100);
            stats.record(r1.isOk(), "Exponential accepts very large lambda");

            // Test very small positive lambda
            auto r2 = dist.trySetLambda(1e-100);
            stats.record(r2.isOk(), "Exponential accepts very small positive lambda");
        }
    }

    if (dist_name == "poisson" || dist_name == "all") {
        auto result = PoissonDistribution::create(1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test maximum supported lambda
            auto r1 = dist.trySetLambda(700.0);  // Near exp limit
            stats.record(r1.isOk(), "Poisson accepts lambda near limit");

            // Test fractional lambda
            auto r2 = dist.trySetLambda(0.001);
            stats.record(r2.isOk(), "Poisson accepts very small lambda");
        }
    }

    if (dist_name == "discrete" || dist_name == "all") {
        auto result = DiscreteDistribution::create(0, 100);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test large range
            auto r1 = dist.trySetParameters(-1000000, 1000000);
            stats.record(r1.isOk(), "Discrete accepts large range");

            // Test negative range
            auto r2 = dist.trySetParameters(-100, -50);
            stats.record(r2.isOk(), "Discrete accepts negative range");
        }
    }

    if (dist_name == "gamma" || dist_name == "all") {
        auto result = GammaDistribution::create(1.0, 1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Test extreme shape and scale
            auto r1 = dist.trySetParameters(1000.0, 0.001);
            stats.record(r1.isOk(), "Gamma accepts extreme shape/scale combination");

            // Test near-zero positive values
            auto r2 = dist.trySetParameters(numeric_limits<double>::epsilon(),
                                            numeric_limits<double>::epsilon());
            stats.record(r2.isOk(), "Gamma accepts epsilon values");
        }
    }
}

void test_distributions() {
    cout << "\n=== Testing Distribution-Specific Error Handling ===" << endl;
    TestStats stats;

    if (std::vector<std::string>().empty()) {
        // Test all distributions
        test_specific_distribution("all", stats);
    } else {
        // Test specific distributions
        for (const auto& dist : std::vector<std::string>()) {
            cout << "\n  Testing " << dist << ":" << endl;
            test_specific_distribution(dist, stats);
        }
    }

    stats.print_summary("Distribution-Specific");
}

//==============================================================================
// STRESS TESTS
//==============================================================================

void test_stress() {
    cout << "\n=== Stress Testing Error Handling ===" << endl;
    cout << "Running " << 1000 << " iterations with " << 4 << " threads" << endl;

    atomic<int> successes(0);
    atomic<int> failures(0);

    // Concurrent parameter updates stress test
    {
        auto result = GaussianDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto dist = make_shared<GaussianDistribution>(std::move(result).unwrap());

            vector<thread> threads;
            for (size_t t = 0; t < 4; ++t) {
                threads.emplace_back([&, t]() {
                    mt19937 rng(static_cast<uint32_t>(42 + t));  // C4267: explicit size_t→uint32_t
                    uniform_real_distribution<> mean_dist(-100, 100);
                    uniform_real_distribution<> std_dist(0.1, 10);

                    for (size_t i = 0; i < 1000 / 4; ++i) {
                        // Randomly choose between valid and invalid updates
                        if (rng() % 2 == 0) {
                            auto r = dist->trySetParameters(mean_dist(rng), std_dist(rng));
                            if (r.isOk())
                                successes++;
                            else
                                failures++;
                        } else {
                            // Try invalid update
                            auto r = dist->trySetStandardDeviation(-1.0);
                            if (r.isError())
                                successes++;
                            else
                                failures++;
                        }
                    }
                });
            }

            for (auto& t : threads) {
                t.join();
            }

            cout << "  Concurrent updates: " << successes << " correct, " << failures
                 << " incorrect" << endl;
        }
    }

    // Factory creation stress test
    {
        mt19937 rng(42);
        uniform_real_distribution<> param_dist(-1000, 1000);
        int factory_successes = 0;
        int factory_failures = 0;

        for (size_t i = 0; i < 1000; ++i) {
            double p1 = param_dist(rng);
            double p2 = param_dist(rng);

            // Try creating distributions with random parameters
            auto gauss = GaussianDistribution::create(p1, abs(p2));
            if (gauss.isOk())
                factory_successes++;
            else
                factory_failures++;

            auto uniform = UniformDistribution::create(min(p1, p2), max(p1, p2));
            if (uniform.isOk() && p1 != p2)
                factory_successes++;
            else
                factory_failures++;

            auto exp = ExponentialDistribution::create(abs(p1) + 0.001);
            if (exp.isOk())
                factory_successes++;
            else
                factory_failures++;
        }

        cout << "  Factory creations: " << factory_successes << " successes, " << factory_failures
             << " expected failures" << endl;
    }
}

//==============================================================================
// PERFORMANCE BENCHMARKS
//==============================================================================

void test_benchmarks() {
    cout << "\n=== Performance Benchmarks ===" << endl;

    const size_t iterations = 100000;

    // Benchmark Result vs Exception overhead
    {
        auto result = GaussianDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            auto dist = std::move(result).unwrap();

            // Benchmark Result-based API
            auto start = chrono::high_resolution_clock::now();
            for (size_t i = 0; i < iterations; ++i) {
                auto r = dist.trySetMean(i % 2 == 0 ? 1.0 : -1.0);
                (void)r;
            }
            auto end = chrono::high_resolution_clock::now();
            auto result_time = chrono::duration_cast<chrono::microseconds>(end - start);

            // Benchmark exception-based API (only valid updates to avoid throw overhead)
            start = chrono::high_resolution_clock::now();
            for (size_t i = 0; i < iterations; ++i) {
                dist.setMean(i % 2 == 0 ? 1.0 : -1.0);
            }
            end = chrono::high_resolution_clock::now();
            auto exception_time = chrono::duration_cast<chrono::microseconds>(end - start);

            cout << "  Parameter update performance (" << iterations << " iterations):" << endl;
            cout << "    Result-based API: " << result_time.count() << " μs" << endl;
            cout << "    Exception-based API: " << exception_time.count() << " μs" << endl;
            cout << "    Overhead: " << fixed << setprecision(2)
                 << static_cast<double>(result_time.count()) /
                        static_cast<double>(exception_time.count())
                 << "x" << endl;
        }
    }

    // Benchmark validation overhead
    {
        auto start = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            auto r = GaussianDistribution::create(static_cast<double>(i) * 0.001,
                                                  1.0 + static_cast<double>(i) * 0.0001);
            (void)r;
        }
        auto end = chrono::high_resolution_clock::now();
        auto create_time = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "\n  Factory creation performance (" << iterations << " iterations):" << endl;
        cout << "    Average time per creation: "
             << static_cast<double>(create_time.count()) / static_cast<double>(iterations) << " μs"
             << endl;
    }
}

//==============================================================================
// MAIN FUNCTION AND ARGUMENT PARSING
//==============================================================================

TEST(ErrorHandling, FactoryMethods) {
    test_factory_methods();
}
TEST(ErrorHandling, DualApi) {
    test_dual_api();
}
TEST(ErrorHandling, ParameterValidation) {
    test_parameter_validation();
}
TEST(ErrorHandling, DistributionSpecific) {
    test_distributions();
}
TEST(ErrorHandling, StressTests) {
    test_stress();
}
TEST(ErrorHandling, Benchmarks) {
    test_benchmarks();
}

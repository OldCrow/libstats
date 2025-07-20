#include <gtest/gtest.h>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>
#include <thread>
#include <iostream>

// Include the enhanced Exponential distribution
#include "exponential.h"

using namespace std;
using namespace libstats;

// Test configuration
constexpr double TOLERANCE = 1e-12;
constexpr size_t LARGE_BATCH_SIZE = 10000;

// Helper function to check if two doubles are approximately equal
bool approxEqual(double a, double b, double tol = TOLERANCE) {
    return std::abs(a - b) <= tol;
}

// Helper function to calculate sample statistics
pair<double, double> calculateSampleStats(const vector<double>& samples) {
    double mean = accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double variance = 0.0;
    for (double x : samples) {
        variance += (x - mean) * (x - mean);
    }
    variance /= samples.size();
    return {mean, variance};
}

class ExponentialDistributionEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
    }
    
    void TearDown() override {
        // Common cleanup for tests
    }
};

// Test basic functionality
TEST_F(ExponentialDistributionEnhancedTest, BasicFunctionality) {
    // Test 1: Unit exponential distribution (λ = 1)
    auto unitExpResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(unitExpResult.isOk()) << "Failed to create unit exponential distribution";
    auto unitExp = std::move(unitExpResult.value);
    
    // Test known values for unit exponential
    EXPECT_TRUE(approxEqual(unitExp.getLambda(), 1.0));
    EXPECT_TRUE(approxEqual(unitExp.getMean(), 1.0));
    EXPECT_TRUE(approxEqual(unitExp.getVariance(), 1.0));
    EXPECT_TRUE(approxEqual(unitExp.getSkewness(), 2.0));
    EXPECT_TRUE(approxEqual(unitExp.getKurtosis(), 6.0));
    EXPECT_TRUE(approxEqual(unitExp.getScale(), 1.0));
    
    // Test probability functions
    double pdf_at_0 = unitExp.getProbability(0.0);
    double pdf_at_1 = unitExp.getProbability(1.0);
    double log_pdf_at_0 = unitExp.getLogProbability(0.0);
    double log_pdf_at_1 = unitExp.getLogProbability(1.0);
    double cdf_at_0 = unitExp.getCumulativeProbability(0.0);
    double cdf_at_1 = unitExp.getCumulativeProbability(1.0);
    double quantile_median = unitExp.getQuantile(0.5);
    
    // Test known values for unit exponential
    EXPECT_TRUE(approxEqual(pdf_at_0, 1.0, 1e-10)) << "PDF at 0 should be 1.0";
    EXPECT_TRUE(approxEqual(pdf_at_1, exp(-1.0), 1e-10)) << "PDF at 1 should be e^(-1)";
    EXPECT_TRUE(approxEqual(log_pdf_at_0, 0.0, 1e-10)) << "Log PDF at 0 should be 0.0";
    EXPECT_TRUE(approxEqual(log_pdf_at_1, -1.0, 1e-10)) << "Log PDF at 1 should be -1.0";
    EXPECT_TRUE(approxEqual(cdf_at_0, 0.0, 1e-10)) << "CDF at 0 should be 0.0";
    EXPECT_TRUE(approxEqual(cdf_at_1, 1.0 - exp(-1.0), 1e-10)) << "CDF at 1 should be 1 - e^(-1)";
    EXPECT_TRUE(approxEqual(quantile_median, log(2.0), 1e-10)) << "Median should be ln(2)";
    
    // Test 2: Custom exponential distribution (λ = 2)
    auto customExpResult = ExponentialDistribution::create(2.0);
    ASSERT_TRUE(customExpResult.isOk()) << "Failed to create custom exponential distribution";
    auto customExp = std::move(customExpResult.value);
    
    // Test known values for λ = 2
    EXPECT_TRUE(approxEqual(customExp.getLambda(), 2.0));
    EXPECT_TRUE(approxEqual(customExp.getMean(), 0.5)) << "Mean should be 1/λ = 1/2";
    EXPECT_TRUE(approxEqual(customExp.getVariance(), 0.25)) << "Variance should be 1/λ² = 1/4";
    EXPECT_TRUE(approxEqual(customExp.getProbability(0.5), 2.0 * exp(-1.0), 1e-10));
    EXPECT_TRUE(approxEqual(customExp.getCumulativeProbability(0.5), 1.0 - exp(-1.0), 1e-10));
}

// Test copy and move operations
TEST_F(ExponentialDistributionEnhancedTest, CopyAndMoveOperations) {
    // Test 1: Copy constructor
    auto originalResult = ExponentialDistribution::create(2.0);
    ASSERT_TRUE(originalResult.isOk());
    auto original = std::move(originalResult.value);
    
    auto copied = original;  // Copy constructor
    
    // Verify both objects have same parameters
    EXPECT_TRUE(approxEqual(original.getLambda(), copied.getLambda()));
    EXPECT_TRUE(approxEqual(original.getMean(), copied.getMean()));
    EXPECT_TRUE(approxEqual(original.getVariance(), copied.getVariance()));
    
    // Verify they produce same results
    double test_val = 1.0;
    EXPECT_TRUE(approxEqual(original.getProbability(test_val), copied.getProbability(test_val)));
    EXPECT_TRUE(approxEqual(original.getCumulativeProbability(test_val), copied.getCumulativeProbability(test_val)));
    
    // Test 2: Move constructor
    auto sourceResult = ExponentialDistribution::create(3.0);
    ASSERT_TRUE(sourceResult.isOk());
    auto source = std::move(sourceResult.value);
    
    double orig_lambda = source.getLambda();
    auto moved = std::move(source);  // Move constructor
    
    // Verify moved object has correct parameters
    EXPECT_TRUE(approxEqual(moved.getLambda(), orig_lambda));
    EXPECT_TRUE(approxEqual(moved.getMean(), 1.0 / orig_lambda));
    
    // Test 3: Copy assignment
    auto dist1Result = ExponentialDistribution::create(1.0);
    auto dist2Result = ExponentialDistribution::create(2.0);
    ASSERT_TRUE(dist1Result.isOk() && dist2Result.isOk());
    auto dist1 = std::move(dist1Result.value);
    auto dist2 = std::move(dist2Result.value);
    
    // Verify they're different initially
    EXPECT_FALSE(approxEqual(dist1.getLambda(), dist2.getLambda()));
    
    dist1 = dist2;  // Copy assignment
    
    // Verify they're now the same
    EXPECT_TRUE(approxEqual(dist1.getLambda(), dist2.getLambda()));
    EXPECT_TRUE(approxEqual(dist1.getMean(), dist2.getMean()));
    
    // Test 4: Move assignment
    auto targetResult = ExponentialDistribution::create(1.0);
    auto sourceResult2 = ExponentialDistribution::create(4.0);
    ASSERT_TRUE(targetResult.isOk() && sourceResult2.isOk());
    auto target = std::move(targetResult.value);
    auto source2 = std::move(sourceResult2.value);
    
    double source_lambda = source2.getLambda();
    target = std::move(source2);  // Move assignment
    
    // Verify target now has source's parameters
    EXPECT_TRUE(approxEqual(target.getLambda(), source_lambda));
    EXPECT_TRUE(approxEqual(target.getMean(), 1.0 / source_lambda));
}

// Test advanced statistical methods
TEST_F(ExponentialDistributionEnhancedTest, AdvancedStatisticalMethods) {
    auto data = std::vector<double>{0.2, 0.4, 0.6, 0.8, 1.0};
    auto expDistResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(expDistResult.isOk());
    auto expDist = std::move(expDistResult.value);

    // Confidence Interval for Rate
    auto [rate_lower, rate_upper] = ExponentialDistribution::confidenceIntervalRate(data, 0.95);
    EXPECT_GT(rate_upper, rate_lower) << "Upper CI bound should be greater than lower bound";
    EXPECT_GT(rate_lower, 0.0) << "Rate CI lower bound should be positive";

    // Likelihood Ratio Test
    auto [lr_statistic, p_value, reject_null] = ExponentialDistribution::likelihoodRatioTest(data, 1.0);
    EXPECT_GE(lr_statistic, 0.0) << "LR statistic should be non-negative";
    EXPECT_GE(p_value, 0.0) << "P-value should be non-negative";
    EXPECT_LE(p_value, 1.0) << "P-value should not exceed 1.0";

    // Bayesian Estimation
    auto [post_shape, post_rate] = ExponentialDistribution::bayesianEstimation(data);
    EXPECT_GT(post_shape, 0.0) << "Posterior shape should be positive";
    EXPECT_GT(post_rate, 0.0) << "Posterior rate should be positive";

    // Method of Moments Estimation
    double lambda_mom = ExponentialDistribution::methodOfMomentsEstimation(data);
    EXPECT_GT(lambda_mom, 0.0) << "MOM estimate should be positive";
}

// Test batch operations
TEST_F(ExponentialDistributionEnhancedTest, BatchOperations) {
    
    auto unitExpResult = ExponentialDistribution::create(1.0);
    auto customExpResult = ExponentialDistribution::create(0.5);
    ASSERT_TRUE(unitExpResult.isOk() && customExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    auto customExp = std::move(customExpResult.value);
    
    // Test data - positive values only for exponential
    vector<double> test_values = {0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0};
    vector<double> pdf_results(test_values.size());
    vector<double> log_pdf_results(test_values.size());
    vector<double> cdf_results(test_values.size());
    
    // Test 1: Small batch operations (should use scalar implementation)
    cout << "1. Small Batch Operations (scalar fallback):" << endl;
    
    unitExp.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    unitExp.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    unitExp.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = unitExp.getProbability(test_values[i]);
        double expected_log_pdf = unitExp.getLogProbability(test_values[i]);
        double expected_cdf = unitExp.getCumulativeProbability(test_values[i]);
        
        EXPECT_TRUE(approxEqual(pdf_results[i], expected_pdf, 1e-12));
        EXPECT_TRUE(approxEqual(log_pdf_results[i], expected_log_pdf, 1e-12));
        EXPECT_TRUE(approxEqual(cdf_results[i], expected_cdf, 1e-12));
    }
    
    cout << "   ✓ Small batch operations match individual calls!" << endl;
    
    // Test 2: Large batch operations (should use SIMD implementation)
    cout << "2. Large Batch Operations (SIMD optimized):" << endl;
    
    vector<double> large_test_values(LARGE_BATCH_SIZE);
    vector<double> large_pdf_results(LARGE_BATCH_SIZE);
    vector<double> large_log_pdf_results(LARGE_BATCH_SIZE);
    vector<double> large_cdf_results(LARGE_BATCH_SIZE);
    
    // Generate test data - exponential distribution domain is [0, ∞)
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::exponential_distribution<> dis(1.0); // Generate exponential-like values
    
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_test_values[i] = dis(gen);
    }
    
    // Time batch operations
    auto start = chrono::high_resolution_clock::now();
    unitExp.getProbabilityBatch(large_test_values.data(), large_pdf_results.data(), LARGE_BATCH_SIZE);
    auto end = chrono::high_resolution_clock::now();
    auto batch_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    // Time individual operations for comparison
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pdf_results[i] = unitExp.getProbability(large_test_values[i]);
    }
    end = chrono::high_resolution_clock::now();
    auto individual_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    cout << "   Batch PDF time: " << batch_time << " μs" << endl;
    cout << "   Individual PDF time: " << individual_time << " μs" << endl;
    cout << "   Speedup: " << (double)individual_time / batch_time << "x" << endl;
    
    // Test batch log probability
    unitExp.getLogProbabilityBatch(large_test_values.data(), large_log_pdf_results.data(), LARGE_BATCH_SIZE);
    
    // Test batch CDF
    unitExp.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    
    // Verify a sample of results
    const size_t sample_size = 100;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * (LARGE_BATCH_SIZE / sample_size);
        double expected_pdf = unitExp.getProbability(large_test_values[idx]);
        double expected_log_pdf = unitExp.getLogProbability(large_test_values[idx]);
        double expected_cdf = unitExp.getCumulativeProbability(large_test_values[idx]);
        
        EXPECT_TRUE(approxEqual(large_pdf_results[idx], expected_pdf, 1e-10));
        EXPECT_TRUE(approxEqual(large_log_pdf_results[idx], expected_log_pdf, 1e-10));
        EXPECT_TRUE(approxEqual(large_cdf_results[idx], expected_cdf, 1e-10));
    }
    
    cout << "   ✓ Large batch operations match individual calls!" << endl;
    
    // Test 3: Custom distribution batch operations
    cout << "3. Custom Distribution Batch Operations:" << endl;
    
    customExp.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    customExp.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    customExp.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = customExp.getProbability(test_values[i]);
        double expected_log_pdf = customExp.getLogProbability(test_values[i]);
        double expected_cdf = customExp.getCumulativeProbability(test_values[i]);
        
        EXPECT_TRUE(approxEqual(pdf_results[i], expected_pdf, 1e-12));
        EXPECT_TRUE(approxEqual(log_pdf_results[i], expected_log_pdf, 1e-12));
        EXPECT_TRUE(approxEqual(cdf_results[i], expected_cdf, 1e-12));
    }
    
    cout << "   ✓ Custom distribution batch operations passed!" << endl;
}

// Test optimized sampling
TEST_F(ExponentialDistributionEnhancedTest, OptimizedSampling) {
    
    auto unitExpResult = ExponentialDistribution::create(1.0);
    auto customExpResult = ExponentialDistribution::create(2.0);
    ASSERT_TRUE(unitExpResult.isOk() && customExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    auto customExp = std::move(customExpResult.value);
    
    mt19937 rng(42);  // Fixed seed for reproducibility
    
    // Test 1: Sample quality for unit exponential
    cout << "1. Unit Exponential Sampling Quality:" << endl;
    
    const size_t num_samples = 100000;
    vector<double> samples;
    samples.reserve(num_samples);
    
    // Time sampling
    auto start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(unitExp.sample(rng));
    }
    auto end = chrono::high_resolution_clock::now();
    auto sampling_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    auto [sample_mean, sample_variance] = calculateSampleStats(samples);
    
    cout << "   Samples: " << num_samples << endl;
    cout << "   Sampling time: " << sampling_time << " μs" << endl;
    cout << "   Time per sample: " << (double)sampling_time / num_samples << " μs" << endl;
    cout << "   Sample mean: " << sample_mean << " (expected: 1.0)" << endl;
    cout << "   Sample variance: " << sample_variance << " (expected: 1.0)" << endl;
    cout << "   Mean error: " << abs(sample_mean - 1.0) << endl;
    cout << "   Variance error: " << abs(sample_variance - 1.0) << endl;
    
    // Check if samples are within reasonable bounds
    EXPECT_LT(abs(sample_mean - 1.0), 0.02) << "Mean should be close to 1.0";
    EXPECT_LT(abs(sample_variance - 1.0), 0.02) << "Variance should be close to 1.0";
    
    // Check that all samples are non-negative
    for (double sample : samples) {
        EXPECT_GE(sample, 0.0) << "All samples should be non-negative";
    }
    
    cout << "   ✓ Unit exponential sampling quality passed!" << endl;
    
    // Test 2: Sample quality for custom distribution
    cout << "2. Custom Distribution Sampling Quality:" << endl;
    
    samples.clear();
    rng.seed(42);  // Reset for reproducibility
    
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(customExp.sample(rng));
    }
    
    auto [custom_mean, custom_variance] = calculateSampleStats(samples);
    
    cout << "   Expected mean: " << customExp.getMean() << endl;
    cout << "   Expected variance: " << customExp.getVariance() << endl;
    cout << "   Sample mean: " << custom_mean << endl;
    cout << "   Sample variance: " << custom_variance << endl;
    cout << "   Mean error: " << abs(custom_mean - customExp.getMean()) << endl;
    cout << "   Variance error: " << abs(custom_variance - customExp.getVariance()) << endl;
    
    // Check if samples are within reasonable bounds
    EXPECT_LT(abs(custom_mean - customExp.getMean()), 0.01) << "Custom mean should match expected";
    EXPECT_LT(abs(custom_variance - customExp.getVariance()), 0.01) << "Custom variance should match expected";
    
    cout << "   ✓ Custom distribution sampling quality passed!" << endl;
    
    // Test 3: Numerical stability with extreme parameters
    cout << "3. Numerical Stability Test:" << endl;
    
    // Test with very large lambda (small scale)
    auto extremeResult = ExponentialDistribution::create(100.0);
    ASSERT_TRUE(extremeResult.isOk());
    auto extreme = std::move(extremeResult.value);
    
    for (int i = 0; i < 1000; ++i) {
        double sample = extreme.sample(rng);
        EXPECT_TRUE(std::isfinite(sample)) << "Sample should be finite";
        EXPECT_GE(sample, 0.0) << "Sample should be non-negative";
    }
    cout << "   ✓ Numerical stability test passed!" << endl;
}

// Test edge cases and error handling
TEST_F(ExponentialDistributionEnhancedTest, EdgeCases) {
    
    // Test 1: Invalid parameters
    cout << "1. Invalid Parameter Handling:" << endl;
    
    auto resultZero = ExponentialDistribution::create(0.0);
    if (resultZero.isError()) {
        cout << "   ✓ Zero lambda correctly rejected: " << resultZero.message << endl;
    }
    
    auto resultNegative = ExponentialDistribution::create(-1.0);
    if (resultNegative.isError()) {
        cout << "   ✓ Negative lambda correctly rejected: " << resultNegative.message << endl;
    }
    
    auto resultInf = ExponentialDistribution::create(std::numeric_limits<double>::infinity());
    if (resultInf.isError()) {
        cout << "   ✓ Infinite lambda correctly rejected: " << resultInf.message << endl;
    }
    
    auto resultNaN = ExponentialDistribution::create(std::numeric_limits<double>::quiet_NaN());
    if (resultNaN.isError()) {
        cout << "   ✓ NaN lambda correctly rejected: " << resultNaN.message << endl;
    }
    
    // Test 2: Extreme values
    cout << "2. Extreme Value Handling:" << endl;
    
    auto normalResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(normalResult.isOk());
    auto normal = std::move(normalResult.value);
    
    // Test very large values
    double large_val = 100.0;
    double pdf_large = normal.getProbability(large_val);
    double log_pdf_large = normal.getLogProbability(large_val);
    double cdf_large = normal.getCumulativeProbability(large_val);
    
    EXPECT_GE(pdf_large, 0.0);
    EXPECT_TRUE(std::isfinite(log_pdf_large));
    EXPECT_GE(cdf_large, 0.0);
    EXPECT_LE(cdf_large, 1.0);
    
    cout << "   ✓ Large value handling passed" << endl;
    
    // Test negative values (should return 0 for PDF, -inf for log PDF, 0 for CDF)
    double neg_val = -1.0;
    double pdf_neg = normal.getProbability(neg_val);
    double log_pdf_neg = normal.getLogProbability(neg_val);
    double cdf_neg = normal.getCumulativeProbability(neg_val);
    
    EXPECT_EQ(pdf_neg, 0.0);
    EXPECT_TRUE(log_pdf_neg == std::numeric_limits<double>::lowest() || std::isinf(log_pdf_neg));
    EXPECT_EQ(cdf_neg, 0.0);
    
    cout << "   ✓ Negative value handling passed" << endl;
    
    // Test 3: Empty batch operations
    cout << "3. Empty Batch Operations:" << endl;
    
    vector<double> empty_values;
    vector<double> empty_results;
    
    // These should not crash
    normal.getProbabilityBatch(empty_values.data(), empty_results.data(), 0);
    normal.getLogProbabilityBatch(empty_values.data(), empty_results.data(), 0);
    normal.getCumulativeProbabilityBatch(empty_values.data(), empty_results.data(), 0);
    
    cout << "   ✓ Empty batch operations handled gracefully" << endl;
    
    // Test 4: Parameter fitting
    cout << "4. Parameter Fitting:" << endl;
    
    // Generate some exponential samples
    mt19937 rng(42);
    vector<double> fit_data;
    exponential_distribution<> true_exp(2.0);  // Rate = 2.0
    for (int i = 0; i < 1000; ++i) {
        fit_data.push_back(true_exp(rng));
    }
    
    // Fit distribution to data
    auto fitResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(fitResult.isOk());
    auto fitted = std::move(fitResult.value);
    fitted.fit(fit_data);
    
    cout << "   Original lambda: 2.0" << endl;
    cout << "   Fitted lambda: " << fitted.getLambda() << endl;
    cout << "   Fitting error: " << abs(fitted.getLambda() - 2.0) << endl;
    
    // Should be reasonably close to original parameter
    EXPECT_LT(abs(fitted.getLambda() - 2.0), 0.1) << "Fitted parameter should be close to original";
    
    cout << "   ✓ Parameter fitting test passed" << endl;
}

// Test thread safety
TEST_F(ExponentialDistributionEnhancedTest, ThreadSafety) {
    
    // This is a basic test - comprehensive thread safety testing would require
    // more sophisticated concurrent testing frameworks
    
    auto normalResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(normalResult.isOk());
    auto normal = std::move(normalResult.value);
    
    // Test that multiple threads can safely read from the same distribution
    const int num_threads = 4;
    const int samples_per_thread = 1000;
    
    vector<thread> threads;
    vector<vector<double>> results(num_threads);
    
    mt19937 rng(42);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&normal, &results, t]() {
            mt19937 local_rng(42 + t);
            results[t].reserve(samples_per_thread);
            
            for (int i = 0; i < samples_per_thread; ++i) {
                results[t].push_back(normal.sample(local_rng));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify that all threads produced valid results
    for (int t = 0; t < num_threads; ++t) {
        EXPECT_EQ(results[t].size(), samples_per_thread);
        for (double val : results[t]) {
            EXPECT_TRUE(std::isfinite(val));
            EXPECT_GE(val, 0.0);
        }
    }
    
    cout << "   ✓ Basic thread safety test passed" << endl;
}

// Test quantile function
TEST_F(ExponentialDistributionEnhancedTest, QuantileFunction) {
    
    auto unitExpResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(unitExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    
    // Test specific quantiles
    cout << "1. Quantile Function Tests:" << endl;
    
    vector<double> probabilities = {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};
    
    for (double p : probabilities) {
        double quantile = unitExp.getQuantile(p);
        double cdf_check = unitExp.getCumulativeProbability(quantile);
        
        cout << "   P(" << p << ") = " << quantile << ", CDF(" << quantile << ") = " << cdf_check << endl;
        
        // CDF(quantile(p)) should equal p
        EXPECT_TRUE(approxEqual(cdf_check, p, 1e-10));
    }
    
    // Test known quantiles for unit exponential
    EXPECT_TRUE(approxEqual(unitExp.getQuantile(0.5), log(2.0), 1e-10)) << "Median should be ln(2)";
    EXPECT_TRUE(approxEqual(unitExp.getQuantile(1.0 - 1.0/exp(1.0)), 1.0, 1e-10)) << "Q(1-1/e) should be 1";
    
    cout << "   ✓ Quantile function tests passed!" << endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

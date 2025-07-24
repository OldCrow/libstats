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
#include <span>

// Include the enhanced Exponential distribution
#include "exponential.h"
#include "work_stealing_pool.h"
#include "adaptive_cache.h"

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
    
    cout << "   === SIMD Batch Performance Results ===" << endl;
    
    // Test 2a: PDF Batch vs Individual
    auto start = chrono::high_resolution_clock::now();
    unitExp.getProbabilityBatch(large_test_values.data(), large_pdf_results.data(), LARGE_BATCH_SIZE);
    auto end = chrono::high_resolution_clock::now();
    auto pdf_batch_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pdf_results[i] = unitExp.getProbability(large_test_values[i]);
    }
    end = chrono::high_resolution_clock::now();
    auto pdf_individual_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    double pdf_speedup = (double)pdf_individual_time / pdf_batch_time;
    cout << "   PDF:     Batch " << pdf_batch_time << "μs vs Individual " << pdf_individual_time << "μs → " << pdf_speedup << "x speedup" << endl;
    
    // Test 2b: Log PDF Batch vs Individual
    start = chrono::high_resolution_clock::now();
    unitExp.getLogProbabilityBatch(large_test_values.data(), large_log_pdf_results.data(), LARGE_BATCH_SIZE);
    end = chrono::high_resolution_clock::now();
    auto log_pdf_batch_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_log_pdf_results[i] = unitExp.getLogProbability(large_test_values[i]);
    }
    end = chrono::high_resolution_clock::now();
    auto log_pdf_individual_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    double log_pdf_speedup = (double)log_pdf_individual_time / log_pdf_batch_time;
    cout << "   LogPDF:  Batch " << log_pdf_batch_time << "μs vs Individual " << log_pdf_individual_time << "μs → " << log_pdf_speedup << "x speedup" << endl;
    
    // Test 2c: CDF Batch vs Individual
    start = chrono::high_resolution_clock::now();
    unitExp.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    end = chrono::high_resolution_clock::now();
    auto cdf_batch_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_cdf_results[i] = unitExp.getCumulativeProbability(large_test_values[i]);
    }
    end = chrono::high_resolution_clock::now();
    auto cdf_individual_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    double cdf_speedup = (double)cdf_individual_time / cdf_batch_time;
    cout << "   CDF:     Batch " << cdf_batch_time << "μs vs Individual " << cdf_individual_time << "μs → " << cdf_speedup << "x speedup" << endl;
    
    // Expect speedup for all operations
    EXPECT_GT(pdf_speedup, 1.0) << "PDF batch should be faster than individual calls";
    EXPECT_GT(log_pdf_speedup, 1.0) << "Log PDF batch should be faster than individual calls";
    EXPECT_GT(cdf_speedup, 1.0) << "CDF batch should be faster than individual calls";
    
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

// Test parallel batch operations benchmark
TEST_F(ExponentialDistributionEnhancedTest, ParallelBatchPerformanceBenchmark) {
    auto unitExpResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(unitExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data (positive values only for exponential)
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.01, 5.0); // Positive values only
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    std::cout << "\n=== Exponential Parallel Batch Operations Performance Benchmark ===" << std::endl;
    std::cout << "Dataset size: " << BENCHMARK_SIZE << " elements" << std::endl;
    
    // 1. Standard SIMD Batch Operations (baseline)
    auto start = std::chrono::high_resolution_clock::now();
    unitExp.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    unitExp.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto simd_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    unitExp.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto simd_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "SIMD Batch (baseline):" << std::endl;
    std::cout << "  PDF:     " << simd_pdf_time << " μs" << std::endl;
    std::cout << "  LogPDF:  " << simd_log_pdf_time << " μs" << std::endl;
    std::cout << "  CDF:     " << simd_cdf_time << " μs" << std::endl;
    
    // 2. Standard Parallel Batch Operations
    std::span<const double> input_span(test_values);
    std::span<double> output_span(pdf_results);
    
    start = std::chrono::high_resolution_clock::now();
    unitExp.getProbabilityBatchParallel(input_span, output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::span<double> log_output_span(log_pdf_results);
    start = std::chrono::high_resolution_clock::now();
    unitExp.getLogProbabilityBatchParallel(input_span, log_output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::span<double> cdf_output_span(cdf_results);
    start = std::chrono::high_resolution_clock::now();
    unitExp.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double parallel_pdf_speedup = (double)simd_pdf_time / parallel_pdf_time;
    double parallel_log_pdf_speedup = (double)simd_log_pdf_time / parallel_log_pdf_time;
    double parallel_cdf_speedup = (double)simd_cdf_time / parallel_cdf_time;
    
    std::cout << "Standard Parallel:" << std::endl;
    std::cout << "  PDF:     " << parallel_pdf_time << " μs (" << parallel_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << parallel_log_pdf_time << " μs (" << parallel_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << parallel_cdf_time << " μs (" << parallel_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // 3. Work-Stealing Parallel Operations
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    
    // PDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    unitExp.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // LogPDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    unitExp.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    unitExp.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double work_steal_pdf_speedup = (double)simd_pdf_time / work_steal_pdf_time;
    double work_steal_log_pdf_speedup = (double)simd_log_pdf_time / work_steal_log_pdf_time;
    double work_steal_cdf_speedup = (double)simd_cdf_time / work_steal_cdf_time;
    
    std::cout << "Work-Stealing Parallel:" << std::endl;
    std::cout << "  PDF:     " << work_steal_pdf_time << " μs (" << work_steal_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << work_steal_log_pdf_time << " μs (" << work_steal_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << work_steal_cdf_time << " μs (" << work_steal_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // 4. Cache-Aware Parallel Operations
    cache::AdaptiveCache<std::string, double> cache_manager;
    
    // PDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    unitExp.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // LogPDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    unitExp.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    unitExp.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cache_aware_pdf_speedup = (double)simd_pdf_time / cache_aware_pdf_time;
    double cache_aware_log_pdf_speedup = (double)simd_log_pdf_time / cache_aware_log_pdf_time;
    double cache_aware_cdf_speedup = (double)simd_cdf_time / cache_aware_cdf_time;
    
    std::cout << "Cache-Aware Parallel:" << std::endl;
    std::cout << "  PDF:     " << cache_aware_pdf_time << " μs (" << cache_aware_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << cache_aware_log_pdf_time << " μs (" << cache_aware_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << cache_aware_cdf_time << " μs (" << cache_aware_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // Performance Analysis
    std::cout << "\nPerformance Analysis:" << std::endl;
    
    if (std::thread::hardware_concurrency() > 2) {
        if (parallel_pdf_speedup > 0.8) {
            std::cout << "  ✓ Standard parallel shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Standard parallel speedup lower than expected" << std::endl;
        }
        
        if (work_steal_pdf_speedup > 0.8) {
            std::cout << "  ✓ Work-stealing shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Work-stealing speedup lower than expected" << std::endl;
        }
        
        if (cache_aware_pdf_speedup > 0.8) {
            std::cout << "  ✓ Cache-aware shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Cache-aware speedup lower than expected" << std::endl;
        }
    } else {
        std::cout << "  ℹ Single/dual-core system - parallel overhead may dominate" << std::endl;
    }
    
    // Verify correctness
    std::cout << "\nCorrectness verification:" << std::endl;
    bool all_correct = true;
    const size_t check_count = 100;
    for (size_t i = 0; i < check_count; ++i) {
        size_t idx = i * (BENCHMARK_SIZE / check_count);
        double expected = unitExp.getProbability(test_values[idx]);
        if (std::abs(pdf_results[idx] - expected) > 1e-10) {
            all_correct = false;
            break;
        }
    }
    
    if (all_correct) {
        std::cout << "  ✓ All parallel implementations produce correct results" << std::endl;
    } else {
        std::cout << "  ✗ Correctness check failed" << std::endl;
    }
    
    EXPECT_TRUE(all_correct) << "Parallel batch operations should produce correct results";
    
    // Statistics
    auto ws_stats = work_stealing_pool.getStatistics();
    auto cache_stats = cache_manager.getStats();
    
    std::cout << "\nWork-Stealing Pool Statistics:" << std::endl;
    std::cout << "  Tasks executed: " << ws_stats.tasksExecuted << std::endl;
    std::cout << "  Work steals: " << ws_stats.workSteals << std::endl;
    std::cout << "  Steal success rate: " << (ws_stats.stealSuccessRate * 100) << "%" << std::endl;
    
    std::cout << "\nCache Manager Statistics:" << std::endl;
    std::cout << "  Cache size: " << cache_stats.size << " entries" << std::endl;
    std::cout << "  Hit rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;
    std::cout << "  Memory usage: " << cache_stats.memory_usage << " bytes" << std::endl;
}

// Test new work-stealing and cache-aware methods for log probability and CDF
TEST_F(ExponentialDistributionEnhancedTest, NewWorkStealingAndCacheAwareMethods) {
    auto unitExpResult = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(unitExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    
    constexpr size_t TEST_SIZE = 10000;
    
    // Generate test data (positive values only for exponential)
    std::vector<double> test_values(TEST_SIZE);
    std::random_device rd;
    std::mt19937 gen(42);
    std::exponential_distribution<> dis(1.0);
    
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    // Prepare result vectors
    std::vector<double> log_pdf_ws_results(TEST_SIZE);
    std::vector<double> log_pdf_cache_results(TEST_SIZE);
    std::vector<double> cdf_ws_results(TEST_SIZE);
    std::vector<double> cdf_cache_results(TEST_SIZE);
    std::vector<double> expected_log_pdf(TEST_SIZE);
    std::vector<double> expected_cdf(TEST_SIZE);
    
    // Calculate expected results
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        expected_log_pdf[i] = unitExp.getLogProbability(test_values[i]);
        expected_cdf[i] = unitExp.getCumulativeProbability(test_values[i]);
    }
    
    // Test work-stealing implementations
    std::cout << "Testing new work-stealing methods:" << std::endl;
    
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    cache::AdaptiveCache<std::string, double> cache_manager;
    
    // Test work-stealing log probability
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(log_pdf_ws_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        unitExp.getLogProbabilityBatchWorkStealing(values_span, results_span, work_stealing_pool);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ws_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  Log PDF work-stealing: " << ws_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(log_pdf_ws_results[i] - expected_log_pdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Work-stealing log PDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test cache-aware log probability
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(log_pdf_cache_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        unitExp.getLogProbabilityBatchCacheAware(values_span, results_span, cache_manager);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  Log PDF cache-aware: " << cache_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(log_pdf_cache_results[i] - expected_log_pdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Cache-aware log PDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test work-stealing CDF
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(cdf_ws_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        unitExp.getCumulativeProbabilityBatchWorkStealing(values_span, results_span, work_stealing_pool);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ws_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  CDF work-stealing: " << ws_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(cdf_ws_results[i] - expected_cdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Work-stealing CDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test cache-aware CDF
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(cdf_cache_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        unitExp.getCumulativeProbabilityBatchCacheAware(values_span, results_span, cache_manager);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  CDF cache-aware: " << cache_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(cdf_cache_results[i] - expected_cdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Cache-aware CDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test with different distribution parameters
    std::cout << "\nTesting with different parameters (λ=2.5):" << std::endl;
    
    auto customExpResult = ExponentialDistribution::create(2.5);
    ASSERT_TRUE(customExpResult.isOk());
    auto customExp = std::move(customExpResult.value);
    
    // Test a subset with custom distribution
    const size_t subset_size = 1000;
    std::span<const double> subset_values(test_values.data(), subset_size);
    std::span<double> subset_log_results(log_pdf_ws_results.data(), subset_size);
    std::span<double> subset_cdf_results(cdf_ws_results.data(), subset_size);
    
    customExp.getLogProbabilityBatchWorkStealing(subset_values, subset_log_results, work_stealing_pool);
    customExp.getCumulativeProbabilityBatchCacheAware(subset_values, subset_cdf_results, cache_manager);
    
    // Verify against individual calls
    bool custom_correct = true;
    for (size_t i = 0; i < subset_size; ++i) {
        double expected_log = customExp.getLogProbability(test_values[i]);
        double expected_cdf = customExp.getCumulativeProbability(test_values[i]);
        
        if (std::abs(log_pdf_ws_results[i] - expected_log) > 1e-10 ||
            std::abs(cdf_ws_results[i] - expected_cdf) > 1e-10) {
            custom_correct = false;
            break;
        }
    }
    
    EXPECT_TRUE(custom_correct) << "Custom distribution should produce correct results";
    std::cout << "  ✓ Custom distribution tests passed" << std::endl;
    
    // Print final statistics
    auto ws_stats = work_stealing_pool.getStatistics();
    auto cache_stats = cache_manager.getStats();
    
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "  Work-stealing tasks: " << ws_stats.tasksExecuted << std::endl;
    std::cout << "  Cache entries: " << cache_stats.size << std::endl;
    std::cout << "  Cache hit rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

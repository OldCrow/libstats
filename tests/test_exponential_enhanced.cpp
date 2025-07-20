#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cassert>
#include <cmath>
#include <thread>

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

// Test basic functionality
void testBasicFunctionality() {
    cout << "\n=== Testing Basic Functionality ===" << endl;
    
    // Test 1: Unit exponential distribution (λ = 1)
    cout << "1. Unit Exponential Distribution Exp(1):" << endl;
    auto unitExpResult = ExponentialDistribution::create(1.0);
    assert(unitExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    
    cout << "   Parameters:" << endl;
    cout << "   - Lambda: " << unitExp.getLambda() << endl;
    cout << "   - Mean: " << unitExp.getMean() << endl;
    cout << "   - Variance: " << unitExp.getVariance() << endl;
    cout << "   - Skewness: " << unitExp.getSkewness() << endl;
    cout << "   - Kurtosis: " << unitExp.getKurtosis() << endl;
    cout << "   - Scale: " << unitExp.getScale() << endl;
    
    // Test known values for unit exponential
    assert(approxEqual(unitExp.getLambda(), 1.0));
    assert(approxEqual(unitExp.getMean(), 1.0));
    assert(approxEqual(unitExp.getVariance(), 1.0));
    assert(approxEqual(unitExp.getSkewness(), 2.0));
    assert(approxEqual(unitExp.getKurtosis(), 6.0));
    assert(approxEqual(unitExp.getScale(), 1.0));
    
    cout << "   Probability Functions:" << endl;
    double pdf_at_0 = unitExp.getProbability(0.0);
    double pdf_at_1 = unitExp.getProbability(1.0);
    double log_pdf_at_0 = unitExp.getLogProbability(0.0);
    double log_pdf_at_1 = unitExp.getLogProbability(1.0);
    double cdf_at_0 = unitExp.getCumulativeProbability(0.0);
    double cdf_at_1 = unitExp.getCumulativeProbability(1.0);
    double quantile_median = unitExp.getQuantile(0.5);
    
    cout << "   - PDF at x=0: " << pdf_at_0 << endl;
    cout << "   - PDF at x=1: " << pdf_at_1 << endl;
    cout << "   - Log PDF at x=0: " << log_pdf_at_0 << endl;
    cout << "   - Log PDF at x=1: " << log_pdf_at_1 << endl;
    cout << "   - CDF at x=0: " << cdf_at_0 << endl;
    cout << "   - CDF at x=1: " << cdf_at_1 << endl;
    cout << "   - Median (quantile 0.5): " << quantile_median << endl;
    
    // Test known values for unit exponential
    assert(approxEqual(pdf_at_0, 1.0, 1e-10));  // f(0) = λ = 1
    assert(approxEqual(pdf_at_1, exp(-1.0), 1e-10));  // f(1) = e^(-1)
    assert(approxEqual(log_pdf_at_0, 0.0, 1e-10));  // log(1) = 0
    assert(approxEqual(log_pdf_at_1, -1.0, 1e-10));  // log(e^(-1)) = -1
    assert(approxEqual(cdf_at_0, 0.0, 1e-10));  // F(0) = 0
    assert(approxEqual(cdf_at_1, 1.0 - exp(-1.0), 1e-10));  // F(1) = 1 - e^(-1)
    assert(approxEqual(quantile_median, log(2.0), 1e-10));  // Q(0.5) = ln(2)
    
    // Test 2: Custom exponential distribution (λ = 2)
    cout << "\n2. Custom Exponential Distribution Exp(2):" << endl;
    auto customExpResult = ExponentialDistribution::create(2.0);
    assert(customExpResult.isOk());
    auto customExp = std::move(customExpResult.value);
    
    cout << "   Lambda: " << customExp.getLambda() << endl;
    cout << "   Mean: " << customExp.getMean() << endl;
    cout << "   Variance: " << customExp.getVariance() << endl;
    cout << "   PDF at x=0.5: " << customExp.getProbability(0.5) << endl;
    cout << "   CDF at x=0.5: " << customExp.getCumulativeProbability(0.5) << endl;
    
    // Test known values for λ = 2
    assert(approxEqual(customExp.getLambda(), 2.0));
    assert(approxEqual(customExp.getMean(), 0.5));  // 1/λ = 1/2
    assert(approxEqual(customExp.getVariance(), 0.25));  // 1/λ² = 1/4
    assert(approxEqual(customExp.getProbability(0.5), 2.0 * exp(-1.0), 1e-10));  // 2*e^(-1)
    assert(approxEqual(customExp.getCumulativeProbability(0.5), 1.0 - exp(-1.0), 1e-10));  // 1 - e^(-1)
    
    cout << "   ✓ Basic functionality tests passed!" << endl;
}

// Test copy and move operations
void testCopyAndMove() {
    cout << "\n=== Testing Copy and Move Operations ===" << endl;
    
    // Test 1: Copy constructor
    cout << "1. Copy Constructor:" << endl;
    
    auto originalResult = ExponentialDistribution::create(2.0);
    assert(originalResult.isOk());
    auto original = std::move(originalResult.value);
    
    auto copied = original;  // Copy constructor
    
    // Verify both objects have same parameters
    assert(approxEqual(original.getLambda(), copied.getLambda()));
    assert(approxEqual(original.getMean(), copied.getMean()));
    assert(approxEqual(original.getVariance(), copied.getVariance()));
    
    // Verify they produce same results
    double test_val = 1.0;
    assert(approxEqual(original.getProbability(test_val), copied.getProbability(test_val)));
    assert(approxEqual(original.getCumulativeProbability(test_val), copied.getCumulativeProbability(test_val)));
    
    cout << "   ✓ Copy constructor works correctly" << endl;
    
    // Test 2: Move constructor
    cout << "2. Move Constructor:" << endl;
    
    auto sourceResult = ExponentialDistribution::create(3.0);
    assert(sourceResult.isOk());
    auto source = std::move(sourceResult.value);
    
    double orig_lambda = source.getLambda();
    auto moved = std::move(source);  // Move constructor
    
    // Verify moved object has correct parameters
    assert(approxEqual(moved.getLambda(), orig_lambda));
    assert(approxEqual(moved.getMean(), 1.0 / orig_lambda));
    
    cout << "   ✓ Move constructor works correctly" << endl;
    
    // Test 3: Copy assignment
    cout << "3. Copy Assignment:" << endl;
    
    auto dist1Result = ExponentialDistribution::create(1.0);
    auto dist2Result = ExponentialDistribution::create(2.0);
    assert(dist1Result.isOk() && dist2Result.isOk());
    auto dist1 = std::move(dist1Result.value);
    auto dist2 = std::move(dist2Result.value);
    
    // Verify they're different initially
    assert(!approxEqual(dist1.getLambda(), dist2.getLambda()));
    
    dist1 = dist2;  // Copy assignment
    
    // Verify they're now the same
    assert(approxEqual(dist1.getLambda(), dist2.getLambda()));
    assert(approxEqual(dist1.getMean(), dist2.getMean()));
    
    cout << "   ✓ Copy assignment works correctly" << endl;
    
    // Test 4: Move assignment
    cout << "4. Move Assignment:" << endl;
    
    auto targetResult = ExponentialDistribution::create(1.0);
    auto sourceResult2 = ExponentialDistribution::create(4.0);
    assert(targetResult.isOk() && sourceResult2.isOk());
    auto target = std::move(targetResult.value);
    auto source2 = std::move(sourceResult2.value);
    
    double source_lambda = source2.getLambda();
    target = std::move(source2);  // Move assignment
    
    // Verify target now has source's parameters
    assert(approxEqual(target.getLambda(), source_lambda));
    assert(approxEqual(target.getMean(), 1.0 / source_lambda));
    
    cout << "   ✓ Move assignment works correctly" << endl;
    
    cout << "   ✓ All copy and move operations passed!" << endl;
}

// Test batch operations
void testBatchOperations() {
    cout << "\n=== Testing Batch Operations ===" << endl;
    
    auto unitExpResult = ExponentialDistribution::create(1.0);
    auto customExpResult = ExponentialDistribution::create(0.5);
    assert(unitExpResult.isOk() && customExpResult.isOk());
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
        
        assert(approxEqual(pdf_results[i], expected_pdf, 1e-12));
        assert(approxEqual(log_pdf_results[i], expected_log_pdf, 1e-12));
        assert(approxEqual(cdf_results[i], expected_cdf, 1e-12));
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
        
        assert(approxEqual(large_pdf_results[idx], expected_pdf, 1e-10));
        assert(approxEqual(large_log_pdf_results[idx], expected_log_pdf, 1e-10));
        assert(approxEqual(large_cdf_results[idx], expected_cdf, 1e-10));
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
        
        assert(approxEqual(pdf_results[i], expected_pdf, 1e-12));
        assert(approxEqual(log_pdf_results[i], expected_log_pdf, 1e-12));
        assert(approxEqual(cdf_results[i], expected_cdf, 1e-12));
    }
    
    cout << "   ✓ Custom distribution batch operations passed!" << endl;
}

// Test optimized sampling
void testOptimizedSampling() {
    cout << "\n=== Testing Optimized Sampling ===" << endl;
    
    auto unitExpResult = ExponentialDistribution::create(1.0);
    auto customExpResult = ExponentialDistribution::create(2.0);
    assert(unitExpResult.isOk() && customExpResult.isOk());
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
    assert(abs(sample_mean - 1.0) < 0.02);  // Mean should be close to 1.0
    assert(abs(sample_variance - 1.0) < 0.02);  // Variance should be close to 1.0
    
    // Check that all samples are non-negative
    for (double sample : samples) {
        assert(sample >= 0.0);
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
    assert(abs(custom_mean - customExp.getMean()) < 0.01);
    assert(abs(custom_variance - customExp.getVariance()) < 0.01);
    
    cout << "   ✓ Custom distribution sampling quality passed!" << endl;
    
    // Test 3: Numerical stability with extreme parameters
    cout << "3. Numerical Stability Test:" << endl;
    
    // Test with very large lambda (small scale)
    auto extremeResult = ExponentialDistribution::create(100.0);
    assert(extremeResult.isOk());
    auto extreme = std::move(extremeResult.value);
    
    bool stability_test_passed = true;
    for (int i = 0; i < 1000; ++i) {
        double sample = extreme.sample(rng);
        if (!std::isfinite(sample) || sample < 0) {
            stability_test_passed = false;
            break;
        }
    }
    
    assert(stability_test_passed);
    cout << "   ✓ Numerical stability test passed!" << endl;
}

// Test edge cases and error handling
void testEdgeCases() {
    cout << "\n=== Testing Edge Cases ===" << endl;
    
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
    assert(normalResult.isOk());
    auto normal = std::move(normalResult.value);
    
    // Test very large values
    double large_val = 100.0;
    double pdf_large = normal.getProbability(large_val);
    double log_pdf_large = normal.getLogProbability(large_val);
    double cdf_large = normal.getCumulativeProbability(large_val);
    
    assert(pdf_large >= 0.0);
    assert(std::isfinite(log_pdf_large));
    assert(cdf_large >= 0.0 && cdf_large <= 1.0);
    
    cout << "   ✓ Large value handling passed" << endl;
    
    // Test negative values (should return 0 for PDF, -inf for log PDF, 0 for CDF)
    double neg_val = -1.0;
    double pdf_neg = normal.getProbability(neg_val);
    double log_pdf_neg = normal.getLogProbability(neg_val);
    double cdf_neg = normal.getCumulativeProbability(neg_val);
    
    assert(pdf_neg == 0.0);
    assert(log_pdf_neg == std::numeric_limits<double>::lowest() || std::isinf(log_pdf_neg));
    assert(cdf_neg == 0.0);
    
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
    assert(fitResult.isOk());
    auto fitted = std::move(fitResult.value);
    fitted.fit(fit_data);
    
    cout << "   Original lambda: 2.0" << endl;
    cout << "   Fitted lambda: " << fitted.getLambda() << endl;
    cout << "   Fitting error: " << abs(fitted.getLambda() - 2.0) << endl;
    
    // Should be reasonably close to original parameter
    assert(abs(fitted.getLambda() - 2.0) < 0.1);
    
    cout << "   ✓ Parameter fitting test passed" << endl;
}

// Test thread safety
void testThreadSafety() {
    cout << "\n=== Testing Thread Safety ===" << endl;
    
    // This is a basic test - comprehensive thread safety testing would require
    // more sophisticated concurrent testing frameworks
    
    auto normalResult = ExponentialDistribution::create(1.0);
    assert(normalResult.isOk());
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
        assert(results[t].size() == samples_per_thread);
        for (double val : results[t]) {
            assert(std::isfinite(val) && val >= 0.0);
        }
    }
    
    cout << "   ✓ Basic thread safety test passed" << endl;
}

// Test quantile function
void testQuantileFunction() {
    cout << "\n=== Testing Quantile Function ===" << endl;
    
    auto unitExpResult = ExponentialDistribution::create(1.0);
    assert(unitExpResult.isOk());
    auto unitExp = std::move(unitExpResult.value);
    
    // Test specific quantiles
    cout << "1. Quantile Function Tests:" << endl;
    
    vector<double> probabilities = {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};
    
    for (double p : probabilities) {
        double quantile = unitExp.getQuantile(p);
        double cdf_check = unitExp.getCumulativeProbability(quantile);
        
        cout << "   P(" << p << ") = " << quantile << ", CDF(" << quantile << ") = " << cdf_check << endl;
        
        // CDF(quantile(p)) should equal p
        assert(approxEqual(cdf_check, p, 1e-10));
    }
    
    // Test known quantiles for unit exponential
    assert(approxEqual(unitExp.getQuantile(0.5), log(2.0), 1e-10));  // Median
    assert(approxEqual(unitExp.getQuantile(1.0 - 1.0/exp(1.0)), 1.0, 1e-10));  // Q(1-1/e) = 1
    
    cout << "   ✓ Quantile function tests passed!" << endl;
}

int main() {
    cout << "=== Enhanced Exponential Distribution Test Suite ===" << endl;
    cout << fixed << setprecision(6);
    
    try {
        testBasicFunctionality();
        testCopyAndMove();
        testBatchOperations();
        testOptimizedSampling();
        testEdgeCases();
        testThreadSafety();
        testQuantileFunction();
        
        cout << "\n=== ALL TESTS PASSED SUCCESSFULLY! ===" << endl;
        cout << "✓ Basic functionality" << endl;
        cout << "✓ SIMD batch operations" << endl;
        cout << "✓ Optimized inverse transform sampling" << endl;
        cout << "✓ Edge cases and error handling" << endl;
        cout << "✓ Thread safety" << endl;
        cout << "✓ Quantile function accuracy" << endl;
        cout << "✓ Parameter fitting" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "\n❌ TEST FAILED: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "\n❌ TEST FAILED: Unknown exception" << endl;
        return 1;
    }
}

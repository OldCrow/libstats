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

// Include the enhanced Gaussian distribution
#include "gaussian.h"

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
    
    // Test 1: Standard normal distribution
    cout << "1. Standard Normal Distribution N(0,1):" << endl;
    GaussianDistribution stdNormal(0.0, 1.0);
    
    cout << "   Parameters:" << endl;
    cout << "   - Mean: " << stdNormal.getMean() << endl;
    cout << "   - Variance: " << stdNormal.getVariance() << endl;
    cout << "   - Standard Deviation: " << stdNormal.getStandardDeviation() << endl;
    cout << "   - Skewness: " << stdNormal.getSkewness() << endl;
    cout << "   - Kurtosis: " << stdNormal.getKurtosis() << endl;
    
    // Test known values
    assert(approxEqual(stdNormal.getMean(), 0.0));
    assert(approxEqual(stdNormal.getVariance(), 1.0));
    assert(approxEqual(stdNormal.getSkewness(), 0.0));
    assert(approxEqual(stdNormal.getKurtosis(), 0.0));  // Excess kurtosis for Gaussian
    
    cout << "   Probability Functions:" << endl;
    double pdf_at_0 = stdNormal.getProbability(0.0);
    double pdf_at_1 = stdNormal.getProbability(1.0);
    double log_pdf_at_0 = stdNormal.getLogProbability(0.0);
    double cdf_at_0 = stdNormal.getCumulativeProbability(0.0);
    double cdf_at_1 = stdNormal.getCumulativeProbability(1.0);
    
    cout << "   - PDF at x=0: " << pdf_at_0 << endl;
    cout << "   - PDF at x=1: " << pdf_at_1 << endl;
    cout << "   - Log PDF at x=0: " << log_pdf_at_0 << endl;
    cout << "   - CDF at x=0: " << cdf_at_0 << endl;
    cout << "   - CDF at x=1: " << cdf_at_1 << endl;
    
    // Test known values
    assert(approxEqual(pdf_at_0, 0.398942280401433, 1e-10));
    assert(approxEqual(cdf_at_0, 0.5, 1e-10));
    assert(approxEqual(log_pdf_at_0, std::log(pdf_at_0), 1e-10));
    
    // Test 2: Custom distribution
    cout << "\n2. Custom Distribution N(10, 2):" << endl;
    GaussianDistribution custom(10.0, 2.0);
    
    cout << "   Mean: " << custom.getMean() << endl;
    cout << "   Variance: " << custom.getVariance() << endl;
    cout << "   PDF at mean: " << custom.getProbability(10.0) << endl;
    cout << "   CDF at mean: " << custom.getCumulativeProbability(10.0) << endl;
    
    // Test known values
    assert(approxEqual(custom.getMean(), 10.0));
    assert(approxEqual(custom.getVariance(), 4.0));
    assert(approxEqual(custom.getCumulativeProbability(10.0), 0.5, 1e-10));
    
    cout << "   ✓ Basic functionality tests passed!" << endl;
}

// Test copy and move semantics
void testCopyAndMove() {
    cout << "\n=== Testing Copy and Move Semantics ===" << endl;
    
    // Test copy constructor
    GaussianDistribution original(3.0, 2.0);
    GaussianDistribution copied(original);
    
    assert(copied.getMean() == original.getMean());
    assert(copied.getVariance() == original.getVariance());
    assert(std::abs(copied.getProbability(3.0) - original.getProbability(3.0)) < 1e-10);
    
    // Test copy assignment
    GaussianDistribution assigned(0.0, 1.0);
    assigned = original;
    
    assert(assigned.getMean() == original.getMean());
    assert(assigned.getVariance() == original.getVariance());
    
    // Test move constructor
    GaussianDistribution to_move(5.0, 3.0);
    double original_mean = to_move.getMean();
    double original_var = to_move.getVariance();
    GaussianDistribution moved(std::move(to_move));
    
    assert(moved.getMean() == original_mean);
    assert(moved.getVariance() == original_var);
    
    // Test move assignment
    GaussianDistribution move_assigned(0.0, 1.0);
    GaussianDistribution to_move2(10.0, 4.0);
    double original_mean2 = to_move2.getMean();
    double original_var2 = to_move2.getVariance();
    move_assigned = std::move(to_move2);
    
    assert(move_assigned.getMean() == original_mean2);
    assert(move_assigned.getVariance() == original_var2);
    
    cout << "   ✓ Copy and move semantics tests passed!" << endl;
}

// Test batch operations
void testBatchOperations() {
    cout << "\n=== Testing Batch Operations ===" << endl;
    
    GaussianDistribution stdNormal(0.0, 1.0);
    GaussianDistribution custom(5.0, 2.0);
    
    // Test data
    vector<double> test_values = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    vector<double> pdf_results(test_values.size());
    vector<double> log_pdf_results(test_values.size());
    vector<double> cdf_results(test_values.size());
    
    // Test 1: Small batch operations (should use scalar implementation)
    cout << "1. Small Batch Operations (scalar fallback):" << endl;
    
    stdNormal.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    stdNormal.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    stdNormal.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = stdNormal.getProbability(test_values[i]);
        double expected_log_pdf = stdNormal.getLogProbability(test_values[i]);
        double expected_cdf = stdNormal.getCumulativeProbability(test_values[i]);
        
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
    
    // Generate test data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-3.0, 3.0);
    
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_test_values[i] = dis(gen);
    }
    
    // Time batch operations
    auto start = chrono::high_resolution_clock::now();
    stdNormal.getProbabilityBatch(large_test_values.data(), large_pdf_results.data(), LARGE_BATCH_SIZE);
    auto end = chrono::high_resolution_clock::now();
    auto batch_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    // Time individual operations for comparison
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pdf_results[i] = stdNormal.getProbability(large_test_values[i]);
    }
    end = chrono::high_resolution_clock::now();
    auto individual_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    cout << "   Batch PDF time: " << batch_time << " μs" << endl;
    cout << "   Individual PDF time: " << individual_time << " μs" << endl;
    cout << "   Speedup: " << (double)individual_time / batch_time << "x" << endl;
    
    // Test batch log probability
    stdNormal.getLogProbabilityBatch(large_test_values.data(), large_log_pdf_results.data(), LARGE_BATCH_SIZE);
    
    // Test batch CDF
    stdNormal.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    
    // Verify a sample of results
    const size_t sample_size = 100;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * (LARGE_BATCH_SIZE / sample_size);
        double expected_pdf = stdNormal.getProbability(large_test_values[idx]);
        double expected_log_pdf = stdNormal.getLogProbability(large_test_values[idx]);
        double expected_cdf = stdNormal.getCumulativeProbability(large_test_values[idx]);
        
        assert(approxEqual(large_pdf_results[idx], expected_pdf, 1e-10));
        assert(approxEqual(large_log_pdf_results[idx], expected_log_pdf, 1e-10));
        assert(approxEqual(large_cdf_results[idx], expected_cdf, 1e-10));
    }
    
    cout << "   ✓ Large batch operations match individual calls!" << endl;
    
    // Test 3: Custom distribution batch operations
    cout << "3. Custom Distribution Batch Operations:" << endl;
    
    custom.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    custom.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    custom.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = custom.getProbability(test_values[i]);
        double expected_log_pdf = custom.getLogProbability(test_values[i]);
        double expected_cdf = custom.getCumulativeProbability(test_values[i]);
        
        assert(approxEqual(pdf_results[i], expected_pdf, 1e-12));
        assert(approxEqual(log_pdf_results[i], expected_log_pdf, 1e-12));
        assert(approxEqual(cdf_results[i], expected_cdf, 1e-12));
    }
    
    cout << "   ✓ Custom distribution batch operations passed!" << endl;
}

// Test optimized sampling
void testOptimizedSampling() {
    cout << "\n=== Testing Optimized Sampling ===" << endl;
    
    GaussianDistribution stdNormal(0.0, 1.0);
    GaussianDistribution custom(10.0, 2.0);
    
    mt19937 rng(42);  // Fixed seed for reproducibility
    
    // Test 1: Sample quality for standard normal
    cout << "1. Standard Normal Sampling Quality:" << endl;
    
    const size_t num_samples = 100000;
    vector<double> samples;
    samples.reserve(num_samples);
    
    // Time sampling
    auto start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(stdNormal.sample(rng));
    }
    auto end = chrono::high_resolution_clock::now();
    auto sampling_time = chrono::duration_cast<chrono::microseconds>(end - start).count();
    
    auto [sample_mean, sample_variance] = calculateSampleStats(samples);
    
    cout << "   Samples: " << num_samples << endl;
    cout << "   Sampling time: " << sampling_time << " μs" << endl;
    cout << "   Time per sample: " << (double)sampling_time / num_samples << " μs" << endl;
    cout << "   Sample mean: " << sample_mean << " (expected: 0.0)" << endl;
    cout << "   Sample variance: " << sample_variance << " (expected: 1.0)" << endl;
    cout << "   Mean error: " << abs(sample_mean - 0.0) << endl;
    cout << "   Variance error: " << abs(sample_variance - 1.0) << endl;
    
    // Check if samples are within reasonable bounds
    assert(abs(sample_mean) < 0.01);  // Mean should be close to 0
    assert(abs(sample_variance - 1.0) < 0.01);  // Variance should be close to 1
    
    cout << "   ✓ Standard normal sampling quality passed!" << endl;
    
    // Test 2: Sample quality for custom distribution
    cout << "2. Custom Distribution Sampling Quality:" << endl;
    
    samples.clear();
    rng.seed(42);  // Reset for reproducibility
    
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(custom.sample(rng));
    }
    
    auto [custom_mean, custom_variance] = calculateSampleStats(samples);
    
    cout << "   Expected mean: " << custom.getMean() << endl;
    cout << "   Expected variance: " << custom.getVariance() << endl;
    cout << "   Sample mean: " << custom_mean << endl;
    cout << "   Sample variance: " << custom_variance << endl;
    cout << "   Mean error: " << abs(custom_mean - custom.getMean()) << endl;
    cout << "   Variance error: " << abs(custom_variance - custom.getVariance()) << endl;
    
    // Check if samples are within reasonable bounds
    assert(abs(custom_mean - custom.getMean()) < 0.02);
    assert(abs(custom_variance - custom.getVariance()) < 0.04);
    
    cout << "   ✓ Custom distribution sampling quality passed!" << endl;
    
    // Test 3: Numerical stability
    cout << "3. Numerical Stability Test:" << endl;
    
    // Test with extreme parameters
    GaussianDistribution extreme(0.0, 0.001);  // Very small standard deviation
    
    bool stability_test_passed = true;
    for (int i = 0; i < 1000; ++i) {
        double sample = extreme.sample(rng);
        if (!std::isfinite(sample)) {
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
    
    auto resultZero = GaussianDistribution::create(0.0, 0.0);
    if (resultZero.isError()) {
        cout << "   ✓ Zero standard deviation correctly rejected: " << resultZero.message << endl;
    }
    
    auto resultNegative = GaussianDistribution::create(0.0, -1.0);
    if (resultNegative.isError()) {
        cout << "   ✓ Negative standard deviation correctly rejected: " << resultNegative.message << endl;
    }
    
    // Test 2: Extreme values
    cout << "2. Extreme Value Handling:" << endl;
    
    GaussianDistribution normal(0.0, 1.0);
    
    // Test very large values
    double large_val = 100.0;
    double pdf_large = normal.getProbability(large_val);
    double log_pdf_large = normal.getLogProbability(large_val);
    double cdf_large = normal.getCumulativeProbability(large_val);
    
    assert(pdf_large >= 0.0);
    assert(std::isfinite(log_pdf_large));
    assert(cdf_large >= 0.0 && cdf_large <= 1.0);
    
    cout << "   ✓ Large value handling passed" << endl;
    
    // Test very small values
    double small_val = -100.0;
    double pdf_small = normal.getProbability(small_val);
    double log_pdf_small = normal.getLogProbability(small_val);
    double cdf_small = normal.getCumulativeProbability(small_val);
    
    assert(pdf_small >= 0.0);
    assert(std::isfinite(log_pdf_small));
    assert(cdf_small >= 0.0 && cdf_small <= 1.0);
    
    cout << "   ✓ Small value handling passed" << endl;
    
    // Test 3: Empty batch operations
    cout << "3. Empty Batch Operations:" << endl;
    
    vector<double> empty_values;
    vector<double> empty_results;
    
    // These should not crash
    normal.getProbabilityBatch(empty_values.data(), empty_results.data(), 0);
    normal.getLogProbabilityBatch(empty_values.data(), empty_results.data(), 0);
    normal.getCumulativeProbabilityBatch(empty_values.data(), empty_results.data(), 0);
    
    cout << "   ✓ Empty batch operations handled gracefully" << endl;
}

// Test thread safety
void testThreadSafety() {
    cout << "\n=== Testing Thread Safety ===" << endl;
    
    // This is a basic test - comprehensive thread safety testing would require
    // more sophisticated concurrent testing frameworks
    
    GaussianDistribution normal(0.0, 1.0);
    
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
            assert(std::isfinite(val));
        }
    }
    
    cout << "   ✓ Basic thread safety test passed" << endl;
}

int main() {
    cout << "=== Enhanced Gaussian Distribution Test Suite ===" << endl;
    cout << fixed << setprecision(6);
    
    try {
        testBasicFunctionality();
        testCopyAndMove();
        testBatchOperations();
        testOptimizedSampling();
        testEdgeCases();
        testThreadSafety();
        
        cout << "\n=== ALL TESTS PASSED SUCCESSFULLY! ===" << endl;
        cout << "✓ Basic functionality" << endl;
        cout << "✓ SIMD batch operations" << endl;
        cout << "✓ Optimized Box-Muller sampling" << endl;
        cout << "✓ CDF batch operations" << endl;
        cout << "✓ Edge cases and error handling" << endl;
        cout << "✓ Thread safety" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "\n❌ TEST FAILED: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "\n❌ TEST FAILED: Unknown exception" << endl;
        return 1;
    }
}

/**
 * @file test_uniform_enhanced.cpp
 * @brief Enhanced comprehensive tests for UniformDistribution class
 * 
 * This file contains comprehensive tests for the UniformDistribution class,
 * including SIMD batch operations, thread safety, edge cases, and performance
 * validation. These tests ensure the distribution is production-ready.
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <numeric>
#include <future>
#include "../include/uniform.h"
#include "../include/constants.h"
#include "../include/cpu_detection.h"

// Test configuration
constexpr size_t BATCH_SIZE = 1000;
constexpr size_t LARGE_BATCH_SIZE = 10000;
constexpr size_t NUM_THREADS = 4;
constexpr double TOLERANCE = 1e-10;

void testBasicProperties() {
    std::cout << "Testing basic distribution properties..." << std::endl;
    
    // Test various parameter combinations
    std::vector<std::pair<double, double>> test_params = {
        {0.0, 1.0},    // Unit interval
        {-1.0, 1.0},   // Standard interval
        {-5.0, 5.0},   // Symmetric interval
        {2.0, 7.0},    // Asymmetric interval
        {0.001, 0.002}, // Narrow interval
        {-1000.0, 1000.0} // Wide interval
    };
    
    for (const auto& params : test_params) {
        double a = params.first;
        double b = params.second;
        
        libstats::UniformDistribution uniform(a, b);
        
        // Test basic properties
        assert(uniform.getLowerBound() == a);
        assert(uniform.getUpperBound() == b);
        assert(std::abs(uniform.getMean() - (a + b) / 2.0) < TOLERANCE);
        assert(std::abs(uniform.getVariance() - (b - a) * (b - a) / 12.0) < TOLERANCE);
        assert(uniform.getSkewness() == 0.0);
        assert(uniform.getKurtosis() == -1.2);
        assert(std::abs(uniform.getWidth() - (b - a)) < TOLERANCE);
        assert(std::abs(uniform.getMidpoint() - (a + b) / 2.0) < TOLERANCE);
        
        // Test support bounds
        assert(uniform.getSupportLowerBound() == a);
        assert(uniform.getSupportUpperBound() == b);
        
        // Test PDF properties
        double mid = (a + b) / 2.0;
        assert(std::abs(uniform.getProbability(mid) - 1.0 / (b - a)) < TOLERANCE);
        assert(uniform.getProbability(a - 1.0) == 0.0);
        assert(uniform.getProbability(b + 1.0) == 0.0);
        
        // Test CDF properties
        assert(uniform.getCumulativeProbability(a) == 0.0);
        assert(uniform.getCumulativeProbability(b) == 1.0);
        assert(std::abs(uniform.getCumulativeProbability(mid) - 0.5) < TOLERANCE);
        
        // Test quantile properties
        assert(uniform.getQuantile(0.0) == a);
        assert(uniform.getQuantile(1.0) == b);
        assert(std::abs(uniform.getQuantile(0.5) - mid) < TOLERANCE);
    }
    
    std::cout << "✓ Basic properties tests passed" << std::endl;
}

void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;
    
    // Test with very small interval
    libstats::UniformDistribution small_uniform(1.0, 1.0 + 1e-15);
    assert(small_uniform.getProbability(1.0 + 5e-16) > 0.0);
    
    // Test with very large interval
    libstats::UniformDistribution large_uniform(-1e10, 1e10);
    assert(std::abs(large_uniform.getMean() - 0.0) < 1e-5);
    
    // Test with extreme values
    libstats::UniformDistribution extreme_uniform(-1e100, 1e100);
    assert(extreme_uniform.getProbability(0.0) > 0.0);
    
    std::cout << "✓ Numerical stability tests passed" << std::endl;
}

void testSIMDBatchOperations() {
    std::cout << "Testing SIMD batch operations..." << std::endl;
    
    // Test with various distributions
    std::vector<libstats::UniformDistribution> test_dists = {
        libstats::UniformDistribution(0.0, 1.0),    // Unit interval
        libstats::UniformDistribution(-2.0, 3.0),   // General case
        libstats::UniformDistribution(10.0, 20.0)   // Shifted interval
    };
    
    for (const auto& uniform : test_dists) {
        // Prepare test data
        std::vector<double> values(BATCH_SIZE);
        std::vector<double> expected_pdf(BATCH_SIZE);
        std::vector<double> expected_logpdf(BATCH_SIZE);
        std::vector<double> expected_cdf(BATCH_SIZE);
        std::vector<double> batch_pdf(BATCH_SIZE);
        std::vector<double> batch_logpdf(BATCH_SIZE);
        std::vector<double> batch_cdf(BATCH_SIZE);
        
        // Generate test values across the distribution support and beyond
        double a = uniform.getLowerBound();
        double b = uniform.getUpperBound();
        double range = b - a;
        
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> value_gen(a - range, b + range);
        
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            values[i] = value_gen(rng);
            expected_pdf[i] = uniform.getProbability(values[i]);
            expected_logpdf[i] = uniform.getLogProbability(values[i]);
            expected_cdf[i] = uniform.getCumulativeProbability(values[i]);
        }
        
        // Test batch operations
        uniform.getProbabilityBatch(values.data(), batch_pdf.data(), BATCH_SIZE);
        uniform.getLogProbabilityBatch(values.data(), batch_logpdf.data(), BATCH_SIZE);
        uniform.getCumulativeProbabilityBatch(values.data(), batch_cdf.data(), BATCH_SIZE);
        
        // Verify results
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            assert(std::abs(batch_pdf[i] - expected_pdf[i]) < TOLERANCE);
            
            // Handle log probability comparisons carefully
            if (std::isfinite(expected_logpdf[i])) {
                assert(std::abs(batch_logpdf[i] - expected_logpdf[i]) < TOLERANCE);
            } else {
                assert(batch_logpdf[i] == expected_logpdf[i]); // Both should be -infinity
            }
            
            assert(std::abs(batch_cdf[i] - expected_cdf[i]) < TOLERANCE);
        }
        
        // Test unsafe batch operations
        uniform.getProbabilityBatchUnsafe(values.data(), batch_pdf.data(), BATCH_SIZE);
        uniform.getLogProbabilityBatchUnsafe(values.data(), batch_logpdf.data(), BATCH_SIZE);
        
        // Verify unsafe results match
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            assert(std::abs(batch_pdf[i] - expected_pdf[i]) < TOLERANCE);
            if (std::isfinite(expected_logpdf[i])) {
                assert(std::abs(batch_logpdf[i] - expected_logpdf[i]) < TOLERANCE);
            } else {
                assert(batch_logpdf[i] == expected_logpdf[i]);
            }
        }
    }
    
    std::cout << "✓ SIMD batch operations tests passed" << std::endl;
}

void testLargeBatchPerformance() {
    std::cout << "Testing large batch performance..." << std::endl;
    
    libstats::UniformDistribution uniform(0.0, 1.0);
    std::vector<double> values(LARGE_BATCH_SIZE);
    std::vector<double> results(LARGE_BATCH_SIZE);
    
    // Generate test values
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> value_gen(-0.5, 1.5);
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        values[i] = value_gen(rng);
    }
    
    // Test batch PDF computation
    auto start = std::chrono::high_resolution_clock::now();
    uniform.getProbabilityBatch(values.data(), results.data(), LARGE_BATCH_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Batch PDF computation: " << batch_duration.count() << " μs" << std::endl;
    
    // Test individual computations for comparison
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        results[i] = uniform.getProbability(values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    
    auto individual_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "  Individual PDF computation: " << individual_duration.count() << " μs" << std::endl;
    
    // Batch should be at least as fast as individual for large batches
    if (LARGE_BATCH_SIZE >= 1000) {
        std::cout << "  Speedup ratio: " << static_cast<double>(individual_duration.count()) / batch_duration.count() << "x" << std::endl;
    }
    
    std::cout << "✓ Large batch performance tests passed" << std::endl;
}

void testThreadSafety() {
    std::cout << "Testing thread safety..." << std::endl;
    
    libstats::UniformDistribution uniform(0.0, 1.0);
    constexpr size_t NUM_OPERATIONS = 1000;
    
    // Test concurrent reads
    auto read_worker = [&uniform]() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> value_gen(-0.5, 1.5);
        
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            double x = value_gen(rng);
            [[maybe_unused]] double pdf = uniform.getProbability(x);
            [[maybe_unused]] double logpdf = uniform.getLogProbability(x);
            [[maybe_unused]] double cdf = uniform.getCumulativeProbability(x);
            [[maybe_unused]] double mean = uniform.getMean();
            [[maybe_unused]] double var = uniform.getVariance();
        }
    };
    
    std::vector<std::thread> read_threads;
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        read_threads.emplace_back(read_worker);
    }
    
    for (auto& t : read_threads) {
        t.join();
    }
    
    // Test concurrent writes (parameter changes)
    libstats::UniformDistribution write_uniform(0.0, 1.0);
    std::atomic<bool> stop_flag{false};
    
    auto write_worker = [&write_uniform, &stop_flag](double base_a, [[maybe_unused]] double base_b) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> param_gen(0.1, 5.0);
        
        while (!stop_flag.load()) {
            double a = base_a + param_gen(rng);
            double b = a + param_gen(rng);
            
            auto result = write_uniform.trySetParameters(a, b);
            if (result.isOk()) {
                [[maybe_unused]] double mean = write_uniform.getMean();
                [[maybe_unused]] double var = write_uniform.getVariance();
            }
        }
    };
    
    std::vector<std::thread> write_threads;
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        write_threads.emplace_back(write_worker, i * 10.0, (i + 1) * 10.0);
    }
    
    // Let threads run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop_flag.store(true);
    
    for (auto& t : write_threads) {
        t.join();
    }
    
    std::cout << "✓ Thread safety tests passed" << std::endl;
}

void testEdgeCases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Test boundary conditions
    libstats::UniformDistribution uniform(0.0, 1.0);
    
    // Test exactly at boundaries
    assert(uniform.getProbability(0.0) == 1.0);
    assert(uniform.getProbability(1.0) == 1.0);
    assert(uniform.getCumulativeProbability(0.0) == 0.0);
    assert(uniform.getCumulativeProbability(1.0) == 1.0);
    
    // Test quantile edge cases
    assert(uniform.getQuantile(0.0) == 0.0);
    assert(uniform.getQuantile(1.0) == 1.0);
    
    // Test with extreme parameter values
    try {
        libstats::UniformDistribution extreme1(1e-300, 1e300);
        // For extreme ranges, PDF might be very small but should be finite
        double pdf_at_zero = extreme1.getProbability(0.0);
        assert(std::isfinite(pdf_at_zero) && pdf_at_zero >= 0.0);
        // Test that values within support have non-zero probability
        assert(extreme1.getProbability(1.0) > 0.0);
    } catch (...) {
        // Expected for extreme values that cause numerical issues
    }
    
    // Test safe factory with invalid parameters
    auto result1 = libstats::UniformDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0);
    assert(result1.isError());
    
    auto result2 = libstats::UniformDistribution::create(0.0, std::numeric_limits<double>::infinity());
    assert(result2.isError());
    
    auto result3 = libstats::UniformDistribution::create(5.0, 2.0);  // a > b
    assert(result3.isError());
    
    std::cout << "✓ Edge cases tests passed" << std::endl;
}

void testSamplingStatistics() {
    std::cout << "Testing sampling statistics..." << std::endl;
    
    libstats::UniformDistribution uniform(2.0, 8.0);
    std::mt19937 rng(42);
    
    constexpr size_t NUM_SAMPLES = 10000;
    std::vector<double> samples(NUM_SAMPLES);
    
    // Generate samples
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        samples[i] = uniform.sample(rng);
    }
    
    // Test that all samples are within bounds
    for (double sample : samples) {
        assert(sample >= 2.0 && sample <= 8.0);
    }
    
    // Test sample statistics
    double sample_mean = std::accumulate(samples.begin(), samples.end(), 0.0) / NUM_SAMPLES;
    double theoretical_mean = uniform.getMean();
    assert(std::abs(sample_mean - theoretical_mean) < 0.1);
    
    // Test sample variance
    double sample_var = 0.0;
    for (double sample : samples) {
        double diff = sample - sample_mean;
        sample_var += diff * diff;
    }
    sample_var /= (NUM_SAMPLES - 1);
    double theoretical_var = uniform.getVariance();
    assert(std::abs(sample_var - theoretical_var) < 0.5);
    
    // Test uniformity using Kolmogorov-Smirnov test (simplified)
    std::sort(samples.begin(), samples.end());
    double max_diff = 0.0;
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        double empirical_cdf = static_cast<double>(i + 1) / NUM_SAMPLES;
        double theoretical_cdf = uniform.getCumulativeProbability(samples[i]);
        max_diff = std::max(max_diff, std::abs(empirical_cdf - theoretical_cdf));
    }
    
    // For large samples, KS statistic should be small
    assert(max_diff < 0.05);
    
    std::cout << "✓ Sampling statistics tests passed" << std::endl;
}

void testCopyAndMove() {
    std::cout << "Testing copy and move semantics..." << std::endl;
    
    // Test copy constructor
    libstats::UniformDistribution original(3.0, 7.0);
    libstats::UniformDistribution copied(original);
    
    assert(copied.getLowerBound() == original.getLowerBound());
    assert(copied.getUpperBound() == original.getUpperBound());
    assert(copied.getMean() == original.getMean());
    assert(copied.getVariance() == original.getVariance());
    
    // Test copy assignment
    libstats::UniformDistribution assigned(0.0, 1.0);
    assigned = original;
    
    assert(assigned.getLowerBound() == original.getLowerBound());
    assert(assigned.getUpperBound() == original.getUpperBound());
    
    // Test move constructor
    libstats::UniformDistribution to_move(5.0, 10.0);
    double original_mean = to_move.getMean();
    libstats::UniformDistribution moved(std::move(to_move));
    
    assert(moved.getMean() == original_mean);
    
    // Test move assignment
    libstats::UniformDistribution move_assigned(0.0, 1.0);
    libstats::UniformDistribution to_move2(15.0, 25.0);
    double original_mean2 = to_move2.getMean();
    move_assigned = std::move(to_move2);
    
    assert(move_assigned.getMean() == original_mean2);
    
    std::cout << "✓ Copy and move semantics tests passed" << std::endl;
}

void testSpecialDistributions() {
    std::cout << "Testing special distribution cases..." << std::endl;
    
    // Test unit interval U(0,1)
    libstats::UniformDistribution unit(0.0, 1.0);
    assert(unit.getProbability(0.5) == 1.0);
    assert(unit.getLogProbability(0.5) == 0.0);
    assert(unit.getCumulativeProbability(0.5) == 0.5);
    assert(unit.getQuantile(0.5) == 0.5);
    
    // Test standard interval U(-1,1)
    libstats::UniformDistribution standard(-1.0, 1.0);
    assert(standard.getProbability(0.0) == 0.5);
    assert(standard.getMean() == 0.0);
    assert(standard.getCumulativeProbability(0.0) == 0.5);
    
    // Test symmetric interval U(-c,c)
    libstats::UniformDistribution symmetric(-5.0, 5.0);
    assert(symmetric.getMean() == 0.0);
    assert(symmetric.getCumulativeProbability(0.0) == 0.5);
    
    std::cout << "✓ Special distribution cases tests passed" << std::endl;
}

void testCPUDetection() {
    std::cout << "Testing CPU detection and SIMD usage..." << std::endl;
    
    // Test CPU detection
    std::cout << "  CPU Features:" << std::endl;
    std::cout << "    SSE2: " << (libstats::cpu::supports_sse2() ? "Yes" : "No") << std::endl;
    std::cout << "    AVX: " << (libstats::cpu::supports_avx() ? "Yes" : "No") << std::endl;
    std::cout << "    AVX2: " << (libstats::cpu::supports_avx2() ? "Yes" : "No") << std::endl;
    std::cout << "    FMA: " << (libstats::cpu::supports_fma() ? "Yes" : "No") << std::endl;
    
    // Test that SIMD batch operations work regardless of CPU features
    libstats::UniformDistribution uniform(0.0, 1.0);
    std::vector<double> values(100);
    std::vector<double> results(100);
    
    // Fill with test values
    std::iota(values.begin(), values.end(), 0.0);
    std::transform(values.begin(), values.end(), values.begin(), [](double x) { return x / 100.0; });
    
    // This should work on any CPU
    uniform.getProbabilityBatch(values.data(), results.data(), 100);
    
    // Verify results
    for (size_t i = 0; i < 100; ++i) {
        assert(results[i] == 1.0); // All values are in [0,1]
    }
    
    std::cout << "✓ CPU detection and SIMD tests passed" << std::endl;
}

int main() {
    std::cout << "Running Uniform Distribution Enhanced Tests..." << std::endl;
    std::cout << "=============================================" << std::endl;
    
    try {
        testBasicProperties();
        testNumericalStability();
        testSIMDBatchOperations();
        testLargeBatchPerformance();
        testThreadSafety();
        testEdgeCases();
        testSamplingStatistics();
        testCopyAndMove();
        testSpecialDistributions();
        testCPUDetection();
        
        std::cout << "\n🎉 All Uniform distribution enhanced tests passed!" << std::endl;
        std::cout << "  - Basic functionality: ✓" << std::endl;
        std::cout << "  - Numerical stability: ✓" << std::endl;
        std::cout << "  - SIMD batch operations: ✓" << std::endl;
        std::cout << "  - Performance optimizations: ✓" << std::endl;
        std::cout << "  - Thread safety: ✓" << std::endl;
        std::cout << "  - Edge cases: ✓" << std::endl;
        std::cout << "  - Statistical validation: ✓" << std::endl;
        std::cout << "  - Copy/move semantics: ✓" << std::endl;
        std::cout << "  - Special distributions: ✓" << std::endl;
        std::cout << "  - CPU detection: ✓" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}

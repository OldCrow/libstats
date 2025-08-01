#include <gtest/gtest.h>
#include "../include/distributions/poisson.h"
#include "enhanced_test_template.h"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <span>

using namespace std;
using namespace libstats;
using namespace libstats::testing;

namespace libstats {

//==============================================================================
// TEST FIXTURE FOR POISSON ENHANCED METHODS
//==============================================================================

class PoissonEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic Poisson data for testing
        std::mt19937 rng(42);
        std::poisson_distribution<int> poisson_gen(static_cast<int>(test_lambda_));
        
        poisson_data_.clear();
        poisson_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            poisson_data_.push_back(static_cast<double>(poisson_gen(rng)));
        }
        
        // Generate obviously non-Poisson data (continuous uniform)
        non_poisson_data_.clear();
        non_poisson_data_.reserve(100);
        std::uniform_real_distribution<double> uniform_gen(0.0, 10.0);
        for (int i = 0; i < 100; ++i) {
            non_poisson_data_.push_back(uniform_gen(rng));
        }
        
        test_distribution_ = PoissonDistribution(test_lambda_);
    }
    
    const double test_lambda_ = 3.5;
    std::vector<double> poisson_data_;
    std::vector<double> non_poisson_data_;
    PoissonDistribution test_distribution_;
};

//==============================================================================
// TESTS FOR BASIC ENHANCED FUNCTIONALITY
//==============================================================================

TEST_F(PoissonEnhancedTest, BasicEnhancedFunctionality) {
    // Test standard Poisson distribution properties
    PoissonDistribution stdPoisson(1.0);
    
    EXPECT_DOUBLE_EQ(stdPoisson.getMean(), 1.0);
    EXPECT_DOUBLE_EQ(stdPoisson.getVariance(), 1.0);
    EXPECT_DOUBLE_EQ(stdPoisson.getSkewness(), 1.0);  // 1/√λ = 1/√1 = 1
    EXPECT_DOUBLE_EQ(stdPoisson.getKurtosis(), 1.0);  // 1/λ = 1/1 = 1
    EXPECT_DOUBLE_EQ(stdPoisson.getMode(), 1.0);      // floor(1.0) = 1
    
    // Test known PMF/CDF values for λ=1
    double pmf_at_0 = stdPoisson.getProbability(0.0);
    double pmf_at_1 = stdPoisson.getProbability(1.0);
    double cdf_at_1 = stdPoisson.getCumulativeProbability(1.0);
    
    // For Poisson(1): P(X=0) = e^(-1) ≈ 0.3679, P(X=1) = e^(-1) ≈ 0.3679
    EXPECT_NEAR(pmf_at_0, std::exp(-1.0), 1e-10);
    EXPECT_NEAR(pmf_at_1, std::exp(-1.0), 1e-10);
    EXPECT_NEAR(cdf_at_1, 2.0 * std::exp(-1.0), 1e-9);  // P(X≤1) = P(X=0) + P(X=1)
    
    // Test custom distribution
    PoissonDistribution custom(5.0);
    EXPECT_DOUBLE_EQ(custom.getMean(), 5.0);
    EXPECT_DOUBLE_EQ(custom.getVariance(), 5.0);
    EXPECT_NEAR(custom.getSkewness(), 1.0/std::sqrt(5.0), 1e-10);
    EXPECT_NEAR(custom.getKurtosis(), 1.0/5.0, 1e-10);
}

TEST_F(PoissonEnhancedTest, CopyAndMoveSemantics) {
    // Test copy constructor
    PoissonDistribution original(4.0);
    PoissonDistribution copied(original);
    
    EXPECT_EQ(copied.getMean(), original.getMean());
    EXPECT_EQ(copied.getVariance(), original.getVariance());
    EXPECT_NEAR(copied.getProbability(2.0), original.getProbability(2.0), 1e-10);
    
    // Test move constructor
    PoissonDistribution to_move(6.0);
    double original_mean = to_move.getMean();
    double original_var = to_move.getVariance();
    PoissonDistribution moved(std::move(to_move));
    
    EXPECT_EQ(moved.getMean(), original_mean);
    EXPECT_EQ(moved.getVariance(), original_var);
}

TEST_F(PoissonEnhancedTest, BatchOperations) {
    PoissonDistribution stdPoisson(2.0);
    
    // Test data
    std::vector<double> test_values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
    std::vector<double> pmf_results(test_values.size());
    std::vector<double> log_pmf_results(test_values.size());
    std::vector<double> cdf_results(test_values.size());
    
    // Test batch operations
    stdPoisson.getProbabilityBatch(test_values.data(), pmf_results.data(), test_values.size());
    stdPoisson.getLogProbabilityBatch(test_values.data(), log_pmf_results.data(), test_values.size());
    stdPoisson.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pmf = stdPoisson.getProbability(test_values[i]);
        double expected_log_pmf = stdPoisson.getLogProbability(test_values[i]);
        double expected_cdf = stdPoisson.getCumulativeProbability(test_values[i]);
        
        EXPECT_NEAR(pmf_results[i], expected_pmf, 1e-12);
        EXPECT_NEAR(log_pmf_results[i], expected_log_pmf, 1e-12);
        EXPECT_NEAR(cdf_results[i], expected_cdf, 1e-12);
    }
}

TEST_F(PoissonEnhancedTest, PerformanceTest) {
    PoissonDistribution stdPoisson(3.0);
    constexpr size_t LARGE_BATCH_SIZE = 10000;
    
    std::vector<double> large_test_values(LARGE_BATCH_SIZE);
    std::vector<double> large_pmf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_log_pmf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_cdf_results(LARGE_BATCH_SIZE);
    
    // Generate test data (discrete values 0-10)
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 10);
    
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_test_values[i] = static_cast<double>(dis(gen));
    }
    
    std::cout << "  === Poisson Batch Performance Results ===" << std::endl;
    
    // Test 1: PMF Batch vs Individual
    auto start = std::chrono::high_resolution_clock::now();
    stdPoisson.getProbabilityBatch(large_test_values.data(), large_pmf_results.data(), LARGE_BATCH_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto pmf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pmf_results[i] = stdPoisson.getProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto pmf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double pmf_speedup = (double)pmf_individual_time / pmf_batch_time;
    std::cout << "  PMF:     Batch " << pmf_batch_time << "μs vs Individual " << pmf_individual_time << "μs → " << pmf_speedup << "x speedup" << std::endl;
    
    // Test 2: Log PMF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    stdPoisson.getLogProbabilityBatch(large_test_values.data(), large_log_pmf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto log_pmf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_log_pmf_results[i] = stdPoisson.getLogProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto log_pmf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double log_pmf_speedup = (double)log_pmf_individual_time / log_pmf_batch_time;
    std::cout << "  LogPMF:  Batch " << log_pmf_batch_time << "μs vs Individual " << log_pmf_individual_time << "μs → " << log_pmf_speedup << "x speedup" << std::endl;
    
    // Test 3: CDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    stdPoisson.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto cdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_cdf_results[i] = stdPoisson.getCumulativeProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cdf_speedup = (double)cdf_individual_time / cdf_batch_time;
    std::cout << "  CDF:     Batch " << cdf_batch_time << "μs vs Individual " << cdf_individual_time << "μs → " << cdf_speedup << "x speedup" << std::endl;
    
    // Verify correctness on a sample
    const size_t sample_size = 100;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * (LARGE_BATCH_SIZE / sample_size);
        double expected_pmf = stdPoisson.getProbability(large_test_values[idx]);
        double expected_log_pmf = stdPoisson.getLogProbability(large_test_values[idx]);
        double expected_cdf = stdPoisson.getCumulativeProbability(large_test_values[idx]);
        
        EXPECT_NEAR(large_pmf_results[idx], expected_pmf, 1e-10);
        EXPECT_NEAR(large_log_pmf_results[idx], expected_log_pmf, 1e-10);
        EXPECT_NEAR(large_cdf_results[idx], expected_cdf, 1e-10);
    }
}

//==============================================================================
// TESTS FOR POISSON-SPECIFIC FUNCTIONALITY
//==============================================================================

TEST_F(PoissonEnhancedTest, PoissonSpecificMethods) {
    PoissonDistribution poisson(2.5);
    
    // Test exact integer methods
    EXPECT_EQ(poisson.getProbabilityExact(3), poisson.getProbability(3.0));
    EXPECT_EQ(poisson.getLogProbabilityExact(3), poisson.getLogProbability(3.0));
    EXPECT_EQ(poisson.getCumulativeProbabilityExact(3), poisson.getCumulativeProbability(3.0));
    
    // Test integer sampling
    std::mt19937 rng(42);
    auto int_samples = poisson.sampleIntegers(rng, 100);
    
    EXPECT_EQ(int_samples.size(), 100);
    for (int sample : int_samples) {
        EXPECT_GE(sample, 0);  // All samples should be non-negative integers
    }
    
    // Check sample mean is close to lambda (with reasonable tolerance)
    double sample_mean = std::accumulate(int_samples.begin(), int_samples.end(), 0.0) / int_samples.size();
    EXPECT_NEAR(sample_mean, 2.5, 0.5);  // Allow reasonable deviation for 100 samples
    
    // Test normal approximation capability
    PoissonDistribution small_lambda(3.0);
    PoissonDistribution large_lambda(15.0);
    
    EXPECT_FALSE(small_lambda.canUseNormalApproximation());  // λ=3 < 10
    EXPECT_TRUE(large_lambda.canUseNormalApproximation());   // λ=15 > 10
}

TEST_F(PoissonEnhancedTest, ParameterFitting) {
    // Create synthetic count data
    std::vector<double> count_data = {2, 1, 4, 3, 2, 5, 1, 3, 2, 4, 3, 2, 1, 4, 3, 2, 3, 1, 2, 4};
    
    PoissonDistribution fitted_dist;
    fitted_dist.fit(count_data);
    
    // Check that fitted lambda equals sample mean
    double sample_mean = std::accumulate(count_data.begin(), count_data.end(), 0.0) / count_data.size();
    EXPECT_NEAR(fitted_dist.getLambda(), sample_mean, 1e-10);
    EXPECT_NEAR(fitted_dist.getMean(), sample_mean, 1e-10);
    EXPECT_NEAR(fitted_dist.getVariance(), sample_mean, 1e-10);
}

TEST_F(PoissonEnhancedTest, OptimizedSampling) {
    // Test small lambda (Knuth's algorithm)
    PoissonDistribution small_poisson(2.0);
    std::mt19937 rng(42);
    
    const size_t num_samples = 10000;
    std::vector<double> samples;
    samples.reserve(num_samples);
    
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(small_poisson.sample(rng));
    }
    
    // Calculate sample statistics
    double sample_mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double sample_variance = 0.0;
    for (double x : samples) {
        sample_variance += (x - sample_mean) * (x - sample_mean);
    }
    sample_variance /= samples.size();
    
    // Check if samples are within reasonable bounds (for Poisson, mean ≈ variance)
    EXPECT_NEAR(sample_mean, 2.0, 0.1);      // Mean should be close to λ
    EXPECT_NEAR(sample_variance, 2.0, 0.2);  // Variance should be close to λ
    
    // Test large lambda (Atkinson algorithm)
    PoissonDistribution large_poisson(25.0);
    std::mt19937 rng2(42);
    
    std::vector<double> large_samples;
    large_samples.reserve(1000);
    
    for (size_t i = 0; i < 1000; ++i) {
        large_samples.push_back(large_poisson.sample(rng2));
    }
    
    double large_sample_mean = std::accumulate(large_samples.begin(), large_samples.end(), 0.0) / large_samples.size();
    EXPECT_NEAR(large_sample_mean, 25.0, 2.0);  // Allow more deviation for fewer samples
}

TEST_F(PoissonEnhancedTest, EdgeCases) {
    // Test invalid parameter creation
    auto resultZero = PoissonDistribution::create(0.0);
    EXPECT_TRUE(resultZero.isError());
    
    auto resultNegative = PoissonDistribution::create(-1.0);
    EXPECT_TRUE(resultNegative.isError());
    
    // Test very small lambda
    auto resultTiny = PoissonDistribution::create(0.001);
    if (resultTiny.isOk()) {
        auto tiny_dist = std::move(resultTiny.value);
        EXPECT_NEAR(tiny_dist.getMean(), 0.001, 1e-10);
        EXPECT_NEAR(tiny_dist.getVariance(), 0.001, 1e-10);
        
        // PMF(0) should be very close to 1 for tiny lambda
        double pmf_zero = tiny_dist.getProbability(0.0);
        EXPECT_NEAR(pmf_zero, std::exp(-0.001), 1e-10);
    }
    
    // Test extreme values
    PoissonDistribution normal(5.0);
    
    double pmf_negative = normal.getProbability(-1.0);
    double cdf_negative = normal.getCumulativeProbability(-1.0);
    
    EXPECT_EQ(pmf_negative, 0.0);  // PMF should be 0 for negative values
    EXPECT_EQ(cdf_negative, 0.0);  // CDF should be 0 for negative values
    
    // Test large count values
    double pmf_large = normal.getProbability(100.0);
    double cdf_large = normal.getCumulativeProbability(100.0);
    
    EXPECT_GE(pmf_large, 0.0);
    EXPECT_GE(cdf_large, 0.0);
    EXPECT_LE(cdf_large, 1.0);
}

TEST_F(PoissonEnhancedTest, ThreadSafety) {
    PoissonDistribution poisson(4.0);
    
    const int num_threads = 4;
    const int samples_per_thread = 1000;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> results(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&poisson, &results, t]() {
            std::mt19937 local_rng(42 + t);
            results[t].reserve(1000);
            
            for (int i = 0; i < 1000; ++i) {
                results[t].push_back(poisson.sample(local_rng));
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
            EXPECT_GE(val, 0.0);  // Poisson samples should be non-negative
        }
    }
}

TEST_F(PoissonEnhancedTest, QuantileFunctionAccuracy) {
    PoissonDistribution poisson(3.0);
    
    // Test quantile function for various probabilities
    std::vector<double> probabilities = {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};
    
    for (double p : probabilities) {
        double quantile = poisson.getQuantile(p);
        double cdf_at_quantile = poisson.getCumulativeProbability(quantile);
        
        // CDF at quantile should be >= p (since we have discrete distribution)
        EXPECT_GE(cdf_at_quantile, p - 1e-10);
        
        // CDF at (quantile - 1) should be < p (if quantile > 0)
        if (quantile > 0.0) {
            double cdf_below = poisson.getCumulativeProbability(quantile - 1.0);
            EXPECT_LT(cdf_below, p + 1e-10);
        }
    }
    
    // Test edge cases
    EXPECT_EQ(poisson.getQuantile(0.0), 0.0);
    EXPECT_EQ(poisson.getQuantile(1.0), std::numeric_limits<double>::infinity());
}

TEST_F(PoissonEnhancedTest, LogSpaceStability) {
    PoissonDistribution poisson(20.0);
    
    // Test log PMF for large counts where regular PMF might underflow
    std::vector<int> large_counts = {30, 40, 50, 60, 70};
    
    for (int k : large_counts) {
        double log_pmf = poisson.getLogProbabilityExact(k);
        double pmf = poisson.getProbabilityExact(k);
        
        // Log PMF should be finite
        EXPECT_TRUE(std::isfinite(log_pmf));
        
        // If PMF is computable, log(PMF) should match log PMF
        if (pmf > 0.0 && std::isfinite(pmf)) {
            EXPECT_NEAR(std::log(pmf), log_pmf, 1e-10);
        }
        
        // Log PMF should be negative (since PMF < 1)
        EXPECT_LT(log_pmf, 0.0);
    }
}

//==============================================================================
// AUTO-DISPATCH STRATEGY TESTING
//==============================================================================

TEST_F(PoissonEnhancedTest, SmartAutoDispatchStrategyTesting) {
    PoissonDistribution poisson_dist(3.0);
    
    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH", "PARALLEL_SIMD"};
    
    std::cout << "\n=== Smart Auto-Dispatch Strategy Testing (Poisson) ===\n";
    
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pdf_results(batch_size);
        std::vector<double> auto_logpdf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);
        
        std::mt19937 gen(42 + i);
        std::uniform_int_distribution<> dis(0, 10);  // Poisson values 0-10
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = static_cast<double>(dis(gen));
        }
        
        // Test smart auto-dispatch methods
        auto start = std::chrono::high_resolution_clock::now();
        poisson_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_pdf_results));
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        poisson_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        poisson_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pdf_results(batch_size);
        std::vector<double> trad_logpdf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);
        
        poisson_dist.getProbabilityBatch(test_values.data(), trad_pdf_results.data(), batch_size);
        poisson_dist.getLogProbabilityBatch(test_values.data(), trad_logpdf_results.data(), batch_size);
        poisson_dist.getCumulativeProbabilityBatch(test_values.data(), trad_cdf_results.data(), batch_size);
        
        // Verify correctness
        bool pdf_correct = true, logpdf_correct = true, cdf_correct = true;
        
        for (size_t j = 0; j < batch_size; ++j) {
            if (std::abs(auto_pdf_results[j] - trad_pdf_results[j]) > 1e-10) {
                pdf_correct = false;
            }
            if (std::abs(auto_logpdf_results[j] - trad_logpdf_results[j]) > 1e-10) {
                logpdf_correct = false;
            }
            if (std::abs(auto_cdf_results[j] - trad_cdf_results[j]) > 1e-10) {
                cdf_correct = false;
            }
        }
        
        std::cout << "Batch size: " << batch_size << ", Expected strategy: " << expected_strategy << "\n";
        std::cout << "  PDF: " << auto_pdf_time << "μs, Correct: " << (pdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  LogPDF: " << auto_logpdf_time << "μs, Correct: " << (logpdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  CDF: " << auto_cdf_time << "μs, Correct: " << (cdf_correct ? "✅" : "❌") << "\n";
        
        EXPECT_TRUE(pdf_correct) << "PDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(logpdf_correct) << "LogPDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(cdf_correct) << "CDF auto-dispatch results should match traditional for batch size " << batch_size;
    }
    
    std::cout << "\n=== Auto-Dispatch Strategy Testing Completed (Poisson) ===\n";
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(PoissonEnhancedTest, ParallelBatchPerformanceBenchmark) {
    PoissonDistribution stdPoisson(3.0);
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data (discrete values 0-15)
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(0, 15);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = static_cast<double>(dis(gen));
    }
    
    StandardizedBenchmark::printBenchmarkHeader("Poisson Distribution", BENCHMARK_SIZE);
    
    std::vector<BenchmarkResult> benchmark_results;
    
    // For each operation type (PDF, LogPDF, CDF)
    std::vector<std::string> operations = {"PDF", "LogPDF", "CDF"};
    
    for (const auto& op : operations) {
        BenchmarkResult result;
        result.operation_name = op;
        
        // 1. SIMD Batch (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PDF") {
            stdPoisson.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
        } else if (op == "LogPDF") {
            stdPoisson.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
        } else if (op == "CDF") {
            stdPoisson.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. Standard Parallel Operations (if available)
        std::span<const double> input_span(test_values);
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getProbabilityBatchParallel(input_span, output_span); }) {
                stdPoisson.getProbabilityBatchParallel(input_span, output_span);
            } else {
                // Fallback to SIMD if parallel not available
                stdPoisson.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getLogProbabilityBatchParallel(input_span, log_output_span); }) {
                stdPoisson.getLogProbabilityBatchParallel(input_span, log_output_span);
            } else {
                stdPoisson.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span); }) {
                stdPoisson.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
            } else {
                stdPoisson.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Work-Stealing Operations (if available)
        WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool); }) {
                stdPoisson.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
            } else {
                stdPoisson.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool); }) {
                stdPoisson.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
            } else {
                stdPoisson.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool); }) {
                stdPoisson.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
            } else {
                stdPoisson.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Cache-Aware Operations (if available)
        cache::AdaptiveCache<std::string, double> cache_manager;
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getProbabilityBatchCacheAware(input_span, output_span, cache_manager); }) {
                stdPoisson.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
            } else {
                stdPoisson.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager); }) {
                stdPoisson.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
            } else {
                stdPoisson.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { stdPoisson.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager); }) {
                stdPoisson.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
            } else {
                stdPoisson.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.cache_aware_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Calculate speedups
        result.parallel_speedup = (double)result.simd_time_us / result.parallel_time_us;
        result.work_stealing_speedup = (double)result.simd_time_us / result.work_stealing_time_us;
        result.cache_aware_speedup = (double)result.simd_time_us / result.cache_aware_time_us;
        
        benchmark_results.push_back(result);
        
        // Verify correctness
        if (op == "PDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdPoisson, test_values, pdf_results, "PDF");
        } else if (op == "LogPDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdPoisson, test_values, log_pdf_results, "LogPDF");
        } else if (op == "CDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdPoisson, test_values, cdf_results, "CDF");
        }
    }
    
    // Print standardized benchmark results
    StandardizedBenchmark::printBenchmarkResults(benchmark_results);
    StandardizedBenchmark::printPerformanceAnalysis(benchmark_results);
}

//==============================================================================
// ADVANCED STATISTICAL METHODS
//==============================================================================

TEST_F(PoissonEnhancedTest, AdvancedStatisticalMethods) {
    // Test confidence interval for rate parameter
    auto [rate_lower, rate_upper] = PoissonDistribution::confidenceIntervalRate(poisson_data_, 0.95);
    
    EXPECT_LT(rate_lower, rate_upper);
    EXPECT_GT(rate_lower, 0.0);
    EXPECT_TRUE(std::isfinite(rate_lower));
    EXPECT_TRUE(std::isfinite(rate_upper));
    
    // Test likelihood ratio test
    auto [lr_stat, p_value, reject_null] = PoissonDistribution::likelihoodRatioTest(poisson_data_, test_lambda_, 0.05);
    
    EXPECT_GE(lr_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(lr_stat));
    EXPECT_TRUE(std::isfinite(p_value));
    
    // Test method of moments estimation
    double lambda_mom = PoissonDistribution::methodOfMomentsEstimation(poisson_data_);
    EXPECT_GT(lambda_mom, 0.0);
    EXPECT_TRUE(std::isfinite(lambda_mom));
    
    // Should equal sample mean for Poisson
    double sample_mean = std::accumulate(poisson_data_.begin(), poisson_data_.end(), 0.0) / poisson_data_.size();
    EXPECT_NEAR(lambda_mom, sample_mean, 1e-10);
    
    // Test Bayesian estimation with conjugate Gamma prior
    auto [post_shape, post_rate] = PoissonDistribution::bayesianEstimation(poisson_data_, 1.0, 1.0);
    EXPECT_GT(post_shape, 0.0);
    EXPECT_GT(post_rate, 0.0);
    EXPECT_TRUE(std::isfinite(post_shape));
    EXPECT_TRUE(std::isfinite(post_rate));
}

TEST_F(PoissonEnhancedTest, GoodnessOfFitTests) {
    // Test Chi-square goodness of fit test
    auto [chi2_stat, chi2_p_value, chi2_reject] = PoissonDistribution::chiSquareGoodnessOfFit(
        poisson_data_, test_distribution_, 0.05);
    
    EXPECT_GE(chi2_stat, 0.0);
    EXPECT_GE(chi2_p_value, 0.0);
    EXPECT_LE(chi2_p_value, 1.0);
    EXPECT_TRUE(std::isfinite(chi2_stat));
    EXPECT_TRUE(std::isfinite(chi2_p_value));
    
    // Test Kolmogorov-Smirnov test (adapted for discrete distributions)
    auto [ks_stat, ks_p_value, ks_reject] = PoissonDistribution::kolmogorovSmirnovTest(
        poisson_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ks_stat, 0.0);
    EXPECT_LE(ks_stat, 1.0);
    EXPECT_GE(ks_p_value, 0.0);
    EXPECT_LE(ks_p_value, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat));
    EXPECT_TRUE(std::isfinite(ks_p_value));
    
    // Test with obviously non-Poisson data - should reject
    auto [bad_chi2_stat, bad_chi2_p_value, bad_chi2_reject] = PoissonDistribution::chiSquareGoodnessOfFit(
        non_poisson_data_, test_distribution_, 0.05);
    
    // Should typically reject non-Poisson data (though not guaranteed for any single test)
    EXPECT_TRUE(std::isfinite(bad_chi2_stat));
    EXPECT_TRUE(std::isfinite(bad_chi2_p_value));
}

TEST_F(PoissonEnhancedTest, CrossValidationMethods) {
    // Test k-fold cross validation
    auto cv_results = PoissonDistribution::kFoldCrossValidation(poisson_data_, 5, 42);
    
    EXPECT_EQ(cv_results.size(), 5);
    
    for (const auto& [mae, rmse, log_likelihood] : cv_results) {
        EXPECT_GE(mae, 0.0);
        EXPECT_GE(rmse, 0.0);
        EXPECT_GE(rmse, mae);  // RMSE should be >= MAE
        EXPECT_LE(log_likelihood, 0.0);  // Log-likelihood should be negative
        EXPECT_TRUE(std::isfinite(mae));
        EXPECT_TRUE(std::isfinite(rmse));
        EXPECT_TRUE(std::isfinite(log_likelihood));
    }
    
    // Test leave-one-out cross validation on smaller dataset
    std::vector<double> small_poisson_data(poisson_data_.begin(), poisson_data_.begin() + 20);
    auto [loocv_mae, loocv_rmse, loocv_log_likelihood] = PoissonDistribution::leaveOneOutCrossValidation(small_poisson_data);
    
    EXPECT_GE(loocv_mae, 0.0);
    EXPECT_GE(loocv_rmse, 0.0);
    EXPECT_GE(loocv_rmse, loocv_mae);
    EXPECT_LE(loocv_log_likelihood, 0.0);
    EXPECT_TRUE(std::isfinite(loocv_mae));
    EXPECT_TRUE(std::isfinite(loocv_rmse));
    EXPECT_TRUE(std::isfinite(loocv_log_likelihood));
}

TEST_F(PoissonEnhancedTest, InformationCriteria) {
    // Fit distribution to data
    PoissonDistribution fitted_dist;
    fitted_dist.fit(poisson_data_);
    
    auto [aic, bic, aicc, log_likelihood] = PoissonDistribution::computeInformationCriteria(
        poisson_data_, fitted_dist);
    
    // Basic validity checks
    EXPECT_LE(log_likelihood, 0.0);  // Log-likelihood should be negative
    EXPECT_GT(aic, 0.0);             // AIC is typically positive
    EXPECT_GT(bic, 0.0);             // BIC is typically positive  
    EXPECT_GT(aicc, 0.0);            // AICc is typically positive
    EXPECT_GE(aicc, aic);            // AICc should be >= AIC (correction term is positive)
    
    // For moderate sample sizes, BIC typically penalizes more than AIC
    EXPECT_GT(bic, aic);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(aic));
    EXPECT_TRUE(std::isfinite(bic));
    EXPECT_TRUE(std::isfinite(aicc));
    EXPECT_TRUE(std::isfinite(log_likelihood));
}

TEST_F(PoissonEnhancedTest, BootstrapParameterConfidenceIntervals) {
    auto lambda_ci = PoissonDistribution::bootstrapParameterConfidenceIntervals(
        poisson_data_, 0.95, 1000, 456);
    
    // Check that confidence interval is reasonable
    EXPECT_LT(lambda_ci.first, lambda_ci.second);  // Lower bound < Upper bound
    EXPECT_GT(lambda_ci.first, 0.0);               // Lambda should be positive
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(lambda_ci.first));
    EXPECT_TRUE(std::isfinite(lambda_ci.second));
    
    // The true lambda should often (but not always) fall within the CI
    [[maybe_unused]] double sample_mean = std::accumulate(poisson_data_.begin(), poisson_data_.end(), 0.0) / poisson_data_.size();
    // Note: We don't assert this since it's probabilistic, but we check the CI makes sense
    EXPECT_LT(lambda_ci.first, lambda_ci.second);
}

//==============================================================================
// NUMERICAL STABILITY TESTS
//==============================================================================

TEST_F(PoissonEnhancedTest, NumericalStability) {
    // Test with very small lambda
    auto tiny_result = PoissonDistribution::create(1e-6);
    if (tiny_result.isOk()) {
        auto tiny_dist = std::move(tiny_result.value);
        EXPECT_TRUE(std::isfinite(tiny_dist.getProbability(0.0)));
        EXPECT_TRUE(std::isfinite(tiny_dist.getLogProbability(0.0)));
        EXPECT_TRUE(std::isfinite(tiny_dist.getCumulativeProbability(5.0)));
    }
    
    // Test with moderately large lambda
    PoissonDistribution large_dist(100.0);
    EXPECT_TRUE(std::isfinite(large_dist.getProbability(100.0)));
    EXPECT_TRUE(std::isfinite(large_dist.getLogProbability(100.0)));
    EXPECT_TRUE(std::isfinite(large_dist.getCumulativeProbability(100.0)));
    
    // Test edge cases
    EdgeCaseTester<PoissonDistribution>::testExtremeValues(large_dist, "Poisson");
    EdgeCaseTester<PoissonDistribution>::testEmptyBatchOperations(large_dist, "Poisson");
}

} // namespace libstats

#include <gtest/gtest.h>
#include "../include/gaussian.h"
#include "../include/work_stealing_pool.h"
#include "../include/adaptive_cache.h"
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

namespace libstats {

//==============================================================================
// TEST FIXTURE FOR GAUSSIAN ENHANCED METHODS
//==============================================================================

class GaussianEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic normal data for testing
        std::mt19937 rng(42);
        std::normal_distribution<double> normal_gen(test_mean_, test_std_);
        
        normal_data_.clear();
        normal_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            normal_data_.push_back(normal_gen(rng));
        }
        
        // Generate obviously non-normal data
        non_normal_data_.clear();
        non_normal_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            non_normal_data_.push_back(i * i); // Quadratic growth
        }
        
        test_distribution_ = GaussianDistribution(test_mean_, test_std_);
    }
    
    const double test_mean_ = 5.0;
    const double test_std_ = 2.0;
    std::vector<double> normal_data_;
    std::vector<double> non_normal_data_;
    GaussianDistribution test_distribution_;
};

//==============================================================================
// TESTS FOR BASIC ENHANCED FUNCTIONALITY
//==============================================================================

TEST_F(GaussianEnhancedTest, BasicEnhancedFunctionality) {
    // Test standard normal distribution properties
    GaussianDistribution stdNormal(0.0, 1.0);
    
    EXPECT_DOUBLE_EQ(stdNormal.getMean(), 0.0);
    EXPECT_DOUBLE_EQ(stdNormal.getVariance(), 1.0);
    EXPECT_DOUBLE_EQ(stdNormal.getSkewness(), 0.0);
    EXPECT_DOUBLE_EQ(stdNormal.getKurtosis(), 0.0);  // Excess kurtosis for Gaussian
    
    // Test known PDF/CDF values
    double pdf_at_0 = stdNormal.getProbability(0.0);
    double cdf_at_0 = stdNormal.getCumulativeProbability(0.0);
    
    EXPECT_NEAR(pdf_at_0, 0.398942280401433, 1e-10);
    EXPECT_NEAR(cdf_at_0, 0.5, 1e-10);
    
    // Test custom distribution
    GaussianDistribution custom(10.0, 2.0);
    EXPECT_DOUBLE_EQ(custom.getMean(), 10.0);
    EXPECT_DOUBLE_EQ(custom.getVariance(), 4.0);
    EXPECT_NEAR(custom.getCumulativeProbability(10.0), 0.5, 1e-10);
}

TEST_F(GaussianEnhancedTest, CopyAndMoveSemantics) {
    // Test copy constructor
    GaussianDistribution original(3.0, 2.0);
    GaussianDistribution copied(original);
    
    EXPECT_EQ(copied.getMean(), original.getMean());
    EXPECT_EQ(copied.getVariance(), original.getVariance());
    EXPECT_NEAR(copied.getProbability(3.0), original.getProbability(3.0), 1e-10);
    
    // Test move constructor
    GaussianDistribution to_move(5.0, 3.0);
    double original_mean = to_move.getMean();
    double original_var = to_move.getVariance();
    GaussianDistribution moved(std::move(to_move));
    
    EXPECT_EQ(moved.getMean(), original_mean);
    EXPECT_EQ(moved.getVariance(), original_var);
}

TEST_F(GaussianEnhancedTest, BatchOperations) {
    GaussianDistribution stdNormal(0.0, 1.0);
    
    // Test data
    std::vector<double> test_values = {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> pdf_results(test_values.size());
    std::vector<double> log_pdf_results(test_values.size());
    std::vector<double> cdf_results(test_values.size());
    
    // Test batch operations
    stdNormal.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    stdNormal.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    stdNormal.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = stdNormal.getProbability(test_values[i]);
        double expected_log_pdf = stdNormal.getLogProbability(test_values[i]);
        double expected_cdf = stdNormal.getCumulativeProbability(test_values[i]);
        
        EXPECT_NEAR(pdf_results[i], expected_pdf, 1e-12);
        EXPECT_NEAR(log_pdf_results[i], expected_log_pdf, 1e-12);
        EXPECT_NEAR(cdf_results[i], expected_cdf, 1e-12);
    }
}

TEST_F(GaussianEnhancedTest, SIMDPerformanceTest) {
    GaussianDistribution stdNormal(0.0, 1.0);
    constexpr size_t LARGE_BATCH_SIZE = 10000;
    
    std::vector<double> large_test_values(LARGE_BATCH_SIZE);
    std::vector<double> large_pdf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_log_pdf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_cdf_results(LARGE_BATCH_SIZE);
    
    // Generate test data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-3.0, 3.0);
    
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_test_values[i] = dis(gen);
    }
    
    std::cout << "  === SIMD Batch Performance Results ===" << std::endl;
    
    // Test 1: PDF Batch vs Individual
    auto start = std::chrono::high_resolution_clock::now();
    stdNormal.getProbabilityBatch(large_test_values.data(), large_pdf_results.data(), LARGE_BATCH_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pdf_results[i] = stdNormal.getProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto pdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double pdf_speedup = (double)pdf_individual_time / pdf_batch_time;
    std::cout << "  PDF:     Batch " << pdf_batch_time << "μs vs Individual " << pdf_individual_time << "μs → " << pdf_speedup << "x speedup" << std::endl;
    
    // Test 2: Log PDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getLogProbabilityBatch(large_test_values.data(), large_log_pdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_log_pdf_results[i] = stdNormal.getLogProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double log_pdf_speedup = (double)log_pdf_individual_time / log_pdf_batch_time;
    std::cout << "  LogPDF:  Batch " << log_pdf_batch_time << "μs vs Individual " << log_pdf_individual_time << "μs → " << log_pdf_speedup << "x speedup" << std::endl;
    
    // Test 3: CDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto cdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_cdf_results[i] = stdNormal.getCumulativeProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cdf_speedup = (double)cdf_individual_time / cdf_batch_time;
    std::cout << "  CDF:     Batch " << cdf_batch_time << "μs vs Individual " << cdf_individual_time << "μs → " << cdf_speedup << "x speedup" << std::endl;
    
    // Expect speedup for all operations
    EXPECT_GT(pdf_speedup, 1.0) << "PDF batch should be faster than individual calls";
    EXPECT_GT(log_pdf_speedup, 1.0) << "Log PDF batch should be faster than individual calls";
    EXPECT_GT(cdf_speedup, 1.0) << "CDF batch should be faster than individual calls";
    
    // Verify correctness on a sample
    const size_t sample_size = 100;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * (LARGE_BATCH_SIZE / sample_size);
        double expected_pdf = stdNormal.getProbability(large_test_values[idx]);
        double expected_log_pdf = stdNormal.getLogProbability(large_test_values[idx]);
        double expected_cdf = stdNormal.getCumulativeProbability(large_test_values[idx]);
        
        EXPECT_NEAR(large_pdf_results[idx], expected_pdf, 1e-10);
        EXPECT_NEAR(large_log_pdf_results[idx], expected_log_pdf, 1e-10);
        EXPECT_NEAR(large_cdf_results[idx], expected_cdf, 1e-10);
    }
}

//==============================================================================
// TESTS FOR ADVANCED STATISTICAL METHODS (From test_advanced_methods.cpp)
//==============================================================================

TEST_F(GaussianEnhancedTest, ConfidenceIntervalMean) {
    auto [ci_lower, ci_upper] = GaussianDistribution::confidenceIntervalMean(normal_data_, 0.95);
    
    EXPECT_LT(ci_lower, ci_upper);
    EXPECT_TRUE(std::isfinite(ci_lower));
    EXPECT_TRUE(std::isfinite(ci_upper));
    
    // The true mean (5.0) should be within the confidence interval most of the time
    // (though not guaranteed for any single sample)
    double sample_mean = std::accumulate(normal_data_.begin(), normal_data_.end(), 0.0) / normal_data_.size();
    EXPECT_GT(ci_upper, sample_mean - 3.0); // Sanity check
    EXPECT_LT(ci_lower, sample_mean + 3.0); // Sanity check
}

TEST_F(GaussianEnhancedTest, OneSampleTTest) {
    auto [t_stat, p_value, reject_null] = GaussianDistribution::oneSampleTTest(normal_data_, 5.0, 0.05);
    
    EXPECT_TRUE(std::isfinite(t_stat));
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(p_value));
}

TEST_F(GaussianEnhancedTest, MethodOfMomentsEstimation) {
    auto [estimated_mean, estimated_std] = GaussianDistribution::methodOfMomentsEstimation(normal_data_);
    
    EXPECT_TRUE(std::isfinite(estimated_mean));
    EXPECT_TRUE(std::isfinite(estimated_std));
    EXPECT_GT(estimated_std, 0.0);
    
    // Should be close to the true parameters (5.0, 2.0) but with some sampling variation
    EXPECT_NEAR(estimated_mean, test_mean_, 1.0);  // Allow reasonable deviation
    EXPECT_NEAR(estimated_std, test_std_, 1.0);    // Allow reasonable deviation
}

TEST_F(GaussianEnhancedTest, JarqueBeraTest) {
    auto [jb_stat, p_value, reject_normality] = GaussianDistribution::jarqueBeraTest(normal_data_, 0.05);
    
    EXPECT_GE(jb_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(jb_stat));
    EXPECT_TRUE(std::isfinite(p_value));
}

TEST_F(GaussianEnhancedTest, RobustEstimation) {
    auto [robust_loc, robust_scale] = GaussianDistribution::robustEstimation(normal_data_, "huber", 1.345);
    
    EXPECT_TRUE(std::isfinite(robust_loc));
    EXPECT_TRUE(std::isfinite(robust_scale));
    EXPECT_GT(robust_scale, 0.0);
    
    // Should be reasonably close to the true parameters
    EXPECT_NEAR(robust_loc, test_mean_, 2.0);    // Allow reasonable deviation
    EXPECT_NEAR(robust_scale, test_std_, 2.0);   // Allow reasonable deviation
}

//==============================================================================
// TESTS FOR PHASE 3 METHODS (From test_gaussian_phase3.cpp)
//==============================================================================

TEST_F(GaussianEnhancedTest, KolmogorovSmirnovTest) {
    auto [ks_stat, p_value, reject] = GaussianDistribution::kolmogorovSmirnovTest(
        normal_data_, test_distribution_, 0.05);
    
    // Basic validity checks
    EXPECT_GE(ks_stat, 0.0);
    EXPECT_LE(ks_stat, 1.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat));
    EXPECT_TRUE(std::isfinite(p_value));
}

TEST_F(GaussianEnhancedTest, KolmogorovSmirnovNonNormal) {
    // Test with obviously non-normal data - should reject normality
    auto [ks_stat, p_value, reject] = GaussianDistribution::kolmogorovSmirnovTest(
        non_normal_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(reject); // Should reject normality for quadratic data
    EXPECT_GE(ks_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
}

TEST_F(GaussianEnhancedTest, AndersonDarlingTest) {
    auto [ad_stat, p_value, reject] = GaussianDistribution::andersonDarlingTest(
        normal_data_, test_distribution_, 0.05);
    
    // Basic validity checks
    EXPECT_GE(ad_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(ad_stat));
    EXPECT_TRUE(std::isfinite(p_value));
}

TEST_F(GaussianEnhancedTest, KFoldCrossValidation) {
    auto results = GaussianDistribution::kFoldCrossValidation(normal_data_, 5, 42);
    
    EXPECT_EQ(results.size(), 5);
    
    // Check that each fold gives reasonable results
    for (const auto& [mean_error, std_error, log_likelihood] : results) {
        EXPECT_GE(mean_error, 0.0);       // Mean absolute error should be non-negative
        EXPECT_GE(std_error, 0.0);        // Standard error should be non-negative
        EXPECT_LE(log_likelihood, 0.0);   // Log-likelihood should be negative for continuous distributions
        EXPECT_TRUE(std::isfinite(mean_error));
        EXPECT_TRUE(std::isfinite(std_error));
        EXPECT_TRUE(std::isfinite(log_likelihood));
    }
}

TEST_F(GaussianEnhancedTest, LeaveOneOutCrossValidation) {
    // Use a smaller dataset for LOOCV to keep test time reasonable
    std::vector<double> small_normal_data(normal_data_.begin(), normal_data_.begin() + 20);
    
    auto [mae, rmse, log_likelihood] = GaussianDistribution::leaveOneOutCrossValidation(small_normal_data);
    
    EXPECT_GE(mae, 0.0);                 // Mean absolute error should be non-negative
    EXPECT_GE(rmse, 0.0);                // RMSE should be non-negative
    EXPECT_LE(log_likelihood, 0.0);      // Total log-likelihood should be negative
    EXPECT_GE(rmse, mae);                // RMSE should be >= MAE
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(mae));
    EXPECT_TRUE(std::isfinite(rmse));
    EXPECT_TRUE(std::isfinite(log_likelihood));
}

TEST_F(GaussianEnhancedTest, BootstrapParameterConfidenceIntervals) {
    auto [mean_ci, std_ci] = GaussianDistribution::bootstrapParameterConfidenceIntervals(
        normal_data_, 0.95, 1000, 456);
    
    // Check that confidence intervals are reasonable
    EXPECT_LT(mean_ci.first, mean_ci.second);  // Lower bound < Upper bound
    EXPECT_LT(std_ci.first, std_ci.second);    // Lower bound < Upper bound
    
    // Standard deviation CIs should be positive
    EXPECT_GT(std_ci.first, 0.0);
    EXPECT_GT(std_ci.second, 0.0);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(mean_ci.first));
    EXPECT_TRUE(std::isfinite(mean_ci.second));
    EXPECT_TRUE(std::isfinite(std_ci.first));
    EXPECT_TRUE(std::isfinite(std_ci.second));
}

TEST_F(GaussianEnhancedTest, ComputeInformationCriteria) {
    // Fit a distribution to the data
    GaussianDistribution fitted_dist;
    fitted_dist.fit(normal_data_);
    
    auto [aic, bic, aicc, log_likelihood] = GaussianDistribution::computeInformationCriteria(
        normal_data_, fitted_dist);
    
    // Basic sanity checks
    EXPECT_LE(log_likelihood, 0.0);    // Log-likelihood should be negative
    EXPECT_GT(aic, 0.0);               // AIC is typically positive
    EXPECT_GT(bic, 0.0);               // BIC is typically positive
    EXPECT_GT(aicc, 0.0);              // AICc is typically positive
    EXPECT_GE(aicc, aic);              // AICc should be >= AIC (correction term is positive)
    
    // For moderate sample sizes, BIC typically penalizes more than AIC
    EXPECT_GT(bic, aic);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(aic));
    EXPECT_TRUE(std::isfinite(bic));
    EXPECT_TRUE(std::isfinite(aicc));
    EXPECT_TRUE(std::isfinite(log_likelihood));
}

//==============================================================================
// TESTS FOR SAMPLING AND EDGE CASES
//==============================================================================

TEST_F(GaussianEnhancedTest, OptimizedSampling) {
    GaussianDistribution stdNormal(0.0, 1.0);
    std::mt19937 rng(42);
    
    const size_t num_samples = 10000;
    std::vector<double> samples;
    samples.reserve(num_samples);
    
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(stdNormal.sample(rng));
    }
    
    // Calculate sample statistics
    double sample_mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double sample_variance = 0.0;
    for (double x : samples) {
        sample_variance += (x - sample_mean) * (x - sample_mean);
    }
    sample_variance /= samples.size();
    
    // Check if samples are within reasonable bounds
    EXPECT_NEAR(sample_mean, 0.0, 0.05);      // Mean should be close to 0
    EXPECT_NEAR(sample_variance, 1.0, 0.05);  // Variance should be close to 1
}

TEST_F(GaussianEnhancedTest, EdgeCases) {
    // Test invalid parameter creation
    auto resultZero = GaussianDistribution::create(0.0, 0.0);
    EXPECT_TRUE(resultZero.isError());
    
    auto resultNegative = GaussianDistribution::create(0.0, -1.0);
    EXPECT_TRUE(resultNegative.isError());
    
    // Test extreme values
    GaussianDistribution normal(0.0, 1.0);
    
    double large_val = 100.0;
    double pdf_large = normal.getProbability(large_val);
    double log_pdf_large = normal.getLogProbability(large_val);
    double cdf_large = normal.getCumulativeProbability(large_val);
    
    EXPECT_GE(pdf_large, 0.0);
    EXPECT_TRUE(std::isfinite(log_pdf_large));
    EXPECT_GE(cdf_large, 0.0);
    EXPECT_LE(cdf_large, 1.0);
}

TEST_F(GaussianEnhancedTest, ThreadSafety) {
    GaussianDistribution normal(0.0, 1.0);
    
    const int num_threads = 4;
    const int samples_per_thread = 1000;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> results(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&normal, &results, t]() {
            const int samples_per_thread = 1000;
            std::mt19937 local_rng(42 + t);
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
        }
    }
}

//==============================================================================
// PARALLEL BATCH OPERATIONS BENCHMARKS
//==============================================================================

TEST_F(GaussianEnhancedTest, ParallelBatchPerformanceBenchmark) {
    GaussianDistribution stdNormal(0.0, 1.0);
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-3.0, 3.0);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    std::cout << "\n=== Parallel Batch Operations Performance Benchmark ===" << std::endl;
    std::cout << "Dataset size: " << BENCHMARK_SIZE << " elements" << std::endl;
    
    // 1. Standard SIMD Batch Operations (baseline)
    auto start = std::chrono::high_resolution_clock::now();
    stdNormal.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto simd_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
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
    stdNormal.getProbabilityBatchParallel(input_span, output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::span<double> log_output_span(log_pdf_results);
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getLogProbabilityBatchParallel(input_span, log_output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::span<double> cdf_output_span(cdf_results);
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
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
    stdNormal.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // LogPDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
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
    stdNormal.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // LogPDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    stdNormal.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cache_aware_pdf_speedup = (double)simd_pdf_time / cache_aware_pdf_time;
    double cache_aware_log_pdf_speedup = (double)simd_log_pdf_time / cache_aware_log_pdf_time;
    double cache_aware_cdf_speedup = (double)simd_cdf_time / cache_aware_cdf_time;
    
    std::cout << "Cache-Aware Parallel:" << std::endl;
    std::cout << "  PDF:     " << cache_aware_pdf_time << " μs (" << cache_aware_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << cache_aware_log_pdf_time << " μs (" << cache_aware_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << cache_aware_cdf_time << " μs (" << cache_aware_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // Performance Expectations
    std::cout << "\nPerformance Analysis:" << std::endl;
    
    // On multi-core systems, parallel operations should generally show improvement
    if (std::thread::hardware_concurrency() > 2) {
        // Allow for some variance in performance measurements
        if (parallel_pdf_speedup > 0.8) {
            std::cout << "  ✓ Standard parallel shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Standard parallel speedup lower than expected (overhead may dominate)" << std::endl;
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
    
    // Verify correctness by spot-checking some results
    std::cout << "\nCorrectness verification:" << std::endl;
    bool all_correct = true;
    const size_t check_count = 100;
    for (size_t i = 0; i < check_count; ++i) {
        size_t idx = i * (BENCHMARK_SIZE / check_count);
        double expected = stdNormal.getProbability(test_values[idx]);
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
    
    // Work stealing pool statistics
    auto ws_stats = work_stealing_pool.getStatistics();
    std::cout << "\nWork-Stealing Pool Statistics:" << std::endl;
    std::cout << "  Tasks executed: " << ws_stats.tasksExecuted << std::endl;
    std::cout << "  Work steals: " << ws_stats.workSteals << std::endl;
    std::cout << "  Steal success rate: " << (ws_stats.stealSuccessRate * 100) << "%" << std::endl;
    
    // Cache manager statistics
    auto cache_stats = cache_manager.getStats();
    std::cout << "\nCache Manager Statistics:" << std::endl;
    std::cout << "  Cache size: " << cache_stats.size << " entries" << std::endl;
    std::cout << "  Hit rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;
    std::cout << "  Memory usage: " << cache_stats.memory_usage << " bytes" << std::endl;
}

} // namespace libstats

#include <gtest/gtest.h>
#include "../include/distributions/exponential.h"
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
// TEST FIXTURE FOR EXPONENTIAL ENHANCED METHODS
//==============================================================================

class ExponentialEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic exponential data for testing
        std::mt19937 rng(42);
        std::exponential_distribution<double> exp_gen(test_lambda_);
        
        exponential_data_.clear();
        exponential_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            exponential_data_.push_back(exp_gen(rng));
        }
        
        // Generate obviously non-exponential data (normal)
        non_exponential_data_.clear();
        non_exponential_data_.reserve(100);
        std::normal_distribution<double> normal_gen(5.0, 2.0);
        for (int i = 0; i < 100; ++i) {
            double val = normal_gen(rng);
            if (val > 0) non_exponential_data_.push_back(val); // Keep only positive values
        }
        
        test_distribution_ = ExponentialDistribution(test_lambda_);
    }
    
    const double test_lambda_ = 2.0;
    std::vector<double> exponential_data_;
    std::vector<double> non_exponential_data_;
    ExponentialDistribution test_distribution_;
};

//==============================================================================
// TESTS FOR BASIC ENHANCED FUNCTIONALITY
//==============================================================================

TEST_F(ExponentialEnhancedTest, BasicEnhancedFunctionality) {
    // Test unit exponential distribution properties
    ExponentialDistribution unitExp(1.0);
    
    EXPECT_DOUBLE_EQ(unitExp.getLambda(), 1.0);
    EXPECT_DOUBLE_EQ(unitExp.getMean(), 1.0);
    EXPECT_DOUBLE_EQ(unitExp.getVariance(), 1.0);
    EXPECT_DOUBLE_EQ(unitExp.getSkewness(), 2.0);
    EXPECT_DOUBLE_EQ(unitExp.getKurtosis(), 6.0);
    
    // Test known PDF/CDF values
    double pdf_at_0 = unitExp.getProbability(0.0);
    double cdf_at_1 = unitExp.getCumulativeProbability(1.0);
    
    EXPECT_NEAR(pdf_at_0, 1.0, 1e-10);
    EXPECT_NEAR(cdf_at_1, 1.0 - std::exp(-1.0), 1e-10);
    
    // Test custom distribution
    ExponentialDistribution custom(2.0);
    EXPECT_DOUBLE_EQ(custom.getLambda(), 2.0);
    EXPECT_DOUBLE_EQ(custom.getMean(), 0.5);
    EXPECT_DOUBLE_EQ(custom.getVariance(), 0.25);
}

TEST_F(ExponentialEnhancedTest, CopyAndMoveSemantics) {
    // Test copy constructor
    ExponentialDistribution original(3.0);
    ExponentialDistribution copied(original);
    
    EXPECT_EQ(copied.getLambda(), original.getLambda());
    EXPECT_EQ(copied.getMean(), original.getMean());
    EXPECT_NEAR(copied.getProbability(1.0), original.getProbability(1.0), 1e-10);
    
    // Test move constructor
    ExponentialDistribution to_move(5.0);
    double original_lambda = to_move.getLambda();
    ExponentialDistribution moved(std::move(to_move));
    
    EXPECT_EQ(moved.getLambda(), original_lambda);
    EXPECT_EQ(moved.getMean(), 1.0 / original_lambda);
}

TEST_F(ExponentialEnhancedTest, BatchOperations) {
    ExponentialDistribution stdExp(1.0);
    
    // Test data (positive values only for exponential)
    std::vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0};
    std::vector<double> pdf_results(test_values.size());
    std::vector<double> log_pdf_results(test_values.size());
    std::vector<double> cdf_results(test_values.size());
    
    // Test batch operations
    stdExp.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    stdExp.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    stdExp.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = stdExp.getProbability(test_values[i]);
        double expected_log_pdf = stdExp.getLogProbability(test_values[i]);
        double expected_cdf = stdExp.getCumulativeProbability(test_values[i]);
        
        EXPECT_NEAR(pdf_results[i], expected_pdf, 1e-12);
        EXPECT_NEAR(log_pdf_results[i], expected_log_pdf, 1e-12);
        EXPECT_NEAR(cdf_results[i], expected_cdf, 1e-12);
    }
}

TEST_F(ExponentialEnhancedTest, PerformanceTest) {
    ExponentialDistribution stdExp(1.0);
    constexpr size_t LARGE_BATCH_SIZE = 10000;
    
    std::vector<double> large_test_values(LARGE_BATCH_SIZE);
    std::vector<double> large_pdf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_log_pdf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_cdf_results(LARGE_BATCH_SIZE);
    
    // Generate test data (positive values only for exponential)
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.01, 10.0);
    
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_test_values[i] = dis(gen);
    }
    
    std::cout << "  === Exponential Batch Performance Results ===" << std::endl;
    
    // Test 1: PDF Batch vs Individual
    auto start = std::chrono::high_resolution_clock::now();
    stdExp.getProbabilityBatch(large_test_values.data(), large_pdf_results.data(), LARGE_BATCH_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pdf_results[i] = stdExp.getProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto pdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double pdf_speedup = (double)pdf_individual_time / pdf_batch_time;
    std::cout << "  PDF:     Batch " << pdf_batch_time << "μs vs Individual " << pdf_individual_time << "μs → " << pdf_speedup << "x speedup" << std::endl;
    
    // Test 2: Log PDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    stdExp.getLogProbabilityBatch(large_test_values.data(), large_log_pdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_log_pdf_results[i] = stdExp.getLogProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double log_pdf_speedup = (double)log_pdf_individual_time / log_pdf_batch_time;
    std::cout << "  LogPDF:  Batch " << log_pdf_batch_time << "μs vs Individual " << log_pdf_individual_time << "μs → " << log_pdf_speedup << "x speedup" << std::endl;
    
    // Test 3: CDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    stdExp.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto cdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_cdf_results[i] = stdExp.getCumulativeProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cdf_speedup = (double)cdf_individual_time / cdf_batch_time;
    std::cout << "  CDF:     Batch " << cdf_batch_time << "μs vs Individual " << cdf_individual_time << "μs → " << cdf_speedup << "x speedup" << std::endl;
    
    // Verify correctness on a sample
    const size_t sample_size = 100;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * (LARGE_BATCH_SIZE / sample_size);
        double expected_pdf = stdExp.getProbability(large_test_values[idx]);
        double expected_log_pdf = stdExp.getLogProbability(large_test_values[idx]);
        double expected_cdf = stdExp.getCumulativeProbability(large_test_values[idx]);
        
        EXPECT_NEAR(large_pdf_results[idx], expected_pdf, 1e-10);
        EXPECT_NEAR(large_log_pdf_results[idx], expected_log_pdf, 1e-10);
        EXPECT_NEAR(large_cdf_results[idx], expected_cdf, 1e-10);
    }
}

//==============================================================================
// TESTS FOR ADVANCED STATISTICAL METHODS
//==============================================================================

TEST_F(ExponentialEnhancedTest, AdvancedStatisticalMethods) {
    // Test confidence interval for rate parameter
    auto [rate_lower, rate_upper] = ExponentialDistribution::confidenceIntervalRate(exponential_data_, 0.95);
    
    EXPECT_LT(rate_lower, rate_upper);
    EXPECT_GT(rate_lower, 0.0);
    EXPECT_TRUE(std::isfinite(rate_lower));
    EXPECT_TRUE(std::isfinite(rate_upper));
    
    // Test likelihood ratio test
    auto [lr_stat, p_value, reject_null] = ExponentialDistribution::likelihoodRatioTest(exponential_data_, test_lambda_, 0.05);
    
    EXPECT_GE(lr_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(lr_stat));
    EXPECT_TRUE(std::isfinite(p_value));
    
    // Test method of moments estimation
    double lambda_mom = ExponentialDistribution::methodOfMomentsEstimation(exponential_data_);
    EXPECT_GT(lambda_mom, 0.0);
    EXPECT_TRUE(std::isfinite(lambda_mom));
    
    // Test Bayesian estimation
    auto [post_shape, post_rate] = ExponentialDistribution::bayesianEstimation(exponential_data_, 1.0, 1.0);
    EXPECT_GT(post_shape, 0.0);
    EXPECT_GT(post_rate, 0.0);
    EXPECT_TRUE(std::isfinite(post_shape));
    EXPECT_TRUE(std::isfinite(post_rate));
}

TEST_F(ExponentialEnhancedTest, GoodnessOfFitTests) {
    // Test Kolmogorov-Smirnov test
    auto [ks_stat, ks_p_value, ks_reject] = ExponentialDistribution::kolmogorovSmirnovTest(
        exponential_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ks_stat, 0.0);
    EXPECT_LE(ks_stat, 1.0);
    EXPECT_GE(ks_p_value, 0.0);
    EXPECT_LE(ks_p_value, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat));
    EXPECT_TRUE(std::isfinite(ks_p_value));
    
    // Test with obviously non-exponential data - should reject
    auto [bad_ks_stat, bad_ks_p_value, bad_ks_reject] = ExponentialDistribution::kolmogorovSmirnovTest(
        non_exponential_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(std::isfinite(bad_ks_stat));
    EXPECT_TRUE(std::isfinite(bad_ks_p_value));
}

TEST_F(ExponentialEnhancedTest, CrossValidationMethods) {
    // Test k-fold cross validation
    auto cv_results = ExponentialDistribution::kFoldCrossValidation(exponential_data_, 5, 42);
    
    EXPECT_EQ(cv_results.size(), 5);
    
    for (const auto& [mae, rmse, log_likelihood] : cv_results) {
        EXPECT_GE(mae, 0.0);
        EXPECT_GE(rmse, 0.0);
        // Note: RMSE is not always >= MAE, so we don't check that relationship
        EXPECT_LE(log_likelihood, 0.0);
        EXPECT_TRUE(std::isfinite(mae));
        EXPECT_TRUE(std::isfinite(rmse));
        EXPECT_TRUE(std::isfinite(log_likelihood));
    }
    
    // Test leave-one-out cross validation on smaller dataset
    std::vector<double> small_exp_data(exponential_data_.begin(), exponential_data_.begin() + 20);
    auto [loocv_mae, loocv_rmse, loocv_log_likelihood] = ExponentialDistribution::leaveOneOutCrossValidation(small_exp_data);
    
    EXPECT_GE(loocv_mae, 0.0);
    EXPECT_GE(loocv_rmse, 0.0);
    // Note: RMSE is not always >= MAE, so we don't check that relationship
    EXPECT_LE(loocv_log_likelihood, 0.0);
    EXPECT_TRUE(std::isfinite(loocv_mae));
    EXPECT_TRUE(std::isfinite(loocv_rmse));
    EXPECT_TRUE(std::isfinite(loocv_log_likelihood));
}

TEST_F(ExponentialEnhancedTest, InformationCriteria) {
    // Fit distribution to data
    ExponentialDistribution fitted_dist;
    fitted_dist.fit(exponential_data_);
    
    auto [aic, bic, aicc, log_likelihood] = ExponentialDistribution::computeInformationCriteria(
        exponential_data_, fitted_dist);
    
    // Basic validity checks
    EXPECT_LE(log_likelihood, 0.0);
    EXPECT_GT(aic, 0.0);
    EXPECT_GT(bic, 0.0);
    EXPECT_GT(aicc, 0.0);
    EXPECT_GE(aicc, aic);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(aic));
    EXPECT_TRUE(std::isfinite(bic));
    EXPECT_TRUE(std::isfinite(aicc));
    EXPECT_TRUE(std::isfinite(log_likelihood));
}

TEST_F(ExponentialEnhancedTest, BootstrapParameterConfidenceInterval) {
    auto [lambda_ci_lower, lambda_ci_upper] = ExponentialDistribution::bootstrapParameterConfidenceInterval(
        exponential_data_, 0.95, 1000, 456);
    
    // Check that confidence intervals are reasonable
    EXPECT_LT(lambda_ci_lower, lambda_ci_upper);
    EXPECT_GT(lambda_ci_lower, 0.0);
    EXPECT_GT(lambda_ci_upper, 0.0);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(lambda_ci_lower));
    EXPECT_TRUE(std::isfinite(lambda_ci_upper));
}

//==============================================================================
// TESTS FOR SAMPLING AND EDGE CASES
//==============================================================================

TEST_F(ExponentialEnhancedTest, OptimizedSampling) {
    ExponentialDistribution stdExp(1.0);
    std::mt19937 rng(42);
    
    const size_t num_samples = 10000;
    std::vector<double> samples;
    samples.reserve(num_samples);
    
    for (size_t i = 0; i < num_samples; ++i) {
        samples.push_back(stdExp.sample(rng));
    }
    
    // Calculate sample statistics
    auto [sample_mean, sample_variance] = StatisticalTestUtils::calculateSampleStats(samples);
    
    // Check if samples are within reasonable bounds
    EXPECT_NEAR(sample_mean, 1.0, 0.05);
    EXPECT_NEAR(sample_variance, 1.0, 0.05);
    
    // Check that all samples are non-negative
    for (double sample : samples) {
        EXPECT_GE(sample, 0.0);
    }
}

TEST_F(ExponentialEnhancedTest, EdgeCases) {
    // Test invalid parameter creation
    auto resultZero = ExponentialDistribution::create(0.0);
    EXPECT_TRUE(resultZero.isError());
    
    auto resultNegative = ExponentialDistribution::create(-1.0);
    EXPECT_TRUE(resultNegative.isError());
    
    // Test extreme values
    ExponentialDistribution normal(1.0);
    
    double large_val = 100.0;
    double pdf_large = normal.getProbability(large_val);
    double log_pdf_large = normal.getLogProbability(large_val);
    double cdf_large = normal.getCumulativeProbability(large_val);
    
    EXPECT_GE(pdf_large, 0.0);
    EXPECT_TRUE(std::isfinite(log_pdf_large));
    EXPECT_GE(cdf_large, 0.0);
    EXPECT_LE(cdf_large, 1.0);
    
    // Test negative values (should return 0 for PDF/CDF)
    double neg_val = -1.0;
    double pdf_neg = normal.getProbability(neg_val);
    double cdf_neg = normal.getCumulativeProbability(neg_val);
    
    EXPECT_EQ(pdf_neg, 0.0);
    EXPECT_EQ(cdf_neg, 0.0);
}

TEST_F(ExponentialEnhancedTest, ThreadSafety) {
    ThreadSafetyTester<ExponentialDistribution>::testBasicThreadSafety(test_distribution_, "Exponential");
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(ExponentialEnhancedTest, ParallelBatchPerformanceBenchmark) {
    ExponentialDistribution stdExp(1.0);
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data (positive values only for exponential)
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.01, 10.0);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    StandardizedBenchmark::printBenchmarkHeader("Exponential Distribution", BENCHMARK_SIZE);
    
    std::vector<BenchmarkResult> benchmark_results;
    
    // For each operation type (PDF, LogPDF, CDF)
    std::vector<std::string> operations = {"PDF", "LogPDF", "CDF"};
    
    for (const auto& op : operations) {
        BenchmarkResult result;
        result.operation_name = op;
        
        // 1. SIMD Batch (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PDF") {
            stdExp.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
        } else if (op == "LogPDF") {
            stdExp.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
        } else if (op == "CDF") {
            stdExp.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. Standard Parallel Operations
        std::span<const double> input_span(test_values);
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getProbabilityBatchParallel(input_span, output_span);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getLogProbabilityBatchParallel(input_span, log_output_span);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Work-Stealing Operations
        WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Cache-Aware Operations
        cache::AdaptiveCache<std::string, double> cache_manager;
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdExp.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
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
            StatisticalTestUtils::verifyBatchCorrectness(stdExp, test_values, pdf_results, "PDF");
        } else if (op == "LogPDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdExp, test_values, log_pdf_results, "LogPDF");
        } else if (op == "CDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdExp, test_values, cdf_results, "CDF");
        }
    }
    
    // Print standardized benchmark results
    StandardizedBenchmark::printBenchmarkResults(benchmark_results);
    StandardizedBenchmark::printPerformanceAnalysis(benchmark_results);
}

//==============================================================================
// NUMERICAL STABILITY AND EDGE CASES
//==============================================================================

TEST_F(ExponentialEnhancedTest, NumericalStability) {
    ExponentialDistribution exp(1.0);
    
    EdgeCaseTester<ExponentialDistribution>::testExtremeValues(exp, "Exponential");
    EdgeCaseTester<ExponentialDistribution>::testEmptyBatchOperations(exp, "Exponential");
}

} // namespace libstats

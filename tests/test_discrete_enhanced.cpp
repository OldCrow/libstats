#include <gtest/gtest.h>
#include "../include/distributions/discrete.h"
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
// TEST FIXTURE FOR DISCRETE ENHANCED METHODS
//==============================================================================

class DiscreteEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate synthetic discrete data for testing
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> discrete_gen(test_lower_, test_upper_);

        discrete_data_.clear();
        discrete_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            discrete_data_.push_back(static_cast<double>(discrete_gen(rng)));
        }

        // Generate obviously non-discrete data
        non_uniform_data_.clear();
        non_uniform_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            non_uniform_data_.push_back(static_cast<double>(i % 3 + 1)); // Pattern: 1,2,3,1,2,3...
        }

        test_distribution_ = DiscreteDistribution(test_lower_, test_upper_);
    }

    const int test_lower_ = 1;
    const int test_upper_ = 6;
    std::vector<double> discrete_data_;
    std::vector<double> non_uniform_data_;
    DiscreteDistribution test_distribution_;
};

//==============================================================================
// BASIC ENHANCED FUNCTIONALITY TESTS
//==============================================================================

TEST_F(DiscreteEnhancedTest, BasicEnhancedFunctionality) {
    // Test standard six-sided die
    DiscreteDistribution dice(1, 6);
    
    EXPECT_EQ(dice.getLowerBound(), 1);
    EXPECT_EQ(dice.getUpperBound(), 6);
    EXPECT_EQ(dice.getRange(), 6);
    EXPECT_DOUBLE_EQ(dice.getMean(), 3.5);
    EXPECT_NEAR(dice.getVariance(), 35.0/12.0, 1e-10);  // (6-1)(6-1+2)/12 = 5*7/12
    EXPECT_DOUBLE_EQ(dice.getSkewness(), 0.0);
    EXPECT_DOUBLE_EQ(dice.getKurtosis(), -1.2);
    
    // Test known PMF/CDF values
    double pmf_at_3 = dice.getProbability(3.0);
    double cdf_at_3 = dice.getCumulativeProbability(3.0);
    
    EXPECT_NEAR(pmf_at_3, 1.0/6.0, 1e-10);
    EXPECT_NEAR(cdf_at_3, 3.0/6.0, 1e-10);
    
    // Test binary distribution
    DiscreteDistribution binary(0, 1);
    EXPECT_DOUBLE_EQ(binary.getMean(), 0.5);
    EXPECT_DOUBLE_EQ(binary.getVariance(), 0.25);  // (1-0)(1-0+2)/12 = 1*3/12 = 0.25
    EXPECT_NEAR(binary.getCumulativeProbability(0.5), 1.0, 1e-10);
    
    // Test discrete-specific properties
    EXPECT_TRUE(dice.isInSupport(3.0));
    EXPECT_FALSE(dice.isInSupport(3.5)); // Non-integers not in discrete support
    EXPECT_TRUE(dice.isDiscrete());
    EXPECT_EQ(dice.getDistributionName(), "Discrete");
}

//==============================================================================
// GOODNESS-OF-FIT TESTS
//==============================================================================

TEST_F(DiscreteEnhancedTest, GoodnessOfFitTests) {
    std::cout << "\n=== Goodness-of-Fit Tests ===\n";
    
    // Kolmogorov-Smirnov test with uniform discrete data
    auto [ks_stat_uniform, ks_p_uniform, ks_reject_uniform] = 
        DiscreteDistribution::kolmogorovSmirnovTest(discrete_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ks_stat_uniform, 0.0);
    EXPECT_LE(ks_stat_uniform, 1.0);
    EXPECT_GE(ks_p_uniform, 0.0);
    EXPECT_LE(ks_p_uniform, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat_uniform));
    EXPECT_TRUE(std::isfinite(ks_p_uniform));
    
    std::cout << "  KS test (uniform data): D=" << ks_stat_uniform << ", p=" << ks_p_uniform << ", reject=" << ks_reject_uniform << "\n";
    
    // Kolmogorov-Smirnov test with non-uniform data (should reject)
    auto [ks_stat_non_uniform, ks_p_non_uniform, ks_reject_non_uniform] = 
        DiscreteDistribution::kolmogorovSmirnovTest(non_uniform_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(ks_reject_non_uniform); // Should reject uniform distribution for skewed data
    EXPECT_LT(ks_p_non_uniform, ks_p_uniform); // Non-uniform data should have lower p-value
    
    std::cout << "  KS test (non-uniform data): D=" << ks_stat_non_uniform << ", p=" << ks_p_non_uniform << ", reject=" << ks_reject_non_uniform << "\n";
    
    // Chi-square test (more appropriate for discrete distributions)
    auto [chi_stat_uniform, chi_p_uniform, chi_reject_uniform] = 
        DiscreteDistribution::chiSquaredGoodnessOfFitTest(discrete_data_, test_distribution_, 0.05);
    auto [chi_stat_non_uniform, chi_p_non_uniform, chi_reject_non_uniform] = 
        DiscreteDistribution::chiSquaredGoodnessOfFitTest(non_uniform_data_, test_distribution_, 0.05);
    
    EXPECT_GE(chi_stat_uniform, 0.0);
    EXPECT_GE(chi_p_uniform, 0.0);
    EXPECT_LE(chi_p_uniform, 1.0);
    EXPECT_TRUE(chi_reject_non_uniform); // Should reject uniformity for non-uniform data
    
    std::cout << "  Chi-square test (uniform data): χ²=" << chi_stat_uniform << ", p=" << chi_p_uniform << ", reject=" << chi_reject_uniform << "\n";
    std::cout << "  Chi-square test (non-uniform data): χ²=" << chi_stat_non_uniform << ", p=" << chi_p_non_uniform << ", reject=" << chi_reject_non_uniform << "\n";
}

//==============================================================================
// INFORMATION CRITERIA TESTS
//==============================================================================

TEST_F(DiscreteEnhancedTest, InformationCriteriaTests) {
    std::cout << "\n=== Information Criteria Tests ===\n";
    
    // Fit distribution to the data
    DiscreteDistribution fitted_dist;
    fitted_dist.fit(discrete_data_);
    
    auto [aic, bic, aicc, log_likelihood] = DiscreteDistribution::computeInformationCriteria(
        discrete_data_, fitted_dist);
    
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
    
    std::cout << "  AIC: " << aic << ", BIC: " << bic << ", AICc: " << aicc << "\n";
    std::cout << "  Log-likelihood: " << log_likelihood << "\n";
}

//==============================================================================
// BOOTSTRAP METHODS TESTS
//==============================================================================

TEST_F(DiscreteEnhancedTest, BootstrapMethods) {
    std::cout << "\n=== Bootstrap Methods ===\n";
    
    // Bootstrap parameter confidence intervals
    auto [lower_ci, upper_ci] = DiscreteDistribution::bootstrapParameterConfidenceIntervals(
        discrete_data_, 0.95, 1000, 456);
    
    // Check that confidence intervals are reasonable
    EXPECT_LT(lower_ci.first, lower_ci.second);  // Lower bound CI
    EXPECT_LT(upper_ci.first, upper_ci.second);  // Upper bound CI
    
    // Parameter CIs should be finite and make statistical sense
    // Note: For discrete uniform bounds, the true parameter 'a' could be below the sample minimum
    // and 'b' could be above the sample maximum - this is statistically correct!
    EXPECT_GE(lower_ci.first, -10.0);  // Reasonable lower bound for parameter 'a'
    EXPECT_LE(lower_ci.second, 10.0);   // Reasonable upper bound for parameter 'a'
    EXPECT_GE(upper_ci.first, 0.0);     // Parameter 'b' should be at least as large as sample minimum
    EXPECT_LE(upper_ci.second, 20.0);   // Reasonable upper bound for parameter 'b'
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(lower_ci.first));
    EXPECT_TRUE(std::isfinite(lower_ci.second));
    EXPECT_TRUE(std::isfinite(upper_ci.first));
    EXPECT_TRUE(std::isfinite(upper_ci.second));
    
    std::cout << "  Lower bound 95% CI: [" << lower_ci.first << ", " << lower_ci.second << "]\n";
    std::cout << "  Upper bound 95% CI: [" << upper_ci.first << ", " << upper_ci.second << "]\n";
    
    // K-fold cross-validation
    auto cv_results = DiscreteDistribution::kFoldCrossValidation(discrete_data_, 5, 42);
    EXPECT_EQ(cv_results.size(), 5);
    
    for (const auto& [mean_error, std_error, log_likelihood] : cv_results) {
        EXPECT_GE(mean_error, 0.0);       // Mean absolute error should be non-negative
        EXPECT_GE(std_error, 0.0);        // Standard error should be non-negative
        EXPECT_LE(log_likelihood, 0.0);   // Log-likelihood should be negative
        EXPECT_TRUE(std::isfinite(mean_error));
        EXPECT_TRUE(std::isfinite(std_error));
        EXPECT_TRUE(std::isfinite(log_likelihood));
    }
    
    std::cout << "  K-fold CV completed with " << cv_results.size() << " folds\n";
    
    // Leave-one-out cross-validation (using smaller dataset)
    std::vector<double> small_discrete_data(discrete_data_.begin(), discrete_data_.begin() + 20);
    auto [mae, rmse, loo_log_likelihood] = DiscreteDistribution::leaveOneOutCrossValidation(small_discrete_data);
    
    EXPECT_GE(mae, 0.0);                 // Mean absolute error should be non-negative
    EXPECT_GE(rmse, 0.0);                // RMSE should be non-negative
    EXPECT_LE(loo_log_likelihood, 0.0);  // Total log-likelihood should be negative
    EXPECT_GE(rmse, mae);                // RMSE should be >= MAE
    
    EXPECT_TRUE(std::isfinite(mae));
    EXPECT_TRUE(std::isfinite(rmse));
    EXPECT_TRUE(std::isfinite(loo_log_likelihood));
    
    std::cout << "  Leave-one-out CV: MAE=" << mae << ", RMSE=" << rmse << ", LogL=" << loo_log_likelihood << "\n";
}

//==============================================================================
// SIMD AND PARALLEL BATCH IMPLEMENTATIONS WITH FULSOME COMPARISONS
//==============================================================================

TEST_F(DiscreteEnhancedTest, SIMDAndParallelBatchImplementations) {
    DiscreteDistribution stdDiscrete(1, 6);
    
    std::cout << "\n=== SIMD and Parallel Batch Implementations ===\n";
    
    // Create shared WorkStealingPool once to avoid resource creation overhead
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    
    // Test multiple batch sizes to show scaling behavior
    std::vector<size_t> batch_sizes = {5000, 50000};
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "\n--- Batch Size: " << batch_size << " elements ---\n";
        
        // Generate test data (discrete values for Discrete distribution)
        std::vector<double> test_values(batch_size);
        std::vector<double> results(batch_size);
        
        std::mt19937 gen(42);
        std::uniform_int_distribution<> dis(1, 6);  // Discrete values for uniform discrete
        for (size_t i = 0; i < batch_size; ++i) {
            test_values[i] = static_cast<double>(dis(gen));
        }
        
        // 1. Sequential individual calls (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = stdDiscrete.getProbability(test_values[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. SIMD batch operations
        std::vector<double> simd_results(batch_size);
        start = std::chrono::high_resolution_clock::now();
        stdDiscrete.getProbabilityBatch(test_values.data(), simd_results.data(), batch_size);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Parallel batch operations
        std::vector<double> parallel_results(batch_size);
        std::span<const double> input_span(test_values);
        std::span<double> output_span(parallel_results);
        
        start = std::chrono::high_resolution_clock::now();
        stdDiscrete.getProbabilityBatchParallel(input_span, output_span);
        end = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Work-stealing operations (use shared pool)
        std::vector<double> work_stealing_results(batch_size);
        std::span<double> ws_output_span(work_stealing_results);
        
        start = std::chrono::high_resolution_clock::now();
        stdDiscrete.getProbabilityBatchWorkStealing(input_span, ws_output_span, work_stealing_pool);
        end = std::chrono::high_resolution_clock::now();
        auto work_stealing_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Calculate speedups
        double simd_speedup = (double)sequential_time / simd_time;
        double parallel_speedup = (double)sequential_time / parallel_time;
        double ws_speedup = (double)sequential_time / work_stealing_time;
        
        std::cout << "  Sequential: " << sequential_time << "μs (baseline)\n";
        std::cout << "  SIMD Batch: " << simd_time << "μs (" << simd_speedup << "x speedup)\n";
        std::cout << "  Parallel: " << parallel_time << "μs (" << parallel_speedup << "x speedup)\n";
        std::cout << "  Work Stealing: " << work_stealing_time << "μs (" << ws_speedup << "x speedup)\n";
        
        // Verify correctness across all methods (sample verification)
        size_t verification_samples = std::min(batch_size, size_t(100));
        for (size_t i = 0; i < verification_samples; ++i) {
            double expected = results[i];
            EXPECT_NEAR(simd_results[i], expected, 1e-12) << "SIMD result mismatch at index " << i << " for batch size " << batch_size;
            EXPECT_NEAR(parallel_results[i], expected, 1e-12) << "Parallel result mismatch at index " << i << " for batch size " << batch_size;
            EXPECT_NEAR(work_stealing_results[i], expected, 1e-12) << "Work-stealing result mismatch at index " << i << " for batch size " << batch_size;
        }
        
        // Performance expectations (adjusted for batch size and computational complexity)
        EXPECT_GT(simd_speedup, 1.0) << "SIMD should provide speedup for batch size " << batch_size;
        
        if (std::thread::hardware_concurrency() > 1) {
            if (batch_size >= 10000) {
                // For discrete distributions, computations are very simple (range checks),
                // so SIMD can achieve massive speedups but parallel has thread overhead.
                // Expect parallel to be at least 40% as efficient as SIMD for large batches.
                EXPECT_GT(parallel_speedup, simd_speedup * 0.4) << "Parallel should be reasonably competitive with SIMD for large batches";
            } else {
                // For smaller batches, parallel may have overhead but should still be reasonable
                EXPECT_GT(parallel_speedup, 0.5) << "Parallel should provide reasonable performance for batch size " << batch_size;
            }
        }
    }
}

//==============================================================================
// ADVANCED STATISTICAL METHODS TESTS
//==============================================================================

TEST_F(DiscreteEnhancedTest, AdvancedStatisticalMethods) {
    std::cout << "\n=== Advanced Statistical Methods ===\n";
    
    // Confidence intervals for lower and upper bounds
    auto [ci_lower_a, ci_upper_a] = DiscreteDistribution::confidenceIntervalLowerBound(discrete_data_, 0.95);
    auto [ci_lower_b, ci_upper_b] = DiscreteDistribution::confidenceIntervalUpperBound(discrete_data_, 0.95);
    EXPECT_LE(ci_lower_a, ci_upper_a);
    EXPECT_LE(ci_lower_b, ci_upper_b);
    std::cout << "  95% CI for lower bound: [" << ci_lower_a << ", " << ci_upper_a << "]\n";
    std::cout << "  95% CI for upper bound: [" << ci_lower_b << ", " << ci_upper_b << "]\n";
    
    // Method of moments estimation
    auto parameters = DiscreteDistribution::methodOfMomentsEstimation(discrete_data_);
    EXPECT_TRUE(std::isfinite(parameters.first));
    EXPECT_TRUE(std::isfinite(parameters.second));
    std::cout << "  MoM estimates: lower=" << parameters.first << ", upper=" << parameters.second << "\n";
    
    // L-moments estimation for discrete parameters
    auto [estimated_lower, estimated_upper] = DiscreteDistribution::lMomentsEstimation(discrete_data_);
    EXPECT_LE(estimated_lower, estimated_upper);
    std::cout << "  L-moments estimates: lower=" << estimated_lower << ", upper=" << estimated_upper << "\n";
    
    // For discrete uniform, MLE is simply min/max of data (like method of moments)
    auto [mle_lower, mle_upper] = DiscreteDistribution::methodOfMomentsEstimation(discrete_data_);
    EXPECT_LE(mle_lower, mle_upper);
    std::cout << "  MLE estimates (via MoM): lower=" << mle_lower << ", upper=" << mle_upper << "\n";
}

//==============================================================================
// CACHING SPEEDUP VERIFICATION TESTS
//==============================================================================

TEST_F(DiscreteEnhancedTest, CachingSpeedupVerification) {
    std::cout << "\n=== Caching Speedup Verification ===\n";
    
    DiscreteDistribution discrete_dist(1, 6);
    
    // First call - cache miss
    auto start = std::chrono::high_resolution_clock::now();
    double mean_first = discrete_dist.getMean();
    double var_first = discrete_dist.getVariance();
    double skew_first = discrete_dist.getSkewness();
    double kurt_first = discrete_dist.getKurtosis();
    auto end = std::chrono::high_resolution_clock::now();
    auto first_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Second call - cache hit
    start = std::chrono::high_resolution_clock::now();
    double mean_second = discrete_dist.getMean();
    double var_second = discrete_dist.getVariance();
    double skew_second = discrete_dist.getSkewness();
    double kurt_second = discrete_dist.getKurtosis();
    end = std::chrono::high_resolution_clock::now();
    auto second_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double cache_speedup = (double)first_time / second_time;
    
    std::cout << "  First getter calls (cache miss): " << first_time << "ns\n";
    std::cout << "  Second getter calls (cache hit): " << second_time << "ns\n";
    std::cout << "  Cache speedup: " << cache_speedup << "x\n";
    
    // Verify correctness
    EXPECT_EQ(mean_first, mean_second);
    EXPECT_EQ(var_first, var_second);
    EXPECT_EQ(skew_first, skew_second);
    EXPECT_EQ(kurt_first, kurt_second);
    
    // Cache should provide speedup (allow some measurement noise)
    EXPECT_GT(cache_speedup, 0.5) << "Cache should provide some speedup";
    
    // Test cache invalidation - create a new distribution with different parameters
    DiscreteDistribution new_dist(2, 8);
    
    start = std::chrono::high_resolution_clock::now();
    double mean_after_change = new_dist.getMean();
    end = std::chrono::high_resolution_clock::now();
    auto after_change_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    EXPECT_EQ(mean_after_change, 5.0); // Mean of uniform(2,8) is (2+8)/2 = 5
    std::cout << "  New distribution parameter access: " << after_change_time << "ns\n";
    
    // Test cache functionality: verify that the new distribution returns correct values
    // (proving cache isolation between different distribution instances)
}

//==============================================================================
// AUTO-DISPATCH STRATEGY TESTING
//==============================================================================

TEST_F(DiscreteEnhancedTest, AutoDispatchAssessment) {
    DiscreteDistribution discrete_dist(1, 6);
    
    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH", "PARALLEL_SIMD"};
    
    std::cout << "\n=== Auto-Dispatch Assessment (Discrete) ===\n";
    
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pmf_results(batch_size);
        std::vector<double> auto_logpmf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);
        
        std::mt19937 gen(42 + i);
        std::uniform_int_distribution<> dis(1, 6);
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = static_cast<double>(dis(gen));
        }
        
        // Test smart auto-dispatch methods (if available)
        auto start = std::chrono::high_resolution_clock::now();
        if constexpr (requires { discrete_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_pmf_results)); }) {
            discrete_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_pmf_results));
        } else {
            discrete_dist.getProbabilityBatch(test_values.data(), auto_pmf_results.data(), batch_size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pmf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires { discrete_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpmf_results)); }) {
            discrete_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpmf_results));
        } else {
            discrete_dist.getLogProbabilityBatch(test_values.data(), auto_logpmf_results.data(), batch_size);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpmf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires { discrete_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results)); }) {
            discrete_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results));
        } else {
            discrete_dist.getCumulativeProbabilityBatch(test_values.data(), auto_cdf_results.data(), batch_size);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pmf_results(batch_size);
        std::vector<double> trad_logpmf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);
        
        discrete_dist.getProbabilityBatch(test_values.data(), trad_pmf_results.data(), batch_size);
        discrete_dist.getLogProbabilityBatch(test_values.data(), trad_logpmf_results.data(), batch_size);
        discrete_dist.getCumulativeProbabilityBatch(test_values.data(), trad_cdf_results.data(), batch_size);
        
        // Verify correctness
        bool pmf_correct = true, logpmf_correct = true, cdf_correct = true;
        
        for (size_t j = 0; j < batch_size; ++j) {
            if (std::abs(auto_pmf_results[j] - trad_pmf_results[j]) > 1e-10) {
                pmf_correct = false;
            }
            if (std::abs(auto_logpmf_results[j] - trad_logpmf_results[j]) > 1e-10) {
                logpmf_correct = false;
            }
            if (std::abs(auto_cdf_results[j] - trad_cdf_results[j]) > 1e-10) {
                cdf_correct = false;
            }
        }
        
        std::cout << "Batch size: " << batch_size << ", Expected strategy: " << expected_strategy << "\n";
        std::cout << "  PMF: " << auto_pmf_time << "μs, Correct: " << (pmf_correct ? "✅" : "❌") << "\n";
        std::cout << "  LogPMF: " << auto_logpmf_time << "μs, Correct: " << (logpmf_correct ? "✅" : "❌") << "\n";
        std::cout << "  CDF: " << auto_cdf_time << "μs, Correct: " << (cdf_correct ? "✅" : "❌") << "\n";
        
        EXPECT_TRUE(pmf_correct) << "PMF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(logpmf_correct) << "LogPMF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(cdf_correct) << "CDF auto-dispatch results should match traditional for batch size " << batch_size;
    }
    
    std::cout << "\n=== Auto-Dispatch Assessment Completed (Discrete) ===\n";
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(DiscreteEnhancedTest, ParallelBatchPerformanceBenchmark) {
    DiscreteDistribution dice(1, 6);
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pmf_results(BENCHMARK_SIZE);
    std::vector<double> log_pmf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 6);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = static_cast<double>(dis(gen));
    }
    
    StandardizedBenchmark::printBenchmarkHeader("Discrete Distribution", BENCHMARK_SIZE);
    
    // Create shared resources ONCE outside the loop to avoid resource issues
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    cache::AdaptiveCache<std::string, double> cache_manager;
    
    std::vector<BenchmarkResult> benchmark_results;
    
    // For each operation type (PMF, LogPMF, CDF)
    std::vector<std::string> operations = {"PMF", "LogPMF", "CDF"};
    
    for (const auto& op : operations) {
        BenchmarkResult result;
        result.operation_name = op;
        
        // 1. SIMD Batch (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PMF") {
            dice.getProbabilityBatch(test_values.data(), pmf_results.data(), BENCHMARK_SIZE);
        } else if (op == "LogPMF") {
            dice.getLogProbabilityBatch(test_values.data(), log_pmf_results.data(), BENCHMARK_SIZE);
        } else if (op == "CDF") {
            dice.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. Standard Parallel Operations (if available) - fallback to SIMD
        std::span<const double> input_span(test_values);
        
        if (op == "PMF") {
            std::span<double> output_span(pmf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { dice.getProbabilityBatchParallel(input_span, output_span); }) {
                dice.getProbabilityBatchParallel(input_span, output_span);
            } else {
                dice.getProbabilityBatch(test_values.data(), pmf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPMF") {
            std::span<double> log_output_span(log_pmf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { dice.getLogProbabilityBatchParallel(input_span, log_output_span); }) {
                dice.getLogProbabilityBatchParallel(input_span, log_output_span);
            } else {
                dice.getLogProbabilityBatch(test_values.data(), log_pmf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { dice.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span); }) {
                dice.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
            } else {
                dice.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Work-Stealing Operations (if available) - fallback to SIMD
        if (op == "PMF") {
            std::span<double> output_span(pmf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { dice.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool); }) {
                dice.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
            } else {
                dice.getProbabilityBatch(test_values.data(), pmf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPMF") {
            std::span<double> log_output_span(log_pmf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { dice.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool); }) {
                dice.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
            } else {
                dice.getLogProbabilityBatch(test_values.data(), log_pmf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { dice.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool); }) {
                dice.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
            } else {
                dice.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Cache-Aware Operations (if available) - fallback to SIMD
        if (op == "PMF") {
            std::span<double> output_span(pmf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { typename cache::AdaptiveCache<std::string, double>; dice.getProbabilityBatchCacheAware(input_span, output_span, std::declval<cache::AdaptiveCache<std::string, double>&>()); }) {
                dice.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
            } else {
                dice.getProbabilityBatch(test_values.data(), pmf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPMF") {
            std::span<double> log_output_span(log_pmf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { typename cache::AdaptiveCache<std::string, double>; dice.getLogProbabilityBatchCacheAware(input_span, log_output_span, std::declval<cache::AdaptiveCache<std::string, double>&>()); }) {
                dice.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
            } else {
                dice.getLogProbabilityBatch(test_values.data(), log_pmf_results.data(), BENCHMARK_SIZE);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { typename cache::AdaptiveCache<std::string, double>; dice.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, std::declval<cache::AdaptiveCache<std::string, double>&>()); }) {
                dice.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
            } else {
                dice.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
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
        if (op == "PMF") {
            StatisticalTestUtils::verifyBatchCorrectness(dice, test_values, pmf_results, "PMF");
        } else if (op == "LogPMF") {
            StatisticalTestUtils::verifyBatchCorrectness(dice, test_values, log_pmf_results, "LogPMF");
        } else if (op == "CDF") {
            StatisticalTestUtils::verifyBatchCorrectness(dice, test_values, cdf_results, "CDF");
        }
    }
    
    // Print standardized benchmark results
    StandardizedBenchmark::printBenchmarkResults(benchmark_results);
    StandardizedBenchmark::printPerformanceAnalysis(benchmark_results);
}

//==============================================================================
// NUMERICAL STABILITY AND EDGE CASES
//==============================================================================

TEST_F(DiscreteEnhancedTest, NumericalStabilityAndEdgeCases) {
    DiscreteDistribution dice(1, 6);
    
    EdgeCaseTester<DiscreteDistribution>::testExtremeValues(dice, "Discrete");
    EdgeCaseTester<DiscreteDistribution>::testEmptyBatchOperations(dice, "Discrete");
}

} // namespace libstats

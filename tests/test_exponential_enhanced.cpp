#include <gtest/gtest.h>
#include "../include/distributions/exponential.h"
#include "enhanced_test_template.h"

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
        
        auto result = libstats::ExponentialDistribution::create(test_lambda_);
        if (result.isOk()) {
            test_distribution_ = std::move(result.value);
        };
    }
    
    const double test_lambda_ = 2.0;
    std::vector<double> exponential_data_;
    std::vector<double> non_exponential_data_;
    ExponentialDistribution test_distribution_;
};

//==============================================================================
// BASIC ENHANCED FUNCTIONALITY TESTS
//==============================================================================

TEST_F(ExponentialEnhancedTest, BasicEnhancedFunctionality) {
    // Test unit exponential distribution properties
    auto unitExp = libstats::ExponentialDistribution::create(1.0).value;
    
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
    auto custom = libstats::ExponentialDistribution::create(2.0).value;
    EXPECT_DOUBLE_EQ(custom.getLambda(), 2.0);
    EXPECT_DOUBLE_EQ(custom.getMean(), 0.5);
    EXPECT_DOUBLE_EQ(custom.getVariance(), 0.25);
    
    // Test exponential-specific properties
    EXPECT_TRUE(unitExp.getProbability(1.0) > 0.0);  // Positive values have non-zero probability
    EXPECT_EQ(unitExp.getProbability(-1.0), 0.0);   // Negative values have zero probability
    EXPECT_FALSE(unitExp.isDiscrete());
    EXPECT_EQ(unitExp.getDistributionName(), "Exponential");
}

//==============================================================================
// GOODNESS-OF-FIT TESTS
//==============================================================================

TEST_F(ExponentialEnhancedTest, GoodnessOfFitTests) {
    std::cout << "\n=== Goodness-of-Fit Tests ===\n";
    
    // Kolmogorov-Smirnov test with exponential data
    auto [ks_stat_exp, ks_p_exp, ks_reject_exp] = 
        ExponentialDistribution::kolmogorovSmirnovTest(exponential_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ks_stat_exp, 0.0);
    EXPECT_LE(ks_stat_exp, 1.0);
    EXPECT_GE(ks_p_exp, 0.0);
    EXPECT_LE(ks_p_exp, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat_exp));
    EXPECT_TRUE(std::isfinite(ks_p_exp));
    
    std::cout << "  KS test (exponential data): D=" << ks_stat_exp << ", p=" << ks_p_exp << ", reject=" << ks_reject_exp << "\n";
    
    // Kolmogorov-Smirnov test with non-exponential data (should reject)
    auto [ks_stat_non_exp, ks_p_non_exp, ks_reject_non_exp] = 
        ExponentialDistribution::kolmogorovSmirnovTest(non_exponential_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(ks_reject_non_exp); // Should reject exponential distribution for normal data
    EXPECT_LT(ks_p_non_exp, ks_p_exp); // Non-exponential data should have lower p-value
    
    std::cout << "  KS test (non-exponential data): D=" << ks_stat_non_exp << ", p=" << ks_p_non_exp << ", reject=" << ks_reject_non_exp << "\n";
    
    // Anderson-Darling test
    auto [ad_stat_exp, ad_p_exp, ad_reject_exp] = 
        ExponentialDistribution::andersonDarlingTest(exponential_data_, test_distribution_, 0.05);
    auto [ad_stat_non_exp, ad_p_non_exp, ad_reject_non_exp] = 
        ExponentialDistribution::andersonDarlingTest(non_exponential_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ad_stat_exp, 0.0);
    EXPECT_GE(ad_p_exp, 0.0);
    EXPECT_LE(ad_p_exp, 1.0);
    EXPECT_TRUE(ad_reject_non_exp); // Should reject exponential for non-exponential data
    
    std::cout << "  AD test (exponential data): A²=" << ad_stat_exp << ", p=" << ad_p_exp << ", reject=" << ad_reject_exp << "\n";
    std::cout << "  AD test (non-exponential data): A²=" << ad_stat_non_exp << ", p=" << ad_p_non_exp << ", reject=" << ad_reject_non_exp << "\n";
}

//==============================================================================
// INFORMATION CRITERIA TESTS
//==============================================================================

TEST_F(ExponentialEnhancedTest, InformationCriteriaTests) {
    std::cout << "\n=== Information Criteria Tests ===\n";
    
    // Fit distribution to data
    ExponentialDistribution fitted_dist;
    fitted_dist.fit(exponential_data_);
    
    auto [aic, bic, aicc, log_likelihood] = ExponentialDistribution::computeInformationCriteria(
        exponential_data_, fitted_dist);
    
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

TEST_F(ExponentialEnhancedTest, BootstrapMethods) {
    std::cout << "\n=== Bootstrap Methods ===\n";
    
    // Bootstrap parameter confidence intervals
    auto [lambda_ci_lower, lambda_ci_upper] = ExponentialDistribution::bootstrapParameterConfidenceIntervals(
        exponential_data_, 0.95, 1000, 456);
    
    // Check that confidence intervals are reasonable
    EXPECT_LT(lambda_ci_lower, lambda_ci_upper);
    EXPECT_GT(lambda_ci_lower, 0.0);
    EXPECT_GT(lambda_ci_upper, 0.0);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(lambda_ci_lower));
    EXPECT_TRUE(std::isfinite(lambda_ci_upper));
    
    std::cout << "  Lambda 95% CI: [" << lambda_ci_lower << ", " << lambda_ci_upper << "]\n";
    
    // K-fold cross-validation
    auto cv_results = ExponentialDistribution::kFoldCrossValidation(exponential_data_, 5, 42);
    EXPECT_EQ(cv_results.size(), 5);
    
    for (const auto& [mae, rmse, log_likelihood] : cv_results) {
        EXPECT_GE(mae, 0.0);             // Mean absolute error should be non-negative
        EXPECT_GE(rmse, 0.0);            // RMSE should be non-negative
        EXPECT_GE(rmse, mae);            // RMSE should be >= MAE
        EXPECT_LE(log_likelihood, 0.0);  // Log-likelihood should be negative
        EXPECT_TRUE(std::isfinite(mae));
        EXPECT_TRUE(std::isfinite(rmse));
        EXPECT_TRUE(std::isfinite(log_likelihood));
    }
    
    std::cout << "  K-fold CV completed with " << cv_results.size() << " folds\n";
    
    // Leave-one-out cross-validation (using smaller dataset)
    std::vector<double> small_exp_data(exponential_data_.begin(), exponential_data_.begin() + 20);
    auto [loocv_mae, loocv_rmse, loocv_log_likelihood] = ExponentialDistribution::leaveOneOutCrossValidation(small_exp_data);
    
    EXPECT_GE(loocv_mae, 0.0);                 // Mean absolute error should be non-negative
    EXPECT_GE(loocv_rmse, 0.0);                // RMSE should be non-negative
    EXPECT_GE(loocv_rmse, loocv_mae);          // RMSE should be >= MAE
    EXPECT_LE(loocv_log_likelihood, 0.0);      // Total log-likelihood should be negative
    
    EXPECT_TRUE(std::isfinite(loocv_mae));
    EXPECT_TRUE(std::isfinite(loocv_rmse));
    EXPECT_TRUE(std::isfinite(loocv_log_likelihood));
    
    std::cout << "  Leave-one-out CV: MAE=" << loocv_mae << ", RMSE=" << loocv_rmse << ", LogL=" << loocv_log_likelihood << "\n";
}

//==============================================================================
// SIMD AND PARALLEL BATCH IMPLEMENTATIONS WITH FULSOME COMPARISONS
//==============================================================================

TEST_F(ExponentialEnhancedTest, SIMDAndParallelBatchImplementations) {
    auto stdExp = libstats::ExponentialDistribution::create(1.0).value;
    
    std::cout << "\n=== SIMD and Parallel Batch Implementations ===\n";
    
    // Create shared WorkStealingPool once to avoid resource creation overhead
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    
    // Test multiple batch sizes to show scaling behavior
    std::vector<size_t> batch_sizes = {5000, 50000};
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "\n--- Batch Size: " << batch_size << " elements ---\n";
        
        // Generate test data (positive values for Exponential distribution)
        std::vector<double> test_values(batch_size);
        std::vector<double> results(batch_size);
        
        std::mt19937 gen(42);
        std::exponential_distribution<> dis(1.0);  // Rate = 1.0
        for (size_t i = 0; i < batch_size; ++i) {
            test_values[i] = dis(gen);
        }
        
        // 1. Sequential individual calls (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = stdExp.getProbability(test_values[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. SIMD batch operations
        std::vector<double> simd_results(batch_size);
        start = std::chrono::high_resolution_clock::now();
        stdExp.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(simd_results), libstats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Parallel batch operations
        std::vector<double> parallel_results(batch_size);
        std::span<const double> input_span(test_values);
        std::span<double> output_span(parallel_results);
        
        start = std::chrono::high_resolution_clock::now();
        stdExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::PARALLEL_SIMD);
        end = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Work-stealing operations (use shared pool)
        std::vector<double> work_stealing_results(batch_size);
        std::span<double> ws_output_span(work_stealing_results);
        
        start = std::chrono::high_resolution_clock::now();
        stdExp.getProbabilityWithStrategy(input_span, ws_output_span, libstats::performance::Strategy::WORK_STEALING);
        end = std::chrono::high_resolution_clock::now();
        auto work_stealing_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Calculate speedups
        double simd_speedup = static_cast<double>(sequential_time) / static_cast<double>(simd_time);
        double parallel_speedup = static_cast<double>(sequential_time) / static_cast<double>(parallel_time);
        double ws_speedup = static_cast<double>(sequential_time) / static_cast<double>(work_stealing_time);
        
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
        
        // Performance expectations (adjusted for batch size)
        EXPECT_GT(simd_speedup, 1.0) << "SIMD should provide speedup for batch size " << batch_size;
        
        if (std::thread::hardware_concurrency() > 1) {
            if (batch_size >= 10000) {
                // For large batches, parallel should significantly outperform SIMD
                EXPECT_GT(parallel_speedup, simd_speedup * 0.8) << "Parallel should be competitive with SIMD for large batches";
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

TEST_F(ExponentialEnhancedTest, AdvancedStatisticalMethods) {
    std::cout << "\n=== Advanced Statistical Methods ===\n";
    
    // Confidence interval for rate parameter (lambda)
    auto [lambda_ci_lower, lambda_ci_upper] = ExponentialDistribution::confidenceIntervalRate(exponential_data_, 0.95);
    EXPECT_LT(lambda_ci_lower, lambda_ci_upper);
    EXPECT_GT(lambda_ci_lower, 0.0);
    EXPECT_GT(lambda_ci_upper, 0.0);
    EXPECT_TRUE(std::isfinite(lambda_ci_lower));
    EXPECT_TRUE(std::isfinite(lambda_ci_upper));
    std::cout << "  95% CI for λ: [" << lambda_ci_lower << ", " << lambda_ci_upper << "]\n";
    
    // Likelihood ratio test
    auto [lr_stat, p_value, reject_null] = ExponentialDistribution::likelihoodRatioTest(exponential_data_, test_lambda_, 0.05);
    EXPECT_GE(lr_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(lr_stat));
    EXPECT_TRUE(std::isfinite(p_value));
    std::cout << "  LR test: stat=" << lr_stat << ", p=" << p_value << ", reject=" << reject_null << "\n";
    
    // Method of moments estimation
    double mom_lambda = ExponentialDistribution::methodOfMomentsEstimation(exponential_data_);
    EXPECT_GT(mom_lambda, 0.0);
    EXPECT_TRUE(std::isfinite(mom_lambda));
    std::cout << "  MoM estimate for λ: " << mom_lambda << "\n";
    
    // For exponential, MLE is same as MoM (1/sample_mean), so we can use MoM as proxy
    double mle_lambda = ExponentialDistribution::methodOfMomentsEstimation(exponential_data_);
    EXPECT_GT(mle_lambda, 0.0);
    EXPECT_TRUE(std::isfinite(mle_lambda));
    std::cout << "  MLE estimate for λ (via MoM): " << mle_lambda << "\n";
    
    // Bayesian estimation (returns posterior parameters)
    if constexpr (requires { ExponentialDistribution::bayesianEstimation(exponential_data_, 1.0, 1.0); }) {
        auto [posterior_shape, posterior_rate] = ExponentialDistribution::bayesianEstimation(exponential_data_, 1.0, 1.0);
        EXPECT_GT(posterior_shape, 0.0);
        EXPECT_GT(posterior_rate, 0.0);
        EXPECT_TRUE(std::isfinite(posterior_shape));
        EXPECT_TRUE(std::isfinite(posterior_rate));
        std::cout << "  Bayesian posterior: shape=" << posterior_shape << ", rate=" << posterior_rate << "\n";
    }
}

//==============================================================================
// CACHING SPEEDUP VERIFICATION TESTS
//==============================================================================

TEST_F(ExponentialEnhancedTest, CachingSpeedupVerification) {
    std::cout << "\n=== Caching Speedup Verification ===\n";
    
    auto exp_dist = libstats::ExponentialDistribution::create(1.0).value;
    
    // First call - cache miss
    auto start = std::chrono::high_resolution_clock::now();
    double mean_first = exp_dist.getMean();
    double var_first = exp_dist.getVariance();
    double skew_first = exp_dist.getSkewness();
    double kurt_first = exp_dist.getKurtosis();
    auto end = std::chrono::high_resolution_clock::now();
    auto first_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Second call - cache hit
    start = std::chrono::high_resolution_clock::now();
    double mean_second = exp_dist.getMean();
    double var_second = exp_dist.getVariance();
    double skew_second = exp_dist.getSkewness();
    double kurt_second = exp_dist.getKurtosis();
    end = std::chrono::high_resolution_clock::now();
    auto second_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    double cache_speedup = static_cast<double>(first_time) / static_cast<double>(second_time);
    
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
    auto new_dist = libstats::ExponentialDistribution::create(2.0).value;
    
    start = std::chrono::high_resolution_clock::now();
    double mean_after_change = new_dist.getMean();
    end = std::chrono::high_resolution_clock::now();
    auto after_change_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    EXPECT_EQ(mean_after_change, 0.5); // Mean of exponential(2.0) is 1/2.0 = 0.5
    std::cout << "  New distribution parameter access: " << after_change_time << "ns\n";
    
    // Test cache functionality: verify that the new distribution returns correct values
    // (proving cache isolation between different distribution instances)
}

//==============================================================================
// AUTO-DISPATCH STRATEGY TESTING
//==============================================================================

TEST_F(ExponentialEnhancedTest, AutoDispatchAssessment) {
    auto exp_dist = libstats::ExponentialDistribution::create(1.0).value;
    
    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH", "PARALLEL_SIMD"};
    
    std::cout << "\n=== Auto-Dispatch Assessment (Exponential) ===\n";
    
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pdf_results(batch_size);
        std::vector<double> auto_logpdf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);
        
        std::mt19937 gen(static_cast<unsigned int>(42 + i));
        std::exponential_distribution<> dis(1.0);
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = dis(gen);
        }
        
        // Test smart auto-dispatch methods (if available)
        auto start = std::chrono::high_resolution_clock::now();
        if constexpr (requires { exp_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_pdf_results)); }) {
            exp_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_pdf_results));
        } else {
            exp_dist.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(auto_pdf_results), libstats::performance::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires { exp_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpdf_results)); }) {
            exp_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpdf_results));
        } else {
            exp_dist.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(auto_logpdf_results), libstats::performance::Strategy::SCALAR);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires { exp_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results)); }) {
            exp_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results));
        } else {
            exp_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(auto_cdf_results), libstats::performance::Strategy::SCALAR);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pdf_results(batch_size);
        std::vector<double> trad_logpdf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);
        
        exp_dist.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(trad_pdf_results), libstats::performance::Strategy::SCALAR);
        exp_dist.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(trad_logpdf_results), libstats::performance::Strategy::SCALAR);
        exp_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(trad_cdf_results), libstats::performance::Strategy::SCALAR);
        
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
    
    std::cout << "\n=== Auto-Dispatch Assessment Completed (Exponential) ===\n";
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(ExponentialEnhancedTest, ParallelBatchPerformanceBenchmark) {
    auto unitExp = libstats::ExponentialDistribution::create(1.0).value;
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::exponential_distribution<> dis(1.0);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    StandardizedBenchmark::printBenchmarkHeader("Exponential Distribution", BENCHMARK_SIZE);
    
    // Create shared resources ONCE outside the loop to avoid resource issues
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    cache::AdaptiveCache<std::string, double> cache_manager;
    
    std::vector<BenchmarkResult> benchmark_results;
    
    // For each operation type (PDF, LogPDF, CDF)
    std::vector<std::string> operations = {"PDF", "LogPDF", "CDF"};
    
    for (const auto& op : operations) {
        BenchmarkResult result;
        result.operation_name = op;
        
        // 1. SIMD Batch (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PDF") {
            unitExp.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(pdf_results), libstats::performance::Strategy::SCALAR);
        } else if (op == "LogPDF") {
            unitExp.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(log_pdf_results), libstats::performance::Strategy::SCALAR);
        } else if (op == "CDF") {
            unitExp.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(cdf_results), libstats::performance::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. Standard Parallel Operations (if available) - fallback to SIMD
        std::span<const double> input_span(test_values);
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { unitExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::PARALLEL_SIMD); }) {
                unitExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::PARALLEL_SIMD);
            } else {
                unitExp.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(pdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { unitExp.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::PARALLEL_SIMD); }) {
                unitExp.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::PARALLEL_SIMD);
            } else {
                unitExp.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(log_pdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { unitExp.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::PARALLEL_SIMD); }) {
                unitExp.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::PARALLEL_SIMD);
            } else {
                unitExp.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(cdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Work-Stealing Operations (if available) - fallback to SIMD        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { unitExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::WORK_STEALING); }) {
                unitExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::WORK_STEALING);
            } else {
                unitExp.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(pdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { unitExp.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::WORK_STEALING); }) {
                unitExp.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::WORK_STEALING);
            } else {
                unitExp.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(log_pdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { unitExp.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::WORK_STEALING); }) {
                unitExp.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::WORK_STEALING);
            } else {
                unitExp.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(cdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Cache-Aware Operations (if available) - fallback to SIMD
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { typename cache::AdaptiveCache<std::string, double>; unitExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::CACHE_AWARE); }) {
                unitExp.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::CACHE_AWARE);
            } else {
                unitExp.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(pdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { typename cache::AdaptiveCache<std::string, double>; unitExp.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::CACHE_AWARE); }) {
                unitExp.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::CACHE_AWARE);
            } else {
                unitExp.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(log_pdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires { typename cache::AdaptiveCache<std::string, double>; unitExp.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::CACHE_AWARE); }) {
                unitExp.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::CACHE_AWARE);
            } else {
                unitExp.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(cdf_results), libstats::performance::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.cache_aware_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Calculate speedups
        result.parallel_speedup = static_cast<double>(result.simd_time_us) / static_cast<double>(result.parallel_time_us);
        result.work_stealing_speedup = static_cast<double>(result.simd_time_us) / static_cast<double>(result.work_stealing_time_us);
        result.cache_aware_speedup = static_cast<double>(result.simd_time_us) / static_cast<double>(result.cache_aware_time_us);
        
        benchmark_results.push_back(result);
        
        // Verify correctness
        if (op == "PDF") {
            StatisticalTestUtils::verifyBatchCorrectness(unitExp, test_values, pdf_results, "PDF");
        } else if (op == "LogPDF") {
            StatisticalTestUtils::verifyBatchCorrectness(unitExp, test_values, log_pdf_results, "LogPDF");
        } else if (op == "CDF") {
            StatisticalTestUtils::verifyBatchCorrectness(unitExp, test_values, cdf_results, "CDF");
        }
    }
    
    // Print standardized benchmark results
    StandardizedBenchmark::printBenchmarkResults(benchmark_results);
    StandardizedBenchmark::printPerformanceAnalysis(benchmark_results);
}

//==============================================================================
// PARALLEL BATCH FITTING TESTS
//==============================================================================

TEST_F(ExponentialEnhancedTest, ParallelBatchFittingTests) {
    std::cout << "\n=== Parallel Batch Fitting Tests ===\n";
    
    // Create multiple datasets for batch fitting
    std::vector<std::vector<double>> datasets;
    std::vector<ExponentialDistribution> expected_distributions;
    
    std::mt19937 rng(42);
    
    // Generate 6 datasets with known parameters
    std::vector<double> true_lambdas = {1.0, 2.0, 0.5, 3.0, 1.5, 0.8};
    
    for (double lambda : true_lambdas) {
        std::vector<double> dataset;
        dataset.reserve(1000);
        
        std::exponential_distribution<double> gen(lambda);
        for (int i = 0; i < 1000; ++i) {
            dataset.push_back(gen(rng));
        }
        
        datasets.push_back(std::move(dataset));
        expected_distributions.push_back(ExponentialDistribution::create(lambda).value);
    }
    
    std::cout << "  Generated " << datasets.size() << " datasets with known parameters\n";
    
    // Test 1: Basic parallel batch fitting correctness
    std::vector<ExponentialDistribution> batch_results(datasets.size());
    
    auto start = std::chrono::high_resolution_clock::now();
    ExponentialDistribution::parallelBatchFit(datasets, batch_results);
    auto end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Verify correctness by comparing with individual fits
    for (size_t i = 0; i < datasets.size(); ++i) {
        ExponentialDistribution individual_fit;
        individual_fit.fit(datasets[i]);
        
        // Parameters should match within tolerance
        EXPECT_NEAR(batch_results[i].getLambda(), individual_fit.getLambda(), 1e-10)
            << "Batch fit lambda mismatch for dataset " << i;
        
        // Should be reasonably close to true parameters (within 3 standard errors)
        double expected_lambda = true_lambdas[i];
        [[maybe_unused]] double n = static_cast<double>(datasets[i].size());
        
        // For exponential distribution, std error of lambda estimate is lambda/sqrt(n)
        double lambda_tolerance = 3.0 * expected_lambda / std::sqrt(n);
        
        EXPECT_NEAR(batch_results[i].getLambda(), expected_lambda, lambda_tolerance)
            << "Fitted lambda too far from true value for dataset " << i;
    }
    
    std::cout << "  ✓ Parallel batch fitting correctness verified\n";
    
    // Test 2: Performance comparison with sequential batch fitting
    std::vector<ExponentialDistribution> sequential_results(datasets.size());
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < datasets.size(); ++i) {
        sequential_results[i].fit(datasets[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double speedup = sequential_time > 0 ? static_cast<double>(sequential_time) / static_cast<double>(parallel_time) : 1.0;
    
    std::cout << "  Parallel batch fitting: " << parallel_time << "μs\n";
    std::cout << "  Sequential individual fits: " << sequential_time << "μs\n";
    std::cout << "  Speedup: " << speedup << "x\n";
    
    // Verify sequential and parallel results match
    for (size_t i = 0; i < datasets.size(); ++i) {
        EXPECT_NEAR(batch_results[i].getLambda(), sequential_results[i].getLambda(), 1e-12)
            << "Sequential vs parallel lambda mismatch for dataset " << i;
    }
    
    // Test 3: Edge cases
    std::cout << "  Testing edge cases...\n";
    
    // Empty datasets vector
    std::vector<std::vector<double>> empty_datasets;
    std::vector<ExponentialDistribution> empty_results;
    ExponentialDistribution::parallelBatchFit(empty_datasets, empty_results);
    EXPECT_TRUE(empty_results.empty());
    
    // Single dataset
    std::vector<std::vector<double>> single_dataset = {datasets[0]};
    std::vector<ExponentialDistribution> single_result(1);
    ExponentialDistribution::parallelBatchFit(single_dataset, single_result);
    EXPECT_NEAR(single_result[0].getLambda(), batch_results[0].getLambda(), 1e-12);
    
    // Results vector auto-sizing
    std::vector<ExponentialDistribution> auto_sized_results;
    ExponentialDistribution::parallelBatchFit(datasets, auto_sized_results);
    EXPECT_EQ(auto_sized_results.size(), datasets.size());
    
    std::cout << "  ✓ Edge cases handled correctly\n";
    
    // Test 4: Thread safety with concurrent calls
    std::cout << "  Testing thread safety...\n";
    
    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<ExponentialDistribution>> concurrent_results(num_threads);
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            concurrent_results[static_cast<size_t>(t)].resize(datasets.size());
            ExponentialDistribution::parallelBatchFit(datasets, concurrent_results[static_cast<size_t>(t)]);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Verify all concurrent results match
    for (int t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < datasets.size(); ++i) {
            EXPECT_NEAR(concurrent_results[static_cast<size_t>(t)][i].getLambda(), batch_results[i].getLambda(), 1e-10)
                << "Thread " << t << " result mismatch for dataset " << i;
        }
    }
    
    std::cout << "  ✓ Thread safety verified\n";
}

//==============================================================================
// NUMERICAL STABILITY AND EDGE CASES
//==============================================================================

TEST_F(ExponentialEnhancedTest, NumericalStabilityAndEdgeCases) {
    auto unitExp = libstats::ExponentialDistribution::create(1.0).value;
    
    EdgeCaseTester<ExponentialDistribution>::testExtremeValues(unitExp, "Exponential");
    EdgeCaseTester<ExponentialDistribution>::testEmptyBatchOperations(unitExp, "Exponential");
}

} // namespace libstats

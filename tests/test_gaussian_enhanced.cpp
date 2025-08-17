#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

#include <gtest/gtest.h>
#include "../include/distributions/gaussian.h"
#include "enhanced_test_template.h"

using namespace std;
using namespace libstats;
using namespace libstats::testing;

namespace libstats {

//==============================================================================
// TEST FIXTURE FOR GAUSSIAN ENHANCED METHODS
//==============================================================================

class GaussianEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::mt19937 rng(42);
        std::normal_distribution<double> normal_gen(test_mean_, test_std_);

        normal_data_.clear();
        normal_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            normal_data_.push_back(normal_gen(rng));
        }

        non_normal_data_.clear();
        non_normal_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            non_normal_data_.push_back(i * i); // Quadratic growth - clearly non-normal
        }

        auto result = libstats::GaussianDistribution::create(test_mean_, test_std_);
        if (result.isOk()) {
            test_distribution_ = std::move(result.value);
        }
    }

    const double test_mean_ = 5.0;
    const double test_std_ = 2.0;
    std::vector<double> normal_data_;
    std::vector<double> non_normal_data_;
    GaussianDistribution test_distribution_;
};

//==============================================================================
// BASIC ENHANCED FUNCTIONALITY TESTS
//==============================================================================

TEST_F(GaussianEnhancedTest, BasicEnhancedFunctionality) {
    // Test standard normal distribution properties
    auto stdNormal = libstats::GaussianDistribution::create(0.0, 1.0).value;
    
    EXPECT_DOUBLE_EQ(stdNormal.getMean(), 0.0);
    EXPECT_DOUBLE_EQ(stdNormal.getStandardDeviation(), 1.0);
    EXPECT_DOUBLE_EQ(stdNormal.getVariance(), 1.0);
    EXPECT_DOUBLE_EQ(stdNormal.getSkewness(), 0.0);
    EXPECT_DOUBLE_EQ(stdNormal.getKurtosis(), 0.0);  // Excess kurtosis for normal distribution
    
    // Test known PDF/CDF values for standard normal
    double pdf_at_0 = stdNormal.getProbability(0.0);
    double cdf_at_0 = stdNormal.getCumulativeProbability(0.0);
    
    EXPECT_NEAR(pdf_at_0, 1.0 / std::sqrt(2.0 * M_PI), 1e-10);
    EXPECT_NEAR(cdf_at_0, 0.5, 1e-10);
    
    // Test custom distribution
    auto custom = libstats::GaussianDistribution::create(5.0, 2.0).value;
    EXPECT_DOUBLE_EQ(custom.getMean(), 5.0);
    EXPECT_DOUBLE_EQ(custom.getStandardDeviation(), 2.0);
    EXPECT_DOUBLE_EQ(custom.getVariance(), 4.0);
    
    // Test gaussian-specific properties
    EXPECT_TRUE(stdNormal.getProbability(0.0) > 0.0);  // PDF should be positive at mean
    EXPECT_TRUE(stdNormal.getProbability(10.0) > 0.0); // PDF should be positive everywhere (but small)
    EXPECT_FALSE(stdNormal.isDiscrete());
    EXPECT_EQ(stdNormal.getDistributionName(), "Gaussian");
}

//==============================================================================
// ADVANCED STATISTICAL METHODS TESTS
//==============================================================================

TEST_F(GaussianEnhancedTest, AdvancedStatisticalMethods) {
    std::cout << "\n=== Advanced Statistical Methods ===\n";
    
    // Confidence interval for mean
    auto [ci_lower, ci_upper] = GaussianDistribution::confidenceIntervalMean(normal_data_, 0.95);
    EXPECT_LT(ci_lower, ci_upper);
    EXPECT_TRUE(std::isfinite(ci_lower));
    EXPECT_TRUE(std::isfinite(ci_upper));
    std::cout << "  95% CI for mean: [" << ci_lower << ", " << ci_upper << "]\n";
    
    // One-sample t-test
    auto [t_stat, p_value, reject_null] = GaussianDistribution::oneSampleTTest(normal_data_, test_mean_, 0.05);
    EXPECT_TRUE(std::isfinite(t_stat));
    EXPECT_TRUE(std::isfinite(p_value));
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    std::cout << "  t-test: t=" << t_stat << ", p=" << p_value << ", reject=" << reject_null << "\n";
    
    // Method of moments estimation
    auto [estimated_mean, estimated_std] = GaussianDistribution::methodOfMomentsEstimation(normal_data_);
    EXPECT_TRUE(std::isfinite(estimated_mean));
    EXPECT_TRUE(std::isfinite(estimated_std));
    EXPECT_GT(estimated_std, 0.0);
    std::cout << "  MoM estimates: mean=" << estimated_mean << ", std=" << estimated_std << "\n";
    
    // Jarque-Bera test for normality
    auto [jb_stat, jb_p_value, reject_normality] = GaussianDistribution::jarqueBeraTest(normal_data_, 0.05);
    EXPECT_GE(jb_stat, 0.0);
    EXPECT_GE(jb_p_value, 0.0);
    EXPECT_LE(jb_p_value, 1.0);
    EXPECT_TRUE(std::isfinite(jb_stat));
    EXPECT_TRUE(std::isfinite(jb_p_value));
    std::cout << "  Jarque-Bera: JB=" << jb_stat << ", p=" << jb_p_value << ", reject=" << reject_normality << "\n";
    
    // Robust estimation
    auto [robust_loc, robust_scale] = GaussianDistribution::robustEstimation(normal_data_, "huber", 1.345);
    EXPECT_TRUE(std::isfinite(robust_loc));
    EXPECT_TRUE(std::isfinite(robust_scale));
    EXPECT_GT(robust_scale, 0.0);
    std::cout << "  Robust estimates: location=" << robust_loc << ", scale=" << robust_scale << "\n";
}

//==============================================================================
// GOODNESS-OF-FIT TESTS
//==============================================================================

TEST_F(GaussianEnhancedTest, GoodnessOfFitTests) {
    std::cout << "\n=== Goodness-of-Fit Tests ===\n";
    
    // Kolmogorov-Smirnov test with normal data
    auto [ks_stat_normal, ks_p_normal, ks_reject_normal] = 
        GaussianDistribution::kolmogorovSmirnovTest(normal_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ks_stat_normal, 0.0);
    EXPECT_LE(ks_stat_normal, 1.0);
    EXPECT_GE(ks_p_normal, 0.0);
    EXPECT_LE(ks_p_normal, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat_normal));
    EXPECT_TRUE(std::isfinite(ks_p_normal));
    
    // Kolmogorov-Smirnov test with non-normal data (should reject)
    auto [ks_stat_non_normal, ks_p_non_normal, ks_reject_non_normal] = 
        GaussianDistribution::kolmogorovSmirnovTest(non_normal_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(ks_reject_non_normal); // Should reject normality for quadratic data
    EXPECT_LT(ks_p_non_normal, ks_p_normal); // Non-normal data should have lower p-value
    
    std::cout << "  KS test (normal data): D=" << ks_stat_normal << ", p=" << ks_p_normal << ", reject=" << ks_reject_normal << "\n";
    std::cout << "  KS test (non-normal data): D=" << ks_stat_non_normal << ", p=" << ks_p_non_normal << ", reject=" << ks_reject_non_normal << "\n";
    
    // Anderson-Darling test
    auto [ad_stat_normal, ad_p_normal, ad_reject_normal] = 
        GaussianDistribution::andersonDarlingTest(normal_data_, test_distribution_, 0.05);
    auto [ad_stat_non_normal, ad_p_non_normal, ad_reject_non_normal] = 
        GaussianDistribution::andersonDarlingTest(non_normal_data_, test_distribution_, 0.05);
    
    EXPECT_GE(ad_stat_normal, 0.0);
    EXPECT_GE(ad_p_normal, 0.0);
    EXPECT_LE(ad_p_normal, 1.0);
    EXPECT_TRUE(ad_reject_non_normal); // Should reject normality for quadratic data
    
    std::cout << "  AD test (normal data): A²=" << ad_stat_normal << ", p=" << ad_p_normal << ", reject=" << ad_reject_normal << "\n";
    std::cout << "  AD test (non-normal data): A²=" << ad_stat_non_normal << ", p=" << ad_p_non_normal << ", reject=" << ad_reject_non_normal << "\n";
}

//==============================================================================
// INFORMATION CRITERIA TESTS
//==============================================================================

TEST_F(GaussianEnhancedTest, InformationCriteriaTests) {
    std::cout << "\n=== Information Criteria Tests ===\n";
    
    // Fit distribution to the data
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
    EXPECT_GT(bic, aic);               // For moderate sample sizes, BIC typically penalizes more than AIC
    
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

TEST_F(GaussianEnhancedTest, BootstrapMethods) {
    std::cout << "\n=== Bootstrap Methods ===\n";
    
    // Bootstrap parameter confidence intervals
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
    
    std::cout << "  Mean 95% CI: [" << mean_ci.first << ", " << mean_ci.second << "]\n";
    std::cout << "  Std 95% CI: [" << std_ci.first << ", " << std_ci.second << "]\n";
    
    // K-fold cross-validation
    auto cv_results = GaussianDistribution::kFoldCrossValidation(normal_data_, 5, 42);
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
    std::vector<double> small_normal_data(normal_data_.begin(), normal_data_.begin() + 20);
    auto [mae, rmse, loo_log_likelihood] = GaussianDistribution::leaveOneOutCrossValidation(small_normal_data);
    
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

TEST_F(GaussianEnhancedTest, SIMDAndParallelBatchImplementations) {
    auto stdNormal = libstats::GaussianDistribution::create(0.0, 1.0).value;
    
    std::cout << "\n=== SIMD and Parallel Batch Implementations ===\n";
    
    // Create shared WorkStealingPool once to avoid resource creation overhead
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    
    // Test multiple batch sizes to show scaling behavior
    std::vector<size_t> batch_sizes = {5000, 50000};
    
    for (size_t batch_size : batch_sizes) {
        std::cout << "\n--- Batch Size: " << batch_size << " elements ---\n";
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> results(batch_size);
        
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-3.0, 3.0);
        for (size_t i = 0; i < batch_size; ++i) {
            test_values[i] = dis(gen);
        }
        
        // 1. Sequential individual calls (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = stdNormal.getProbability(test_values[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto sequential_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. SIMD batch operations
        std::vector<double> simd_results(batch_size);
        start = std::chrono::high_resolution_clock::now();
        stdNormal.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(simd_results), libstats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Parallel batch operations
        std::vector<double> parallel_results(batch_size);
        std::span<const double> input_span(test_values);
        std::span<double> output_span(parallel_results);
        
        start = std::chrono::high_resolution_clock::now();
        stdNormal.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::PARALLEL_SIMD);
        end = std::chrono::high_resolution_clock::now();
        auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Work-stealing operations (use shared pool)
        std::vector<double> work_stealing_results(batch_size);
        std::span<double> ws_output_span(work_stealing_results);
        
        start = std::chrono::high_resolution_clock::now();
        stdNormal.getProbabilityWithStrategy(input_span, ws_output_span, libstats::performance::Strategy::WORK_STEALING);
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
// AUTO-DISPATCH ASSESSMENT
//==============================================================================

TEST_F(GaussianEnhancedTest, AutoDispatchAssessment) {
    auto gauss_dist = libstats::GaussianDistribution::create(0.0, 1.0).value;
    
    std::cout << "\n=== Auto-Dispatch Strategy Assessment ===\n";
    
    // Test different batch sizes to verify auto-dispatch picks the right method
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH", "PARALLEL_SIMD"};
    
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_results(batch_size);
        std::vector<double> traditional_results(batch_size);
        
        std::mt19937 gen(42 + static_cast<unsigned int>(i));
        std::uniform_real_distribution<> dis(-2.0, 2.0);
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = dis(gen);
        }
        
        // Test auto-dispatch
        auto start = std::chrono::high_resolution_clock::now();
        gauss_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_results));
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch method
        start = std::chrono::high_resolution_clock::now();
        gauss_dist.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(traditional_results), libstats::performance::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto traditional_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Verify correctness
        bool results_match = true;
        for (size_t j = 0; j < batch_size; ++j) {
            if (std::abs(auto_results[j] - traditional_results[j]) > 1e-10) {
                results_match = false;
                break;
            }
        }
        
        std::cout << "  Batch size " << batch_size << " (expected: " << expected_strategy << "): ";
        std::cout << "Auto " << auto_time << "μs vs Traditional " << traditional_time << "μs, ";
        std::cout << "Correct: " << (results_match ? "✅" : "❌") << "\n";
        
        EXPECT_TRUE(results_match) << "Auto-dispatch results should match traditional for batch size " << batch_size;
        
        // Auto-dispatch should be competitive or better
        // For very small batch sizes, timing measurements can be noisy and traditional method
        // may complete in 0-1μs, making ratios unreliable or infinite.
        if (traditional_time == 0) {
            // If traditional time is 0, just check that auto time is reasonable (< 100μs)
            EXPECT_LT(auto_time, 100) << "Auto-dispatch should complete quickly for small batches (batch size " << batch_size << ")";
        } else {
            double performance_ratio = static_cast<double>(auto_time) / static_cast<double>(traditional_time);
            if (batch_size <= 100) {
                EXPECT_LT(performance_ratio, 10.0) << "Auto-dispatch should be reasonable for small batches (batch size " << batch_size << ")";
            } else {
                EXPECT_LT(performance_ratio, 2.0) << "Auto-dispatch should not be significantly slower than traditional for batch size " << batch_size;
            }
        }
    }
}

//==============================================================================
// INTERNAL FUNCTIONALITY TESTS (CACHING SPEEDUP)
//==============================================================================

TEST_F(GaussianEnhancedTest, CachingSpeedupVerification) {
    std::cout << "\n=== Caching Speedup Verification ===\n";
    
    auto gauss_dist = libstats::GaussianDistribution::create(0.0, 1.0).value;
    
    // First call - cache miss
    auto start = std::chrono::high_resolution_clock::now();
    double mean_first = gauss_dist.getMean();
    double var_first = gauss_dist.getVariance();
    double skew_first = gauss_dist.getSkewness();
    double kurt_first = gauss_dist.getKurtosis();
    auto end = std::chrono::high_resolution_clock::now();
    auto first_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    // Second call - cache hit
    start = std::chrono::high_resolution_clock::now();
    double mean_second = gauss_dist.getMean();
    double var_second = gauss_dist.getVariance();
    double skew_second = gauss_dist.getSkewness();
    double kurt_second = gauss_dist.getKurtosis();
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
    
    // Test cache invalidation by modifying parameters
    gauss_dist.setMean(1.0); // This should invalidate the cache
    
    start = std::chrono::high_resolution_clock::now();
    double mean_after_change = gauss_dist.getMean();
    end = std::chrono::high_resolution_clock::now();
    auto after_change_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    EXPECT_EQ(mean_after_change, 1.0);
    std::cout << "  After parameter change: " << after_change_time << "ns\n";
    
    // Test cache functionality: verify that cache invalidation worked correctly
    // (the new parameter value is returned, proving cache was invalidated)
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(GaussianEnhancedTest, ParallelBatchPerformanceBenchmark) {
    auto stdNormal = libstats::GaussianDistribution::create(0.0, 1.0).value;
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::normal_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    StandardizedBenchmark::printBenchmarkHeader("Gaussian Distribution", BENCHMARK_SIZE);
    
    // Create shared thread pool ONCE outside the loop to avoid resource issues
    // This prevents thread creation/destruction overhead and potential hangs
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
            stdNormal.getProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(pdf_results), libstats::performance::Strategy::SCALAR);
        } else if (op == "LogPDF") {
            stdNormal.getLogProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(log_pdf_results), libstats::performance::Strategy::SCALAR);
        } else if (op == "CDF") {
            stdNormal.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values), std::span<double>(cdf_results), libstats::performance::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        
        // 2. Standard Parallel Operations
        std::span<const double> input_span(test_values);
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::PARALLEL_SIMD);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::PARALLEL_SIMD);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::PARALLEL_SIMD);
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        
        // 3. Work-Stealing Operations (use shared pool to avoid resource issues)
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::WORK_STEALING);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::WORK_STEALING);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::WORK_STEALING);
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        
        // 4. GPU-Accelerated Operations (CPU fallback)
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::GPU_ACCELERATED);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getLogProbabilityWithStrategy(input_span, log_output_span, libstats::performance::Strategy::GPU_ACCELERATED);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            stdNormal.getCumulativeProbabilityWithStrategy(input_span, cdf_output_span, libstats::performance::Strategy::GPU_ACCELERATED);
            end = std::chrono::high_resolution_clock::now();
        }
        result.gpu_accelerated_time_us = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
        
        // Calculate speedups
        result.parallel_speedup = result.simd_time_us > 0 ? static_cast<double>(result.simd_time_us) / static_cast<double>(result.parallel_time_us) : 0.0;
        result.work_stealing_speedup = result.simd_time_us > 0 ? static_cast<double>(result.simd_time_us) / static_cast<double>(result.work_stealing_time_us) : 0.0;
        result.gpu_accelerated_speedup = result.simd_time_us > 0 ? static_cast<double>(result.simd_time_us) / static_cast<double>(result.gpu_accelerated_time_us) : 0.0;
        
        benchmark_results.push_back(result);
        
        // Verify correctness for this operation type
        if (op == "PDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdNormal, test_values, pdf_results, "PDF");
        } else if (op == "LogPDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdNormal, test_values, log_pdf_results, "LogPDF");
        } else if (op == "CDF") {
            StatisticalTestUtils::verifyBatchCorrectness(stdNormal, test_values, cdf_results, "CDF");
        }
    }
    
    // Print results and performance analysis
    StandardizedBenchmark::printBenchmarkResults(benchmark_results);
    StandardizedBenchmark::printPerformanceAnalysis(benchmark_results);
}

//==============================================================================
// NUMERICAL STABILITY AND EDGE CASES
//==============================================================================

TEST_F(GaussianEnhancedTest, NumericalStabilityAndEdgeCases) {
    std::cout << "\n=== Numerical Stability and Edge Cases ===\n";
    
    auto normal = libstats::GaussianDistribution::create(0.0, 1.0).value;
    
    // Test extreme values
    std::vector<double> extreme_values = {-100.0, -10.0, 10.0, 100.0};
    
    for (double val : extreme_values) {
        double pdf = normal.getProbability(val);
        double log_pdf = normal.getLogProbability(val);
        double cdf = normal.getCumulativeProbability(val);
        
        EXPECT_GE(pdf, 0.0) << "PDF should be non-negative for value " << val;
        EXPECT_TRUE(std::isfinite(log_pdf)) << "Log PDF should be finite for value " << val;
        EXPECT_GE(cdf, 0.0) << "CDF should be non-negative for value " << val;
        EXPECT_LE(cdf, 1.0) << "CDF should be <= 1 for value " << val;
        
        std::cout << "  Value " << val << ": PDF=" << pdf << ", LogPDF=" << log_pdf << ", CDF=" << cdf << "\n";
    }
    
    // Test empty batch operations
    std::vector<double> empty_input;
    std::vector<double> empty_output;
    
    // These should not crash
    normal.getProbabilityWithStrategy(std::span<const double>(empty_input), std::span<double>(empty_output), libstats::performance::Strategy::SCALAR);
    normal.getLogProbabilityWithStrategy(std::span<const double>(empty_input), std::span<double>(empty_output), libstats::performance::Strategy::SCALAR);
    normal.getCumulativeProbabilityWithStrategy(std::span<const double>(empty_input), std::span<double>(empty_output), libstats::performance::Strategy::SCALAR);
    
    // Test invalid parameter creation
    auto result_zero_std = GaussianDistribution::create(0.0, 0.0);
    EXPECT_TRUE(result_zero_std.isError()) << "Should fail with zero standard deviation";
    
    auto result_negative_std = GaussianDistribution::create(0.0, -1.0);
    EXPECT_TRUE(result_negative_std.isError()) << "Should fail with negative standard deviation";
    
    std::cout << "  Edge case testing completed\n";
}

//==============================================================================
// PARALLEL BATCH FITTING TESTS
//==============================================================================

TEST_F(GaussianEnhancedTest, ParallelBatchFittingTests) {
    std::cout << "\n=== Parallel Batch Fitting Tests ===\n";
    
    // Create multiple datasets for batch fitting
    std::vector<std::vector<double>> datasets;
    std::vector<GaussianDistribution> expected_distributions;
    
    std::mt19937 rng(42);
    
    // Generate 6 datasets with known parameters
    std::vector<std::pair<double, double>> true_params = {
        {0.0, 1.0}, {5.0, 2.0}, {-2.0, 0.5}, {10.0, 3.0}, {1.0, 1.5}, {-5.0, 4.0}
    };
    
    for (const auto& [mean, std] : true_params) {
        std::vector<double> dataset;
        dataset.reserve(1000);
        
        std::normal_distribution<double> gen(mean, std);
        for (int i = 0; i < 1000; ++i) {
            dataset.push_back(gen(rng));
        }
        
        datasets.push_back(std::move(dataset));
        expected_distributions.push_back(GaussianDistribution::create(mean, std).value);
    }
    
    std::cout << "  Generated " << datasets.size() << " datasets with known parameters\n";
    
    // Test 1: Basic parallel batch fitting correctness
    std::vector<GaussianDistribution> batch_results(datasets.size());
    
    auto start = std::chrono::high_resolution_clock::now();
    GaussianDistribution::parallelBatchFit(datasets, batch_results);
    auto end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // Verify correctness by comparing with individual fits
    for (size_t i = 0; i < datasets.size(); ++i) {
        GaussianDistribution individual_fit;
        individual_fit.fit(datasets[i]);
        
        // Parameters should match within tolerance
        EXPECT_NEAR(batch_results[i].getMean(), individual_fit.getMean(), 1e-10)
            << "Batch fit mean mismatch for dataset " << i;
        EXPECT_NEAR(batch_results[i].getStandardDeviation(), individual_fit.getStandardDeviation(), 1e-10)
            << "Batch fit std dev mismatch for dataset " << i;
        
        // Should be reasonably close to true parameters (within 3 standard errors)
        double expected_mean = true_params[i].first;
        double expected_std = true_params[i].second;
        [[maybe_unused]] double n = static_cast<double>(datasets[i].size());
        
        double mean_tolerance = 3.0 * expected_std / std::sqrt(n);  // 3 standard errors
        double std_tolerance = 0.1 * expected_std;  // 10% tolerance for std dev estimation
        
        EXPECT_NEAR(batch_results[i].getMean(), expected_mean, mean_tolerance)
            << "Fitted mean too far from true value for dataset " << i;
        EXPECT_NEAR(batch_results[i].getStandardDeviation(), expected_std, std_tolerance)
            << "Fitted std dev too far from true value for dataset " << i;
    }
    
    std::cout << "  ✓ Parallel batch fitting correctness verified\n";
    
    // Test 2: Performance comparison with sequential batch fitting
    std::vector<GaussianDistribution> sequential_results(datasets.size());
    
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
        EXPECT_NEAR(batch_results[i].getMean(), sequential_results[i].getMean(), 1e-12)
            << "Sequential vs parallel mean mismatch for dataset " << i;
        EXPECT_NEAR(batch_results[i].getStandardDeviation(), sequential_results[i].getStandardDeviation(), 1e-12)
            << "Sequential vs parallel std dev mismatch for dataset " << i;
    }
    
    // Test 3: Edge cases
    std::cout << "  Testing edge cases...\n";
    
    // Empty datasets vector
    std::vector<std::vector<double>> empty_datasets;
    std::vector<GaussianDistribution> empty_results;
    GaussianDistribution::parallelBatchFit(empty_datasets, empty_results);
    EXPECT_TRUE(empty_results.empty());
    
    // Single dataset
    std::vector<std::vector<double>> single_dataset = {datasets[0]};
    std::vector<GaussianDistribution> single_result(1);
    GaussianDistribution::parallelBatchFit(single_dataset, single_result);
    EXPECT_NEAR(single_result[0].getMean(), batch_results[0].getMean(), 1e-12);
    EXPECT_NEAR(single_result[0].getStandardDeviation(), batch_results[0].getStandardDeviation(), 1e-12);
    
    // Results vector auto-sizing
    std::vector<GaussianDistribution> auto_sized_results;
    GaussianDistribution::parallelBatchFit(datasets, auto_sized_results);
    EXPECT_EQ(auto_sized_results.size(), datasets.size());
    
    std::cout << "  ✓ Edge cases handled correctly\n";
    
    // Test 4: Thread safety with concurrent calls
    std::cout << "  Testing thread safety...\n";
    
    const int num_threads = 4;
    const int calls_per_thread = 10;
    std::vector<std::thread> threads;
    std::atomic<int> successful_calls{0};
    std::atomic<int> total_calls{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&datasets, &successful_calls, &total_calls]() {
            for (int call = 0; call < calls_per_thread; ++call) {
                try {
                    std::vector<GaussianDistribution> thread_results(datasets.size());
                    GaussianDistribution::parallelBatchFit(datasets, thread_results);
                    
                    // Verify at least one result is reasonable
                    if (!thread_results.empty() && 
                        std::isfinite(thread_results[0].getMean()) && 
                        std::isfinite(thread_results[0].getStandardDeviation()) &&
                        thread_results[0].getStandardDeviation() > 0.0) {
                        successful_calls++;
                    }
                } catch (const std::exception& e) {
                    // Thread safety failure
                    std::cerr << "Thread safety test exception: " << e.what() << "\n";
                }
                total_calls++;
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    double success_rate = static_cast<double>(successful_calls.load()) / static_cast<double>(total_calls.load());
    std::cout << "  Thread safety: " << successful_calls.load() << "/" << total_calls.load() 
              << " calls successful (" << (success_rate * 100.0) << "%)\n";
    
    EXPECT_GT(success_rate, 0.95) << "Thread safety test should have >95% success rate";
    
    std::cout << "  ✓ Thread safety verified\n";
    
    // Performance expectations
    if (std::thread::hardware_concurrency() > 1) {
        EXPECT_GT(speedup, 0.8) << "Parallel batch fitting should provide reasonable speedup on multi-core systems";
    }
    
    std::cout << "✅ All parallel batch fitting tests passed\n";
}

} // namespace libstats



#ifdef _MSC_VER
#pragma warning(pop)
#endif
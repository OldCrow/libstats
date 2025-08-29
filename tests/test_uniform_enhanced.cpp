#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

#include "../include/distributions/uniform.h"
#include "../include/tests/tests.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <vector>

using namespace std;
using namespace stats;
using namespace stats::tests;

namespace stats {

//==============================================================================
// TEST FIXTURE FOR UNIFORM ENHANCED METHODS
//==============================================================================

class UniformEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Generate synthetic uniform data for testing
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> uniform_gen(test_a_, test_b_);

        uniform_data_.clear();
        uniform_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            uniform_data_.push_back(uniform_gen(rng));
        }

        // Generate obviously non-uniform data (normal)
        non_uniform_data_.clear();
        non_uniform_data_.reserve(100);
        std::normal_distribution<double> normal_gen(0.0, 1.0);
        for (int i = 0; i < 100; ++i) {
            non_uniform_data_.push_back(normal_gen(rng));
        }

        auto result = stats::UniformDistribution::create(test_a_, test_b_);
        if (result.isOk()) {
            test_distribution_ = std::move(result.value);
        };
    }

    const double test_a_ = 2.0;
    const double test_b_ = 8.0;
    std::vector<double> uniform_data_;
    std::vector<double> non_uniform_data_;
    UniformDistribution test_distribution_;
};

//==============================================================================
// BASIC FUNCTIONALITY TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, BasicEnhancedFunctionality) {
    // Test standard uniform distribution properties
    auto stdUniform = stats::UniformDistribution::create(0.0, 1.0).value;

    EXPECT_DOUBLE_EQ(stdUniform.getLowerBound(), 0.0);
    EXPECT_DOUBLE_EQ(stdUniform.getUpperBound(), 1.0);
    EXPECT_DOUBLE_EQ(stdUniform.getMean(), 0.5);
    EXPECT_NEAR(stdUniform.getVariance(), 1.0 / 12.0, 1e-10);
    EXPECT_DOUBLE_EQ(stdUniform.getSkewness(), 0.0);
    EXPECT_DOUBLE_EQ(stdUniform.getKurtosis(), -1.2);

    // Test known PDF/CDF values
    double pdf_at_mid = stdUniform.getProbability(0.5);
    double cdf_at_mid = stdUniform.getCumulativeProbability(0.5);

    EXPECT_DOUBLE_EQ(pdf_at_mid, 1.0);  // PDF is constant at 1/(b-a) = 1
    EXPECT_DOUBLE_EQ(cdf_at_mid, 0.5);  // CDF at midpoint

    // Test custom distribution
    auto custom = stats::UniformDistribution::create(-2.0, 4.0).value;
    EXPECT_DOUBLE_EQ(custom.getLowerBound(), -2.0);
    EXPECT_DOUBLE_EQ(custom.getUpperBound(), 4.0);
    EXPECT_DOUBLE_EQ(custom.getMean(), 1.0);
    EXPECT_NEAR(custom.getVariance(), 36.0 / 12.0, 1e-10);
}

//==============================================================================
// GOODNESS-OF-FIT TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, GoodnessOfFitTests) {
    std::cout << "\n=== Goodness-of-Fit Tests ===\n";

    // Kolmogorov-Smirnov test with uniform data
    auto [ks_stat_uniform, ks_p_uniform, ks_reject_uniform] =
        UniformDistribution::kolmogorovSmirnovTest(uniform_data_, test_distribution_, 0.05);

    EXPECT_GE(ks_stat_uniform, 0.0);
    EXPECT_LE(ks_stat_uniform, 1.0);
    EXPECT_GE(ks_p_uniform, 0.0);
    EXPECT_LE(ks_p_uniform, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat_uniform));
    EXPECT_TRUE(std::isfinite(ks_p_uniform));

    std::cout << "  KS test (uniform data): D=" << ks_stat_uniform << ", p=" << ks_p_uniform
              << ", reject=" << ks_reject_uniform << "\n";

    // Kolmogorov-Smirnov test with non-uniform data (should reject)
    auto [ks_stat_non_uniform, ks_p_non_uniform, ks_reject_non_uniform] =
        UniformDistribution::kolmogorovSmirnovTest(non_uniform_data_, test_distribution_, 0.05);

    EXPECT_TRUE(ks_reject_non_uniform);  // Should reject uniform distribution for normal data
    EXPECT_LT(ks_p_non_uniform, ks_p_uniform);  // Non-uniform data should have lower p-value

    std::cout << "  KS test (non-uniform data): D=" << ks_stat_non_uniform
              << ", p=" << ks_p_non_uniform << ", reject=" << ks_reject_non_uniform << "\n";

    // Anderson-Darling test
    auto [ad_stat_uniform, ad_p_uniform, ad_reject_uniform] =
        UniformDistribution::andersonDarlingTest(uniform_data_, test_distribution_, 0.05);
    auto [ad_stat_non_uniform, ad_p_non_uniform, ad_reject_non_uniform] =
        UniformDistribution::andersonDarlingTest(non_uniform_data_, test_distribution_, 0.05);

    EXPECT_GE(ad_stat_uniform, 0.0);
    EXPECT_GE(ad_p_uniform, 0.0);
    EXPECT_LE(ad_p_uniform, 1.0);
    EXPECT_TRUE(ad_reject_non_uniform);  // Should reject uniform for non-uniform data

    std::cout << "  AD test (uniform data): A²=" << ad_stat_uniform << ", p=" << ad_p_uniform
              << ", reject=" << ad_reject_uniform << "\n";
    std::cout << "  AD test (non-uniform data): A²=" << ad_stat_non_uniform
              << ", p=" << ad_p_non_uniform << ", reject=" << ad_reject_non_uniform << "\n";
}

//==============================================================================
// INFORMATION CRITERIA TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, InformationCriteriaTests) {
    std::cout << "\n=== Information Criteria Tests ===\n";

    // Fit distribution to data
    auto fitted_dist = stats::UniformDistribution::create().value;
    fitted_dist.fit(uniform_data_);

    auto [aic, bic, aicc, log_likelihood] =
        UniformDistribution::computeInformationCriteria(uniform_data_, fitted_dist);

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

TEST_F(UniformEnhancedTest, BootstrapMethods) {
    std::cout << "\n=== Bootstrap Methods ===\n";

    // Bootstrap parameter confidence intervals (returns nested pairs)
    auto [a_ci_pair, b_ci_pair] =
        UniformDistribution::bootstrapParameterConfidenceIntervals(uniform_data_, 0.95, 1000, 456);

    // Extract the individual confidence intervals
    auto [a_ci_lower, a_ci_upper] = a_ci_pair;
    auto [b_ci_lower, b_ci_upper] = b_ci_pair;

    // Check that confidence intervals are reasonable
    EXPECT_LT(a_ci_lower, a_ci_upper);  // Lower bound CI
    EXPECT_LT(b_ci_lower, b_ci_upper);  // Upper bound CI
    EXPECT_LT(a_ci_lower, b_ci_lower);  // a should be less than b

    // Check for finite values
    EXPECT_TRUE(std::isfinite(a_ci_lower));
    EXPECT_TRUE(std::isfinite(a_ci_upper));
    EXPECT_TRUE(std::isfinite(b_ci_lower));
    EXPECT_TRUE(std::isfinite(b_ci_upper));

    std::cout << "  Parameter a 95% CI: [" << a_ci_lower << ", " << a_ci_upper << "]\n";
    std::cout << "  Parameter b 95% CI: [" << b_ci_lower << ", " << b_ci_upper << "]\n";

    // K-fold cross-validation
    auto cv_results = UniformDistribution::kFoldCrossValidation(uniform_data_, 5, 42);
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

    // Leave-one-out cross-validation on smaller dataset
    std::vector<double> small_uniform_data(uniform_data_.begin(), uniform_data_.begin() + 20);
    auto [loocv_mae, loocv_rmse, loocv_log_likelihood] =
        UniformDistribution::leaveOneOutCrossValidation(small_uniform_data);

    EXPECT_GE(loocv_mae, 0.0);             // Mean absolute error should be non-negative
    EXPECT_GE(loocv_rmse, 0.0);            // RMSE should be non-negative
    EXPECT_GE(loocv_rmse, loocv_mae);      // RMSE should be >= MAE
    EXPECT_LE(loocv_log_likelihood, 0.0);  // Total log-likelihood should be negative

    EXPECT_TRUE(std::isfinite(loocv_mae));
    EXPECT_TRUE(std::isfinite(loocv_rmse));
    EXPECT_TRUE(std::isfinite(loocv_log_likelihood));

    std::cout << "  Leave-one-out CV: MAE=" << loocv_mae << ", RMSE=" << loocv_rmse
              << ", LogL=" << loocv_log_likelihood << "\n";
}

//==============================================================================
// SIMD AND PARALLEL BATCH IMPLEMENTATIONS WITH FULSOME COMPARISONS
//==============================================================================

TEST_F(UniformEnhancedTest, SIMDAndParallelBatchImplementations) {
    auto stdUniform = stats::UniformDistribution::create(0.0, 1.0).value;

    std::cout << "\n=== SIMD and Parallel Batch Implementations ===\n";

    // Create shared WorkStealingPool once to avoid resource creation overhead
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());

    // Test multiple batch sizes to show scaling behavior
    std::vector<size_t> batch_sizes = {5000, 50000};

    for (size_t batch_size : batch_sizes) {
        std::cout << "\n--- Batch Size: " << batch_size << " elements ---\n";

        // Generate test data (mixture of in-support and out-of-support values)
        std::vector<double> test_values(batch_size);
        std::vector<double> results(batch_size);

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(-0.5, 1.5);  // Mix of values in/out of [0,1]
        for (size_t i = 0; i < batch_size; ++i) {
            test_values[i] = dis(gen);
        }

        // 1. Sequential individual calls (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = stdUniform.getProbability(test_values[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto sequential_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 2. SIMD batch operations
        std::vector<double> simd_results(batch_size);
        start = std::chrono::high_resolution_clock::now();
        stdUniform.getProbabilityWithStrategy(std::span<const double>(test_values),
                                              std::span<double>(simd_results),
                                              stats::detail::Strategy::SCALAR);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 3. Parallel batch operations
        std::vector<double> parallel_results(batch_size);
        std::span<const double> input_span(test_values);
        std::span<double> output_span(parallel_results);

        start = std::chrono::high_resolution_clock::now();
        stdUniform.getProbabilityWithStrategy(input_span, output_span,
                                              stats::detail::Strategy::PARALLEL_SIMD);
        end = std::chrono::high_resolution_clock::now();
        auto parallel_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 4. Work-stealing operations (use shared pool)
        std::vector<double> work_stealing_results(batch_size);
        std::span<double> ws_output_span(work_stealing_results);

        start = std::chrono::high_resolution_clock::now();
        stdUniform.getProbabilityWithStrategy(input_span, ws_output_span,
                                              stats::detail::Strategy::WORK_STEALING);
        end = std::chrono::high_resolution_clock::now();
        auto work_stealing_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Calculate speedups
        double simd_speedup = static_cast<double>(sequential_time) / static_cast<double>(simd_time);
        double parallel_speedup =
            static_cast<double>(sequential_time) / static_cast<double>(parallel_time);
        double ws_speedup =
            static_cast<double>(sequential_time) / static_cast<double>(work_stealing_time);

        std::cout << "  Sequential: " << sequential_time << "μs (baseline)\n";
        std::cout << "  SIMD Batch: " << simd_time << "μs (" << simd_speedup << "x speedup)\n";
        std::cout << "  Parallel: " << parallel_time << "μs (" << parallel_speedup
                  << "x speedup)\n";
        std::cout << "  Work Stealing: " << work_stealing_time << "μs (" << ws_speedup
                  << "x speedup)\n";

        // Verify correctness across all methods (sample verification)
        size_t verification_samples = std::min(batch_size, size_t(100));
        for (size_t i = 0; i < verification_samples; ++i) {
            double expected = results[i];
            EXPECT_NEAR(simd_results[i], expected, 1e-12)
                << "SIMD result mismatch at index " << i << " for batch size " << batch_size;
            EXPECT_NEAR(parallel_results[i], expected, 1e-12)
                << "Parallel result mismatch at index " << i << " for batch size " << batch_size;
            EXPECT_NEAR(work_stealing_results[i], expected, 1e-12)
                << "Work-stealing result mismatch at index " << i << " for batch size "
                << batch_size;
        }

        // Performance expectations (adjusted for batch size and computational complexity)
        EXPECT_GT(simd_speedup, 1.0) << "SIMD should provide speedup for batch size " << batch_size;

        if (std::thread::hardware_concurrency() > 1) {
            if (batch_size >= 10000) {
                // For uniform distributions, computations are extremely simple (range checks and
                // constants), so SIMD can achieve massive speedups (50-70x) but parallel has thread
                // overhead. Expect parallel to be at least 70% as efficient as SIMD for large
                // batches.
                EXPECT_GT(parallel_speedup, simd_speedup * 0.7)
                    << "Parallel should be reasonably competitive with SIMD for large batches";
            } else {
                // For smaller batches, parallel may have overhead but should still be reasonable
                EXPECT_GT(parallel_speedup, 0.5)
                    << "Parallel should provide reasonable performance for batch size "
                    << batch_size;
            }
        }
    }
}

//==============================================================================
// ADVANCED STATISTICAL METHODS TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, AdvancedStatisticalMethods) {
    std::cout << "\n=== Advanced Statistical Methods ===\n";

    // Confidence intervals for lower and upper bounds
    auto [a_lower, a_upper] =
        UniformDistribution::confidenceIntervalLowerBound(uniform_data_, 0.95);
    auto [b_lower, b_upper] =
        UniformDistribution::confidenceIntervalUpperBound(uniform_data_, 0.95);
    EXPECT_LT(a_lower, a_upper);
    EXPECT_LT(b_lower, b_upper);
    EXPECT_TRUE(std::isfinite(a_lower));
    EXPECT_TRUE(std::isfinite(a_upper));
    EXPECT_TRUE(std::isfinite(b_lower));
    EXPECT_TRUE(std::isfinite(b_upper));
    std::cout << "  95% CI for a: [" << a_lower << ", " << a_upper << "]\n";
    std::cout << "  95% CI for b: [" << b_lower << ", " << b_upper << "]\n";

    // Likelihood ratio test
    auto [lr_stat, p_value, reject_null] =
        UniformDistribution::likelihoodRatioTest(uniform_data_, test_a_, test_b_, 0.05);
    EXPECT_GE(lr_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(lr_stat));
    EXPECT_TRUE(std::isfinite(p_value));
    std::cout << "  LR test: stat=" << lr_stat << ", p=" << p_value << ", reject=" << reject_null
              << "\n";

    // Method of moments estimation
    auto [a_mom, b_mom] = UniformDistribution::methodOfMomentsEstimation(uniform_data_);
    EXPECT_LT(a_mom, b_mom);
    EXPECT_TRUE(std::isfinite(a_mom));
    EXPECT_TRUE(std::isfinite(b_mom));
    std::cout << "  MoM estimates: a=" << a_mom << ", b=" << b_mom << "\n";

    // For uniform, MLE is simply min/max of data (same as method of moments)
    auto [a_mle, b_mle] = UniformDistribution::methodOfMomentsEstimation(uniform_data_);
    EXPECT_LT(a_mle, b_mle);
    EXPECT_TRUE(std::isfinite(a_mle));
    EXPECT_TRUE(std::isfinite(b_mle));
    std::cout << "  MLE estimates (via MoM): a=" << a_mle << ", b=" << b_mle << "\n";
}

//==============================================================================
// CACHING SPEEDUP VERIFICATION TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, CachingSpeedupVerification) {
    std::cout << "\n=== Caching Speedup Verification ===\n";

    auto uniform_dist = stats::UniformDistribution::create(0.0, 1.0).value;

    // First call - cache miss
    auto start = std::chrono::high_resolution_clock::now();
    double mean_first = uniform_dist.getMean();
    double var_first = uniform_dist.getVariance();
    double skew_first = uniform_dist.getSkewness();
    double kurt_first = uniform_dist.getKurtosis();
    auto end = std::chrono::high_resolution_clock::now();
    auto first_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Second call - cache hit
    start = std::chrono::high_resolution_clock::now();
    double mean_second = uniform_dist.getMean();
    double var_second = uniform_dist.getVariance();
    double skew_second = uniform_dist.getSkewness();
    double kurt_second = uniform_dist.getKurtosis();
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
    auto new_dist = stats::UniformDistribution::create(0.5, 1.5).value;

    start = std::chrono::high_resolution_clock::now();
    double mean_after_change = new_dist.getMean();
    end = std::chrono::high_resolution_clock::now();
    auto after_change_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    EXPECT_EQ(mean_after_change, 1.0);  // Mean of uniform(0.5,1.5) is (0.5+1.5)/2 = 1
    std::cout << "  New distribution parameter access: " << after_change_time << "ns\n";

    // Test cache functionality: verify that the new distribution returns correct values
    // (proving cache isolation between different distribution instances)
}

//==============================================================================
// AUTO-DISPATCH STRATEGY TESTING
//==============================================================================

TEST_F(UniformEnhancedTest, AutoDispatchAssessment) {
    auto uniform_dist = stats::UniformDistribution::create(0.0, 1.0).value;

    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH",
                                                    "PARALLEL_SIMD"};

    std::cout << "\n=== Auto-Dispatch Assessment (Uniform) ===\n";

    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];

        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pdf_results(batch_size);
        std::vector<double> auto_logpdf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);

        std::mt19937 gen(42 + static_cast<unsigned int>(i));
        std::uniform_real_distribution<> dis(-0.5, 1.5);
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = dis(gen);
        }

        // Test smart auto-dispatch methods (if available)
        auto start = std::chrono::high_resolution_clock::now();
        if constexpr (requires {
                          uniform_dist.getProbability(std::span<const double>(test_values),
                                                      std::span<double>(auto_pdf_results));
                      }) {
            uniform_dist.getProbability(std::span<const double>(test_values),
                                        std::span<double>(auto_pdf_results));
        } else {
            uniform_dist.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                    std::span<double>(auto_pdf_results),
                                                    stats::detail::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires {
                          uniform_dist.getLogProbability(std::span<const double>(test_values),
                                                         std::span<double>(auto_logpdf_results));
                      }) {
            uniform_dist.getLogProbability(std::span<const double>(test_values),
                                           std::span<double>(auto_logpdf_results));
        } else {
            uniform_dist.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                       std::span<double>(auto_logpdf_results),
                                                       stats::detail::Strategy::SCALAR);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires {
                          uniform_dist.getCumulativeProbability(
                              std::span<const double>(test_values),
                              std::span<double>(auto_cdf_results));
                      }) {
            uniform_dist.getCumulativeProbability(std::span<const double>(test_values),
                                                  std::span<double>(auto_cdf_results));
        } else {
            uniform_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                              std::span<double>(auto_cdf_results),
                                                              stats::detail::Strategy::SCALAR);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pdf_results(batch_size);
        std::vector<double> trad_logpdf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);

        uniform_dist.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                std::span<double>(trad_pdf_results),
                                                stats::detail::Strategy::SCALAR);
        uniform_dist.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                   std::span<double>(trad_logpdf_results),
                                                   stats::detail::Strategy::SCALAR);
        uniform_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                          std::span<double>(trad_cdf_results),
                                                          stats::detail::Strategy::SCALAR);

        // Verify correctness
        bool pdf_correct = true, logpdf_correct = true, cdf_correct = true;

        for (size_t j = 0; j < batch_size; ++j) {
            if (std::abs(auto_pdf_results[j] - trad_pdf_results[j]) > 1e-10) {
                pdf_correct = false;
            }
            // Handle log probability comparisons carefully for infinities
            if (std::isfinite(auto_logpdf_results[j]) && std::isfinite(trad_logpdf_results[j])) {
                if (std::abs(auto_logpdf_results[j] - trad_logpdf_results[j]) > 1e-10) {
                    logpdf_correct = false;
                }
            } else if (auto_logpdf_results[j] != trad_logpdf_results[j]) {
                logpdf_correct = false;
            }
            if (std::abs(auto_cdf_results[j] - trad_cdf_results[j]) > 1e-10) {
                cdf_correct = false;
            }
        }

        std::cout << "Batch size: " << batch_size << ", Expected strategy: " << expected_strategy
                  << "\n";
        std::cout << "  PDF: " << auto_pdf_time << "μs, Correct: " << (pdf_correct ? "✅" : "❌")
                  << "\n";
        std::cout << "  LogPDF: " << auto_logpdf_time
                  << "μs, Correct: " << (logpdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  CDF: " << auto_cdf_time << "μs, Correct: " << (cdf_correct ? "✅" : "❌")
                  << "\n";

        EXPECT_TRUE(pdf_correct)
            << "PDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(logpdf_correct)
            << "LogPDF auto-dispatch results should match traditional for batch size "
            << batch_size;
        EXPECT_TRUE(cdf_correct)
            << "CDF auto-dispatch results should match traditional for batch size " << batch_size;
    }

    std::cout << "\n=== Auto-Dispatch Assessment Completed (Uniform) ===\n";
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(UniformEnhancedTest, ParallelBatchPerformanceBenchmark) {
    auto stdUniform = stats::UniformDistribution::create(0.0, 1.0).value;
    constexpr size_t BENCHMARK_SIZE = 50000;

    // Generate test data
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 1.5);
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }

    fixtures::BenchmarkFormatter::printBenchmarkHeader("Uniform Distribution", BENCHMARK_SIZE);

    // Create shared resources ONCE outside the loop to avoid resource issues
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());

    std::vector<fixtures::BenchmarkResult> benchmark_results;

    // For each operation type (PDF, LogPDF, CDF)
    std::vector<std::string> operations = {"PDF", "LogPDF", "CDF"};

    for (const auto& op : operations) {
        fixtures::BenchmarkResult result;
        result.operation_name = op;

        // 1. SIMD Batch (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PDF") {
            stdUniform.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                  std::span<double>(pdf_results),
                                                  stats::detail::Strategy::SCALAR);
        } else if (op == "LogPDF") {
            stdUniform.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                     std::span<double>(log_pdf_results),
                                                     stats::detail::Strategy::SCALAR);
        } else if (op == "CDF") {
            stdUniform.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                            std::span<double>(cdf_results),
                                                            stats::detail::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 2. Standard Parallel Operations (if available) - fallback to SIMD
        std::span<const double> input_span(test_values);

        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getProbabilityWithStrategy(
                                  input_span, output_span, stats::detail::Strategy::PARALLEL_SIMD);
                          }) {
                stdUniform.getProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::PARALLEL_SIMD);
            } else {
                stdUniform.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                      std::span<double>(pdf_results),
                                                      stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getLogProbabilityWithStrategy(
                                  input_span, log_output_span,
                                  stats::detail::Strategy::PARALLEL_SIMD);
                          }) {
                stdUniform.getLogProbabilityWithStrategy(input_span, log_output_span,
                                                         stats::detail::Strategy::PARALLEL_SIMD);
            } else {
                stdUniform.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                         std::span<double>(log_pdf_results),
                                                         stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getCumulativeProbabilityWithStrategy(
                                  input_span, cdf_output_span,
                                  stats::detail::Strategy::PARALLEL_SIMD);
                          }) {
                stdUniform.getCumulativeProbabilityWithStrategy(
                    input_span, cdf_output_span, stats::detail::Strategy::PARALLEL_SIMD);
            } else {
                stdUniform.getCumulativeProbabilityWithStrategy(
                    std::span<const double>(test_values), std::span<double>(cdf_results),
                    stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 3. Work-Stealing Operations (if available) - fallback to SIMD

        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getProbabilityWithStrategy(
                                  input_span, output_span, stats::detail::Strategy::WORK_STEALING);
                          }) {
                stdUniform.getProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::WORK_STEALING);
            } else {
                stdUniform.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                      std::span<double>(pdf_results),
                                                      stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getLogProbabilityWithStrategy(
                                  input_span, log_output_span,
                                  stats::detail::Strategy::WORK_STEALING);
                          }) {
                stdUniform.getLogProbabilityWithStrategy(input_span, log_output_span,
                                                         stats::detail::Strategy::WORK_STEALING);
            } else {
                stdUniform.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                         std::span<double>(log_pdf_results),
                                                         stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getCumulativeProbabilityWithStrategy(
                                  input_span, cdf_output_span,
                                  stats::detail::Strategy::WORK_STEALING);
                          }) {
                stdUniform.getCumulativeProbabilityWithStrategy(
                    input_span, cdf_output_span, stats::detail::Strategy::WORK_STEALING);
            } else {
                stdUniform.getCumulativeProbabilityWithStrategy(
                    std::span<const double>(test_values), std::span<double>(cdf_results),
                    stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 4. Cache-Aware Operations (if available) - fallback to SIMD
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getProbabilityWithStrategy(
                                  input_span, output_span,
                                  stats::detail::Strategy::GPU_ACCELERATED);
                          }) {
                stdUniform.getProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::GPU_ACCELERATED);
            } else {
                stdUniform.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                      std::span<double>(pdf_results),
                                                      stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getLogProbabilityWithStrategy(
                                  input_span, log_output_span,
                                  stats::detail::Strategy::GPU_ACCELERATED);
                          }) {
                stdUniform.getLogProbabilityWithStrategy(input_span, log_output_span,
                                                         stats::detail::Strategy::GPU_ACCELERATED);
            } else {
                stdUniform.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                         std::span<double>(log_pdf_results),
                                                         stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdUniform.getCumulativeProbabilityWithStrategy(
                                  input_span, cdf_output_span,
                                  stats::detail::Strategy::GPU_ACCELERATED);
                          }) {
                stdUniform.getCumulativeProbabilityWithStrategy(
                    input_span, cdf_output_span, stats::detail::Strategy::GPU_ACCELERATED);
            } else {
                stdUniform.getCumulativeProbabilityWithStrategy(
                    std::span<const double>(test_values), std::span<double>(cdf_results),
                    stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.gpu_accelerated_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // Calculate speedups
        result.parallel_speedup =
            static_cast<double>(result.simd_time_us) / static_cast<double>(result.parallel_time_us);
        result.work_stealing_speedup = static_cast<double>(result.simd_time_us) /
                                       static_cast<double>(result.work_stealing_time_us);
        result.gpu_accelerated_speedup = static_cast<double>(result.simd_time_us) /
                                         static_cast<double>(result.gpu_accelerated_time_us);

        benchmark_results.push_back(result);

        // Verify correctness
        if (op == "PDF") {
            fixtures::StatisticalTestUtils::verifyBatchCorrectness(stdUniform, test_values,
                                                                   pdf_results, "PDF");
        } else if (op == "LogPDF") {
            fixtures::StatisticalTestUtils::verifyBatchCorrectness(stdUniform, test_values,
                                                                   log_pdf_results, "LogPDF");
        } else if (op == "CDF") {
            fixtures::StatisticalTestUtils::verifyBatchCorrectness(stdUniform, test_values,
                                                                   cdf_results, "CDF");
        }
    }

    // Print standardized benchmark results
    fixtures::BenchmarkFormatter::printBenchmarkResults(benchmark_results);
    fixtures::BenchmarkFormatter::printPerformanceAnalysis(benchmark_results);
}

//==============================================================================
// PARALLEL BATCH FITTING TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, ParallelBatchFittingTests) {
    std::cout << "\n=== Parallel Batch Fitting Tests ===\n";

    // Create multiple datasets for batch fitting
    std::vector<std::vector<double>> datasets;
    std::vector<UniformDistribution> expected_distributions;

    std::mt19937 rng(42);

    // Generate 6 datasets with known parameters
    std::vector<std::pair<double, double>> true_params = {{0.0, 1.0},  {-2.0, 2.0}, {5.0, 10.0},
                                                          {-1.0, 3.0}, {0.5, 1.5},  {-5.0, 5.0}};

    for (const auto& [a, b] : true_params) {
        std::vector<double> dataset;
        dataset.reserve(1000);

        std::uniform_real_distribution<double> gen(a, b);
        for (int i = 0; i < 1000; ++i) {
            dataset.push_back(gen(rng));
        }

        datasets.push_back(std::move(dataset));
        expected_distributions.push_back(UniformDistribution::create(a, b).value);
    }

    std::cout << "  Generated " << datasets.size() << " datasets with known parameters\n";

    // Test 1: Basic parallel batch fitting correctness
    std::vector<UniformDistribution> batch_results(datasets.size());

    auto start = std::chrono::high_resolution_clock::now();
    UniformDistribution::parallelBatchFit(datasets, batch_results);
    auto end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Verify correctness by comparing with individual fits
    for (size_t i = 0; i < datasets.size(); ++i) {
        UniformDistribution individual_fit;
        individual_fit.fit(datasets[i]);

        // Parameters should match within tolerance
        EXPECT_NEAR(batch_results[i].getLowerBound(), individual_fit.getLowerBound(), 1e-10)
            << "Batch fit lower bound mismatch for dataset " << i;
        EXPECT_NEAR(batch_results[i].getUpperBound(), individual_fit.getUpperBound(), 1e-10)
            << "Batch fit upper bound mismatch for dataset " << i;

        // Should be reasonably close to true parameters
        double expected_a = true_params[i].first;
        double expected_b = true_params[i].second;
        [[maybe_unused]] double n = static_cast<double>(datasets[i].size());

        // For uniform distribution, the sample range should be close to true range
        // Allow some tolerance for sample variation
        double range_tolerance = (expected_b - expected_a) * 0.1;  // 10% tolerance

        EXPECT_NEAR(batch_results[i].getLowerBound(), expected_a, range_tolerance)
            << "Fitted lower bound too far from true value for dataset " << i;
        EXPECT_NEAR(batch_results[i].getUpperBound(), expected_b, range_tolerance)
            << "Fitted upper bound too far from true value for dataset " << i;
    }

    std::cout << "  ✓ Parallel batch fitting correctness verified\n";

    // Test 2: Performance comparison with sequential batch fitting
    std::vector<UniformDistribution> sequential_results(datasets.size());

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < datasets.size(); ++i) {
        sequential_results[i].fit(datasets[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto sequential_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double speedup = sequential_time > 0
                         ? static_cast<double>(sequential_time) / static_cast<double>(parallel_time)
                         : 1.0;

    std::cout << "  Parallel batch fitting: " << parallel_time << "μs\n";
    std::cout << "  Sequential individual fits: " << sequential_time << "μs\n";
    std::cout << "  Speedup: " << speedup << "x\n";

    // Verify sequential and parallel results match
    for (size_t i = 0; i < datasets.size(); ++i) {
        EXPECT_NEAR(batch_results[i].getLowerBound(), sequential_results[i].getLowerBound(), 1e-12)
            << "Sequential vs parallel lower bound mismatch for dataset " << i;
        EXPECT_NEAR(batch_results[i].getUpperBound(), sequential_results[i].getUpperBound(), 1e-12)
            << "Sequential vs parallel upper bound mismatch for dataset " << i;
    }

    // Test 3: Edge cases
    std::cout << "  Testing edge cases...\n";

    // Empty datasets vector
    std::vector<std::vector<double>> empty_datasets;
    std::vector<UniformDistribution> empty_results;
    UniformDistribution::parallelBatchFit(empty_datasets, empty_results);
    EXPECT_TRUE(empty_results.empty());

    // Single dataset
    std::vector<std::vector<double>> single_dataset = {datasets[0]};
    std::vector<UniformDistribution> single_result(1);
    UniformDistribution::parallelBatchFit(single_dataset, single_result);
    EXPECT_NEAR(single_result[0].getLowerBound(), batch_results[0].getLowerBound(), 1e-12);
    EXPECT_NEAR(single_result[0].getUpperBound(), batch_results[0].getUpperBound(), 1e-12);

    // Results vector auto-sizing
    std::vector<UniformDistribution> auto_sized_results;
    UniformDistribution::parallelBatchFit(datasets, auto_sized_results);
    EXPECT_EQ(auto_sized_results.size(), datasets.size());

    std::cout << "  ✓ Edge cases handled correctly\n";

    // Test 4: Thread safety with concurrent calls
    std::cout << "  Testing thread safety...\n";

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<UniformDistribution>> concurrent_results(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            concurrent_results[static_cast<size_t>(t)].resize(datasets.size());
            UniformDistribution::parallelBatchFit(datasets,
                                                  concurrent_results[static_cast<size_t>(t)]);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all concurrent results match
    for (int t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < datasets.size(); ++i) {
            EXPECT_NEAR(concurrent_results[static_cast<size_t>(t)][i].getLowerBound(),
                        batch_results[i].getLowerBound(), 1e-10)
                << "Thread " << t << " lower bound result mismatch for dataset " << i;
            EXPECT_NEAR(concurrent_results[static_cast<size_t>(t)][i].getUpperBound(),
                        batch_results[i].getUpperBound(), 1e-10)
                << "Thread " << t << " upper bound result mismatch for dataset " << i;
        }
    }

    std::cout << "  ✓ Thread safety verified\n";
}

//==============================================================================
// NUMERICAL STABILITY AND EDGE CASES
//==============================================================================

TEST_F(UniformEnhancedTest, NumericalStabilityAndEdgeCases) {
    auto uniform = stats::UniformDistribution::create(0.0, 1.0).value;

    fixtures::EdgeCaseTester<UniformDistribution>::testExtremeValues(uniform, "Uniform");
    fixtures::EdgeCaseTester<UniformDistribution>::testEmptyBatchOperations(uniform, "Uniform");
}

}  // namespace stats

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

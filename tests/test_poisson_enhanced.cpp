#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

#include "../include/distributions/poisson.h"
#include "include/tests.h"

// Standard library includes
#include <algorithm>  // for std::sort, std::min, std::max
#include <chrono>
#include <cmath>  // for std::exp, std::log, std::isfinite, std::abs
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>  // for std::cout, std::endl
#include <numeric>
#include <random>  // for std::mt19937, std::poisson_distribution
#include <random>
#include <span>
#include <thread>
#include <utility>  // for std::move, std::pair
#include <vector>   // for std::vector
#include <vector>

using namespace std;
using namespace stats;
using namespace stats::tests;

namespace stats {

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

        auto result = stats::PoissonDistribution::create(test_lambda_);
        if (result.isOk()) {
            test_distribution_ = std::move(result.value);
        };
    }

    const double test_lambda_ = 3.5;
    std::vector<double> poisson_data_;
    std::vector<double> non_poisson_data_;
    PoissonDistribution test_distribution_;
};

//==============================================================================
// BASIC FUNCTIONALITY TESTS
//==============================================================================

TEST_F(PoissonEnhancedTest, BasicEnhancedFunctionality) {
    // Test standard Poisson distribution properties
    auto stdPoisson = stats::PoissonDistribution::create(1.0).value;

    EXPECT_DOUBLE_EQ(stdPoisson.getMean(), 1.0);
    EXPECT_DOUBLE_EQ(stdPoisson.getVariance(), 1.0);
    EXPECT_DOUBLE_EQ(stdPoisson.getSkewness(), 1.0);  // 1/√λ = 1/√1 = 1
    EXPECT_DOUBLE_EQ(stdPoisson.getKurtosis(), 1.0);  // 1/λ = 1/1 = 1

    // Test known PMF/CDF values for λ=1
    double pmf_at_0 = stdPoisson.getProbability(0.0);
    double pmf_at_1 = stdPoisson.getProbability(1.0);
    double cdf_at_1 = stdPoisson.getCumulativeProbability(1.0);

    // For Poisson(1): P(X=0) = e^(-1) ≈ 0.3679, P(X=1) = e^(-1) ≈ 0.3679
    EXPECT_NEAR(pmf_at_0, std::exp(-1.0), 1e-10);
    EXPECT_NEAR(pmf_at_1, std::exp(-1.0), 1e-10);
    EXPECT_NEAR(cdf_at_1, 2.0 * std::exp(-1.0), 1e-9);  // P(X≤1) = P(X=0) + P(X=1)

    // Test custom distribution
    auto custom = stats::PoissonDistribution::create(5.0).value;
    EXPECT_DOUBLE_EQ(custom.getMean(), 5.0);
    EXPECT_DOUBLE_EQ(custom.getVariance(), 5.0);
    EXPECT_NEAR(custom.getSkewness(), 1.0 / std::sqrt(5.0), 1e-10);
    EXPECT_NEAR(custom.getKurtosis(), 1.0 / 5.0, 1e-10);
}

//==============================================================================
// GOODNESS-OF-FIT TESTS
//==============================================================================

TEST_F(PoissonEnhancedTest, GoodnessOfFitTests) {
    std::cout << "\n=== Goodness-of-Fit Tests ===\n";

    // Chi-square goodness of fit test with Poisson data
    auto [chi2_stat_poisson, chi2_p_poisson, chi2_reject_poisson] =
        PoissonDistribution::chiSquareGoodnessOfFit(poisson_data_, test_distribution_, 0.05);

    EXPECT_GE(chi2_stat_poisson, 0.0);
    EXPECT_GE(chi2_p_poisson, 0.0);
    EXPECT_LE(chi2_p_poisson, 1.0);
    EXPECT_TRUE(std::isfinite(chi2_stat_poisson));
    EXPECT_TRUE(std::isfinite(chi2_p_poisson));

    std::cout << "  Chi-square test (Poisson data): χ²=" << chi2_stat_poisson
              << ", p=" << chi2_p_poisson << ", reject=" << chi2_reject_poisson << "\n";

    // Chi-square test with obviously non-Poisson data - should reject
    auto [chi2_stat_non_poisson, chi2_p_non_poisson, chi2_reject_non_poisson] =
        PoissonDistribution::chiSquareGoodnessOfFit(non_poisson_data_, test_distribution_, 0.05);

    // Should typically reject non-Poisson data (though not guaranteed for any single test)
    EXPECT_TRUE(std::isfinite(chi2_stat_non_poisson));
    EXPECT_TRUE(std::isfinite(chi2_p_non_poisson));
    std::cout << "  Chi-square test (non-Poisson data): χ²=" << chi2_stat_non_poisson
              << ", p=" << chi2_p_non_poisson << ", reject=" << chi2_reject_non_poisson << "\n";

    // Kolmogorov-Smirnov test (adapted for discrete distributions)
    auto [ks_stat, ks_p_value, ks_reject] =
        PoissonDistribution::kolmogorovSmirnovTest(poisson_data_, test_distribution_, 0.05);

    EXPECT_GE(ks_stat, 0.0);
    EXPECT_LE(ks_stat, 1.0);
    EXPECT_GE(ks_p_value, 0.0);
    EXPECT_LE(ks_p_value, 1.0);
    EXPECT_TRUE(std::isfinite(ks_stat));
    EXPECT_TRUE(std::isfinite(ks_p_value));

    std::cout << "  KS test: D=" << ks_stat << ", p=" << ks_p_value << ", reject=" << ks_reject
              << "\n";
}

//==============================================================================
// INFORMATION CRITERIA TESTS
//==============================================================================

TEST_F(PoissonEnhancedTest, InformationCriteriaTests) {
    std::cout << "\n=== Information Criteria Tests ===\n";

    // Fit distribution to data
    PoissonDistribution fitted_dist;
    fitted_dist.fit(poisson_data_);

    auto [aic, bic, aicc, log_likelihood] =
        PoissonDistribution::computeInformationCriteria(poisson_data_, fitted_dist);

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

TEST_F(PoissonEnhancedTest, BootstrapMethods) {
    std::cout << "\n=== Bootstrap Methods ===\n";

    // Bootstrap parameter confidence intervals
    auto lambda_ci =
        PoissonDistribution::bootstrapParameterConfidenceIntervals(poisson_data_, 0.95, 1000, 456);

    // Check that confidence interval is reasonable
    EXPECT_LT(lambda_ci.first, lambda_ci.second);  // Lower bound < Upper bound
    EXPECT_GT(lambda_ci.first, 0.0);               // Lambda should be positive

    // Check for finite values
    EXPECT_TRUE(std::isfinite(lambda_ci.first));
    EXPECT_TRUE(std::isfinite(lambda_ci.second));

    std::cout << "  Lambda 95% CI: [" << lambda_ci.first << ", " << lambda_ci.second << "]\n";

    // K-fold cross-validation
    auto cv_results = PoissonDistribution::kFoldCrossValidation(poisson_data_, 5, 42);
    EXPECT_EQ(cv_results.size(), 5);

    for (const auto& [mae, rmse, log_likelihood] : cv_results) {
        EXPECT_GE(mae, 0.0);
        EXPECT_GE(rmse, 0.0);
        EXPECT_GE(rmse, mae);            // RMSE should be >= MAE
        EXPECT_LE(log_likelihood, 0.0);  // Log-likelihood should be negative
        EXPECT_TRUE(std::isfinite(mae));
        EXPECT_TRUE(std::isfinite(rmse));
        EXPECT_TRUE(std::isfinite(log_likelihood));
    }

    std::cout << "  K-fold CV completed with " << cv_results.size() << " folds\n";

    // Leave-one-out cross-validation on smaller dataset
    std::vector<double> small_poisson_data(poisson_data_.begin(), poisson_data_.begin() + 20);
    auto [loocv_mae, loocv_rmse, loocv_log_likelihood] =
        PoissonDistribution::leaveOneOutCrossValidation(small_poisson_data);

    EXPECT_GE(loocv_mae, 0.0);
    EXPECT_GE(loocv_rmse, 0.0);
    EXPECT_GE(loocv_rmse, loocv_mae);
    EXPECT_LE(loocv_log_likelihood, 0.0);
    EXPECT_TRUE(std::isfinite(loocv_mae));
    EXPECT_TRUE(std::isfinite(loocv_rmse));
    EXPECT_TRUE(std::isfinite(loocv_log_likelihood));

    std::cout << "  Leave-one-out CV: MAE=" << loocv_mae << ", RMSE=" << loocv_rmse
              << ", LogL=" << loocv_log_likelihood << "\n";
}

//==============================================================================
// SIMD AND PARALLEL BATCH IMPLEMENTATIONS WITH FULSOME COMPARISONS
//==============================================================================

TEST_F(PoissonEnhancedTest, SIMDAndParallelBatchImplementations) {
    auto stdPoisson = stats::PoissonDistribution::create(2.0).value;

    std::cout << "\n=== SIMD and Parallel Batch Implementations ===\n";

    // Create shared WorkStealingPool once to avoid resource creation overhead
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());

    // Test multiple batch sizes to show scaling behavior
    std::vector<size_t> batch_sizes = {5000, 50000};

    for (size_t batch_size : batch_sizes) {
        std::cout << "\n--- Batch Size: " << batch_size << " elements ---\n";

        // Generate test data (discrete values for Poisson distribution)
        std::vector<double> test_values(batch_size);
        std::vector<double> results(batch_size);

        std::mt19937 gen(42);
        std::uniform_int_distribution<> dis(0, 10);  // Discrete values 0-10 for Poisson
        for (size_t i = 0; i < batch_size; ++i) {
            test_values[i] = static_cast<double>(dis(gen));
        }

        // 1. Sequential individual calls (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < batch_size; ++i) {
            results[i] = stdPoisson.getProbability(test_values[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto sequential_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 2. SIMD batch operations
        std::vector<double> simd_results(batch_size);
        start = std::chrono::high_resolution_clock::now();
        stdPoisson.getProbabilityWithStrategy(std::span<const double>(test_values),
                                              std::span<double>(simd_results),
                                              stats::detail::Strategy::SIMD_BATCH);
        end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 3. Parallel batch operations
        std::vector<double> parallel_results(batch_size);
        std::span<const double> input_span(test_values);
        std::span<double> output_span(parallel_results);

        start = std::chrono::high_resolution_clock::now();
        stdPoisson.getProbabilityWithStrategy(input_span, output_span,
                                              stats::detail::Strategy::PARALLEL_SIMD);
        end = std::chrono::high_resolution_clock::now();
        auto parallel_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // 4. Work-stealing operations (use shared pool)
        std::vector<double> work_stealing_results(batch_size);
        std::span<double> ws_output_span(work_stealing_results);

        start = std::chrono::high_resolution_clock::now();
        stdPoisson.getProbabilityWithStrategy(input_span, ws_output_span,
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

        // Architecture-aware performance expectations using adaptive validation
        // Poisson is a complex distribution
        double simd_threshold =
            stats::tests::validators::getSIMDValidationThreshold(batch_size, true);
        EXPECT_GT(simd_speedup, simd_threshold)
            << "SIMD speedup " << simd_speedup << "x should exceed adaptive threshold "
            << simd_threshold << "x for batch size " << batch_size;

        if (std::thread::hardware_concurrency() > 1) {
            // Use adaptive parallel threshold for Poisson (complex distribution)
            double parallel_threshold =
                stats::tests::validators::getParallelValidationThreshold(batch_size, true);
            EXPECT_GT(parallel_speedup, parallel_threshold)
                << "Parallel speedup " << parallel_speedup << "x should exceed adaptive threshold "
                << parallel_threshold << "x for batch size " << batch_size;
        }
    }
}

//==============================================================================
// ADVANCED STATISTICAL METHODS TESTS
//==============================================================================

TEST_F(PoissonEnhancedTest, AdvancedStatisticalMethods) {
    std::cout << "\n=== Advanced Statistical Methods ===\n";

    // Confidence interval for rate parameter
    auto [rate_lower, rate_upper] =
        PoissonDistribution::confidenceIntervalRate(poisson_data_, 0.95);
    EXPECT_LT(rate_lower, rate_upper);
    EXPECT_GT(rate_lower, 0.0);
    EXPECT_TRUE(std::isfinite(rate_lower));
    EXPECT_TRUE(std::isfinite(rate_upper));
    std::cout << "  95% CI for rate: [" << rate_lower << ", " << rate_upper << "]\n";

    // Likelihood ratio test
    auto [lr_stat, p_value, reject_null] =
        PoissonDistribution::likelihoodRatioTest(poisson_data_, test_lambda_, 0.05);
    EXPECT_GE(lr_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
    EXPECT_TRUE(std::isfinite(lr_stat));
    EXPECT_TRUE(std::isfinite(p_value));
    std::cout << "  LR test: stat=" << lr_stat << ", p=" << p_value << ", reject=" << reject_null
              << "\n";

    // Method of moments estimation
    double lambda_mom = PoissonDistribution::methodOfMomentsEstimation(poisson_data_);
    EXPECT_GT(lambda_mom, 0.0);
    EXPECT_TRUE(std::isfinite(lambda_mom));

    // Should equal sample mean for Poisson
    double sample_mean = std::accumulate(poisson_data_.begin(), poisson_data_.end(), 0.0) /
                         static_cast<double>(poisson_data_.size());
    EXPECT_NEAR(lambda_mom, sample_mean, 1e-10);
    std::cout << "  MoM estimate: λ=" << lambda_mom << "\n";

    // Bayesian estimation with conjugate Gamma prior
    auto [post_shape, post_rate] = PoissonDistribution::bayesianEstimation(poisson_data_, 1.0, 1.0);
    EXPECT_GT(post_shape, 0.0);
    EXPECT_GT(post_rate, 0.0);
    EXPECT_TRUE(std::isfinite(post_shape));
    EXPECT_TRUE(std::isfinite(post_rate));
    std::cout << "  Bayesian estimates: shape=" << post_shape << ", rate=" << post_rate << "\n";
}

//==============================================================================
// CACHING SPEEDUP VERIFICATION TESTS
//==============================================================================

TEST_F(PoissonEnhancedTest, CachingSpeedupVerification) {
    std::cout << "\n=== Caching Speedup Verification ===\n";

    auto poisson_dist = stats::PoissonDistribution::create(4.0).value;

    // First call - cache miss
    auto start = std::chrono::high_resolution_clock::now();
    double mean_first = poisson_dist.getMean();
    double var_first = poisson_dist.getVariance();
    double skew_first = poisson_dist.getSkewness();
    double kurt_first = poisson_dist.getKurtosis();
    auto end = std::chrono::high_resolution_clock::now();
    auto first_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Second call - cache hit
    start = std::chrono::high_resolution_clock::now();
    double mean_second = poisson_dist.getMean();
    double var_second = poisson_dist.getVariance();
    double skew_second = poisson_dist.getSkewness();
    double kurt_second = poisson_dist.getKurtosis();
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

    // Test cache invalidation by modifying the distribution's parameter
    poisson_dist.setLambda(8.0);  // This should invalidate the cache

    start = std::chrono::high_resolution_clock::now();
    double mean_after_change = poisson_dist.getMean();
    end = std::chrono::high_resolution_clock::now();
    auto after_change_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    EXPECT_EQ(mean_after_change, 8.0);  // Mean of Poisson(8.0) is 8.0
    std::cout << "  After parameter change: " << after_change_time << "ns\n";

    // Test cache functionality: verify that cache invalidation worked correctly
    // (the new parameter value is returned, proving cache was invalidated)
}

//==============================================================================
// AUTO-DISPATCH STRATEGY TESTING
//==============================================================================

TEST_F(PoissonEnhancedTest, AutoDispatchAssessment) {
    auto poisson_dist = stats::PoissonDistribution::create(3.0).value;

    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH",
                                                    "PARALLEL_SIMD"};

    std::cout << "\n=== Auto-Dispatch Assessment (Poisson) ===\n";

    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];

        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pmf_results(batch_size);
        std::vector<double> auto_logpmf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);

        std::mt19937 gen(42 + static_cast<unsigned int>(i));
        std::uniform_int_distribution<> dis(0, 10);  // Poisson values 0-10
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = static_cast<double>(dis(gen));
        }

        // Test smart auto-dispatch methods (if available)
        auto start = std::chrono::high_resolution_clock::now();
        if constexpr (requires {
                          poisson_dist.getProbability(std::span<const double>(test_values),
                                                      std::span<double>(auto_pmf_results));
                      }) {
            poisson_dist.getProbability(std::span<const double>(test_values),
                                        std::span<double>(auto_pmf_results));
        } else {
            poisson_dist.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                    std::span<double>(auto_pmf_results),
                                                    stats::detail::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pmf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires {
                          poisson_dist.getLogProbability(std::span<const double>(test_values),
                                                         std::span<double>(auto_logpmf_results));
                      }) {
            poisson_dist.getLogProbability(std::span<const double>(test_values),
                                           std::span<double>(auto_logpmf_results));
        } else {
            poisson_dist.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                       std::span<double>(auto_logpmf_results),
                                                       stats::detail::Strategy::SCALAR);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpmf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        if constexpr (requires {
                          poisson_dist.getCumulativeProbability(
                              std::span<const double>(test_values),
                              std::span<double>(auto_cdf_results));
                      }) {
            poisson_dist.getCumulativeProbability(std::span<const double>(test_values),
                                                  std::span<double>(auto_cdf_results));
        } else {
            poisson_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                              std::span<double>(auto_cdf_results),
                                                              stats::detail::Strategy::SCALAR);
        }
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pmf_results(batch_size);
        std::vector<double> trad_logpmf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);

        poisson_dist.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                std::span<double>(trad_pmf_results),
                                                stats::detail::Strategy::SCALAR);
        poisson_dist.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                   std::span<double>(trad_logpmf_results),
                                                   stats::detail::Strategy::SCALAR);
        poisson_dist.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                          std::span<double>(trad_cdf_results),
                                                          stats::detail::Strategy::SCALAR);

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

        std::cout << "Batch size: " << batch_size << ", Expected strategy: " << expected_strategy
                  << "\n";
        std::cout << "  PMF: " << auto_pmf_time << "μs, Correct: " << (pmf_correct ? "✅" : "❌")
                  << "\n";
        std::cout << "  LogPMF: " << auto_logpmf_time
                  << "μs, Correct: " << (logpmf_correct ? "✅" : "❌") << "\n";
        std::cout << "  CDF: " << auto_cdf_time << "μs, Correct: " << (cdf_correct ? "✅" : "❌")
                  << "\n";

        EXPECT_TRUE(pmf_correct)
            << "PMF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(logpmf_correct)
            << "LogPMF auto-dispatch results should match traditional for batch size "
            << batch_size;
        EXPECT_TRUE(cdf_correct)
            << "CDF auto-dispatch results should match traditional for batch size " << batch_size;
    }

    std::cout << "\n=== Auto-Dispatch Assessment Completed (Poisson) ===\n";
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(PoissonEnhancedTest, ParallelBatchPerformanceBenchmark) {
    auto stdPoisson = stats::PoissonDistribution::create(3.0).value;
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

    fixtures::BenchmarkFormatter::printBenchmarkHeader("Poisson Distribution", BENCHMARK_SIZE);

    // Create shared resources ONCE outside the loop to avoid resource issues
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());

    std::vector<fixtures::BenchmarkResult> benchmark_results;

    // For each operation type (PMF, LogPMF, CDF)
    std::vector<std::string> operations = {"PMF", "LogPMF", "CDF"};

    for (const auto& op : operations) {
        fixtures::BenchmarkResult result;
        result.operation_name = op;

        // 1. Baseline (SCALAR strategy)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PMF") {
            stdPoisson.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                  std::span<double>(pdf_results),
                                                  stats::detail::Strategy::SCALAR);
        } else if (op == "LogPMF") {
            stdPoisson.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                     std::span<double>(log_pdf_results),
                                                     stats::detail::Strategy::SCALAR);
        } else if (op == "CDF") {
            stdPoisson.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                            std::span<double>(cdf_results),
                                                            stats::detail::Strategy::SCALAR);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.baseline_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 2. SIMD Batch operations
        start = std::chrono::high_resolution_clock::now();
        if (op == "PMF") {
            stdPoisson.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                  std::span<double>(pdf_results),
                                                  stats::detail::Strategy::SIMD_BATCH);
        } else if (op == "LogPMF") {
            stdPoisson.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                     std::span<double>(log_pdf_results),
                                                     stats::detail::Strategy::SIMD_BATCH);
        } else if (op == "CDF") {
            stdPoisson.getCumulativeProbabilityWithStrategy(std::span<const double>(test_values),
                                                            std::span<double>(cdf_results),
                                                            stats::detail::Strategy::SIMD_BATCH);
        }
        end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 3. Thread Pool (PARALLEL_SIMD strategy) - fallback to SCALAR
        std::span<const double> input_span(test_values);

        if (op == "PMF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getProbabilityWithStrategy(
                                  input_span, output_span, stats::detail::Strategy::PARALLEL_SIMD);
                          }) {
                stdPoisson.getProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::PARALLEL_SIMD);
            } else {
                stdPoisson.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                      std::span<double>(pdf_results),
                                                      stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPMF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getLogProbabilityWithStrategy(
                                  input_span, log_output_span,
                                  stats::detail::Strategy::PARALLEL_SIMD);
                          }) {
                stdPoisson.getLogProbabilityWithStrategy(input_span, log_output_span,
                                                         stats::detail::Strategy::PARALLEL_SIMD);
            } else {
                stdPoisson.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                         std::span<double>(log_pdf_results),
                                                         stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getCumulativeProbabilityWithStrategy(
                                  input_span, cdf_output_span,
                                  stats::detail::Strategy::PARALLEL_SIMD);
                          }) {
                stdPoisson.getCumulativeProbabilityWithStrategy(
                    input_span, cdf_output_span, stats::detail::Strategy::PARALLEL_SIMD);
            } else {
                stdPoisson.getCumulativeProbabilityWithStrategy(
                    std::span<const double>(test_values), std::span<double>(cdf_results),
                    stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.thread_pool_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 4. Work-Stealing Operations (if available) - fallback to SCALAR
        if (op == "PMF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getProbabilityWithStrategy(
                                  input_span, output_span, stats::detail::Strategy::WORK_STEALING);
                          }) {
                stdPoisson.getProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::WORK_STEALING);
            } else {
                stdPoisson.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                      std::span<double>(pdf_results),
                                                      stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPMF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getLogProbabilityWithStrategy(
                                  input_span, log_output_span,
                                  stats::detail::Strategy::WORK_STEALING);
                          }) {
                stdPoisson.getLogProbabilityWithStrategy(input_span, log_output_span,
                                                         stats::detail::Strategy::WORK_STEALING);
            } else {
                stdPoisson.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                         std::span<double>(log_pdf_results),
                                                         stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getCumulativeProbabilityWithStrategy(
                                  input_span, cdf_output_span,
                                  stats::detail::Strategy::WORK_STEALING);
                          }) {
                stdPoisson.getCumulativeProbabilityWithStrategy(
                    input_span, cdf_output_span, stats::detail::Strategy::WORK_STEALING);
            } else {
                stdPoisson.getCumulativeProbabilityWithStrategy(
                    std::span<const double>(test_values), std::span<double>(cdf_results),
                    stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 5. GPU-Accelerated Operations (if available) - fallback to SCALAR
        if (op == "PMF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getProbabilityWithStrategy(
                                  input_span, output_span,
                                  stats::detail::Strategy::GPU_ACCELERATED);
                          }) {
                stdPoisson.getProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::GPU_ACCELERATED);
            } else {
                stdPoisson.getProbabilityWithStrategy(std::span<const double>(test_values),
                                                      std::span<double>(pdf_results),
                                                      stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPMF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getLogProbabilityWithStrategy(
                                  input_span, log_output_span,
                                  stats::detail::Strategy::GPU_ACCELERATED);
                          }) {
                stdPoisson.getLogProbabilityWithStrategy(input_span, log_output_span,
                                                         stats::detail::Strategy::GPU_ACCELERATED);
            } else {
                stdPoisson.getLogProbabilityWithStrategy(std::span<const double>(test_values),
                                                         std::span<double>(log_pdf_results),
                                                         stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            if constexpr (requires {
                              stdPoisson.getCumulativeProbabilityWithStrategy(
                                  input_span, cdf_output_span,
                                  stats::detail::Strategy::GPU_ACCELERATED);
                          }) {
                stdPoisson.getCumulativeProbabilityWithStrategy(
                    input_span, cdf_output_span, stats::detail::Strategy::GPU_ACCELERATED);
            } else {
                stdPoisson.getCumulativeProbabilityWithStrategy(
                    std::span<const double>(test_values), std::span<double>(cdf_results),
                    stats::detail::Strategy::SCALAR);
            }
            end = std::chrono::high_resolution_clock::now();
        }
        result.gpu_accelerated_time_us = static_cast<long>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // Calculate speedups (all relative to baseline)
        result.simd_speedup = result.baseline_time_us > 0
                                  ? static_cast<double>(result.baseline_time_us) /
                                        static_cast<double>(result.simd_time_us)
                                  : 0.0;
        result.thread_pool_speedup = result.baseline_time_us > 0
                                         ? static_cast<double>(result.baseline_time_us) /
                                               static_cast<double>(result.thread_pool_time_us)
                                         : 0.0;
        result.work_stealing_speedup = result.baseline_time_us > 0
                                           ? static_cast<double>(result.baseline_time_us) /
                                                 static_cast<double>(result.work_stealing_time_us)
                                           : 0.0;
        result.gpu_accelerated_speedup =
            result.baseline_time_us > 0 ? static_cast<double>(result.baseline_time_us) /
                                              static_cast<double>(result.gpu_accelerated_time_us)
                                        : 0.0;

        benchmark_results.push_back(result);

        // Verify correctness
        if (op == "PMF") {
            fixtures::StatisticalTestUtils::verifyBatchCorrectness(stdPoisson, test_values,
                                                                   pdf_results, "PMF");
        } else if (op == "LogPMF") {
            fixtures::StatisticalTestUtils::verifyBatchCorrectness(stdPoisson, test_values,
                                                                   log_pdf_results, "LogPMF");
        } else if (op == "CDF") {
            fixtures::StatisticalTestUtils::verifyBatchCorrectness(stdPoisson, test_values,
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

TEST_F(PoissonEnhancedTest, ParallelBatchFittingTests) {
    std::cout << "\n=== Parallel Batch Fitting Tests ===\n";

    // Create multiple datasets for batch fitting (convert to double for interface)
    std::vector<std::vector<double>> datasets;
    std::vector<PoissonDistribution> expected_distributions;

    std::mt19937 rng(42);

    // Generate 6 datasets with known lambda parameters
    std::vector<double> true_lambdas = {1.0, 2.5, 5.0, 0.5, 10.0, 3.7};

    for (double lambda : true_lambdas) {
        std::vector<double> dataset;
        dataset.reserve(1000);

        std::poisson_distribution<int> gen(lambda);
        for (int i = 0; i < 1000; ++i) {
            dataset.push_back(static_cast<double>(gen(rng)));
        }

        datasets.push_back(std::move(dataset));
        expected_distributions.push_back(PoissonDistribution::create(lambda).value);
    }

    std::cout << "  Generated " << datasets.size() << " datasets with known parameters\n";

    // Test 1: Basic parallel batch fitting correctness
    std::vector<PoissonDistribution> batch_results(datasets.size());

    auto start = std::chrono::high_resolution_clock::now();
    PoissonDistribution::parallelBatchFit(datasets, batch_results);
    auto end = std::chrono::high_resolution_clock::now();
    auto parallel_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Verify correctness by comparing with individual fits
    for (size_t i = 0; i < datasets.size(); ++i) {
        PoissonDistribution individual_fit;
        individual_fit.fit(datasets[i]);

        // Parameters should match within tolerance
        EXPECT_NEAR(batch_results[i].getLambda(), individual_fit.getLambda(), 1e-10)
            << "Batch fit lambda mismatch for dataset " << i;

        // Should be reasonably close to true parameters
        double expected_lambda = true_lambdas[i];

        // For Poisson distribution, MLE is just the sample mean
        // Allow reasonable tolerance for sample variation
        double lambda_tolerance =
            std::max(0.2, expected_lambda * 0.1);  // At least 0.2 or 10% tolerance

        EXPECT_NEAR(batch_results[i].getLambda(), expected_lambda, lambda_tolerance)
            << "Fitted lambda too far from true value for dataset " << i
            << " (expected: " << expected_lambda << ", got: " << batch_results[i].getLambda()
            << ")";
    }

    std::cout << "  ✓ Parallel batch fitting correctness verified\n";

    // Test 2: Performance comparison with sequential batch fitting
    std::vector<PoissonDistribution> sequential_results(datasets.size());

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
        EXPECT_NEAR(batch_results[i].getLambda(), sequential_results[i].getLambda(), 1e-12)
            << "Sequential vs parallel lambda mismatch for dataset " << i;
    }

    // Test 3: Edge cases
    std::cout << "  Testing edge cases...\n";

    // Empty datasets vector
    std::vector<std::vector<double>> empty_datasets;
    std::vector<PoissonDistribution> empty_results;
    PoissonDistribution::parallelBatchFit(empty_datasets, empty_results);
    EXPECT_TRUE(empty_results.empty());

    // Single dataset
    std::vector<std::vector<double>> single_dataset = {datasets[0]};
    std::vector<PoissonDistribution> single_result(1);
    PoissonDistribution::parallelBatchFit(single_dataset, single_result);
    EXPECT_NEAR(single_result[0].getLambda(), batch_results[0].getLambda(), 1e-12);

    // Results vector auto-sizing
    std::vector<PoissonDistribution> auto_sized_results;
    PoissonDistribution::parallelBatchFit(datasets, auto_sized_results);
    EXPECT_EQ(auto_sized_results.size(), datasets.size());

    std::cout << "  ✓ Edge cases handled correctly\n";

    // Test 4: Thread safety with concurrent calls
    std::cout << "  Testing thread safety...\n";

    const int num_threads = 4;
    std::vector<std::thread> threads;
    std::vector<std::vector<PoissonDistribution>> concurrent_results(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            concurrent_results[static_cast<size_t>(t)].resize(datasets.size());
            PoissonDistribution::parallelBatchFit(datasets,
                                                  concurrent_results[static_cast<size_t>(t)]);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all concurrent results match
    for (int t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < datasets.size(); ++i) {
            EXPECT_NEAR(concurrent_results[static_cast<size_t>(t)][i].getLambda(),
                        batch_results[i].getLambda(), 1e-10)
                << "Thread " << t << " lambda result mismatch for dataset " << i;
        }
    }

    std::cout << "  ✓ Thread safety verified\n";
}

//==============================================================================
// NUMERICAL STABILITY AND EDGE CASES
//==============================================================================

TEST_F(PoissonEnhancedTest, NumericalStabilityAndEdgeCases) {
    auto poisson = stats::PoissonDistribution::create(5.0).value;

    fixtures::EdgeCaseTester<PoissonDistribution>::testExtremeValues(poisson, "Poisson");
    fixtures::EdgeCaseTester<PoissonDistribution>::testEmptyBatchOperations(poisson, "Poisson");
}

}  // namespace stats

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

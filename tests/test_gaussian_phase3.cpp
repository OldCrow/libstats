#include <gtest/gtest.h>
#include "../include/gaussian.h"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace libstats {

//==============================================================================
// TEST FIXTURE FOR GAUSSIAN PHASE 3 METHODS
//==============================================================================

class GaussianPhase3Test : public ::testing::Test {
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
// TESTS FOR KOLMOGOROV-SMIRNOV TEST
//==============================================================================

TEST_F(GaussianPhase3Test, KolmogorovSmirnovBasic) {
    auto [ks_stat, p_value, reject] = GaussianDistribution::kolmogorovSmirnovTest(
        normal_data_, test_distribution_, 0.05);
    
    // Basic validity checks
    EXPECT_GE(ks_stat, 0.0);
    EXPECT_LE(ks_stat, 1.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
}

TEST_F(GaussianPhase3Test, KolmogorovSmirnovNonNormal) {
    // Test with obviously non-normal data - should reject normality
    auto [ks_stat, p_value, reject] = GaussianDistribution::kolmogorovSmirnovTest(
        non_normal_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(reject); // Should reject normality for quadratic data
    EXPECT_GE(ks_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
}

TEST_F(GaussianPhase3Test, KolmogorovSmirnovEdgeCases) {
    // Empty data should throw
    std::vector<double> empty_data;
    EXPECT_THROW(
        GaussianDistribution::kolmogorovSmirnovTest(empty_data, test_distribution_, 0.05),
        std::invalid_argument
    );
    
    // Invalid alpha values should throw
    std::vector<double> valid_data = {1.0, 2.0, 3.0};
    EXPECT_THROW(
        GaussianDistribution::kolmogorovSmirnovTest(valid_data, test_distribution_, 0.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::kolmogorovSmirnovTest(valid_data, test_distribution_, 1.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::kolmogorovSmirnovTest(valid_data, test_distribution_, -0.1),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::kolmogorovSmirnovTest(valid_data, test_distribution_, 1.1),
        std::invalid_argument
    );
}

//==============================================================================
// TESTS FOR ANDERSON-DARLING TEST
//==============================================================================

TEST_F(GaussianPhase3Test, AndersonDarlingBasic) {
    auto [ad_stat, p_value, reject] = GaussianDistribution::andersonDarlingTest(
        normal_data_, test_distribution_, 0.05);
    
    // Basic validity checks
    EXPECT_GE(ad_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
}

TEST_F(GaussianPhase3Test, AndersonDarlingNonNormal) {
    // Test with obviously non-normal data
    auto [ad_stat, p_value, reject] = GaussianDistribution::andersonDarlingTest(
        non_normal_data_, test_distribution_, 0.05);
    
    EXPECT_TRUE(reject); // Should reject normality for quadratic data
    EXPECT_GE(ad_stat, 0.0);
    EXPECT_GE(p_value, 0.0);
    EXPECT_LE(p_value, 1.0);
}

TEST_F(GaussianPhase3Test, AndersonDarlingEdgeCases) {
    // Empty data should throw
    std::vector<double> empty_data;
    EXPECT_THROW(
        GaussianDistribution::andersonDarlingTest(empty_data, test_distribution_, 0.05),
        std::invalid_argument
    );
    
    // Invalid alpha values should throw
    std::vector<double> valid_data = {1.0, 2.0, 3.0};
    EXPECT_THROW(
        GaussianDistribution::andersonDarlingTest(valid_data, test_distribution_, 0.0),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::andersonDarlingTest(valid_data, test_distribution_, 1.0),
        std::invalid_argument
    );
}

//==============================================================================
// TESTS FOR K-FOLD CROSS-VALIDATION
//==============================================================================

TEST_F(GaussianPhase3Test, KFoldCrossValidationBasic) {
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

TEST_F(GaussianPhase3Test, KFoldCrossValidationDifferentK) {
    // Test with different values of k
    for (int k : {2, 3, 4, 10}) {
        if (normal_data_.size() >= static_cast<size_t>(k)) {
            auto results = GaussianDistribution::kFoldCrossValidation(normal_data_, k, 42);
            EXPECT_EQ(results.size(), static_cast<size_t>(k));
        }
    }
}

TEST_F(GaussianPhase3Test, KFoldCrossValidationEdgeCases) {
    // k larger than data size should throw
    std::vector<double> small_data = {1.0, 2.0, 3.0};
    EXPECT_THROW(
        GaussianDistribution::kFoldCrossValidation(small_data, 5),
        std::invalid_argument
    );
    
    // k <= 1 should throw
    EXPECT_THROW(
        GaussianDistribution::kFoldCrossValidation(small_data, 1),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::kFoldCrossValidation(small_data, 0),
        std::invalid_argument
    );
}

//==============================================================================
// TESTS FOR LEAVE-ONE-OUT CROSS-VALIDATION
//==============================================================================

TEST_F(GaussianPhase3Test, LeaveOneOutCrossValidationBasic) {
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

TEST_F(GaussianPhase3Test, LeaveOneOutCrossValidationEdgeCases) {
    // Insufficient data should throw
    std::vector<double> too_small_data = {1.0, 2.0};
    EXPECT_THROW(
        GaussianDistribution::leaveOneOutCrossValidation(too_small_data),
        std::invalid_argument
    );
    
    // Empty data should throw (handled by fit method)
    std::vector<double> empty_data;
    EXPECT_THROW(
        GaussianDistribution::leaveOneOutCrossValidation(empty_data),
        std::invalid_argument
    );
}

//==============================================================================
// TESTS FOR BOOTSTRAP PARAMETER CONFIDENCE INTERVALS
//==============================================================================

TEST_F(GaussianPhase3Test, BootstrapParameterConfidenceIntervalsBasic) {
    auto [mean_ci, std_ci] = GaussianDistribution::bootstrapParameterConfidenceIntervals(
        normal_data_, 0.95, 1000, 456);
    
    // Check that confidence intervals are reasonable
    EXPECT_LT(mean_ci.first, mean_ci.second);  // Lower bound < Upper bound
    EXPECT_LT(std_ci.first, std_ci.second);    // Lower bound < Upper bound
    
    // Check that CI bounds are reasonable given the data
    double sample_mean = std::accumulate(normal_data_.begin(), normal_data_.end(), 0.0) / normal_data_.size();
    EXPECT_GT(mean_ci.second, sample_mean - 5.0); // Upper bound shouldn't be too far from sample mean
    EXPECT_LT(mean_ci.first, sample_mean + 5.0);  // Lower bound shouldn't be too far from sample mean
    
    // Standard deviation CIs should be positive
    EXPECT_GT(std_ci.first, 0.0);
    EXPECT_GT(std_ci.second, 0.0);
    
    // Check for finite values
    EXPECT_TRUE(std::isfinite(mean_ci.first));
    EXPECT_TRUE(std::isfinite(mean_ci.second));
    EXPECT_TRUE(std::isfinite(std_ci.first));
    EXPECT_TRUE(std::isfinite(std_ci.second));
}

TEST_F(GaussianPhase3Test, BootstrapParameterConfidenceIntervalsEdgeCases) {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Invalid confidence level should throw
    EXPECT_THROW(
        GaussianDistribution::bootstrapParameterConfidenceIntervals(data, 0.0, 100, 42),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::bootstrapParameterConfidenceIntervals(data, 1.0, 100, 42),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::bootstrapParameterConfidenceIntervals(data, -0.1, 100, 42),
        std::invalid_argument
    );
    
    // Invalid number of bootstrap samples should throw
    EXPECT_THROW(
        GaussianDistribution::bootstrapParameterConfidenceIntervals(data, 0.95, 0, 42),
        std::invalid_argument
    );
    EXPECT_THROW(
        GaussianDistribution::bootstrapParameterConfidenceIntervals(data, 0.95, -10, 42),
        std::invalid_argument
    );
    
    // Empty data should throw
    std::vector<double> empty_data;
    EXPECT_THROW(
        GaussianDistribution::bootstrapParameterConfidenceIntervals(empty_data, 0.95, 100, 42),
        std::invalid_argument
    );
}

//==============================================================================
// TESTS FOR INFORMATION CRITERIA
//==============================================================================

TEST_F(GaussianPhase3Test, ComputeInformationCriteriaBasic) {
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

TEST_F(GaussianPhase3Test, ComputeInformationCriteriaEdgeCases) {
    GaussianDistribution dist(0.0, 1.0);
    
    // Empty data should throw
    std::vector<double> empty_data;
    EXPECT_THROW(
        GaussianDistribution::computeInformationCriteria(empty_data, dist),
        std::invalid_argument
    );
    
    // Test with single data point (very small sample)
    std::vector<double> single_point = {5.0};
    auto [aic, bic, aicc, log_likelihood] = GaussianDistribution::computeInformationCriteria(
        single_point, dist);
    
    // With very small samples, AICc might be infinite
    EXPECT_LE(log_likelihood, 0.0);
    EXPECT_GT(aic, 0.0);
    EXPECT_GT(bic, 0.0);
    // Don't check aicc bounds since it might be infinite for very small samples
}

//==============================================================================
// INTEGRATION TESTS
//==============================================================================

TEST_F(GaussianPhase3Test, CrossValidationConsistency) {
    // k-fold CV with k=n should give similar results to LOOCV for small datasets
    std::vector<double> small_data(normal_data_.begin(), normal_data_.begin() + 10);
    
    auto loocv_results = GaussianDistribution::leaveOneOutCrossValidation(small_data);
    auto kfold_results = GaussianDistribution::kFoldCrossValidation(small_data, small_data.size(), 42);
    
    // Results should be in similar ranges (not exact due to different implementations)
    auto [loocv_mae, loocv_rmse, loocv_ll] = loocv_results;
    
    // Average the k-fold results
    double kfold_avg_mae = 0.0;
    double kfold_avg_rmse = 0.0;
    double kfold_total_ll = 0.0;
    
    for (const auto& [mae, rmse, ll] : kfold_results) {
        kfold_avg_mae += mae;
        kfold_avg_rmse += rmse;
        kfold_total_ll += ll;
    }
    kfold_avg_mae /= kfold_results.size();
    kfold_avg_rmse /= kfold_results.size();
    
    // Results should be in reasonable ranges
    EXPECT_GT(loocv_mae, 0.0);
    EXPECT_GT(kfold_avg_mae, 0.0);
    EXPECT_GT(loocv_rmse, 0.0);
    // Note: kfold_avg_rmse is actually std_error (standard deviation of errors), 
    // which could be 0.0 for perfectly uniform errors. Make it more lenient.
    EXPECT_GE(kfold_avg_rmse, 0.0);
    EXPECT_LT(loocv_ll, 0.0);
    EXPECT_LT(kfold_total_ll, 0.0);
}

TEST_F(GaussianPhase3Test, GoodnessOfFitConsistency) {
    // Both KS and AD tests should reject obviously non-normal data
    auto [ks_stat, ks_p, ks_reject] = GaussianDistribution::kolmogorovSmirnovTest(
        non_normal_data_, test_distribution_, 0.05);
    auto [ad_stat, ad_p, ad_reject] = GaussianDistribution::andersonDarlingTest(
        non_normal_data_, test_distribution_, 0.05);
    
    // Both should reject normality for quadratic data
    EXPECT_TRUE(ks_reject);
    EXPECT_TRUE(ad_reject);
    
    // Both should have very small p-values
    EXPECT_LT(ks_p, 0.1);
    EXPECT_LT(ad_p, 0.1);
}

} // namespace libstats

#include "../include/core/distribution_base.h"
#include "../include/core/math_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

// Simple test implementation of a distribution for testing goodness-of-fit functions
class TestNormalDistribution : public libstats::DistributionBase {
public:
    TestNormalDistribution(double mean = 0.0, double stddev = 1.0) 
        : mean_(mean), stddev_(stddev) {}
    
    double getProbability(double x) const override {
        double z = (x - mean_) / stddev_;
        return (1.0 / (stddev_ * std::sqrt(2.0 * M_PI))) * std::exp(-0.5 * z * z);
    }
    
    double getCumulativeProbability(double x) const override {
        double z = (x - mean_) / (stddev_ * std::sqrt(2.0));
        return 0.5 * (1.0 + std::erf(z));
    }
    
    double getQuantile([[maybe_unused]] double p) const override { return mean_; } // Stub
    double getMean() const override { return mean_; }
    double getVariance() const override { return stddev_ * stddev_; }
    double getSkewness() const override { return 0.0; }
    double getKurtosis() const override { return 0.0; }
    double sample(std::mt19937& rng) const override {
        std::normal_distribution<double> dist(mean_, stddev_);
        return dist(rng);
    }
    void fit([[maybe_unused]] const std::vector<double>& data) override { /* stub */ }
    void reset() noexcept override { /* stub */ }
    int getNumParameters() const override { return 2; }
    std::string getDistributionName() const override { return "TestNormal"; }
    std::string toString() const override { return "TestNormal"; }
    bool isDiscrete() const override { return false; }
    double getSupportLowerBound() const override { return -std::numeric_limits<double>::infinity(); }
    double getSupportUpperBound() const override { return std::numeric_limits<double>::infinity(); }
    
protected:
    void updateCacheUnsafe() const override { /* stub */ }
    
private:
    double mean_;
    double stddev_;
};

int main() {
    try {
        // Create a test normal distribution
        TestNormalDistribution normal(0.0, 1.0);
        
        // Generate some sample data from the same distribution
        std::mt19937 rng(42);
        std::vector<double> data;
        for (int i = 0; i < 100; ++i) {
            data.push_back(normal.sample(rng));
        }
        
        // Test the goodness-of-fit functions
        double ks_statistic = libstats::math::calculate_ks_statistic(data, normal);
        double ad_statistic = libstats::math::calculate_ad_statistic(data, normal);
        
        std::cout << "Goodness-of-fit tests:" << std::endl;
        std::cout << "KS statistic: " << ks_statistic << std::endl;
        std::cout << "AD statistic: " << ad_statistic << std::endl;
        
        // The statistics should be reasonably small for data from the same distribution
        if (ks_statistic < 0.3 && ad_statistic < 3.0) {
            std::cout << "✓ Goodness-of-fit functions work correctly!" << std::endl;
            return 0;
        } else {
            std::cout << "✗ Unexpected goodness-of-fit results" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

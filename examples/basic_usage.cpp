/**
 * @file basic_usage.cpp
 * @brief Basic usage example for libstats
 */

#include "../include/libstats.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <span>

int main() {
    std::cout << "=== libstats Basic Usage Example ===" << std::endl;
    std::cout << "Version: " << libstats::VERSION_STRING << std::endl << std::endl;
    
    // Create a random number generator
    std::mt19937 rng(42);  // Fixed seed for reproducible results
    
    // Create different distributions
    std::cout << "1. Creating distributions:" << std::endl;
    libstats::Gaussian normal(0.0, 1.0);        // Standard normal
    libstats::Exponential exponential(2.0);     // Rate parameter = 2.0
    
    std::cout << "   - Gaussian N(0,1)" << std::endl;
    std::cout << "   - Exponential(λ=2.0)" << std::endl << std::endl;
    
    // Statistical properties
    std::cout << "2. Statistical properties:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "   Normal distribution:" << std::endl;
    std::cout << "     Mean: " << normal.getMean() << std::endl;
    std::cout << "     Variance: " << normal.getVariance() << std::endl;
    std::cout << "     Standard deviation: " << normal.getStandardDeviation() << std::endl;
    std::cout << "     Skewness: " << normal.getSkewness() << std::endl;
    std::cout << "     Kurtosis: " << normal.getKurtosis() << std::endl;
    
    std::cout << "   Exponential distribution:" << std::endl;
    std::cout << "     Mean: " << exponential.getMean() << std::endl;
    std::cout << "     Variance: " << exponential.getVariance() << std::endl;
    std::cout << "     Standard deviation: " << exponential.getStandardDeviation() << std::endl << std::endl;
    
    // PDF evaluation
    std::cout << "3. Probability density function (PDF):" << std::endl;
    std::cout << "   P(X=0.0) for N(0,1): " << normal.getProbability(0.0) << std::endl;
    std::cout << "   P(X=1.0) for N(0,1): " << normal.getProbability(1.0) << std::endl;
    std::cout << "   P(X=0.0) for Exp(2): " << exponential.getProbability(0.0) << std::endl;
    std::cout << "   P(X=0.5) for Exp(2): " << exponential.getProbability(0.5) << std::endl << std::endl;
    
    // CDF evaluation
    std::cout << "4. Cumulative distribution function (CDF):" << std::endl;
    std::cout << "   P(X<=0.0) for N(0,1): " << normal.getCumulativeProbability(0.0) << std::endl;
    std::cout << "   P(X<=1.0) for N(0,1): " << normal.getCumulativeProbability(1.0) << std::endl;
    std::cout << "   P(X<=0.5) for Exp(2): " << exponential.getCumulativeProbability(0.5) << std::endl << std::endl;
    
    // Quantiles
    std::cout << "5. Quantile function (inverse CDF):" << std::endl;
    std::cout << "   95th percentile of N(0,1): " << normal.getQuantile(0.95) << std::endl;
    std::cout << "   Median of N(0,1): " << normal.getQuantile(0.5) << std::endl;
    std::cout << "   75th percentile of Exp(2): " << exponential.getQuantile(0.75) << std::endl << std::endl;
    
    // Random sampling
    std::cout << "6. Random sampling:" << std::endl;
    std::cout << "   10 samples from N(0,1): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << normal.sample(rng);
    }
    std::cout << std::endl;
    
    std::cout << "   10 samples from Exp(2): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << exponential.sample(rng);
    }
    std::cout << std::endl << std::endl;
    
    // Bulk sampling
    std::cout << "7. Bulk sampling and fitting:" << std::endl;
    auto samples = normal.sample(rng, 1000);
    std::cout << "   Generated 1000 samples from N(0,1)" << std::endl;
    
    // Fit a new distribution to the samples
    libstats::Gaussian fitted_normal;
    fitted_normal.fit(samples);
    
    std::cout << "   Fitted Gaussian parameters:" << std::endl;
    std::cout << "     Estimated mean: " << fitted_normal.getMean() << std::endl;
    std::cout << "     Estimated std dev: " << fitted_normal.getStandardDeviation() << std::endl << std::endl;
    
    std::cout << "=== Example completed successfully ===" << std::endl;
    
    return 0;
}

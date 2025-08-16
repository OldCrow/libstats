/**
 * @file quick_start_tutorial.cpp
 * @brief Quick Start Tutorial - 5-minute getting started guide for libstats
 * 
 * This tutorial covers the essential operations in just a few minutes:
 * - Creating distributions
 * - Computing probabilities 
 * - Sampling random values
 * - Basic parameter fitting
 * 
 * Perfect for new users who want to get up and running quickly!
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats.h"
#include <iostream>
#include <random>

int main() {
    std::cout << "=== libstats Quick Start Tutorial ===" << std::endl;
    std::cout << "Learn the basics in just 5 minutes!\n" << std::endl;
    
    // Step 1: Create distributions
    std::cout << "📝 Step 1: Creating Distributions" << std::endl;
    auto normal = libstats::GaussianDistribution::create(0.0, 1.0).value;      // Standard normal N(0,1)
    auto exponential = libstats::ExponentialDistribution::create(1.5).value;   // Exponential with rate 1.5
    auto uniform = libstats::UniformDistribution::create(0.0, 10.0).value;     // Uniform on [0, 10]
    std::cout << "   ✓ Created Gaussian N(0,1), Exponential(1.5), and Uniform(0,10)\n" << std::endl;
    
    // Step 2: Basic properties
    std::cout << "📊 Step 2: Basic Properties" << std::endl;
    std::cout << "   Normal distribution mean: " << normal.getMean() << std::endl;
    std::cout << "   Exponential distribution mean: " << exponential.getMean() << std::endl;
    std::cout << "   Uniform distribution mean: " << uniform.getMean() << std::endl;
    std::cout << "   Uniform distribution range: [" << uniform.getSupportLowerBound() 
              << ", " << uniform.getSupportUpperBound() << "]\n" << std::endl;
    
    // Step 3: Compute probabilities
    std::cout << "🎯 Step 3: Computing Probabilities" << std::endl;
    double x = 1.0;
    std::cout << "   At x = " << x << ":" << std::endl;
    std::cout << "   - Normal PDF: " << normal.getProbability(x) << std::endl;
    std::cout << "   - Normal CDF: " << normal.getCumulativeProbability(x) << std::endl;
    std::cout << "   - Exponential PDF: " << exponential.getProbability(x) << std::endl;
    std::cout << "   - Uniform PDF: " << uniform.getProbability(x) << std::endl;
    std::cout << "   Tip: PDF = probability density, CDF = cumulative probability\n" << std::endl;
    
    // Step 4: Find percentiles (quantiles)
    std::cout << "📈 Step 4: Finding Percentiles" << std::endl;
    std::cout << "   Normal 95th percentile: " << normal.getQuantile(0.95) << std::endl;
    std::cout << "   Exponential median (50th percentile): " << exponential.getQuantile(0.5) << std::endl;
    std::cout << "   Uniform 25th percentile: " << uniform.getQuantile(0.25) << std::endl;
    std::cout << "   Tip: getQuantile(p) finds x where P(X ≤ x) = p\n" << std::endl;
    
    // Step 5: Generate random samples
    std::cout << "🎲 Step 5: Random Sampling" << std::endl;
    std::mt19937 rng(42);  // Random number generator with seed
    
    // Single samples
    std::cout << "   Single random samples:" << std::endl;
    std::cout << "   - Normal: " << normal.sample(rng) << std::endl;
    std::cout << "   - Exponential: " << exponential.sample(rng) << std::endl;
    std::cout << "   - Uniform: " << uniform.sample(rng) << std::endl;
    
    // Multiple samples
    auto normal_samples = normal.sample(rng, 1000);
    std::cout << "   Generated 1000 normal samples, first 5: ";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << normal_samples[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "   Tip: Use sample(rng, n) to generate n samples at once\n" << std::endl;
    
    // Step 6: Parameter fitting
    std::cout << "🔧 Step 6: Parameter Fitting" << std::endl;
    // Create some sample data from a known distribution
    auto true_dist = libstats::GaussianDistribution::create(5.0, 2.0).value;  // Mean=5, StdDev=2
    auto sample_data = true_dist.sample(rng, 500);
    
    // Fit a new distribution to this data
    libstats::Gaussian fitted_dist;
    fitted_dist.fit(sample_data);
    
    std::cout << "   Generated 500 samples from N(5.0, 2.0)" << std::endl;
    std::cout << "   Fitted distribution parameters:" << std::endl;
    std::cout << "   - Estimated mean: " << fitted_dist.getMean() 
              << " (should be ≈ 5.0)" << std::endl;
    std::cout << "   - Estimated std dev: " << fitted_dist.getStandardDeviation() 
              << " (should be ≈ 2.0)" << std::endl;
    std::cout << "   Tip: fit() automatically estimates parameters from data\n" << std::endl;
    
    // Step 7: Batch processing for performance
    std::cout << "⚡ Step 7: High-Performance Batch Processing" << std::endl;
    std::vector<double> input_values = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> output_values(input_values.size());
    
    // Process all values at once for better performance
    normal.getProbability(input_values, output_values);
    
    std::cout << "   Batch PDF computation results:" << std::endl;
    for (size_t i = 0; i < input_values.size(); ++i) {
        std::cout << "   - N(0,1) PDF(" << input_values[i] << ") = " 
                  << output_values[i] << std::endl;
    }
    std::cout << "   Tip: Batch operations are much faster for large datasets!\n" << std::endl;
    
    // Summary
    std::cout << "🎉 Congratulations! You've learned the essentials:" << std::endl;
    std::cout << "   ✓ Creating distributions: libstats::Gaussian(mean, stddev)" << std::endl;
    std::cout << "   ✓ Computing probabilities: getProbability(), getCumulativeProbability()" << std::endl;
    std::cout << "   ✓ Finding percentiles: getQuantile(probability)" << std::endl;
    std::cout << "   ✓ Random sampling: sample(rng) or sample(rng, count)" << std::endl;
    std::cout << "   ✓ Parameter fitting: fit(data_vector)" << std::endl;
    std::cout << "   ✓ Batch processing: Pass vectors for high performance" << std::endl;
    
    std::cout << "\n🔍 Next steps:" << std::endl;
    std::cout << "   - Try other distributions: Poisson, Gamma, Discrete" << std::endl;
    std::cout << "   - Explore statistical_validation_demo.cpp for advanced analysis" << std::endl;
    std::cout << "   - Check parallel_execution_demo.cpp for performance optimization" << std::endl;
    std::cout << "   - See basic_usage.cpp for comprehensive examples" << std::endl;
    
    return 0;
}

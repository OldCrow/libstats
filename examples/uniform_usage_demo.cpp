/**
 * @file uniform_usage_demo.cpp
 * @brief Basic usage demonstration for Uniform distribution
 *
 * This example demonstrates the core functionality of the Uniform distribution including:
 * - Distribution creation with bounds parameters
 * - Statistical property calculations (mean, variance, etc.)
 * - Probability density and cumulative distribution functions
 * - Quantile calculations (inverse CDF)
 * - Random sampling
 * - Parameter estimation from data
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats.h"

#include <iomanip>
#include <iostream>
#include <random>

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

int main() {
    std::cout << "=== libstats Uniform Distribution Usage Demo ===" << std::endl;
    std::cout << "Demonstration of Uniform distribution operations\n" << std::endl;

    // Create a random number generator
    std::mt19937 rng(42);  // Fixed seed for reproducible results

    print_separator("1. Distribution Creation");
    std::cout << "\nCreating Uniform distributions with different bounds:\n"
              << "  - U(0, 1): Standard uniform distribution on [0, 1]\n"
              << "  - U(-5, 10): Custom uniform distribution on [-5, 10]\n"
              << std::endl;

    auto standard = stats::UniformDistribution::create(0.0, 1.0).value;  // Standard uniform [0, 1]
    auto custom = stats::UniformDistribution::create(-5.0, 10.0).value;  // Custom uniform [-5, 10]

    std::cout << "✓ Standard uniform U(0,1) created" << std::endl;
    std::cout << "✓ Custom uniform U(-5,10) created" << std::endl;

    print_separator("2. Statistical Properties");
    std::cout << "\nComputing theoretical statistical properties:\n" << std::endl;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "📊 Standard Uniform U(0,1) properties:" << std::endl;
    std::cout << "     Mean: " << standard.getMean() << " [Expected: 0.5000]" << std::endl;
    std::cout << "     Variance: " << standard.getVariance() << " [Expected: 0.0833]" << std::endl;
    std::cout << "     Standard deviation: " << stats::getStandardDeviation(standard)
              << " [Expected: 0.2887]" << std::endl;
    std::cout << "     Skewness: " << standard.getSkewness() << " [Expected: 0.0000 - symmetric]"
              << std::endl;
    std::cout << "     Kurtosis: " << standard.getKurtosis() << " [Expected: 1.8000 - platykurtic]"
              << std::endl;

    std::cout << "\n📈 Custom Uniform U(-5,10) properties:" << std::endl;
    double expected_mean = (-5.0 + 10.0) / 2.0;                      // 2.5
    double expected_var = (10.0 - (-5.0)) * (10.0 - (-5.0)) / 12.0;  // 18.75
    std::cout << "     Mean: " << custom.getMean() << " [Expected: " << expected_mean << "]"
              << std::endl;
    std::cout << "     Variance: " << custom.getVariance() << " [Expected: " << expected_var << "]"
              << std::endl;
    std::cout << "     Standard deviation: " << stats::getStandardDeviation(custom)
              << " [Expected: " << std::sqrt(expected_var) << "]" << std::endl;
    std::cout << "     Support: [" << custom.getSupportLowerBound() << ", "
              << custom.getSupportUpperBound() << "]" << std::endl;

    print_separator("3. Probability Density Function (PDF)");
    std::cout << "\nEvaluating probability density at specific points:\n"
              << "(For uniform distributions: constant density within bounds, zero outside)\n"
              << std::endl;

    std::cout << "🎯 Standard Uniform U(0,1) PDF evaluations:" << std::endl;
    std::cout << "   f(x=0.5): " << standard.getProbability(0.5)
              << " [Within bounds - should be 1.0]" << std::endl;
    std::cout << "   f(x=0.0): " << standard.getProbability(0.0) << " [Lower boundary]"
              << std::endl;
    std::cout << "   f(x=1.0): " << standard.getProbability(1.0) << " [Upper boundary]"
              << std::endl;
    std::cout << "   f(x=-0.5): " << standard.getProbability(-0.5)
              << " [Outside bounds - should be 0.0]" << std::endl;

    std::cout << "\n📊 Custom Uniform U(-5,10) PDF evaluations:" << std::endl;
    double expected_density = 1.0 / (10.0 - (-5.0));  // 1/15 ≈ 0.0667
    std::cout << "   f(x=2.5): " << custom.getProbability(2.5) << " [Within bounds - should be "
              << expected_density << "]" << std::endl;
    std::cout << "   f(x=-5.0): " << custom.getProbability(-5.0) << " [Lower boundary]"
              << std::endl;
    std::cout << "   f(x=10.0): " << custom.getProbability(10.0) << " [Upper boundary]"
              << std::endl;
    std::cout << "   f(x=15.0): " << custom.getProbability(15.0)
              << " [Outside bounds - should be 0.0]" << std::endl;

    print_separator("4. Cumulative Distribution Function (CDF)");
    std::cout << "\nComputing cumulative probabilities P(X ≤ x):\n"
              << "(Linear function from 0 to 1 within bounds)\n"
              << std::endl;

    std::cout << "📊 Standard Uniform U(0,1) CDF evaluations:" << std::endl;
    std::cout << "   P(X≤0.25): " << standard.getCumulativeProbability(0.25)
              << " [Expected: 0.2500]" << std::endl;
    std::cout << "   P(X≤0.50): " << standard.getCumulativeProbability(0.50)
              << " [Expected: 0.5000]" << std::endl;
    std::cout << "   P(X≤0.75): " << standard.getCumulativeProbability(0.75)
              << " [Expected: 0.7500]" << std::endl;
    std::cout << "   P(X≤-1.0): " << standard.getCumulativeProbability(-1.0)
              << " [Expected: 0.0000 - below range]" << std::endl;
    std::cout << "   P(X≤2.0): " << standard.getCumulativeProbability(2.0)
              << " [Expected: 1.0000 - above range]" << std::endl;

    std::cout << "\n📈 Custom Uniform U(-5,10) CDF evaluations:" << std::endl;
    std::cout << "   P(X≤-2.5): " << custom.getCumulativeProbability(-2.5)
              << " [Expected: " << (-2.5 - (-5.0)) / (10.0 - (-5.0)) << "]" << std::endl;
    std::cout << "   P(X≤2.5): " << custom.getCumulativeProbability(2.5)
              << " [Expected: 0.5000 - at mean]" << std::endl;
    std::cout << "   P(X≤7.5): " << custom.getCumulativeProbability(7.5)
              << " [Expected: " << (7.5 - (-5.0)) / (10.0 - (-5.0)) << "]" << std::endl;

    print_separator("5. Quantile Function (Inverse CDF)");
    std::cout << "\nFinding values x such that P(X ≤ x) = p:\n"
              << "(Linear interpolation within bounds)\n"
              << std::endl;

    std::cout << "🎯 Standard Uniform U(0,1) quantiles:" << std::endl;
    std::cout << "   25th percentile: " << standard.getQuantile(0.25) << " [Expected: 0.2500]"
              << std::endl;
    std::cout << "   50th percentile (median): " << standard.getQuantile(0.5)
              << " [Expected: 0.5000]" << std::endl;
    std::cout << "   75th percentile: " << standard.getQuantile(0.75) << " [Expected: 0.7500]"
              << std::endl;
    std::cout << "   95th percentile: " << standard.getQuantile(0.95) << " [Expected: 0.9500]"
              << std::endl;

    std::cout << "\n📊 Custom Uniform U(-5,10) quantiles:" << std::endl;
    std::cout << "   25th percentile: " << custom.getQuantile(0.25)
              << " [Expected: " << (-5.0 + 0.25 * (10.0 - (-5.0))) << "]" << std::endl;
    std::cout << "   50th percentile (median): " << custom.getQuantile(0.5) << " [Expected: 2.5000]"
              << std::endl;
    std::cout << "   75th percentile: " << custom.getQuantile(0.75)
              << " [Expected: " << (-5.0 + 0.75 * (10.0 - (-5.0))) << "]" << std::endl;

    print_separator("6. Random Sampling");
    std::cout << "\nGenerating random samples from distributions:\n"
              << "(Using fixed seed 42 for reproducible results)\n"
              << std::endl;

    std::cout << "🎲 10 samples from Standard Uniform U(0,1):";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << std::setprecision(3) << standard.sample(rng);
    }
    std::cout << std::endl;

    std::cout << "🎲 10 samples from Custom Uniform U(-5,10):";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << std::setprecision(2) << custom.sample(rng);
    }
    std::cout << std::endl;

    print_separator("7. Parameter Estimation");
    std::cout << "\nGenerating large samples and fitting new distributions:\n"
              << "(Demonstrates parameter recovery from sample data)\n"
              << std::endl;

    auto samples = custom.sample(rng, 5000);
    std::cout << "📦 Generated 5000 samples from Uniform U(-5,10)" << std::endl;

    // Fit a new distribution to the samples
    stats::Uniform fitted_uniform;
    fitted_uniform.fit(samples);

    std::cout << "\n🔍 Parameter estimation results:" << std::endl;
    std::cout << "     Original: a=-5.0000, b=10.0000" << std::endl;
    std::cout << "     Estimated a: " << fitted_uniform.getLowerBound()
              << " [Should be close to -5.0000]" << std::endl;
    std::cout << "     Estimated b: " << fitted_uniform.getUpperBound()
              << " [Should be close to 10.0000]" << std::endl;
    std::cout << "     Estimated mean: " << fitted_uniform.getMean()
              << " [Should be close to 2.5000]" << std::endl;

    std::cout << "\n=== Example completed successfully ===" << std::endl;

    return 0;
}

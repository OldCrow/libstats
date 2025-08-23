/**
 * @file basic_usage.cpp
 * @brief Comprehensive basic usage example for libstats
 *
 * This example demonstrates the core functionality of libstats including:
 * - Distribution creation and parameter access
 * - Statistical property calculations
 * - Probability density and cumulative distribution functions
 * - Quantile calculations (inverse CDF)
 * - Random sampling (single and bulk)
 * - Parameter estimation from data
 * - Smart auto-dispatch with performance hints
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

int main() {
    // Initialize performance systems for optimal batch operation performance
    stats::initialize_performance_systems();

    std::cout << "=== libstats Basic Usage Guide ===" << std::endl;
    std::cout << "Comprehensive demonstration of statistical distribution operations\n"
              << std::endl;

    // Create a random number generator
    std::mt19937 rng(42);  // Fixed seed for reproducible results

    print_separator("1. Distribution Creation");
    std::cout << "\nCreating statistical distributions with specific parameters:\n"
              << "  - Gaussian N(μ=0, σ=1): Standard normal distribution\n"
              << "  - Exponential(λ=2.0): Rate parameter controls decay speed\n"
              << std::endl;

    auto normal = stats::GaussianDistribution::create(0.0, 1.0).value;     // Standard normal
    auto exponential = stats::ExponentialDistribution::create(2.0).value;  // Rate parameter = 2.0

    std::cout << "✓ Gaussian N(0,1) created" << std::endl;
    std::cout << "✓ Exponential(λ=2.0) created" << std::endl;

    print_separator("2. Statistical Properties");
    std::cout << "\nComputing theoretical statistical properties for each distribution:\n"
              << std::endl;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "📊 Gaussian N(0,1) properties:" << std::endl;
    std::cout << "     Mean (μ): " << normal.getMean() << " [Expected: 0.0000]" << std::endl;
    std::cout << "     Variance (σ²): " << normal.getVariance() << " [Expected: 1.0000]"
              << std::endl;
    std::cout << "     Standard deviation (σ): " << normal.getStandardDeviation()
              << " [Expected: 1.0000]" << std::endl;
    std::cout << "     Skewness: " << normal.getSkewness() << " [Expected: 0.0000 - symmetric]"
              << std::endl;
    std::cout << "     Kurtosis: " << normal.getKurtosis() << " [Expected: 3.0000 - mesokurtic]"
              << std::endl;

    std::cout << "\n📈 Exponential(λ=2.0) properties:" << std::endl;
    std::cout << "     Mean (1/λ): " << exponential.getMean() << " [Expected: 0.5000]" << std::endl;
    std::cout << "     Variance (1/λ²): " << exponential.getVariance() << " [Expected: 0.2500]"
              << std::endl;
    std::cout << "     Standard deviation: " << stats::getStandardDeviation(exponential)
              << " [Expected: 0.5000]" << std::endl;
    std::cout << "     Skewness: " << exponential.getSkewness()
              << " [Expected: 2.0000 - right-skewed]" << std::endl;
    std::cout << "     Kurtosis: " << exponential.getKurtosis()
              << " [Expected: 9.0000 - leptokurtic]" << std::endl;

    print_separator("3. Probability Density Function (PDF)");
    std::cout << "\nEvaluating probability density at specific points:\n"
              << "(For continuous distributions: relative likelihood at each point)\n"
              << std::endl;

    std::cout << "🎯 Gaussian N(0,1) PDF evaluations:" << std::endl;
    std::cout << "   f(x=0.0): " << normal.getProbability(0.0) << " [Peak density at mean]"
              << std::endl;
    std::cout << "   f(x=1.0): " << normal.getProbability(1.0) << " [1 std dev from mean]"
              << std::endl;

    std::cout << "\n⚡ Exponential(λ=2.0) PDF evaluations:" << std::endl;
    std::cout << "   f(x=0.0): " << exponential.getProbability(0.0) << " [Maximum density at x=0]"
              << std::endl;
    std::cout << "   f(x=0.5): " << exponential.getProbability(0.5) << " [At the mean]"
              << std::endl;

    print_separator("4. Cumulative Distribution Function (CDF)");
    std::cout << "\nComputing cumulative probabilities P(X ≤ x):\n"
              << "(Area under the curve from -∞ to x)\n"
              << std::endl;

    std::cout << "📊 Gaussian N(0,1) CDF evaluations:" << std::endl;
    std::cout << "   P(X≤0.0): " << normal.getCumulativeProbability(0.0)
              << " [Expected: 0.5000 - 50th percentile]" << std::endl;
    std::cout << "   P(X≤1.0): " << normal.getCumulativeProbability(1.0)
              << " [Expected: ~0.8413 - 84th percentile]" << std::endl;

    std::cout << "\n📈 Exponential(λ=2.0) CDF evaluations:" << std::endl;
    std::cout << "   P(X≤0.5): " << exponential.getCumulativeProbability(0.5)
              << " [Expected: ~0.6321]" << std::endl;

    print_separator("5. Quantile Function (Inverse CDF)");
    std::cout << "\nFinding values x such that P(X ≤ x) = p:\n"
              << "(Inverse of the CDF - percentile calculations)\n"
              << std::endl;

    std::cout << "🎯 Gaussian N(0,1) quantiles:" << std::endl;
    std::cout << "   95th percentile: " << normal.getQuantile(0.95) << " [Expected: ~1.6449]"
              << std::endl;
    std::cout << "   50th percentile (median): " << normal.getQuantile(0.5) << " [Expected: 0.0000]"
              << std::endl;

    std::cout << "\n⚡ Exponential(λ=2.0) quantiles:" << std::endl;
    std::cout << "   75th percentile: " << exponential.getQuantile(0.75) << " [Expected: ~0.6932]"
              << std::endl;

    print_separator("6. Random Sampling");
    std::cout << "\nGenerating random samples from distributions:\n"
              << "(Using fixed seed 42 for reproducible results)\n"
              << std::endl;

    std::cout << "🎲 10 samples from Gaussian N(0,1):";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << normal.sample(rng);
    }
    std::cout << std::endl;

    std::cout << "🎲 10 samples from Exponential(λ=2.0):";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::setw(8) << exponential.sample(rng);
    }
    std::cout << std::endl;

    print_separator("7. Bulk Sampling & Parameter Estimation");
    std::cout << "\nGenerating large samples and fitting new distributions:\n"
              << "(Demonstrates efficient bulk operations and parameter recovery)\n"
              << std::endl;

    auto samples = normal.sample(rng, 1000);
    std::cout << "📦 Generated 1000 samples from Gaussian N(0,1)" << std::endl;

    // Fit a new distribution to the samples
    stats::Gaussian fitted_normal;
    fitted_normal.fit(samples);

    std::cout << "\n🔍 Maximum likelihood parameter estimation results:" << std::endl;
    std::cout << "     Original: μ=0.0000, σ=1.0000" << std::endl;
    std::cout << "     Estimated μ: " << fitted_normal.getMean() << " [Should be close to 0.0000]"
              << std::endl;
    std::cout << "     Estimated σ: " << fitted_normal.getStandardDeviation()
              << " [Should be close to 1.0000]" << std::endl;

    print_separator("8. Smart Auto-Dispatch");
    std::cout << "\nDemonstrating adaptive performance optimization:\n"
              << "(libstats automatically chooses optimal execution strategy)\n"
              << std::endl;

    // Create 16 test values spanning the Normal distribution
    std::vector<double> test_values = {-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5,
                                       1.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0, 4.5};
    std::vector<double> results(test_values.size());

    // Use smart dispatch with performance hints and measure timing
    auto hint = stats::performance::PerformanceHint::minimal_latency();
    std::cout << "🚀 Computing batch Normal distribution PDFs with minimal_latency hint..."
              << std::endl;
    std::cout << "   Processing 16 input values: {-3.0 to 4.5 in 0.5 increments}" << std::endl;

    // Time the batch PDF computation
    auto start_time = std::chrono::high_resolution_clock::now();

    normal.getProbability(std::span<const double>(test_values), std::span<double>(results), hint);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);

    std::cout << "\n   📊 Batch PDF results for Gaussian N(0,1):" << std::endl;
    std::cout << "   Input:  ";
    for (size_t i = 0; i < test_values.size(); ++i) {
        std::cout << std::setw(7) << std::setprecision(1) << test_values[i];
    }
    std::cout << std::endl;

    std::cout << "   Output: ";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << std::setw(7) << std::setprecision(4) << results[i];
    }
    std::cout << std::endl;

    std::cout << "\n   ⚡ Performance: Processed 16 PDFs in " << duration.count() << " nanoseconds"
              << " (" << std::setprecision(2) << static_cast<double>(duration.count()) / 16.0
              << " ns per PDF)" << std::endl;
    std::cout
        << "   ✓ libstats selected optimal strategy based on data size and system capabilities"
        << std::endl;

    std::cout << "=== Example completed successfully ===" << std::endl;

    return 0;
}

/**
 * @file poisson_usage_demo.cpp
 * @brief Comprehensive demonstration of Poisson distribution operations
 *
 * The Poisson distribution models the number of events occurring in a fixed interval
 * when events occur at a known constant rate and independently of the time since
 * the last event. Common applications include:
 * - Number of phone calls received per hour
 * - Number of defects in manufacturing per batch
 * - Number of website visits per minute
 * - Number of radioactive decays per second
 *
 * Features demonstrated:
 * - Distribution creation and parameter validation
 * - Statistical properties (mean, variance, skewness, kurtosis)
 * - Probability mass function (PMF) evaluations
 * - Cumulative distribution function (CDF) operations
 * - Quantile function and percentile calculations
 * - Random sampling with reproducible seeds
 * - Parameter estimation from sample data
 * - Batch operations for performance
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats.h"

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void demonstrate_creation_and_properties() {
    print_separator("1. Distribution Creation and Properties");

    std::cout << "\nCreating Poisson distributions with different rate parameters:\n"
              << "  - Poisson(λ=2): Low rate, suitable for rare events\n"
              << "  - Poisson(λ=5): Moderate rate, common in practice\n"
              << "  - Poisson(λ=10): Higher rate, approaching normal approximation\n"
              << std::endl;

    // Create distributions with different parameters
    auto low_rate = libstats::PoissonDistribution::create(2.0).value;       // Low rate
    auto moderate_rate = libstats::PoissonDistribution::create(5.0).value;  // Moderate rate
    auto high_rate = libstats::PoissonDistribution::create(10.0).value;     // Higher rate

    std::cout << "✓ Low rate Poisson(λ=2) created" << std::endl;
    std::cout << "✓ Moderate rate Poisson(λ=5) created" << std::endl;
    std::cout << "✓ High rate Poisson(λ=10) created" << std::endl;

    print_separator("2. Statistical Properties");

    std::cout << "\nComputing theoretical statistical properties:\n"
              << "(For Poisson distributions: mean = variance = λ)\n"
              << std::endl;

    auto display_properties = [](const libstats::Poisson& dist, const std::string& name,
                                 double lambda) {
        std::cout << "📊 " << name << " properties:\n";
        std::cout << "     Rate parameter (λ): " << std::fixed << std::setprecision(4) << lambda
                  << std::endl;
        std::cout << "     Mean: " << std::fixed << std::setprecision(4) << dist.getMean()
                  << " [Expected: " << lambda << "]" << std::endl;
        std::cout << "     Variance: " << std::fixed << std::setprecision(4) << dist.getVariance()
                  << " [Expected: " << lambda << "]" << std::endl;
        std::cout << "     Standard deviation: " << std::fixed << std::setprecision(4)
                  << libstats::getStandardDeviation(dist) << " [Expected: " << std::sqrt(lambda)
                  << "]" << std::endl;
        std::cout << "     Skewness: " << std::fixed << std::setprecision(4) << dist.getSkewness()
                  << " [Expected: " << 1.0 / std::sqrt(lambda) << " - right-skewed]" << std::endl;
        std::cout << "     Kurtosis: " << std::fixed << std::setprecision(4) << dist.getKurtosis()
                  << " [Expected: " << 3.0 + 1.0 / lambda << " - leptokurtic]" << std::endl;
        std::cout << std::endl;
    };

    display_properties(low_rate, "Low Rate Poisson(λ=2)", 2.0);
    display_properties(moderate_rate, "Moderate Rate Poisson(λ=5)", 5.0);
    display_properties(high_rate, "High Rate Poisson(λ=10)", 10.0);
}

void demonstrate_pmf_evaluations() {
    print_separator("3. Probability Mass Function (PMF)");

    std::cout << "\nEvaluating probability mass at specific integer values:\n"
              << "(Poisson PMF: P(X = k) = (λ^k * e^(-λ)) / k!)\n"
              << std::endl;

    auto dist = libstats::PoissonDistribution::create(3.0).value;  // λ = 3 for demonstration

    std::cout << "🎯 Poisson(λ=3) PMF evaluations:\n";
    std::vector<int> test_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    for (int k : test_values) {
        double pmf_value = dist.getProbability(k);
        std::cout << "   P(X=" << k << "): " << std::fixed << std::setprecision(6) << pmf_value;

        if (k == 3) {
            std::cout << " [Mode - most likely value]";
        } else if (k == 0) {
            std::cout << " [P(no events)]";
        } else if (k > 8) {
            std::cout << " [Tail probability - rare events]";
        }
        std::cout << std::endl;
    }

    // Show that probabilities sum to 1 (approximately for finite range)
    double sum_prob = 0.0;
    for (int k = 0; k <= 20; ++k) {
        sum_prob += dist.getProbability(k);
    }
    std::cout << "\n📈 Sum of P(X=k) for k=0 to 20: " << std::fixed << std::setprecision(6)
              << sum_prob << " [Should approach 1.0]" << std::endl;
}

void demonstrate_cdf_operations() {
    print_separator("4. Cumulative Distribution Function (CDF)");

    std::cout << "\nComputing cumulative probabilities P(X ≤ k):\n"
              << "(Useful for finding probability of 'at most k events')\n"
              << std::endl;

    auto dist = libstats::PoissonDistribution::create(4.0).value;  // λ = 4 for demonstration

    std::cout << "📊 Poisson(λ=4) CDF evaluations:\n";
    std::vector<int> cdf_values = {0, 2, 4, 6, 8, 10, 12};

    for (int k : cdf_values) {
        double cdf_value = dist.getCumulativeProbability(k);
        std::cout << "   P(X≤" << k << "): " << std::fixed << std::setprecision(6) << cdf_value;

        if (k == 4) {
            std::cout << " [P(X ≤ mean)]";
        } else if (k == 0) {
            std::cout << " [P(no events)]";
        } else if (cdf_value > 0.95) {
            std::cout << " [High confidence interval]";
        }
        std::cout << std::endl;
    }

    // Demonstrate tail probabilities
    std::cout << "\n🎯 Tail probability examples:\n";
    std::cout << "   P(X > 6) = 1 - P(X≤6) = " << std::fixed << std::setprecision(6)
              << (1.0 - dist.getCumulativeProbability(6)) << std::endl;
    std::cout << "   P(X > 10) = 1 - P(X≤10) = " << std::fixed << std::setprecision(6)
              << (1.0 - dist.getCumulativeProbability(10)) << std::endl;
}

void demonstrate_quantiles() {
    print_separator("5. Quantile Function and Percentiles");

    std::cout << "\nFinding values k such that P(X ≤ k) = p:\n"
              << "(Useful for determining thresholds and confidence intervals)\n"
              << std::endl;

    auto dist = libstats::PoissonDistribution::create(6.0).value;  // λ = 6 for demonstration

    std::cout << "🎯 Poisson(λ=6) quantiles:\n";
    std::vector<double> percentiles = {0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99};

    for (double p : percentiles) {
        int quantile = static_cast<int>(dist.getQuantile(p));
        double actual_p = dist.getCumulativeProbability(quantile);

        std::cout << "   " << std::fixed << std::setprecision(0) << (p * 100)
                  << "th percentile: k=" << quantile << " [P(X≤" << quantile << ")=" << std::fixed
                  << std::setprecision(4) << actual_p << "]";

        if (p == 0.5) {
            std::cout << " - Median";
        } else if (p >= 0.95) {
            std::cout << " - Upper tail";
        }
        std::cout << std::endl;
    }

    std::cout << "\n📈 Confidence intervals:\n";
    int lower_95 = static_cast<int>(dist.getQuantile(0.025));
    int upper_95 = static_cast<int>(dist.getQuantile(0.975));
    std::cout << "   95% confidence interval: [" << lower_95 << ", " << upper_95 << "]"
              << std::endl;

    int lower_99 = static_cast<int>(dist.getQuantile(0.005));
    int upper_99 = static_cast<int>(dist.getQuantile(0.995));
    std::cout << "   99% confidence interval: [" << lower_99 << ", " << upper_99 << "]"
              << std::endl;
}

void demonstrate_sampling() {
    print_separator("6. Random Sampling");

    std::cout << "\nGenerating random samples from Poisson distributions:\n"
              << "(Using fixed seed 42 for reproducible results)\n"
              << std::endl;

    // Use libstats Poisson distributions for sampling
    auto low_rate_dist = libstats::PoissonDistribution::create(1.5).value;
    auto high_rate_dist = libstats::PoissonDistribution::create(8.0).value;

    std::mt19937 rng(42);

    std::cout << "🎲 10 samples from Poisson(λ=1.5): ";
    for (int i = 0; i < 10; ++i) {
        int sample = static_cast<int>(low_rate_dist.sample(rng));
        std::cout << std::setw(3) << sample;
    }
    std::cout << std::endl;

    std::cout << "🎲 10 samples from Poisson(λ=8.0):  ";
    rng.seed(42);  // Reset for comparison
    for (int i = 0; i < 10; ++i) {
        int sample = static_cast<int>(high_rate_dist.sample(rng));
        std::cout << std::setw(3) << sample;
    }
    std::cout << std::endl;
}

void demonstrate_parameter_estimation() {
    print_separator("7. Parameter Estimation");

    std::cout << "\nGenerating large samples and estimating parameters:\n"
              << "(Demonstrates maximum likelihood estimation for λ)\n"
              << std::endl;

    // Generate sample data from known Poisson distribution
    const double true_lambda = 7.5;
    auto true_dist = libstats::PoissonDistribution::create(true_lambda).value;

    std::mt19937 rng(12345);
    auto sample_data = true_dist.sample(rng, 5000);

    std::cout << "📦 Generated 5000 samples from Poisson(λ=" << true_lambda << ")" << std::endl;

    // For Poisson distribution, MLE of λ is simply the sample mean
    double estimated_lambda = std::accumulate(sample_data.begin(), sample_data.end(), 0.0) /
                              static_cast<double>(sample_data.size());

    std::cout << "\n🔍 Parameter estimation results:\n";
    std::cout << "     True λ: " << std::fixed << std::setprecision(4) << true_lambda << std::endl;
    std::cout << "     Estimated λ (sample mean): " << std::fixed << std::setprecision(4)
              << estimated_lambda << " [Should be close to " << true_lambda << "]" << std::endl;

    // Create distribution with estimated parameter
    auto estimated_dist = libstats::PoissonDistribution::create(estimated_lambda).value;

    std::cout << "     Estimated mean: " << std::fixed << std::setprecision(4)
              << estimated_dist.getMean() << " [Should be close to " << true_lambda << "]"
              << std::endl;
    std::cout << "     Estimation error: " << std::fixed << std::setprecision(4)
              << std::abs(estimated_lambda - true_lambda) << std::endl;

    // Sample statistics
    auto minmax = std::minmax_element(sample_data.begin(), sample_data.end());
    std::cout << "\n📊 Sample statistics:\n";
    std::cout << "     Sample size: " << sample_data.size() << std::endl;
    std::cout << "     Sample range: [" << static_cast<int>(*minmax.first) << ", "
              << static_cast<int>(*minmax.second) << "]" << std::endl;

    // Count frequency of values
    std::map<int, int> frequency;
    for (double val : sample_data) {
        frequency[static_cast<int>(val)]++;
    }

    std::cout << "     Most common values:\n";
    std::vector<std::pair<int, int>> freq_pairs(frequency.begin(), frequency.end());
    std::sort(freq_pairs.begin(), freq_pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    for (int i = 0; i < std::min(5, static_cast<int>(freq_pairs.size())); ++i) {
        std::cout << "       " << freq_pairs[static_cast<std::size_t>(i)].first << ": "
                  << freq_pairs[static_cast<std::size_t>(i)].second << " times (" << std::fixed
                  << std::setprecision(1)
                  << (100.0 * static_cast<double>(freq_pairs[static_cast<std::size_t>(i)].second) /
                      static_cast<double>(sample_data.size()))
                  << "%)" << std::endl;
    }
}

void demonstrate_batch_operations() {
    print_separator("8. Batch Operations and Performance");

    std::cout << "\nDemonstrating batch operations for high-performance computing:\n"
              << "(Processing arrays of values efficiently with SIMD optimization)\n"
              << std::endl;

    auto dist = libstats::PoissonDistribution::create(5.0).value;

    // Test batch PMF evaluation
    std::vector<double> input_values = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<double> pmf_results(input_values.size());

    std::cout << "🚀 Batch PMF evaluation for " << input_values.size() << " values:\n";

    // Use batch operation
    dist.getProbability(std::span<const double>(input_values), std::span<double>(pmf_results));

    std::cout << "   Value    PMF\n";
    std::cout << "   -----    -------\n";
    for (size_t i = 0; i < input_values.size(); ++i) {
        std::cout << "   " << std::setw(5) << static_cast<int>(input_values[i]) << "    "
                  << std::fixed << std::setprecision(6) << pmf_results[i] << std::endl;
    }

    // Test batch CDF evaluation
    std::vector<double> cdf_results(input_values.size());
    dist.getCumulativeProbability(std::span<const double>(input_values),
                                  std::span<double>(cdf_results));

    std::cout << "\n🚀 Batch CDF evaluation for the same values:\n";
    std::cout << "   Value    CDF\n";
    std::cout << "   -----    -------\n";
    for (size_t i = 0; i < input_values.size(); ++i) {
        std::cout << "   " << std::setw(5) << static_cast<int>(input_values[i]) << "    "
                  << std::fixed << std::setprecision(6) << cdf_results[i] << std::endl;
    }

    std::cout << "\n💡 Batch operations provide significant performance benefits for:\n"
              << "   - Large-scale statistical analysis\n"
              << "   - Monte Carlo simulations\n"
              << "   - Real-time event processing\n"
              << "   - Machine learning applications\n";
}

int main() {
    std::cout << "=== libstats Poisson Distribution Usage Demo ===" << std::endl;
    std::cout << "Comprehensive demonstration of Poisson distribution operations\n" << std::endl;

    try {
        demonstrate_creation_and_properties();
        demonstrate_pmf_evaluations();
        demonstrate_cdf_operations();
        demonstrate_quantiles();
        demonstrate_sampling();
        demonstrate_parameter_estimation();
        demonstrate_batch_operations();

        print_separator("Summary");
        std::cout << "✅ Distribution creation and parameter validation" << std::endl;
        std::cout << "✅ Statistical properties computation" << std::endl;
        std::cout << "✅ PMF evaluations for discrete probabilities" << std::endl;
        std::cout << "✅ CDF operations for cumulative probabilities" << std::endl;
        std::cout << "✅ Quantile calculations and confidence intervals" << std::endl;
        std::cout << "✅ Random sampling with reproducible results" << std::endl;
        std::cout << "✅ Maximum likelihood parameter estimation" << std::endl;
        std::cout << "✅ High-performance batch operations" << std::endl;
        std::cout << "\nThe Poisson distribution is ready for statistical analysis!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

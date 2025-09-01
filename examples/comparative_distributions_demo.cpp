/**
 * @file comparative_distributions_demo.cpp
 * @brief Comparative analysis of multiple libstats distributions
 *
 * This demo showcases different probability distributions side-by-side to help
 * users understand their characteristics, use cases, and how they compare in
 * practical scenarios. Features demonstrated:
 *
 * - Statistical property comparisons across distributions
 * - Probability function evaluations at common points
 * - Random sampling and empirical vs theoretical comparisons
 * - Real-world modeling scenarios and distribution selection
 * - Performance benchmarking across distribution types
 */

#define LIBSTATS_FULL_INTERFACE
#include "../include/libstats.h"

// Standard library includes
#include <algorithm>  // for std::minmax_element, std::min_element, std::accumulate
#include <chrono>     // for timing operations
#include <cmath>      // for std::sqrt
#include <exception>  // for std::exception
#include <iomanip>    // for std::setw, std::setprecision, std::fixed
#include <iostream>   // for std::cout, std::cerr
#include <numeric>    // for std::accumulate
#include <random>     // for std::mt19937
#include <string>     // for std::string
#include <tuple>      // for std::make_tuple, structured bindings
#include <utility>    // for std::pair
#include <vector>     // for std::vector

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_subsection(const std::string& title) {
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

void compare_statistical_properties() {
    print_separator("1. Statistical Properties Comparison");

    std::cout << "\nComparing fundamental statistical properties across distributions:\n"
              << std::endl;

    // Create comparable distributions (similar means where possible)
    auto uniform = stats::UniformDistribution::create(0.0, 10.0).value;  // Mean ≈ 5.0
    auto poisson = stats::PoissonDistribution::create(5.0).value;        // Mean = 5.0
    auto exponential =
        stats::ExponentialDistribution::create(0.2).value;  // Mean = 5.0 (rate = 1/mean)
    auto gaussian = stats::GaussianDistribution::create(5.0, 2.0).value;  // Mean = 5.0, σ = 2.0

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Distribution    | Mean    | Variance | Std Dev  | Skewness | Kurtosis | Support"
              << std::endl;
    std::cout
        << "----------------|---------|----------|----------|----------|----------|----------------"
        << std::endl;

    std::cout << "Uniform(0,10)   | " << std::setw(7) << uniform.getMean() << " | " << std::setw(8)
              << uniform.getVariance() << " | " << std::setw(8)
              << stats::getStandardDeviation(uniform) << " | " << std::setw(8)
              << uniform.getSkewness() << " | " << std::setw(8) << uniform.getKurtosis() << " | "
              << "[0, 10]" << std::endl;

    std::cout << "Poisson(λ=5)    | " << std::setw(7) << poisson.getMean() << " | " << std::setw(8)
              << poisson.getVariance() << " | " << std::setw(8)
              << stats::getStandardDeviation(poisson) << " | " << std::setw(8)
              << poisson.getSkewness() << " | " << std::setw(8) << poisson.getKurtosis() << " | "
              << "[0, ∞)" << std::endl;

    std::cout << "Exponential(0.2)| " << std::setw(7) << exponential.getMean() << " | "
              << std::setw(8) << exponential.getVariance() << " | " << std::setw(8)
              << stats::getStandardDeviation(exponential) << " | " << std::setw(8)
              << exponential.getSkewness() << " | " << std::setw(8) << exponential.getKurtosis()
              << " | "
              << "[0, ∞)" << std::endl;

    std::cout << "Gaussian(5,2)   | " << std::setw(7) << gaussian.getMean() << " | " << std::setw(8)
              << gaussian.getVariance() << " | " << std::setw(8)
              << stats::getStandardDeviation(gaussian) << " | " << std::setw(8)
              << gaussian.getSkewness() << " | " << std::setw(8) << gaussian.getKurtosis() << " | "
              << "(-∞, ∞)" << std::endl;

    std::cout << "\n📊 Key observations:" << std::endl;
    std::cout << "   • Uniform: Zero skewness (symmetric), platykurtic (negative excess kurtosis)"
              << std::endl;
    std::cout << "   • Poisson: Right-skewed, leptokurtic (positive excess kurtosis)" << std::endl;
    std::cout << "   • Exponential: Highly right-skewed, very leptokurtic" << std::endl;
    std::cout << "   • Gaussian: Zero skewness (symmetric), mesokurtic (zero excess kurtosis)"
              << std::endl;
}

void compare_probability_functions() {
    print_separator("2. Probability Function Comparisons");

    // Continuous distributions
    print_subsection("Continuous Distributions - PDF at x = 3.0");

    auto uniform = stats::UniformDistribution::create(0.0, 10.0).value;
    auto exponential = stats::ExponentialDistribution::create(0.2).value;
    auto gaussian = stats::GaussianDistribution::create(5.0, 2.0).value;

    double x = 3.0;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nEvaluating PDF at x = " << x << ":" << std::endl;
    std::cout << "  Uniform(0,10):      f(" << x << ") = " << uniform.getProbability(x)
              << std::endl;
    std::cout << "  Exponential(0.2):   f(" << x << ") = " << exponential.getProbability(x)
              << std::endl;
    std::cout << "  Gaussian(5,2):      f(" << x << ") = " << gaussian.getProbability(x)
              << std::endl;

    // Discrete distribution
    print_subsection("Discrete Distribution - PMF at k = 3");

    auto poisson = stats::PoissonDistribution::create(5.0).value;
    int k = 3;
    std::cout << "\nEvaluating PMF at k = " << k << ":" << std::endl;
    std::cout << "  Poisson(λ=5):       P(X=" << k << ") = " << poisson.getProbability(k)
              << std::endl;

    // CDF comparisons
    print_subsection("Cumulative Distribution Functions at x/k = 5");

    double cdf_point = 5.0;
    std::cout << "\nCDF evaluations P(X ≤ 5):" << std::endl;
    std::cout << "  Uniform(0,10):      " << std::setprecision(4)
              << uniform.getCumulativeProbability(cdf_point) << std::endl;
    std::cout << "  Exponential(0.2):   " << exponential.getCumulativeProbability(cdf_point)
              << std::endl;
    std::cout << "  Gaussian(5,2):      " << gaussian.getCumulativeProbability(cdf_point)
              << std::endl;
    std::cout << "  Poisson(λ=5):       " << poisson.getCumulativeProbability(5) << std::endl;
}

void compare_sampling_behavior() {
    print_separator("3. Random Sampling and Empirical Analysis");

    std::cout << "\nGenerating 1000 samples from each distribution and comparing:" << std::endl;

    std::mt19937 rng(12345);  // Fixed seed for reproducibility

    // Create distributions
    auto uniform = stats::UniformDistribution::create(0.0, 10.0).value;
    auto exponential = stats::ExponentialDistribution::create(0.2).value;
    auto gaussian = stats::GaussianDistribution::create(5.0, 2.0).value;
    auto poisson = stats::PoissonDistribution::create(5.0).value;

    const int n_samples = 1000;

    // Generate samples
    auto uniform_samples = uniform.sample(rng, n_samples);
    rng.seed(12345);  // Reset for fair comparison
    auto exp_samples = exponential.sample(rng, n_samples);
    rng.seed(12345);
    auto gaussian_samples = gaussian.sample(rng, n_samples);
    rng.seed(12345);
    auto poisson_samples = poisson.sample(rng, n_samples);

    // Calculate empirical statistics
    auto calc_stats = [](const auto& samples) {
        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                      static_cast<double>(samples.size());
        double variance = 0.0;
        for (auto val : samples) {
            variance += (val - mean) * (val - mean);
        }
        variance /= static_cast<double>(samples.size() - 1);
        auto [min_it, max_it] = std::minmax_element(samples.begin(), samples.end());
        return std::make_tuple(mean, variance, std::sqrt(variance), *min_it, *max_it);
    };

    auto [u_mean, u_var, u_std, u_min, u_max] = calc_stats(uniform_samples);
    auto [e_mean, e_var, e_std, e_min, e_max] = calc_stats(exp_samples);
    auto [g_mean, g_var, g_std, g_min, g_max] = calc_stats(gaussian_samples);
    auto [p_mean, p_var, p_std, p_min, p_max] = calc_stats(poisson_samples);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nEmpirical Statistics from " << n_samples << " samples:" << std::endl;
    std::cout << "Distribution    | Sample Mean | Sample Std | Sample Range" << std::endl;
    std::cout << "----------------|-------------|------------|----------------" << std::endl;
    std::cout << "Uniform(0,10)   | " << std::setw(11) << u_mean << " | " << std::setw(10) << u_std
              << " | [" << std::setprecision(1) << u_min << ", " << u_max << "]" << std::endl;
    std::cout << "Exponential(0.2)| " << std::setw(11) << std::setprecision(3) << e_mean << " | "
              << std::setw(10) << e_std << " | [" << std::setprecision(1) << e_min << ", " << e_max
              << "]" << std::endl;
    std::cout << "Gaussian(5,2)   | " << std::setw(11) << std::setprecision(3) << g_mean << " | "
              << std::setw(10) << g_std << " | [" << std::setprecision(1) << g_min << ", " << g_max
              << "]" << std::endl;
    std::cout << "Poisson(λ=5)    | " << std::setw(11) << std::setprecision(3) << p_mean << " | "
              << std::setw(10) << p_std << " | [" << static_cast<int>(p_min) << ", "
              << static_cast<int>(p_max) << "]" << std::endl;

    std::cout << "\n📈 Sample behavior insights:" << std::endl;
    std::cout << "   • Uniform: Bounded samples, consistent spread" << std::endl;
    std::cout << "   • Exponential: Unbounded, occasional large values (heavy right tail)"
              << std::endl;
    std::cout << "   • Gaussian: Unbounded but concentrated around mean" << std::endl;
    std::cout << "   • Poisson: Discrete values, bounded below by 0" << std::endl;
}

void demonstrate_real_world_scenarios() {
    print_separator("4. Real-World Modeling Scenarios");

    print_subsection("Scenario 1: Customer Service Response Times");
    std::cout << "\nModeling response times with different assumptions:" << std::endl;

    // Exponential: memoryless service times
    auto service_exp = stats::ExponentialDistribution::create(0.1).value;  // Mean = 10 minutes
    std::cout << "📞 Exponential model (memoryless service):" << std::endl;
    std::cout << "   Mean response time: " << std::fixed << std::setprecision(1)
              << service_exp.getMean() << " minutes" << std::endl;
    std::cout << "   P(response ≤ 5 min): " << std::setprecision(3)
              << service_exp.getCumulativeProbability(5.0) << std::endl;
    std::cout << "   P(response > 20 min): " << (1.0 - service_exp.getCumulativeProbability(20.0))
              << std::endl;

    // Gaussian: normally distributed service times
    auto service_normal =
        stats::GaussianDistribution::create(10.0, 3.0).value;  // Mean = 10, σ = 3 minutes
    std::cout << "\n📊 Gaussian model (normally distributed service):" << std::endl;
    std::cout << "   Mean response time: " << std::setprecision(1) << service_normal.getMean()
              << " minutes" << std::endl;
    std::cout << "   P(response ≤ 5 min): " << std::setprecision(3)
              << service_normal.getCumulativeProbability(5.0) << std::endl;
    std::cout << "   P(response > 20 min): "
              << (1.0 - service_normal.getCumulativeProbability(20.0)) << std::endl;

    print_subsection("Scenario 2: Daily Event Counting");
    std::cout << "\nComparing discrete models for daily event counts:" << std::endl;

    // Poisson: independent events at constant rate
    auto daily_events = stats::PoissonDistribution::create(8.0).value;  // Average 8 events/day
    std::cout << "🎯 Poisson model (constant rate, independent events):" << std::endl;
    std::cout << "   Average events/day: " << std::setprecision(1) << daily_events.getMean()
              << std::endl;
    std::cout << "   P(no events): " << std::setprecision(4) << daily_events.getProbability(0)
              << std::endl;
    std::cout << "   P(≥ 12 events): " << (1.0 - daily_events.getCumulativeProbability(11))
              << std::endl;

    print_subsection("Scenario 3: Quality Control Measurements");
    std::cout << "\nModeling measurement variations in manufacturing:" << std::endl;

    // Uniform: measurements within tolerance bounds
    auto tolerance = stats::UniformDistribution::create(49.8, 50.2).value;  // Target: 50.0 ± 0.2
    std::cout << "📏 Uniform model (within tolerance bounds):" << std::endl;
    std::cout << "   Target specification: 50.0 ± 0.2 units" << std::endl;
    std::cout << "   P(within spec): " << std::setprecision(4) << 1.0 << " (by design)"
              << std::endl;
    std::cout << "   Standard deviation: " << tolerance.getVariance() << std::endl;

    // Gaussian: process with natural variation
    auto process_normal =
        stats::GaussianDistribution::create(50.0, 0.1).value;  // Mean = 50.0, σ = 0.1
    std::cout << "\n🎯 Gaussian model (natural process variation):" << std::endl;
    std::cout << "   Process mean: " << process_normal.getMean() << std::endl;
    std::cout << "   P(within spec): " << std::setprecision(4)
              << (process_normal.getCumulativeProbability(50.2) -
                  process_normal.getCumulativeProbability(49.8))
              << std::endl;
    std::cout << "   Standard deviation: " << stats::getStandardDeviation(process_normal)
              << std::endl;
}

void compare_performance() {
    print_separator("5. Performance Benchmarking");

    std::cout << "\nBenchmarking computation performance across distributions:" << std::endl;

    const int n_operations = 10000;
    std::mt19937 rng(42);

    // Create distributions
    auto uniform = stats::UniformDistribution::create(0.0, 10.0).value;
    auto exponential = stats::ExponentialDistribution::create(0.2).value;
    auto gaussian = stats::GaussianDistribution::create(5.0, 2.0).value;
    auto poisson = stats::PoissonDistribution::create(5.0).value;

    auto benchmark_sampling = [&](auto& dist, const std::string& name) {
        rng.seed(42);  // Reset for fair comparison
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_operations; ++i) {
            [[maybe_unused]] volatile auto sample =
                dist.sample(rng);  // volatile prevents optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "  " << std::left << std::setw(15) << name << ": " << std::right
                  << std::setw(6) << duration.count() << " μs (" << std::setprecision(2)
                  << std::fixed
                  << (static_cast<double>(duration.count()) / static_cast<double>(n_operations))
                  << " μs/sample)" << std::endl;
        return duration.count();
    };

    std::cout << "\nSampling Performance (" << n_operations << " samples):" << std::endl;
    auto uniform_time = benchmark_sampling(uniform, "Uniform");
    auto exp_time = benchmark_sampling(exponential, "Exponential");
    auto gaussian_time = benchmark_sampling(gaussian, "Gaussian");
    auto poisson_time = benchmark_sampling(poisson, "Poisson");

    // Find fastest
    std::vector<std::pair<std::string, long>> times = {
        {"Uniform", static_cast<long>(uniform_time)},
        {"Exponential", static_cast<long>(exp_time)},
        {"Gaussian", static_cast<long>(gaussian_time)},
        {"Poisson", static_cast<long>(poisson_time)}};

    auto fastest = *std::min_element(times.begin(), times.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });

    std::cout << "\n⚡ Fastest sampling: " << fastest.first << " (" << fastest.second
              << " μs total)" << std::endl;

    std::cout << "\n💡 Performance insights:" << std::endl;
    std::cout << "   • Uniform: Typically fastest (simple transformation)" << std::endl;
    std::cout << "   • Exponential: Fast (inverse transform method)" << std::endl;
    std::cout << "   • Gaussian: Moderate (Box-Muller or similar algorithms)" << std::endl;
    std::cout << "   • Poisson: Variable (depends on λ, uses different algorithms)" << std::endl;
}

int main() {
    std::cout << "=== libstats Comparative Distributions Demo ===" << std::endl;
    std::cout << "Side-by-side analysis of multiple probability distributions" << std::endl;

    try {
        compare_statistical_properties();
        compare_probability_functions();
        compare_sampling_behavior();
        demonstrate_real_world_scenarios();
        compare_performance();

        print_separator("Summary");
        std::cout << "✅ Statistical properties compared across distribution types" << std::endl;
        std::cout << "✅ Probability functions evaluated at common points" << std::endl;
        std::cout << "✅ Empirical sampling behavior analyzed and compared" << std::endl;
        std::cout << "✅ Real-world modeling scenarios demonstrated" << std::endl;
        std::cout << "✅ Performance characteristics benchmarked" << std::endl;
        std::cout << "\n🎯 This comparison helps in selecting the right distribution for your data!"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during comparative analysis: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

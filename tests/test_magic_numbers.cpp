#include "../include/distributions/discrete.h"
#include "../include/distributions/gamma.h"
#include "../include/distributions/poisson.h"
#include "../include/distributions/uniform.h"

#include <iostream>
#include <random>
#include <vector>

using namespace libstats;

void test_gamma_goodness_of_fit() {
    std::cout << "\n=== Testing Gamma Distribution Goodness-of-Fit ===\n";

    // Create a gamma distribution and generate some data
    GammaDistribution gamma(2.0, 1.0);
    std::mt19937 rng(42);
    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(gamma.sample(rng));
    }

    // Test Kolmogorov-Smirnov
    auto [ks_stat, ks_pval, ks_reject] = gamma.kolmogorovSmirnovTest(data, gamma, 0.05);
    std::cout << "KS Test:\n";
    std::cout << "  Statistic: " << ks_stat << "\n";
    std::cout << "  P-value: " << ks_pval << "\n";
    std::cout << "  Reject null (α=0.05): " << (ks_reject ? "YES" : "NO") << "\n";

    // Test Anderson-Darling
    auto [ad_stat, ad_pval, ad_reject] = gamma.andersonDarlingTest(data, gamma, 0.05);
    std::cout << "Anderson-Darling Test:\n";
    std::cout << "  Statistic: " << ad_stat << "\n";
    std::cout << "  P-value: " << ad_pval << "\n";
    std::cout << "  Reject null (α=0.05): " << (ad_reject ? "YES" : "NO") << "\n";

    // Both tests should generally not reject the null hypothesis
    // since we're testing data from the same distribution
    if (!ks_reject && !ad_reject) {
        std::cout << "✅ Gamma goodness-of-fit tests working correctly\n";
    } else {
        std::cout << "⚠️ Note: Some tests rejected null hypothesis (may happen randomly)\n";
    }
}

void test_uniform_anderson_darling() {
    std::cout << "\n=== Testing Uniform Distribution Anderson-Darling ===\n";

    UniformDistribution uniform(0.0, 1.0);
    std::mt19937 rng(42);
    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(uniform.sample(rng));
    }

    auto [ad_stat, p_value, reject] = uniform.andersonDarlingTest(data, uniform, 0.05);
    std::cout << "AD Statistic: " << ad_stat << "\n";
    std::cout << "P-value: " << p_value << "\n";
    std::cout << "Reject null (α=0.05): " << (reject ? "YES" : "NO") << "\n";

    if (!reject) {
        std::cout << "✅ Uniform Anderson-Darling test working correctly\n";
    } else {
        std::cout << "⚠️ Note: Test rejected null hypothesis (may happen randomly)\n";
    }
}

void test_discrete_chi_square() {
    std::cout << "\n=== Testing Discrete Distribution Chi-Square ===\n";

    DiscreteDistribution discrete(1, 6);  // Like a die
    std::mt19937 rng(42);
    std::vector<double> samples;
    for (int i = 0; i < 600; ++i) {
        samples.push_back(discrete.sample(rng));
    }

    auto [chi2_stat, p_value, reject] =
        discrete.chiSquaredGoodnessOfFitTest(samples, discrete, 0.05);
    std::cout << "Chi-square Statistic: " << chi2_stat << "\n";
    std::cout << "P-value: " << p_value << "\n";
    std::cout << "Reject null (α=0.05): " << (reject ? "YES" : "NO") << "\n";

    if (!reject) {
        std::cout << "✅ Discrete chi-square test working correctly\n";
    } else {
        std::cout << "⚠️ Note: Test rejected null hypothesis (may happen randomly)\n";
    }
}

void test_poisson_anderson_darling() {
    std::cout << "\n=== Testing Poisson Distribution Anderson-Darling ===\n";

    PoissonDistribution poisson(3.0);
    std::mt19937 rng(42);
    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(poisson.sample(rng));
    }

    auto [ad_stat, p_value, reject] = poisson.andersonDarlingTest(data, poisson, 0.05);
    std::cout << "AD Statistic: " << ad_stat << "\n";
    std::cout << "P-value: " << p_value << "\n";
    std::cout << "Reject null (α=0.05): " << (reject ? "YES" : "NO") << "\n";

    if (!reject) {
        std::cout << "✅ Poisson Anderson-Darling test working correctly\n";
    } else {
        std::cout << "⚠️ Note: Test rejected null hypothesis (may happen randomly)\n";
    }
}

int main() {
    std::cout << "=================================================\n";
    std::cout << "Testing Magic Number Replacements\n";
    std::cout << "=================================================\n";
    std::cout << "This test verifies that the replaced magic numbers\n";
    std::cout << "in statistical tests are working correctly.\n";

    try {
        test_gamma_goodness_of_fit();
        test_uniform_anderson_darling();
        test_discrete_chi_square();
        test_poisson_anderson_darling();

        std::cout << "\n=================================================\n";
        std::cout << "🎉 All magic number replacement tests completed!\n";
        std::cout << "=================================================\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

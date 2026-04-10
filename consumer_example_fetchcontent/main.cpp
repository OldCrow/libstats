/// @file main.cpp
/// @brief Minimal consumer example for libstats via find_package.
///
/// Verifies that an installed libstats can be found, linked, and used
/// by an external project.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <iostream>

int main() {
    // Create a standard normal distribution using the safe factory API
    auto result = stats::GaussianDistribution::create(0.0, 1.0);
    if (result.isError()) {
        std::cerr << "Failed to create distribution: " << result.message << "\n";
        return 1;
    }

    auto& gaussian = result.value;

    // Evaluate PDF and CDF at a few points
    std::cout << "libstats consumer example\n";
    std::cout << "========================\n";
    std::cout << "Distribution: Gaussian(mu=0, sigma=1)\n\n";

    const double points[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    for (double x : points) {
        double pdf = gaussian.getProbability(x);
        double cdf = gaussian.getCumulativeProbability(x);
        std::cout << "  x=" << x << "  PDF=" << pdf << "  CDF=" << cdf << "\n";
    }

    // Verify a known value: PDF(0) for N(0,1) = 1/sqrt(2*pi) ≈ 0.3989
    double pdf_at_zero = gaussian.getProbability(0.0);
    double expected = 1.0 / std::sqrt(2.0 * M_PI);
    if (std::abs(pdf_at_zero - expected) > 1e-10) {
        std::cerr << "Verification failed: PDF(0) = " << pdf_at_zero << ", expected " << expected
                  << "\n";
        return 1;
    }

    std::cout << "\nVerification passed.\n";
    return 0;
}

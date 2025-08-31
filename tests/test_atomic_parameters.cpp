/**
 * @file test_atomic_parameters.cpp
 * @brief Test atomic parameter management for lock-free access in libstats distributions
 *
 * This test validates that the atomic parameter access methods work correctly,
 * provide proper performance characteristics for lock-free parameter access,
 * and properly handle atomic parameter invalidation when parameters change.
 *
 * @author libstats Development Team
 * @version 1.0.0
 * @since 1.0.0
 */
// Use focused headers for atomic parameters testing
#include "../include/distributions/discrete.h"
#include "../include/distributions/exponential.h"
#include "../include/distributions/gaussian.h"

// Standard library includes
#include <cassert>   // for assert
#include <cmath>     // for std::abs
#include <iostream>  // for std::cout, std::endl
#include <utility>   // for std::move

using namespace stats;

/**
 * @brief Test basic atomic getter functionality for Exponential distribution
 */
void test_exponential_atomic_getter() {
    std::cout << "Testing Exponential distribution atomic getter..." << std::endl;

    // Create exponential distribution
    auto result = ExponentialDistribution::create(2.5);
    assert(result.isOk());
    auto exp_dist = std::move(result.value);

    // Test that atomic getter returns same value as regular getter
    double lambda_regular = exp_dist.getLambda();
    double lambda_atomic = exp_dist.getLambdaAtomic();

    assert(std::abs(lambda_regular - lambda_atomic) < 1e-15);
    assert(std::abs(lambda_atomic - 2.5) < 1e-15);

    std::cout << "  ✓ Regular getter: " << lambda_regular << std::endl;
    std::cout << "  ✓ Atomic getter: " << lambda_atomic << std::endl;
    std::cout << "  ✓ Values match perfectly" << std::endl;
}

/**
 * @brief Test atomic getter consistency under parameter changes
 */
void test_atomic_getter_consistency() {
    std::cout << "Testing atomic getter consistency under parameter changes..." << std::endl;

    auto result = ExponentialDistribution::create(1.0);
    assert(result.isOk());
    auto exp_dist = std::move(result.value);

    // Initial check
    assert(std::abs(exp_dist.getLambdaAtomic() - 1.0) < 1e-15);

    // Change parameter and verify atomic getter updates
    exp_dist.setLambda(3.5);

    // Both getters should return the same updated value
    [[maybe_unused]] double lambda_regular = exp_dist.getLambda();
    double lambda_atomic = exp_dist.getLambdaAtomic();

    assert(std::abs(lambda_regular - lambda_atomic) < 1e-15);
    assert(std::abs(lambda_atomic - 3.5) < 1e-15);

    std::cout << "  ✓ Parameter change handled correctly" << std::endl;
    std::cout << "  ✓ Updated atomic getter: " << lambda_atomic << std::endl;
}

/**
 * @brief Test Gaussian distribution atomic parameter invalidation
 */
void test_gaussian_atomic_invalidation() {
    std::cout << "Testing Gaussian atomic parameter invalidation..." << std::endl;

    auto result = GaussianDistribution::create(1.0, 1.0);
    assert(result.isOk());
    auto gauss_dist = std::move(result.value);

    // Verify initial values
    assert(std::abs(gauss_dist.getMeanAtomic() - 1.0) < 1e-15);
    assert(std::abs(gauss_dist.getStandardDeviationAtomic() - 1.0) < 1e-15);
    std::cout << "  ✓ Initial atomic values correct" << std::endl;

    // Test setMean invalidation
    gauss_dist.setMean(5.0);
    assert(std::abs(gauss_dist.getMean() - 5.0) < 1e-15);
    assert(std::abs(gauss_dist.getMeanAtomic() - 5.0) < 1e-15);
    assert(std::abs(gauss_dist.getStandardDeviationAtomic() - 1.0) < 1e-15);
    std::cout << "  ✓ setMean() properly invalidates and updates atomic parameters" << std::endl;

    // Test setStandardDeviation invalidation
    gauss_dist.setStandardDeviation(2.0);
    assert(std::abs(gauss_dist.getStandardDeviation() - 2.0) < 1e-15);
    assert(std::abs(gauss_dist.getMeanAtomic() - 5.0) < 1e-15);
    assert(std::abs(gauss_dist.getStandardDeviationAtomic() - 2.0) < 1e-15);
    std::cout << "  ✓ setStandardDeviation() properly invalidates and updates atomic parameters"
              << std::endl;

    // Test setParameters invalidation
    gauss_dist.setParameters(10.0, 3.0);
    assert(std::abs(gauss_dist.getMean() - 10.0) < 1e-15);
    assert(std::abs(gauss_dist.getStandardDeviation() - 3.0) < 1e-15);
    assert(std::abs(gauss_dist.getMeanAtomic() - 10.0) < 1e-15);
    assert(std::abs(gauss_dist.getStandardDeviationAtomic() - 3.0) < 1e-15);
    std::cout << "  ✓ setParameters() properly invalidates and updates atomic parameters"
              << std::endl;
}

/**
 * @brief Test Exponential distribution atomic parameter invalidation
 */
void test_exponential_atomic_invalidation() {
    std::cout << "Testing Exponential atomic parameter invalidation..." << std::endl;

    auto result = ExponentialDistribution::create(1.0);
    assert(result.isOk());
    auto exp_dist = std::move(result.value);

    // Verify initial value
    assert(std::abs(exp_dist.getLambdaAtomic() - 1.0) < 1e-15);
    std::cout << "  ✓ Initial atomic value correct" << std::endl;

    // Test setLambda invalidation
    exp_dist.setLambda(2.5);
    assert(std::abs(exp_dist.getLambda() - 2.5) < 1e-15);
    assert(std::abs(exp_dist.getLambdaAtomic() - 2.5) < 1e-15);
    std::cout << "  ✓ setLambda() properly invalidates and updates atomic parameters" << std::endl;

    // Test trySetParameters invalidation
    auto try_result = exp_dist.trySetParameters(4.0);
    assert(try_result.isOk());
    assert(std::abs(exp_dist.getLambda() - 4.0) < 1e-15);
    assert(std::abs(exp_dist.getLambdaAtomic() - 4.0) < 1e-15);
    std::cout << "  ✓ trySetParameters() properly invalidates and updates atomic parameters"
              << std::endl;
}

/**
 * @brief Test Discrete distribution atomic parameter invalidation
 */
void test_discrete_atomic_invalidation() {
    std::cout << "Testing Discrete atomic parameter invalidation..." << std::endl;

    auto result = DiscreteDistribution::create(1, 5);
    assert(result.isOk());
    auto discrete_dist = std::move(result.value);

    // Verify initial values
    assert(discrete_dist.getLowerBoundAtomic() == 1);
    assert(discrete_dist.getUpperBoundAtomic() == 5);
    std::cout << "  ✓ Initial atomic values correct" << std::endl;

    // Test setLowerBound invalidation
    discrete_dist.setLowerBound(0);
    assert(discrete_dist.getLowerBound() == 0);
    assert(discrete_dist.getLowerBoundAtomic() == 0);
    assert(discrete_dist.getUpperBoundAtomic() == 5);
    std::cout << "  ✓ setLowerBound() properly invalidates and updates atomic parameters"
              << std::endl;

    // Test setUpperBound invalidation
    discrete_dist.setUpperBound(10);
    assert(discrete_dist.getUpperBound() == 10);
    assert(discrete_dist.getLowerBoundAtomic() == 0);
    assert(discrete_dist.getUpperBoundAtomic() == 10);
    std::cout << "  ✓ setUpperBound() properly invalidates and updates atomic parameters"
              << std::endl;

    // Test setBounds invalidation
    discrete_dist.setBounds(2, 8);
    assert(discrete_dist.getLowerBound() == 2);
    assert(discrete_dist.getUpperBound() == 8);
    assert(discrete_dist.getLowerBoundAtomic() == 2);
    assert(discrete_dist.getUpperBoundAtomic() == 8);
    std::cout << "  ✓ setBounds() properly invalidates and updates atomic parameters" << std::endl;

    // Test trySetParameters invalidation
    auto try_result = discrete_dist.trySetParameters(1, 6);
    assert(try_result.isOk());
    assert(discrete_dist.getLowerBound() == 1);
    assert(discrete_dist.getUpperBound() == 6);
    assert(discrete_dist.getLowerBoundAtomic() == 1);
    assert(discrete_dist.getUpperBoundAtomic() == 6);
    std::cout << "  ✓ trySetParameters() properly invalidates and updates atomic parameters"
              << std::endl;
}

/**
 * @brief Test Gaussian distribution atomic getters if available
 */
void test_gaussian_atomic_getters() {
    std::cout << "Testing Gaussian distribution atomic getters..." << std::endl;

    auto result = GaussianDistribution::create(5.0, 2.0);
    assert(result.isOk());
    auto gauss_dist = std::move(result.value);

    // Test atomic getters with correct method names
    double mean_regular = gauss_dist.getMean();
    double stddev_regular = gauss_dist.getStandardDeviation();

    // Check atomic getters
    double mean_atomic = gauss_dist.getMeanAtomic();
    double stddev_atomic = gauss_dist.getStandardDeviationAtomic();

    assert(std::abs(mean_regular - mean_atomic) < 1e-15);
    assert(std::abs(stddev_regular - stddev_atomic) < 1e-15);
    assert(std::abs(mean_atomic - 5.0) < 1e-15);
    assert(std::abs(stddev_atomic - 2.0) < 1e-15);

    std::cout << "  ✓ Mean - Regular: " << mean_regular << ", Atomic: " << mean_atomic << std::endl;
    std::cout << "  ✓ StdDev - Regular: " << stddev_regular << ", Atomic: " << stddev_atomic
              << std::endl;
}

/**
 * @brief Test Discrete distribution atomic getters if available
 */
void test_discrete_atomic_getters() {
    std::cout << "Testing Discrete distribution atomic getters..." << std::endl;

    auto result = DiscreteDistribution::create(1, 10);
    assert(result.isOk());
    auto discrete_dist = std::move(result.value);

    // Test atomic getters with correct method names
    int lower_regular = discrete_dist.getLowerBound();
    int upper_regular = discrete_dist.getUpperBound();

    // Check atomic getters
    int lower_atomic = discrete_dist.getLowerBoundAtomic();
    int upper_atomic = discrete_dist.getUpperBoundAtomic();

    assert(lower_regular == lower_atomic);
    assert(upper_regular == upper_atomic);
    assert(lower_atomic == 1);
    assert(upper_atomic == 10);

    std::cout << "  ✓ Lower - Regular: " << lower_regular << ", Atomic: " << lower_atomic
              << std::endl;
    std::cout << "  ✓ Upper - Regular: " << upper_regular << ", Atomic: " << upper_atomic
              << std::endl;
}

/**
 * @brief Simple performance comparison test
 */
void test_performance_comparison() {
    std::cout << "Testing basic performance comparison..." << std::endl;

    auto result = ExponentialDistribution::create(1.5);
    assert(result.isOk());
    auto exp_dist = std::move(result.value);

    const int iterations = 100000;

    // Warm up the cache
    for (int i = 0; i < 1000; ++i) {
        volatile double lambda = exp_dist.getLambdaAtomic();
        (void)lambda;  // Suppress unused variable warning
    }

    // Time regular getter
    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum_regular = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum_regular = sum_regular + exp_dist.getLambda();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto regular_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Time atomic getter
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_atomic = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum_atomic = sum_atomic + exp_dist.getLambdaAtomic();
    }
    end = std::chrono::high_resolution_clock::now();
    auto atomic_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double regular_per_call = static_cast<double>(regular_time.count()) / iterations;
    double atomic_per_call = static_cast<double>(atomic_time.count()) / iterations;

    std::cout << "  Regular getter: " << regular_per_call << " ns/call" << std::endl;
    std::cout << "  Atomic getter:  " << atomic_per_call << " ns/call" << std::endl;

    // Atomic should typically be faster or comparable (when cache is valid)
    if (atomic_per_call <= regular_per_call * 1.2) {  // Allow 20% margin
        std::cout << "  ✓ Atomic getter performance is good" << std::endl;
    } else {
        std::cout << "  ⚠ Atomic getter is slower than expected (this may be normal during testing)"
                  << std::endl;
    }

    // Verify the results are the same (within precision)
    assert(std::abs(sum_regular - sum_atomic) < 1e-10);
}

/**
 * @brief Test thread safety of atomic getters
 */
void test_thread_safety() {
    std::cout << "Testing thread safety of atomic getters..." << std::endl;

    auto result = ExponentialDistribution::create(2.0);
    assert(result.isOk());
    auto exp_dist = std::move(result.value);

    const int num_threads = 4;
    const int iterations_per_thread = 10000;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(
            [&exp_dist, &success_count](int iterations) {
                bool local_success = true;
                for (int i = 0; i < iterations; ++i) {
                    double lambda = exp_dist.getLambdaAtomic();
                    if (std::abs(lambda - 2.0) > 1e-14) {
                        local_success = false;
                        break;
                    }
                }
                if (local_success) {
                    success_count++;
                }
            },
            iterations_per_thread);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    assert(success_count.load() == num_threads);
    std::cout << "  ✓ All " << num_threads << " threads completed successfully" << std::endl;
    std::cout << "  ✓ Thread safety verified with " << (num_threads * iterations_per_thread)
              << " total operations" << std::endl;
}

int main() {
    std::cout << "=== Testing Atomic Parameter Management ===" << std::endl;
    std::cout << std::endl;

    try {
        test_exponential_atomic_getter();
        std::cout << std::endl;

        test_atomic_getter_consistency();
        std::cout << std::endl;

        test_gaussian_atomic_getters();
        std::cout << std::endl;

        test_discrete_atomic_getters();
        std::cout << std::endl;

        test_gaussian_atomic_invalidation();
        std::cout << std::endl;

        test_exponential_atomic_invalidation();
        std::cout << std::endl;

        test_discrete_atomic_invalidation();
        std::cout << std::endl;

        test_performance_comparison();
        std::cout << std::endl;

        test_thread_safety();
        std::cout << std::endl;

        std::cout << "=== All Atomic Parameter Tests Passed! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}

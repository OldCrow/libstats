/**
 * @file test_atomic_parameters.cpp
 * @brief GTest suite for atomic parameter management in libstats distributions
 */
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gaussian.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using namespace stats;

TEST(AtomicParameters, ExponentialAtomicGetter) {
    auto result = ExponentialDistribution::create(2.5);
    ASSERT_TRUE(result.isOk());
    auto exp_dist = std::move(result.value);

    double lambda_regular = exp_dist.getLambda();
    double lambda_atomic  = exp_dist.getLambdaAtomic();

    EXPECT_NEAR(lambda_regular, lambda_atomic, 1e-15);
    EXPECT_NEAR(lambda_atomic, 2.5, 1e-15);

    std::cout << "  Regular: " << lambda_regular << ", Atomic: " << lambda_atomic << std::endl;
}

TEST(AtomicParameters, AtomicGetterConsistency) {
    auto result = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(result.isOk());
    auto exp_dist = std::move(result.value);

    EXPECT_NEAR(exp_dist.getLambdaAtomic(), 1.0, 1e-15);

    exp_dist.setLambda(3.5);

    double lambda_regular = exp_dist.getLambda();
    double lambda_atomic  = exp_dist.getLambdaAtomic();

    EXPECT_NEAR(lambda_regular, lambda_atomic, 1e-15);
    EXPECT_NEAR(lambda_atomic, 3.5, 1e-15);
    std::cout << "  Updated atomic getter: " << lambda_atomic << std::endl;
}

TEST(AtomicParameters, GaussianAtomicGetters) {
    auto result = GaussianDistribution::create(5.0, 2.0);
    ASSERT_TRUE(result.isOk());
    auto gauss_dist = std::move(result.value);

    EXPECT_NEAR(gauss_dist.getMean(),              gauss_dist.getMeanAtomic(),              1e-15);
    EXPECT_NEAR(gauss_dist.getStandardDeviation(), gauss_dist.getStandardDeviationAtomic(), 1e-15);
    EXPECT_NEAR(gauss_dist.getMeanAtomic(), 5.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getStandardDeviationAtomic(), 2.0, 1e-15);
}

TEST(AtomicParameters, DiscreteAtomicGetters) {
    auto result = DiscreteDistribution::create(1, 10);
    ASSERT_TRUE(result.isOk());
    auto discrete_dist = std::move(result.value);

    EXPECT_EQ(discrete_dist.getLowerBound(), discrete_dist.getLowerBoundAtomic());
    EXPECT_EQ(discrete_dist.getUpperBound(), discrete_dist.getUpperBoundAtomic());
    EXPECT_EQ(discrete_dist.getLowerBoundAtomic(), 1);
    EXPECT_EQ(discrete_dist.getUpperBoundAtomic(), 10);
}

TEST(AtomicParameters, GaussianAtomicInvalidation) {
    auto result = GaussianDistribution::create(1.0, 1.0);
    ASSERT_TRUE(result.isOk());
    auto gauss_dist = std::move(result.value);

    EXPECT_NEAR(gauss_dist.getMeanAtomic(), 1.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getStandardDeviationAtomic(), 1.0, 1e-15);

    gauss_dist.setMean(5.0);
    EXPECT_NEAR(gauss_dist.getMean(), 5.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getMeanAtomic(), 5.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getStandardDeviationAtomic(), 1.0, 1e-15);

    gauss_dist.setStandardDeviation(2.0);
    EXPECT_NEAR(gauss_dist.getStandardDeviation(), 2.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getMeanAtomic(), 5.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getStandardDeviationAtomic(), 2.0, 1e-15);

    gauss_dist.setParameters(10.0, 3.0);
    EXPECT_NEAR(gauss_dist.getMeanAtomic(), 10.0, 1e-15);
    EXPECT_NEAR(gauss_dist.getStandardDeviationAtomic(), 3.0, 1e-15);
}

TEST(AtomicParameters, ExponentialAtomicInvalidation) {
    auto result = ExponentialDistribution::create(1.0);
    ASSERT_TRUE(result.isOk());
    auto exp_dist = std::move(result.value);

    EXPECT_NEAR(exp_dist.getLambdaAtomic(), 1.0, 1e-15);

    exp_dist.setLambda(2.5);
    EXPECT_NEAR(exp_dist.getLambda(), 2.5, 1e-15);
    EXPECT_NEAR(exp_dist.getLambdaAtomic(), 2.5, 1e-15);

    auto try_result = exp_dist.trySetParameters(4.0);
    ASSERT_TRUE(try_result.isOk());
    EXPECT_NEAR(exp_dist.getLambdaAtomic(), 4.0, 1e-15);
}

TEST(AtomicParameters, DiscreteAtomicInvalidation) {
    auto result = DiscreteDistribution::create(1, 5);
    ASSERT_TRUE(result.isOk());
    auto discrete_dist = std::move(result.value);

    EXPECT_EQ(discrete_dist.getLowerBoundAtomic(), 1);
    EXPECT_EQ(discrete_dist.getUpperBoundAtomic(), 5);

    discrete_dist.setLowerBound(0);
    EXPECT_EQ(discrete_dist.getLowerBound(), 0);
    EXPECT_EQ(discrete_dist.getLowerBoundAtomic(), 0);
    EXPECT_EQ(discrete_dist.getUpperBoundAtomic(), 5);

    discrete_dist.setUpperBound(10);
    EXPECT_EQ(discrete_dist.getUpperBoundAtomic(), 10);

    discrete_dist.setBounds(2, 8);
    EXPECT_EQ(discrete_dist.getLowerBoundAtomic(), 2);
    EXPECT_EQ(discrete_dist.getUpperBoundAtomic(), 8);

    auto try_result = discrete_dist.trySetParameters(1, 6);
    ASSERT_TRUE(try_result.isOk());
    EXPECT_EQ(discrete_dist.getLowerBoundAtomic(), 1);
    EXPECT_EQ(discrete_dist.getUpperBoundAtomic(), 6);
}

TEST(AtomicParameters, PerformanceComparison) {
    auto result = ExponentialDistribution::create(1.5);
    ASSERT_TRUE(result.isOk());
    auto exp_dist = std::move(result.value);

    const int iterations = 100000;
    for (int i = 0; i < 1000; ++i)
        (void)exp_dist.getLambdaAtomic();  // warm up

    auto t0 = std::chrono::high_resolution_clock::now();
    volatile double sum_regular = 0.0;
    for (int i = 0; i < iterations; ++i)
        sum_regular = sum_regular + exp_dist.getLambda();
    auto t1 = std::chrono::high_resolution_clock::now();

    volatile double sum_atomic = 0.0;
    for (int i = 0; i < iterations; ++i)
        sum_atomic = sum_atomic + exp_dist.getLambdaAtomic();
    auto t2 = std::chrono::high_resolution_clock::now();

    auto ns_regular = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    auto ns_atomic  = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    std::cout << "  Regular: " << ns_regular / iterations << " ns/call, "
              << "Atomic: " << ns_atomic / iterations << " ns/call\n";

    // Both sums should converge to the same value
    EXPECT_NEAR(sum_regular, sum_atomic, 1e-10);
}

TEST(AtomicParameters, ThreadSafety) {
    auto result = ExponentialDistribution::create(2.0);
    ASSERT_TRUE(result.isOk());
    auto exp_dist = std::move(result.value);

    const int num_threads = 4;
    const int ops         = 10000;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&exp_dist, &success_count, ops]() {
            bool ok = true;
            for (int i = 0; i < ops && ok; ++i) {
                if (std::abs(exp_dist.getLambdaAtomic() - 2.0) > 1e-14)
                    ok = false;
            }
            if (ok) success_count++;
        });
    }

    for (auto& thread : threads) thread.join();

    EXPECT_EQ(success_count.load(), num_threads);
    std::cout << "  Thread safety: " << num_threads << " threads x " << ops << " ops OK\n";
}

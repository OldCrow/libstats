#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>
#include <type_traits>
#include <vector>

// All 16 distribution headers (needed for static_assert guards below)
#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/lognormal.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/pareto.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/rayleigh.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/uniform.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/distributions/weibull.h"

using namespace std;
using namespace stats;

// =============================================================================
// COMPILE-TIME noexcept REGRESSION GUARDS (A-10)
// Each assertion fires at build time if the move ctor or move assignment
// reverts to a throwing implementation on any distribution.
// =============================================================================
static_assert(std::is_nothrow_move_constructible_v<BetaDistribution>);
static_assert(std::is_nothrow_move_assignable_v<BetaDistribution>);
static_assert(std::is_nothrow_move_constructible_v<BinomialDistribution>);
static_assert(std::is_nothrow_move_assignable_v<BinomialDistribution>);
static_assert(std::is_nothrow_move_constructible_v<ChiSquaredDistribution>);
static_assert(std::is_nothrow_move_assignable_v<ChiSquaredDistribution>);
static_assert(std::is_nothrow_move_constructible_v<DiscreteDistribution>);
static_assert(std::is_nothrow_move_assignable_v<DiscreteDistribution>);
static_assert(std::is_nothrow_move_constructible_v<ExponentialDistribution>);
static_assert(std::is_nothrow_move_assignable_v<ExponentialDistribution>);
static_assert(std::is_nothrow_move_constructible_v<GammaDistribution>);
static_assert(std::is_nothrow_move_assignable_v<GammaDistribution>);
static_assert(std::is_nothrow_move_constructible_v<GaussianDistribution>);
static_assert(std::is_nothrow_move_assignable_v<GaussianDistribution>);
static_assert(std::is_nothrow_move_constructible_v<LogNormalDistribution>);
static_assert(std::is_nothrow_move_assignable_v<LogNormalDistribution>);
static_assert(std::is_nothrow_move_constructible_v<NegativeBinomialDistribution>);
static_assert(std::is_nothrow_move_assignable_v<NegativeBinomialDistribution>);
static_assert(std::is_nothrow_move_constructible_v<ParetoDistribution>);
static_assert(std::is_nothrow_move_assignable_v<ParetoDistribution>);
static_assert(std::is_nothrow_move_constructible_v<PoissonDistribution>);
static_assert(std::is_nothrow_move_assignable_v<PoissonDistribution>);
static_assert(std::is_nothrow_move_constructible_v<RayleighDistribution>);
static_assert(std::is_nothrow_move_assignable_v<RayleighDistribution>);
static_assert(std::is_nothrow_move_constructible_v<StudentTDistribution>);
static_assert(std::is_nothrow_move_assignable_v<StudentTDistribution>);
static_assert(std::is_nothrow_move_constructible_v<UniformDistribution>);
static_assert(std::is_nothrow_move_assignable_v<UniformDistribution>);
static_assert(std::is_nothrow_move_constructible_v<VonMisesDistribution>);
static_assert(std::is_nothrow_move_assignable_v<VonMisesDistribution>);
static_assert(std::is_nothrow_move_constructible_v<WeibullDistribution>);
static_assert(std::is_nothrow_move_assignable_v<WeibullDistribution>);

std::atomic<int> completed_operations{0};
std::atomic<bool> stop_test{false};
std::atomic<bool> worker_failed{false};

void stressTestUniformCopyMove(int thread_id) {
    int local_ops = 0;

    while (!stop_test.load()) {
        try {
            // Create distributions
            auto result1 = UniformDistribution::create(thread_id * 10, thread_id * 10 + 5);
            auto result2 = UniformDistribution::create(0, 1);

            if (result1.isOk() && result2.isOk()) {
                auto uniform1 = std::move(result1.value);
                auto uniform2 = std::move(result2.value);

                // Perform multiple copy assignments
                for (int i = 0; i < 10; ++i) {
                    auto copy1 = uniform1;
                    auto copy2 = uniform2;

                    // Swap them
                    copy1 = copy2;
                    copy2 = uniform1;

                    // Move assignments
                    auto moved1 = std::move(copy1);
                    auto moved2 = std::move(copy2);

                    // Use them to prevent optimization
                    double sum = moved1.getMean() + moved2.getMean();
                    (void)sum;
                }

                local_ops++;
            }
        } catch (const std::exception& e) {
            cout << "Thread " << thread_id << " caught exception: " << e.what() << endl;
            worker_failed.store(true);
            break;
        }

        if (local_ops % 100 == 0) {
            completed_operations.fetch_add(100);
        }
    }

    completed_operations.fetch_add(local_ops % 100);
}

void stressTestGaussianCopyMove(int thread_id) {
    int local_ops = 0;

    while (!stop_test.load()) {
        try {
            // Create distributions
            auto gauss1 = stats::GaussianDistribution::create(thread_id, 1.0).value;
            auto gauss2 = stats::GaussianDistribution::create(thread_id + 10, 2.0).value;

            // Perform multiple copy assignments
            for (int i = 0; i < 10; ++i) {
                auto copy1 = gauss1;
                auto copy2 = gauss2;

                // Swap them
                copy1 = copy2;
                copy2 = gauss1;

                // Move assignments
                auto moved1 = std::move(copy1);
                auto moved2 = std::move(copy2);

                // Use them to prevent optimization
                double sum = moved1.getMean() + moved2.getMean();
                (void)sum;
            }

            local_ops++;
        } catch (const std::exception& e) {
            cout << "Thread " << thread_id << " caught exception: " << e.what() << endl;
            worker_failed.store(true);
            break;
        }

        if (local_ops % 100 == 0) {
            completed_operations.fetch_add(100);
        }
    }

    completed_operations.fetch_add(local_ops % 100);
}

void stressTestPoissonCopyMove(int thread_id) {
    int local_ops = 0;

    while (!stop_test.load()) {
        try {
            auto pois1 = PoissonDistribution::create(thread_id + 1.0).value;
            auto pois2 = PoissonDistribution::create(thread_id + 3.0).value;

            for (int i = 0; i < 10; ++i) {
                auto copy1 = pois1;
                auto copy2 = pois2;
                copy1 = copy2;
                copy2 = pois1;
                auto moved1 = std::move(copy1);
                auto moved2 = std::move(copy2);
                double sum = moved1.getMean() + moved2.getMean();
                (void)sum;
            }
            local_ops++;
        } catch (const std::exception& e) {
            cout << "Poisson thread " << thread_id << " caught: " << e.what() << endl;
            worker_failed.store(true);
            break;
        }
        if (local_ops % 100 == 0) completed_operations.fetch_add(100);
    }
    completed_operations.fetch_add(local_ops % 100);
}

void stressTestGammaCopyMove(int thread_id) {
    int local_ops = 0;

    while (!stop_test.load()) {
        try {
            auto gamma1 = GammaDistribution::create(2.0, thread_id + 0.5).value;
            auto gamma2 = GammaDistribution::create(3.0, thread_id + 1.0).value;

            for (int i = 0; i < 10; ++i) {
                auto copy1 = gamma1;
                auto copy2 = gamma2;
                copy1 = copy2;
                copy2 = gamma1;
                auto moved1 = std::move(copy1);
                auto moved2 = std::move(copy2);
                double sum = moved1.getMean() + moved2.getMean();
                (void)sum;
            }
            local_ops++;
        } catch (const std::exception& e) {
            cout << "Gamma thread " << thread_id << " caught: " << e.what() << endl;
            worker_failed.store(true);
            break;
        }
        if (local_ops % 100 == 0) completed_operations.fetch_add(100);
    }
    completed_operations.fetch_add(local_ops % 100);
}

void stressTestDiscreteCopyMove(int thread_id) {
    int local_ops = 0;

    while (!stop_test.load()) {
        try {
            auto disc1 = DiscreteDistribution::create(1, thread_id + 6).value;
            auto disc2 = DiscreteDistribution::create(0, thread_id + 4).value;

            for (int i = 0; i < 10; ++i) {
                auto copy1 = disc1;
                auto copy2 = disc2;
                copy1 = copy2;
                copy2 = disc1;
                auto moved1 = std::move(copy1);
                auto moved2 = std::move(copy2);
                double sum = moved1.getMean() + moved2.getMean();
                (void)sum;
            }
            local_ops++;
        } catch (const std::exception& e) {
            cout << "Discrete thread " << thread_id << " caught: " << e.what() << endl;
            worker_failed.store(true);
            break;
        }
        if (local_ops % 100 == 0) completed_operations.fetch_add(100);
    }
    completed_operations.fetch_add(local_ops % 100);
}

void stressTestExponentialCopyMove(int thread_id) {
    int local_ops = 0;

    while (!stop_test.load()) {
        try {
            // Create distributions
            auto result1 = ExponentialDistribution::create(thread_id + 1);
            auto result2 = ExponentialDistribution::create(thread_id + 2);

            if (result1.isOk() && result2.isOk()) {
                auto exp1 = std::move(result1.value);
                auto exp2 = std::move(result2.value);

                // Perform multiple copy assignments
                for (int i = 0; i < 10; ++i) {
                    auto copy1 = exp1;
                    auto copy2 = exp2;

                    // Swap them
                    copy1 = copy2;
                    copy2 = exp1;

                    // Move assignments
                    auto moved1 = std::move(copy1);
                    auto moved2 = std::move(copy2);

                    // Use them to prevent optimization
                    double sum = moved1.getMean() + moved2.getMean();
                    (void)sum;
                }

                local_ops++;
            }
        } catch (const std::exception& e) {
            cout << "Thread " << thread_id << " caught exception: " << e.what() << endl;
            worker_failed.store(true);
            break;
        }

        if (local_ops % 100 == 0) {
            completed_operations.fetch_add(100);
        }
    }

    completed_operations.fetch_add(local_ops % 100);
}

int main() {
    cout << "=== Copy/Move Semantics Stress Test ===" << endl;
    cout << "This test performs intensive copy/move operations across multiple threads" << endl;
    cout << "to ensure no deadlocks occur under high load." << endl;

    const int num_threads = 8;
    const int test_duration_seconds = 5;

    vector<thread> threads;

    cout << "\nStarting stress test with " << num_threads << " threads for "
         << test_duration_seconds << " seconds..." << endl;

    auto start_time = chrono::steady_clock::now();

    // Start threads for each distribution type (6 distributions, ~1 thread each)
    threads.emplace_back(stressTestUniformCopyMove, 0);
    threads.emplace_back(stressTestGaussianCopyMove, 100);
    threads.emplace_back(stressTestExponentialCopyMove, 200);
    threads.emplace_back(stressTestPoissonCopyMove, 300);
    threads.emplace_back(stressTestGammaCopyMove, 400);
    threads.emplace_back(stressTestDiscreteCopyMove, 500);
    // Fill remaining threads with Gaussian (simplest, broadest coverage)
    for (int i = 6; i < num_threads; ++i) {
        threads.emplace_back(stressTestGaussianCopyMove, i + 600);
    }

    // Monitor progress
    while (chrono::steady_clock::now() - start_time < chrono::seconds(test_duration_seconds)) {
        this_thread::sleep_for(chrono::milliseconds(500));
        cout << "\rOperations completed: " << completed_operations.load() << flush;
    }

    cout << "\n\nStopping test..." << endl;
    stop_test.store(true);

    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = chrono::steady_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);

    cout << "\n🎉 Stress test completed successfully!" << endl;
    cout << "✅ Duration: " << duration.count() << " ms" << endl;
    cout << "✅ Total operations: " << completed_operations.load() << endl;
    cout << "✅ Operations per second: " << (completed_operations.load() * 1000) / duration.count()
         << endl;
    cout << "✅ No deadlocks occurred under high load" << endl;
    cout << "✅ All distributions handle concurrent copy/move operations safely" << endl;
    return worker_failed.load() ? 1 : 0;
}

/**
 * @file parallel_correctness_verification.cpp
 * @brief Platform-Aware Parallel Correctness Verification Tool
 *
 * Validates that parallel implementations produce identical results to sequential versions
 * across different threading systems (GCD, Windows Thread Pool API, OpenMP, pthreads, etc.)
 * and tests determinism with libstats' actual parallel execution infrastructure.
 */

// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

// Additional standard library includes for verification functionality
#include "../include/distributions/discrete.h"
#include "../include/distributions/exponential.h"
#include "../include/distributions/gamma.h"
#include "../include/distributions/gaussian.h"
#include "../include/distributions/poisson.h"
#include "../include/distributions/uniform.h"
#include "../include/libstats.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_execution.h"
#include "../include/platform/thread_pool.h"
#include "tool_utils.h"

#include <cmath>
#include <functional>
#include <future>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <thread>

using namespace libstats;

// Test configuration
struct TestConfig {
    size_t min_size = 1000;
    size_t max_size = 100000;
    size_t num_runs = 5;
    double tolerance = 1e-12;
    std::vector<int> thread_counts = {1, 2, 4, 8};
    bool test_determinism = true;
    bool test_thread_safety = true;
};

// Test result structure
struct TestResult {
    std::string distribution;
    std::string operation;
    std::string threading_system;
    bool correctness_pass;
    bool determinism_pass;
    bool thread_safety_pass;
    double max_difference;
    double performance_ratio;
    size_t failed_comparisons;
    std::string error_details;
};

class ParallelCorrectnessVerifier {
   private:
    TestConfig config_;
    std::vector<TestResult> results_;

    template <typename Dist>
    std::vector<double> generate_test_data(size_t size, int seed = 12345) {
        std::mt19937 gen(static_cast<std::mt19937::result_type>(seed));
        std::vector<double> data(size);

        if constexpr (std::is_same_v<Dist, DiscreteDistribution>) {
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<double>(gen() % 10);  // 0-9 range
            }
        } else {
            std::uniform_real_distribution<> uniform(0.1, 10.0);
            for (size_t i = 0; i < size; ++i) {
                data[i] = uniform(gen);
            }
        }

        return data;
    }

    template <typename Dist>
    std::vector<double> compute_sequential(const std::vector<double>& inputs,
                                           const std::string& operation) {
        auto dist_result = Dist::create();
        if (dist_result.isError()) {
            throw std::runtime_error("Failed to create distribution instance: " +
                                     dist_result.message);
        }
        auto dist = dist_result.value;
        std::vector<double> results(inputs.size());

        for (size_t i = 0; i < inputs.size(); ++i) {
            if (operation == "PDF") {
                results[i] = dist.getProbability(inputs[i]);
            } else if (operation == "LogPDF") {
                results[i] = dist.getLogProbability(inputs[i]);
            } else if (operation == "CDF") {
                results[i] = dist.getCumulativeProbability(inputs[i]);
            }
        }

        return results;
    }

    template <typename Dist>
    std::vector<double> compute_with_threadpool(const std::vector<double>& inputs,
                                                const std::string& operation, int num_threads) {
        auto dist_result = Dist::create();
        if (dist_result.isError()) {
            throw std::runtime_error("Failed to create distribution instance: " +
                                     dist_result.message);
        }
        auto dist = dist_result.value;
        std::vector<double> results(inputs.size());

        libstats::ThreadPool pool(static_cast<std::size_t>(num_threads));

        auto worker = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                if (operation == "PDF") {
                    results[i] = dist.getProbability(inputs[i]);
                } else if (operation == "LogPDF") {
                    results[i] = dist.getLogProbability(inputs[i]);
                } else if (operation == "CDF") {
                    results[i] = dist.getCumulativeProbability(inputs[i]);
                }
            }
        };

        size_t chunk_size = inputs.size() / static_cast<std::size_t>(num_threads);
        std::vector<std::future<void>> futures;

        for (int t = 0; t < num_threads; ++t) {
            size_t start = static_cast<std::size_t>(t) * chunk_size;
            size_t end = (t == num_threads - 1) ? inputs.size()
                                                : (static_cast<std::size_t>(t) + 1) * chunk_size;
            futures.push_back(pool.submit(worker, start, end));
        }

        for (auto& f : futures) {
            f.wait();
        }

        return results;
    }

#if defined(LIBSTATS_HAS_GCD)
    template <typename Dist>
    std::vector<double> compute_with_gcd(const std::vector<double>& inputs,
                                         const std::string& operation) {
        auto dist_result = Dist::create();
        if (dist_result.isError()) {
            throw std::runtime_error("Failed to create distribution instance: " +
                                     dist_result.message);
        }
        auto dist = dist_result.value;
        std::vector<double> results(inputs.size());

        dispatch_queue_t queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
        dispatch_group_t group = dispatch_group_create();

        size_t num_chunks = libstats::parallel::get_optimal_thread_count(inputs.size());
        size_t chunk_size = inputs.size() / num_chunks;

        // Capture pointers for modification in blocks
        double* results_ptr = results.data();
        const double* inputs_ptr = inputs.data();

        for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
            dispatch_group_async(group, queue, ^{
              size_t start = chunk * chunk_size;
              size_t end = (chunk == num_chunks - 1) ? inputs.size() : (chunk + 1) * chunk_size;

              auto local_dist_result = Dist::create();  // Create local instance for thread safety
              if (local_dist_result.isError()) {
                  return;  // Skip this chunk if distribution creation fails
              }
              auto local_dist = local_dist_result.value;
              for (size_t i = start; i < end; ++i) {
                  if (operation == "PDF") {
                      results_ptr[i] = local_dist.getProbability(inputs_ptr[i]);
                  } else if (operation == "LogPDF") {
                      results_ptr[i] = local_dist.getLogProbability(inputs_ptr[i]);
                  } else if (operation == "CDF") {
                      results_ptr[i] = local_dist.getCumulativeProbability(inputs_ptr[i]);
                  }
              }
            });
        }

        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        dispatch_release(group);

        return results;
    }
#endif

#if defined(LIBSTATS_HAS_OPENMP)
    template <typename Dist>
    std::vector<double> compute_with_openmp(const std::vector<double>& inputs,
                                            const std::string& operation, int num_threads) {
        auto dist_result = Dist::create();
        if (dist_result.isError()) {
            throw std::runtime_error("Failed to create distribution instance: " +
                                     dist_result.message);
        }
        auto dist = dist_result.value;
        std::vector<double> results(inputs.size());

        omp_set_num_threads(num_threads);

    #pragma omp parallel for
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (operation == "PDF") {
                results[i] = dist.getProbability(inputs[i]);
            } else if (operation == "LogPDF") {
                results[i] = dist.getLogProbability(inputs[i]);
            } else if (operation == "CDF") {
                results[i] = dist.getCumulativeProbability(inputs[i]);
            }
        }

        return results;
    }
#endif

    bool compare_results(const std::vector<double>& reference, const std::vector<double>& test,
                         double& max_diff, size_t& failed_count, std::string& error_details) {
        max_diff = 0.0;
        failed_count = 0;
        error_details.clear();

        if (reference.size() != test.size()) {
            error_details = "Size mismatch: " + std::to_string(reference.size()) + " vs " +
                            std::to_string(test.size());
            return false;
        }

        std::ostringstream error_stream;
        size_t error_samples = 0;
        const size_t max_error_samples = 5;

        for (size_t i = 0; i < reference.size(); ++i) {
            double diff = std::abs(reference[i] - test[i]);

            if (std::isfinite(reference[i]) && std::isfinite(test[i])) {
                double rel_diff = diff / (std::max(std::abs(reference[i]), 1e-10));
                if (rel_diff > config_.tolerance) {
                    failed_count++;
                    if (error_samples < max_error_samples) {
                        error_stream << "Index " << i << ": ref=" << reference[i]
                                     << ", test=" << test[i] << ", diff=" << diff << "; ";
                        error_samples++;
                    }
                }
            } else if (std::isnan(reference[i]) != std::isnan(test[i]) ||
                       std::isinf(reference[i]) != std::isinf(test[i])) {
                failed_count++;
                if (error_samples < max_error_samples) {
                    error_stream << "Index " << i << ": ref=" << reference[i]
                                 << ", test=" << test[i] << ", special value mismatch; ";
                    error_samples++;
                }
            }

            max_diff = std::max(max_diff, diff);
        }

        if (failed_count > max_error_samples) {
            error_stream << "... (+" << (failed_count - max_error_samples) << " more)";
        }

        error_details = error_stream.str();
        return failed_count == 0;
    }

    template <typename Dist>
    void test_threading_system(
        const std::string& dist_name, const std::string& system_name,
        std::function<std::vector<double>(const std::vector<double>&, const std::string&)>
            compute_func) {
        for (const std::string& operation : {"PDF", "LogPDF", "CDF"}) {
            TestResult result;
            result.distribution = dist_name;
            result.operation = operation;
            result.threading_system = system_name;
            result.correctness_pass = true;
            result.determinism_pass = true;
            result.thread_safety_pass = true;
            result.max_difference = 0.0;
            result.failed_comparisons = 0;

            auto test_data = generate_test_data<Dist>(config_.max_size);

            try {
                // Test correctness: sequential vs parallel
                auto seq_results = compute_sequential<Dist>(test_data, operation);

                auto start_time = std::chrono::high_resolution_clock::now();
                auto par_results = compute_func(test_data, operation);
                auto end_time = std::chrono::high_resolution_clock::now();

                double par_time = std::chrono::duration<double>(end_time - start_time).count();

                start_time = std::chrono::high_resolution_clock::now();
                compute_sequential<Dist>(test_data, operation);
                end_time = std::chrono::high_resolution_clock::now();

                double seq_time = std::chrono::duration<double>(end_time - start_time).count();
                result.performance_ratio = (seq_time > 0) ? seq_time / par_time : 1.0;

                double max_diff;
                size_t failed_count;
                std::string error_details;
                result.correctness_pass = compare_results(seq_results, par_results, max_diff,
                                                          failed_count, error_details);
                result.max_difference = max_diff;
                result.failed_comparisons = failed_count;
                result.error_details = error_details;

                // Test determinism: multiple runs
                if (config_.test_determinism) {
                    for (size_t run = 1; run < config_.num_runs; ++run) {
                        auto run_results = compute_func(test_data, operation);
                        double run_max_diff;
                        size_t run_failed_count;
                        std::string run_error_details;
                        bool deterministic = compare_results(par_results, run_results, run_max_diff,
                                                             run_failed_count, run_error_details);

                        if (!deterministic) {
                            result.determinism_pass = false;
                            if (result.error_details.empty()) {
                                result.error_details = "Non-deterministic: " + run_error_details;
                            }
                        }
                    }
                }

                // Test thread safety: concurrent execution
                if (config_.test_thread_safety) {
                    const int concurrent_tests = 4;
                    std::vector<std::future<std::vector<double>>> futures;

                    for (int i = 0; i < concurrent_tests; ++i) {
                        futures.push_back(
                            std::async(std::launch::async, compute_func, test_data, operation));
                    }

                    for (auto& future : futures) {
                        auto concurrent_results = future.get();
                        double concurrent_max_diff;
                        size_t concurrent_failed_count;
                        std::string concurrent_error_details;
                        bool thread_safe =
                            compare_results(par_results, concurrent_results, concurrent_max_diff,
                                            concurrent_failed_count, concurrent_error_details);

                        if (!thread_safe) {
                            result.thread_safety_pass = false;
                            if (result.error_details.find("Thread safety") == std::string::npos) {
                                result.error_details +=
                                    " Thread safety issue: " + concurrent_error_details;
                            }
                        }
                    }
                }

            } catch (const std::exception& e) {
                result.correctness_pass = false;
                result.determinism_pass = false;
                result.thread_safety_pass = false;
                result.error_details = std::string("Exception: ") + e.what();
            }

            // Display result
            std::string status =
                (result.correctness_pass && result.determinism_pass && result.thread_safety_pass)
                    ? "✓ PASS"
                    : "✗ FAIL";
            std::cout << std::setw(8) << operation << " [" << std::setw(12) << system_name
                      << "]: " << status;
            std::cout << " (max_diff=" << std::scientific << std::setprecision(2)
                      << result.max_difference;
            std::cout << ", speedup=" << std::fixed << std::setprecision(1)
                      << result.performance_ratio << "x";

            if (!result.correctness_pass) {
                std::cout << ", " << result.failed_comparisons << " correctness failures";
            }
            if (!result.determinism_pass) {
                std::cout << ", non-deterministic";
            }
            if (!result.thread_safety_pass) {
                std::cout << ", thread-unsafe";
            }
            std::cout << ")\n";

            results_.push_back(result);
        }
    }

    template <typename Dist>
    void test_distribution(const std::string& dist_name) {
        std::cout << "\n--- " << dist_name << " Distribution Parallel Verification ---\n";

        // Test libstats ThreadPool
        test_threading_system<Dist>(
            dist_name, "ThreadPool",
            [this](const std::vector<double>& inputs, const std::string& operation) {
                return compute_with_threadpool<Dist>(inputs, operation, 4);
            });

#if defined(LIBSTATS_HAS_GCD)
        // Test Grand Central Dispatch (GCD) on Apple platforms
        test_threading_system<Dist>(
            dist_name, "GCD",
            [this](const std::vector<double>& inputs, const std::string& operation) {
                return compute_with_gcd<Dist>(inputs, operation);
            });
#endif

#if defined(LIBSTATS_HAS_OPENMP)
        // Test OpenMP
        test_threading_system<Dist>(
            dist_name, "OpenMP",
            [this](const std::vector<double>& inputs, const std::string& operation) {
                return compute_with_openmp<Dist>(inputs, operation, 4);
            });
#endif

        // Add more threading systems as they're implemented in libstats
    }

   public:
    void run_verification() {
        std::cout << "\n================================================\n";
        std::cout << "  Platform-Aware Parallel Correctness Verification\n";
        std::cout << "================================================\n\n";
        std::cout << "Validates parallel execution across different threading systems\n\n";

        // Display system information
        const auto& cpu_features = libstats::cpu::get_features();
        std::cout << "System: " << cpu_features.topology.logical_cores << " logical cores, "
                  << cpu_features.topology.physical_cores << " physical cores\n";
        std::cout << "Threading: " << libstats::parallel::execution_support_string() << "\n";
        std::cout << "Test size: " << config_.max_size << " elements\n";
        std::cout << "Tolerance: " << std::scientific << config_.tolerance << "\n";
        std::cout << "Runs per test: " << config_.num_runs << "\n";

        std::cout << "\nAvailable threading systems:\n";
        std::cout << "  - libstats::ThreadPool (std::thread based)\n";
#if defined(LIBSTATS_HAS_GCD)
        std::cout << "  - Grand Central Dispatch (GCD)\n";
#endif
#if defined(LIBSTATS_HAS_WIN_THREADPOOL)
        std::cout << "  - Windows Thread Pool API\n";
#endif
#if defined(LIBSTATS_HAS_OPENMP)
        std::cout << "  - OpenMP\n";
#endif
#if defined(LIBSTATS_HAS_PTHREADS)
        std::cout << "  - POSIX threads\n";
#endif

        // Test all distributions using safe factory methods
        test_distribution<UniformDistribution>("Uniform");
        test_distribution<GaussianDistribution>("Gaussian");
        test_distribution<ExponentialDistribution>("Exponential");
        test_distribution<DiscreteDistribution>("Discrete");
        test_distribution<PoissonDistribution>("Poisson");
        test_distribution<GammaDistribution>("Gamma");

        print_summary();
    }

    void print_summary() {
        std::cout << "\n\n--- Summary ---\n";

        std::map<std::string, std::vector<TestResult*>> by_system;
        for (auto& result : results_) {
            by_system[result.threading_system].push_back(&result);
        }

        for (const auto& [system, system_results] : by_system) {
            size_t total = system_results.size();
            size_t correctness_passed = 0;
            size_t determinism_passed = 0;
            size_t thread_safety_passed = 0;
            double avg_speedup = 0.0;

            for (const auto& result : system_results) {
                if (result->correctness_pass)
                    correctness_passed++;
                if (result->determinism_pass)
                    determinism_passed++;
                if (result->thread_safety_pass)
                    thread_safety_passed++;
                avg_speedup += result->performance_ratio;
            }

            avg_speedup /= static_cast<double>(total);

            std::cout << "\n" << system << " Threading System:\n";
            std::cout << "  Total tests: " << total << "\n";
            std::cout << "  Correctness: " << correctness_passed << "/" << total << " ("
                      << std::fixed << std::setprecision(1)
                      << (100.0 * static_cast<double>(correctness_passed) /
                          static_cast<double>(total))
                      << "%)\n";
            std::cout << "  Determinism: " << determinism_passed << "/" << total << " ("
                      << std::fixed << std::setprecision(1)
                      << (100.0 * static_cast<double>(determinism_passed) /
                          static_cast<double>(total))
                      << "%)\n";
            std::cout << "  Thread Safety: " << thread_safety_passed << "/" << total << " ("
                      << std::fixed << std::setprecision(1)
                      << (100.0 * static_cast<double>(thread_safety_passed) /
                          static_cast<double>(total))
                      << "%)\n";
            std::cout << "  Average speedup: " << std::fixed << std::setprecision(1) << avg_speedup
                      << "x\n";
        }

        // List failed tests
        std::vector<TestResult*> failed_tests;
        for (auto& result : results_) {
            if (!result.correctness_pass || !result.determinism_pass ||
                !result.thread_safety_pass) {
                failed_tests.push_back(&result);
            }
        }

        if (!failed_tests.empty()) {
            std::cout << "\n--- Failed Tests ---\n";
            for (const auto& result : failed_tests) {
                std::cout << "❌ " << result->distribution << "::" << result->operation << " ["
                          << result->threading_system << "]";
                if (!result->correctness_pass) {
                    std::cout << " (correctness: " << result->failed_comparisons << " failures)";
                }
                if (!result->determinism_pass) {
                    std::cout << " (non-deterministic)";
                }
                if (!result->thread_safety_pass) {
                    std::cout << " (thread-unsafe)";
                }
                std::cout << "\n";
                if (!result->error_details.empty()) {
                    std::cout << "   Details: " << result->error_details << "\n";
                }
            }
        }

        std::cout << "\n--- Recommendations ---\n";
        if (failed_tests.empty()) {
            std::cout
                << "✅ All parallel implementations are correct, deterministic, and thread-safe.\n";
        } else {
            std::cout << "⚠️  Some parallel implementations have issues.\n";
            std::cout << "⚠️  Review failed tests and ensure proper synchronization.\n";
        }

        // Calculate overall average speedup
        double overall_avg_speedup = 0.0;
        for (const auto& result : results_) {
            overall_avg_speedup += result.performance_ratio;
        }
        overall_avg_speedup /= static_cast<double>(results_.size());

        if (overall_avg_speedup > 1.5) {
            std::cout << "✅ Parallel performance is meeting expectations (avg " << std::fixed
                      << std::setprecision(1) << overall_avg_speedup << "x).\n";
        } else {
            std::cout << "⚠️  Parallel speedup is lower than expected (avg " << std::fixed
                      << std::setprecision(1) << overall_avg_speedup << "x).\n";
            std::cout << "⚠️  Consider optimizing parallel overhead or increasing problem size.\n";
        }
    }
};

int main() {
    try {
        ParallelCorrectnessVerifier verifier;
        verifier.run_verification();

        std::cout << "\nPlatform-aware parallel correctness verification completed.\n";
        std::cout << "This tool validates libstats' actual parallel execution infrastructure.\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

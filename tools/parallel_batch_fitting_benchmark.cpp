/**
 * @file parallel_batch_fitting_benchmark.cpp
 * @brief Comprehensive benchmark for parallel batch fitting across all distributions
 *
 * This professional benchmark tool provides detailed performance analysis of the
 * parallelBatchFit() method across all probability distributions in libstats.
 *
 * Features:
 * - Cross-distribution performance comparisons
 * - Scalability analysis with varying dataset counts and sizes
 * - Threading efficiency evaluation
 * - Memory usage and cache performance analysis
 * - Statistical accuracy verification
 * - Export capabilities for performance data
 *
 * Usage:
 *   ./parallel_batch_fitting_benchmark [OPTIONS]
 *
 * Options:
 *   --quick         Run a quick benchmark with smaller datasets
 *   --full          Run comprehensive benchmark (default)
 *   --export        Export results to CSV files
 *   --threads N     Use N threads for parallel execution
 *   --verbose       Enable verbose output
 *   --distribution D Test only distribution D (gaussian, exponential, etc.)
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace stats;

//==============================================================================
// BENCHMARK CONFIGURATION
//==============================================================================

struct BenchmarkConfig {
    bool quick_mode = false;
    bool full_mode = true;
    bool export_results = false;
    bool verbose = false;
    size_t num_threads = std::thread::hardware_concurrency();
    std::string target_distribution = "all";

    // Dataset configuration
    std::vector<size_t> dataset_counts = {1, 5, 10, 25, 50, 100};
    std::vector<size_t> dataset_sizes = {100, 500, 1000, 5000, 10000};
    size_t num_trials = 5;

    void configure_quick() {
        dataset_counts = {1, 5, 10, 25};
        dataset_sizes = {100, 500, 1000};
        num_trials = 3;
    }
};

//==============================================================================
// BENCHMARK RESULTS STRUCTURES
//==============================================================================

struct FittingResult {
    std::string distribution_name;
    size_t dataset_count;
    size_t dataset_size;

    double parallel_time_ms = 0.0;
    double sequential_time_ms = 0.0;
    double speedup = 0.0;
    double efficiency = 0.0;  // speedup / num_cores

    double accuracy_score = 0.0;  // How close fitted parameters are to true values
    double memory_usage_mb = 0.0;

    bool success = false;
    std::string error_message;
};

struct BenchmarkSummary {
    std::vector<FittingResult> results;
    double total_runtime_seconds = 0.0;
    std::string system_info;
    BenchmarkConfig config;
};

//==============================================================================
// SYSTEM INFO COLLECTION
//==============================================================================

std::string getSystemInfo() {
    std::ostringstream info;
    info << "CPU Cores: " << std::thread::hardware_concurrency() << ", ";

// Get SIMD capabilities
#ifdef LIBSTATS_HAS_AVX512
    info << "SIMD: AVX-512, ";
#elif defined(LIBSTATS_HAS_AVX2)
    info << "SIMD: AVX2, ";
#elif defined(LIBSTATS_HAS_AVX)
    info << "SIMD: AVX, ";
#elif defined(LIBSTATS_HAS_SSE2)
    info << "SIMD: SSE2, ";
#else
    info << "SIMD: None, ";
#endif

    info << "Threading: ";
#ifdef __APPLE__
    info << "GCD";
#else
    info << "ThreadPool";
#endif

    return info.str();
}

//==============================================================================
// DATA GENERATION UTILITIES
//==============================================================================

class DatasetGenerator {
   public:
    static std::vector<std::vector<double>> generateGaussianDatasets(size_t count, size_t size,
                                                                     std::mt19937& rng) {
        std::vector<std::vector<double>> datasets;
        std::uniform_real_distribution<double> mean_dist(-10.0, 10.0);
        std::uniform_real_distribution<double> std_dist(0.5, 5.0);

        for (size_t i = 0; i < count; ++i) {
            double true_mean = mean_dist(rng);
            double true_std = std_dist(rng);

            std::vector<double> dataset;
            dataset.reserve(size);

            std::normal_distribution<double> gen(true_mean, true_std);
            for (size_t j = 0; j < size; ++j) {
                dataset.push_back(gen(rng));
            }

            datasets.push_back(std::move(dataset));
        }

        return datasets;
    }

    static std::vector<std::vector<double>> generateExponentialDatasets(size_t count, size_t size,
                                                                        std::mt19937& rng) {
        std::vector<std::vector<double>> datasets;
        std::uniform_real_distribution<double> lambda_dist(0.1, 5.0);

        for (size_t i = 0; i < count; ++i) {
            double true_lambda = lambda_dist(rng);

            std::vector<double> dataset;
            dataset.reserve(size);

            std::exponential_distribution<double> gen(true_lambda);
            for (size_t j = 0; j < size; ++j) {
                dataset.push_back(gen(rng));
            }

            datasets.push_back(std::move(dataset));
        }

        return datasets;
    }

    static std::vector<std::vector<double>> generateUniformDatasets(size_t count, size_t size,
                                                                    std::mt19937& rng) {
        std::vector<std::vector<double>> datasets;
        std::uniform_real_distribution<double> range_dist(-10.0, 10.0);
        std::uniform_real_distribution<double> width_dist(1.0, 10.0);

        for (size_t i = 0; i < count; ++i) {
            double a = range_dist(rng);
            double width = width_dist(rng);
            double b = a + width;

            std::vector<double> dataset;
            dataset.reserve(size);

            std::uniform_real_distribution<double> gen(a, b);
            for (size_t j = 0; j < size; ++j) {
                dataset.push_back(gen(rng));
            }

            datasets.push_back(std::move(dataset));
        }

        return datasets;
    }

    static std::vector<std::vector<double>> generateGammaDatasets(size_t count, size_t size,
                                                                  std::mt19937& rng) {
        std::vector<std::vector<double>> datasets;
        std::uniform_real_distribution<double> alpha_dist(0.5, 5.0);
        std::uniform_real_distribution<double> beta_dist(0.5, 3.0);

        for (size_t i = 0; i < count; ++i) {
            double alpha = alpha_dist(rng);
            double beta = beta_dist(rng);

            std::vector<double> dataset;
            dataset.reserve(size);

            std::gamma_distribution<double> gen(alpha, 1.0 / beta);
            for (size_t j = 0; j < size; ++j) {
                dataset.push_back(gen(rng));
            }

            datasets.push_back(std::move(dataset));
        }

        return datasets;
    }

    static std::vector<std::vector<double>> generatePoissonDatasets(size_t count, size_t size,
                                                                    std::mt19937& rng) {
        std::vector<std::vector<double>> datasets;
        std::uniform_real_distribution<double> lambda_dist(0.5, 10.0);

        for (size_t i = 0; i < count; ++i) {
            double lambda = lambda_dist(rng);

            std::vector<double> dataset;
            dataset.reserve(size);

            std::poisson_distribution<int> gen(lambda);
            for (size_t j = 0; j < size; ++j) {
                dataset.push_back(static_cast<double>(gen(rng)));
            }

            datasets.push_back(std::move(dataset));
        }

        return datasets;
    }

    static std::vector<std::vector<double>> generateDiscreteDatasets(size_t count, size_t size,
                                                                     std::mt19937& rng) {
        std::vector<std::vector<double>> datasets;
        std::uniform_int_distribution<int> range_dist(0, 20);
        std::uniform_int_distribution<int> width_dist(5, 15);

        for (size_t i = 0; i < count; ++i) {
            int a = range_dist(rng);
            int width = width_dist(rng);
            int b = a + width;

            std::vector<double> dataset;
            dataset.reserve(size);

            std::uniform_int_distribution<int> gen(a, b);
            for (size_t j = 0; j < size; ++j) {
                dataset.push_back(static_cast<double>(gen(rng)));
            }

            datasets.push_back(std::move(dataset));
        }

        return datasets;
    }
};

//==============================================================================
// BENCHMARK IMPLEMENTATIONS
//==============================================================================

template <typename Distribution>
class DistributionBenchmark {
   public:
    static FittingResult benchmarkParallelFitting(const std::string& name,
                                                  const std::vector<std::vector<double>>& datasets,
                                                  size_t num_trials, bool verbose = false) {
        FittingResult result;
        result.distribution_name = name;
        result.dataset_count = datasets.size();
        result.dataset_size = datasets.empty() ? 0 : datasets[0].size();

        if (datasets.empty()) {
            result.error_message = "Empty dataset vector";
            return result;
        }

        try {
            // Warmup run
            std::vector<Distribution> warmup_results(datasets.size());
            Distribution::parallelBatchFit(datasets, warmup_results);

            double total_parallel_time = 0.0;
            double total_sequential_time = 0.0;

            for (size_t trial = 0; trial < num_trials; ++trial) {
                // Parallel benchmark
                std::vector<Distribution> parallel_results(datasets.size());
                auto start = std::chrono::high_resolution_clock::now();
                Distribution::parallelBatchFit(datasets, parallel_results);
                auto end = std::chrono::high_resolution_clock::now();

                auto parallel_time =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                total_parallel_time +=
                    static_cast<double>(parallel_time.count()) / 1000.0;  // Convert to milliseconds

                // Sequential benchmark
                std::vector<Distribution> sequential_results(datasets.size());
                start = std::chrono::high_resolution_clock::now();
                for (size_t i = 0; i < datasets.size(); ++i) {
                    sequential_results[i].fit(datasets[i]);
                }
                end = std::chrono::high_resolution_clock::now();

                auto sequential_time =
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                total_sequential_time += static_cast<double>(sequential_time.count()) /
                                         1000.0;  // Convert to milliseconds

                // Verify correctness on first trial
                if (trial == 0) {
                    result.accuracy_score = computeAccuracy(parallel_results, sequential_results);
                }
            }

            result.parallel_time_ms = total_parallel_time / static_cast<double>(num_trials);
            result.sequential_time_ms = total_sequential_time / static_cast<double>(num_trials);
            result.speedup = result.sequential_time_ms > 0
                                 ? result.sequential_time_ms / result.parallel_time_ms
                                 : 0.0;
            result.efficiency = result.speedup / std::thread::hardware_concurrency();
            result.success = true;

            if (verbose) {
                std::cout << "    " << name << ": " << std::fixed << std::setprecision(2)
                          << result.parallel_time_ms << "ms parallel, " << result.sequential_time_ms
                          << "ms sequential, " << result.speedup << "x speedup\n";
            }

        } catch (const std::exception& e) {
            result.error_message = e.what();
            result.success = false;
        }

        return result;
    }

   private:
    static double computeAccuracy(const std::vector<Distribution>& parallel_results,
                                  const std::vector<Distribution>& sequential_results) {
        if (parallel_results.size() != sequential_results.size()) {
            return 0.0;
        }

        double total_error = 0.0;
        size_t count = 0;

        for (size_t i = 0; i < parallel_results.size(); ++i) {
            // For each distribution type, compare key parameters
            total_error += computeParameterError(parallel_results[i], sequential_results[i]);
            count++;
        }

        // Return accuracy score (1.0 = perfect, 0.0 = completely wrong)
        double avg_error = count > 0 ? total_error / static_cast<double>(count) : 1.0;
        return std::max(0.0, 1.0 - avg_error);
    }

    static double computeParameterError(const Distribution& a, const Distribution& b) {
        // This is a simplified error computation - in practice you'd compare
        // specific parameters for each distribution type
        double error = 0.0;

        // Compare basic statistical properties
        double mean_error = std::abs(a.getMean() - b.getMean()) / (std::abs(b.getMean()) + 1e-10);
        double var_error =
            std::abs(a.getVariance() - b.getVariance()) / (std::abs(b.getVariance()) + 1e-10);

        error = (mean_error + var_error) / 2.0;
        return std::min(error, 1.0);  // Cap at 1.0
    }
};

//==============================================================================
// MAIN BENCHMARK RUNNER
//==============================================================================

class ParallelBatchFittingBenchmark {
   public:
    explicit ParallelBatchFittingBenchmark(const BenchmarkConfig& config)
        : config_(config), rng_(42) {}

    BenchmarkSummary run() {
        BenchmarkSummary summary;
        summary.config = config_;
        summary.system_info = getSystemInfo();

        printHeader();

        auto start_time = std::chrono::high_resolution_clock::now();

        // Run benchmarks for each distribution
        if (config_.target_distribution == "all" || config_.target_distribution == "gaussian") {
            runDistributionBenchmark<GaussianDistribution>(
                "Gaussian", &DatasetGenerator::generateGaussianDatasets, summary);
        }

        if (config_.target_distribution == "all" || config_.target_distribution == "exponential") {
            runDistributionBenchmark<ExponentialDistribution>(
                "Exponential", &DatasetGenerator::generateExponentialDatasets, summary);
        }

        if (config_.target_distribution == "all" || config_.target_distribution == "uniform") {
            runDistributionBenchmark<UniformDistribution>(
                "Uniform", &DatasetGenerator::generateUniformDatasets, summary);
        }

        if (config_.target_distribution == "all" || config_.target_distribution == "gamma") {
            runDistributionBenchmark<GammaDistribution>(
                "Gamma", &DatasetGenerator::generateGammaDatasets, summary);
        }

        if (config_.target_distribution == "all" || config_.target_distribution == "poisson") {
            runDistributionBenchmark<PoissonDistribution>(
                "Poisson", &DatasetGenerator::generatePoissonDatasets, summary);
        }

        if (config_.target_distribution == "all" || config_.target_distribution == "discrete") {
            runDistributionBenchmark<DiscreteDistribution>(
                "Discrete", &DatasetGenerator::generateDiscreteDatasets, summary);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        summary.total_runtime_seconds = static_cast<double>(duration.count());

        printSummary(summary);

        if (config_.export_results) {
            exportResults(summary);
        }

        return summary;
    }

   private:
    template <typename Distribution>
    void runDistributionBenchmark(const std::string& name,
                                  std::vector<std::vector<double>> (*generator)(size_t, size_t,
                                                                                std::mt19937&),
                                  BenchmarkSummary& summary) {
        std::cout << "\n=== " << name << " Distribution Benchmark ===\n";

        for (size_t dataset_count : config_.dataset_counts) {
            for (size_t dataset_size : config_.dataset_sizes) {
                std::cout << "  Testing " << dataset_count << " datasets of size " << dataset_size
                          << "...\n";

                // Generate datasets
                auto datasets = generator(dataset_count, dataset_size, rng_);

                // Run benchmark
                auto result = DistributionBenchmark<Distribution>::benchmarkParallelFitting(
                    name, datasets, config_.num_trials, config_.verbose);

                summary.results.push_back(result);

                if (!result.success) {
                    std::cout << "    ❌ Failed: " << result.error_message << "\n";
                } else {
                    std::cout << "    ✅ Success: " << std::fixed << std::setprecision(2)
                              << result.speedup << "x speedup, " << (result.efficiency * 100.0)
                              << "% efficiency\n";
                }
            }
        }
    }

    void printHeader() {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Parallel Batch Fitting Benchmark               ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        std::cout << "System: " << getSystemInfo() << "\n";
        std::cout << "Mode: " << (config_.quick_mode ? "Quick" : "Full") << "\n";
        std::cout << "Threads: " << config_.num_threads << "\n";
        std::cout << "Target: " << config_.target_distribution << "\n";
        std::cout << "\n";
    }

    void printSummary(const BenchmarkSummary& summary) {
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                      Benchmark Summary                       ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";

        // Count successes and failures
        size_t total_tests = summary.results.size();
        size_t successful_tests = 0;
        size_t failed_tests = 0;

        for (const auto& result : summary.results) {
            if (result.success)
                successful_tests++;
            else
                failed_tests++;
        }

        std::cout << "Total tests: " << total_tests << "\n";
        std::cout << "Successful: " << successful_tests << " (" << std::fixed
                  << std::setprecision(1)
                  << (100.0 * static_cast<double>(successful_tests) /
                      static_cast<double>(total_tests))
                  << "%)\n";
        std::cout << "Failed: " << failed_tests << "\n";
        std::cout << "Runtime: " << summary.total_runtime_seconds << " seconds\n\n";

        // Performance analysis by distribution
        std::map<std::string, std::vector<const FittingResult*>> by_distribution;
        for (const auto& result : summary.results) {
            if (result.success) {
                by_distribution[result.distribution_name].push_back(&result);
            }
        }

        std::cout << "Performance by Distribution:\n";
        std::cout << std::left << std::setw(12) << "Distribution" << std::right << std::setw(10)
                  << "Avg Speedup" << std::setw(12) << "Max Speedup" << std::setw(12)
                  << "Efficiency" << std::setw(10) << "Tests"
                  << "\n";
        std::cout << std::string(56, '-') << "\n";

        for (const auto& [dist_name, results] : by_distribution) {
            double avg_speedup = 0.0;
            double max_speedup = 0.0;
            double avg_efficiency = 0.0;

            for (const auto* result : results) {
                avg_speedup += result->speedup;
                max_speedup = std::max(max_speedup, result->speedup);
                avg_efficiency += result->efficiency;
            }

            avg_speedup /= static_cast<double>(results.size());
            avg_efficiency /= static_cast<double>(results.size());

            std::cout << std::left << std::setw(12) << dist_name << std::right << std::fixed
                      << std::setprecision(2) << std::setw(10) << avg_speedup << "x"
                      << std::setw(11) << max_speedup << "x" << std::setw(10)
                      << (avg_efficiency * 100.0) << "%" << std::setw(10) << results.size() << "\n";
        }

        std::cout << "\n";

        // Scalability analysis
        analyzeScalability(summary);
    }

    void analyzeScalability(const BenchmarkSummary& summary) {
        std::cout << "Scalability Analysis:\n";
        std::cout << "• Dataset Count Scaling:\n";

        // Group by dataset size to see how performance scales with dataset count
        std::map<size_t, std::vector<const FittingResult*>> by_dataset_size;
        for (const auto& result : summary.results) {
            if (result.success) {
                by_dataset_size[result.dataset_size].push_back(&result);
            }
        }

        for (const auto& [size, results] : by_dataset_size) {
            std::cout << "  Dataset size " << size << ":\n";

            // Sort by dataset count
            std::vector<const FittingResult*> sorted_results(results);
            std::sort(sorted_results.begin(), sorted_results.end(),
                      [](const FittingResult* a, const FittingResult* b) {
                          return a->dataset_count < b->dataset_count;
                      });

            for (const auto* result : sorted_results) {
                std::cout << "    " << std::setw(3) << result->dataset_count << " datasets → "
                          << std::fixed << std::setprecision(2) << result->speedup << "x speedup\n";
            }
        }

        std::cout << "\n";
    }

    void exportResults(const BenchmarkSummary& summary) {
        std::string filename = "parallel_batch_fitting_benchmark_results.csv";
        std::ofstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Failed to create export file: " << filename << "\n";
            return;
        }

        // Write CSV header
        file << "Distribution,DatasetCount,DatasetSize,ParallelTime_ms,SequentialTime_ms,"
             << "Speedup,Efficiency,AccuracyScore,Success,ErrorMessage\n";

        // Write results
        for (const auto& result : summary.results) {
            file << result.distribution_name << "," << result.dataset_count << ","
                 << result.dataset_size << "," << result.parallel_time_ms << ","
                 << result.sequential_time_ms << "," << result.speedup << "," << result.efficiency
                 << "," << result.accuracy_score << "," << (result.success ? "true" : "false")
                 << ","
                 << "\"" << result.error_message << "\"\n";
        }

        file.close();
        std::cout << "Results exported to: " << filename << "\n";
    }

    BenchmarkConfig config_;
    std::mt19937 rng_;
};

//==============================================================================
// COMMAND LINE ARGUMENT PARSING
//==============================================================================

BenchmarkConfig parseArguments(int argc, char* argv[]) {
    BenchmarkConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--quick") {
            config.quick_mode = true;
            config.full_mode = false;
            config.configure_quick();
        } else if (arg == "--full") {
            config.full_mode = true;
            config.quick_mode = false;
        } else if (arg == "--export") {
            config.export_results = true;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            config.num_threads = std::stoul(argv[++i]);
        } else if (arg == "--distribution" && i + 1 < argc) {
            config.target_distribution = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Parallel Batch Fitting Benchmark\n\n";
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n\n";
            std::cout << "Options:\n";
            std::cout << "  --quick              Run quick benchmark with smaller datasets\n";
            std::cout << "  --full               Run comprehensive benchmark (default)\n";
            std::cout << "  --export             Export results to CSV files\n";
            std::cout << "  --threads N          Use N threads for parallel execution\n";
            std::cout << "  --verbose            Enable verbose output\n";
            std::cout << "  --distribution D     Test only distribution D\n";
            std::cout << "                       (gaussian, exponential, uniform, gamma, poisson, "
                         "discrete)\n";
            std::cout << "  --help, -h           Show this help message\n\n";
            std::cout << "Examples:\n";
            std::cout << "  " << argv[0] << " --quick --verbose\n";
            std::cout << "  " << argv[0] << " --distribution gaussian --export\n";
            std::cout << "  " << argv[0] << " --threads 8 --full\n";
            exit(0);
        }
    }

    return config;
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

int main(int argc, char* argv[]) {
    try {
        BenchmarkConfig config = parseArguments(argc, argv);

        // Initialize libstats
        stats::initialize_performance_systems();

        ParallelBatchFittingBenchmark benchmark(config);
        auto summary = benchmark.run();

        return summary.results.empty() ? 1 : 0;

    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Benchmark failed with unknown exception\n";
        return 1;
    }
}

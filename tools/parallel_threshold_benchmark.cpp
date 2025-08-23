/**
 * @file parallel_threshold_benchmark.cpp
 * @brief Enhanced Benchmark tool for determining dynamic thresholds using PerformanceHistory
 *
 * This tool benchmarks different data sizes to find the optimal thresholds
 * for parallel execution, utilizing adaptive learning from PerformanceHistory.
 */

// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

#include <fstream>
#include <map>
#include <span>
#include <thread>

// Include the distribution headers
#include "../include/core/constants.h"
#include "../include/core/performance_dispatcher.h"
#include "../include/distributions/discrete.h"
#include "../include/distributions/exponential.h"
#include "../include/distributions/gamma.h"
#include "../include/distributions/gaussian.h"
#include "../include/distributions/poisson.h"
#include "../include/distributions/uniform.h"
#include "../include/libstats.h"
#include "tool_utils.h"

using namespace std::chrono;
using namespace stats;
using namespace stats::constants;

// Tool-specific benchmark constants
namespace {
// Benchmark timing constants
constexpr int DEFAULT_RNG_SEED = 42;
constexpr double SPEEDUP_SLOWDOWN_THRESHOLD = 0.5;  // Below this is "extreme slowdown"

// Distribution-specific test parameters
namespace distribution_params {
// Poisson parameters
constexpr double DEFAULT_POISSON_LAMBDA = 3.5;
constexpr int POISSON_TEST_LAMBDA = 3;

// Discrete distribution range
constexpr int DISCRETE_MIN = 0;
constexpr int DISCRETE_MAX = 10;
constexpr int DISCRETE_TEST_MIN = -2;
constexpr int DISCRETE_TEST_MAX = 12;

// Uniform distribution range
constexpr double UNIFORM_MIN = 0.0;
constexpr double UNIFORM_MAX = 1.0;
constexpr double UNIFORM_TEST_MIN = -0.5;
constexpr double UNIFORM_TEST_MAX = 1.5;

// Gaussian distribution parameters
constexpr double GAUSSIAN_MEAN = 0.0;
constexpr double GAUSSIAN_STDDEV = 1.0;
constexpr double GAUSSIAN_TEST_STDDEV = 2.0;  // Wider range for testing

// Exponential distribution parameter
constexpr double EXPONENTIAL_LAMBDA = 1.0;
constexpr double EXPONENTIAL_TEST_LAMBDA = 0.5;

// Gamma distribution parameters
constexpr double GAMMA_ALPHA = 2.0;
constexpr double GAMMA_BETA = 1.0;
constexpr double GAMMA_TEST_ALPHA = 1.5;
constexpr double GAMMA_TEST_BETA = 2.0;
}  // namespace distribution_params

// Output file configuration
constexpr const char* RESULTS_CSV_FILENAME = "parallel_threshold_benchmark_results.csv";
}  // namespace

struct ToolBenchmarkResult {
    std::size_t data_size;
    std::string distribution_type;
    std::string operation_type;
    double serial_time_us;
    double parallel_time_us;
    double simd_time_us;
    double parallel_speedup;
    double simd_speedup;
    bool parallel_beneficial;
};

class ParallelThresholdBenchmark {
   private:
    std::mt19937 gen_;
    std::vector<ToolBenchmarkResult> results_;
    std::vector<std::size_t> test_sizes_;

    void initializeTestSizes(bool include_large) {
        // Base test sizes - start small and work up to 524K elements
        test_sizes_ = {64,   128,   256,   512,   1024,   2048,   4096,
                       8192, 16384, 32768, 65536, 131072, 262144, 524288};

        // Add the large (and slow) test sizes only if requested
        if (include_large) {
            test_sizes_.push_back(1048576);  // 1M elements
            test_sizes_.push_back(2097152);  // 2M elements
        }
    }

    // Number of iterations for timing stability
    static constexpr int TIMING_ITERATIONS = 10;
    static constexpr int WARMUP_ITERATIONS = 3;

   public:
    ParallelThresholdBenchmark(bool include_large = false) : gen_(DEFAULT_RNG_SEED) {
        initializeTestSizes(include_large);
    }

    void runAllBenchmarks() {
        using namespace stats::tools;

        // Initialize performance systems for accurate threshold determination
        stats::initialize_performance_systems();

        // Display tool header with system information
        system_info::displayToolHeader(
            "Parallel Threshold Benchmark",
            "Distribution-specific threshold optimization with adaptive learning");

        benchmarkUniformDistribution();
        benchmarkPoissonDistribution();
        benchmarkDiscreteDistribution();
        benchmarkGaussianDistribution();
        benchmarkExponentialDistribution();
        benchmarkGammaDistribution();

        analyzeResults();
        saveResults();
    }

   private:
    void benchmarkUniformDistribution() {
        using namespace stats::tools;

        display::subsectionHeader("Uniform Distribution Benchmark");
        auto uniform = stats::UniformDistribution::create(distribution_params::UNIFORM_MIN,
                                                          distribution_params::UNIFORM_MAX)
                           .value;

        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;

            // Generate test data
            std::vector<double> test_data(size);
            std::uniform_real_distribution<double> dis(distribution_params::UNIFORM_TEST_MIN,
                                                       distribution_params::UNIFORM_TEST_MAX);
            for (auto& val : test_data) {
                val = dis(gen_);
            }

            // Benchmark PDF
            auto pdf_result = benchmarkOperation(uniform, test_data, "PDF", "Uniform");
            results_.push_back(pdf_result);

            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(uniform, test_data, "LogPDF", "Uniform");
            results_.push_back(logpdf_result);

            // Benchmark CDF
            auto cdf_result = benchmarkOperation(uniform, test_data, "CDF", "Uniform");
            results_.push_back(cdf_result);

            std::cout << " ✓\n";
        }
    }

    void benchmarkPoissonDistribution() {
        using namespace stats::tools;

        display::subsectionHeader("Poisson Distribution Benchmark");
        auto poisson =
            stats::PoissonDistribution::create(distribution_params::DEFAULT_POISSON_LAMBDA).value;

        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;

            // Generate test data (integer values for Poisson)
            std::vector<double> test_data(size);
            std::poisson_distribution<int> dis(distribution_params::POISSON_TEST_LAMBDA);
            for (auto& val : test_data) {
                val = static_cast<double>(dis(gen_));
            }

            // Benchmark PDF (PMF)
            auto pdf_result = benchmarkOperation(poisson, test_data, "PDF", "Poisson");
            results_.push_back(pdf_result);

            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(poisson, test_data, "LogPDF", "Poisson");
            results_.push_back(logpdf_result);

            // Benchmark CDF
            auto cdf_result = benchmarkOperation(poisson, test_data, "CDF", "Poisson");
            results_.push_back(cdf_result);

            std::cout << " ✓\n";
        }
    }

    void benchmarkDiscreteDistribution() {
        using namespace stats::tools;

        display::subsectionHeader("Discrete Distribution Benchmark");
        auto discrete = stats::DiscreteDistribution::create(distribution_params::DISCRETE_MIN,
                                                            distribution_params::DISCRETE_MAX)
                            .value;

        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;

            // Generate test data (integer values)
            std::vector<double> test_data(size);
            std::uniform_int_distribution<int> dis(distribution_params::DISCRETE_TEST_MIN,
                                                   distribution_params::DISCRETE_TEST_MAX);
            for (auto& val : test_data) {
                val = static_cast<double>(dis(gen_));
            }

            // Benchmark PDF (PMF)
            auto pdf_result = benchmarkOperation(discrete, test_data, "PDF", "Discrete");
            results_.push_back(pdf_result);

            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(discrete, test_data, "LogPDF", "Discrete");
            results_.push_back(logpdf_result);

            // Benchmark CDF
            auto cdf_result = benchmarkOperation(discrete, test_data, "CDF", "Discrete");
            results_.push_back(cdf_result);

            std::cout << " ✓\n";
        }
    }

    void benchmarkGaussianDistribution() {
        using namespace stats::tools;

        display::subsectionHeader("Gaussian Distribution Benchmark");
        auto gaussian = stats::GaussianDistribution::create(distribution_params::GAUSSIAN_MEAN,
                                                            distribution_params::GAUSSIAN_STDDEV)
                            .value;

        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;

            // Generate test data (normal distribution values)
            std::vector<double> test_data(size);
            std::normal_distribution<double> dis(
                distribution_params::GAUSSIAN_MEAN,
                distribution_params::GAUSSIAN_TEST_STDDEV);  // Wider range
            for (auto& val : test_data) {
                val = dis(gen_);
            }

            // Benchmark PDF
            auto pdf_result = benchmarkOperation(gaussian, test_data, "PDF", "Gaussian");
            results_.push_back(pdf_result);

            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(gaussian, test_data, "LogPDF", "Gaussian");
            results_.push_back(logpdf_result);

            // Benchmark CDF
            auto cdf_result = benchmarkOperation(gaussian, test_data, "CDF", "Gaussian");
            results_.push_back(cdf_result);

            std::cout << " ✓\n";
        }
    }

    void benchmarkExponentialDistribution() {
        using namespace stats::tools;

        display::subsectionHeader("Exponential Distribution Benchmark");
        auto exponential =
            stats::ExponentialDistribution::create(distribution_params::EXPONENTIAL_LAMBDA).value;

        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;

            // Generate test data (exponential distribution values)
            std::vector<double> test_data(size);
            std::exponential_distribution<double> dis(distribution_params::EXPONENTIAL_TEST_LAMBDA);
            for (auto& val : test_data) {
                val = dis(gen_);
            }

            // Benchmark PDF
            auto pdf_result = benchmarkOperation(exponential, test_data, "PDF", "Exponential");
            results_.push_back(pdf_result);

            // Benchmark LogPDF
            auto logpdf_result =
                benchmarkOperation(exponential, test_data, "LogPDF", "Exponential");
            results_.push_back(logpdf_result);

            // Benchmark CDF
            auto cdf_result = benchmarkOperation(exponential, test_data, "CDF", "Exponential");
            results_.push_back(cdf_result);

            std::cout << " ✓\n";
        }
    }

    void benchmarkGammaDistribution() {
        using namespace stats::tools;

        display::subsectionHeader("Gamma Distribution Benchmark");
        auto gamma = stats::GammaDistribution::create(distribution_params::GAMMA_ALPHA,
                                                      distribution_params::GAMMA_BETA)
                         .value;

        for (auto size : test_sizes_) {
            std::cout << "  Testing size: " << size << std::flush;

            // Generate test data (gamma distribution values)
            std::vector<double> test_data(size);
            std::gamma_distribution<double> dis(distribution_params::GAMMA_TEST_ALPHA,
                                                distribution_params::GAMMA_TEST_BETA);
            for (auto& val : test_data) {
                val = dis(gen_);
            }

            // Benchmark PDF
            auto pdf_result = benchmarkOperation(gamma, test_data, "PDF", "Gamma");
            results_.push_back(pdf_result);

            // Benchmark LogPDF
            auto logpdf_result = benchmarkOperation(gamma, test_data, "LogPDF", "Gamma");
            results_.push_back(logpdf_result);

            // Benchmark CDF
            auto cdf_result = benchmarkOperation(gamma, test_data, "CDF", "Gamma");
            results_.push_back(cdf_result);

            std::cout << " ✓\n";
        }
    }

    template <typename Distribution>
    ToolBenchmarkResult benchmarkOperation(const Distribution& dist,
                                           const std::vector<double>& test_data,
                                           const std::string& operation,
                                           const std::string& dist_type) {
        ToolBenchmarkResult result;
        result.data_size = test_data.size();
        result.distribution_type = dist_type;
        result.operation_type = operation;

        std::vector<double> results_buffer(test_data.size());
        std::span<const double> input_span(test_data);
        std::span<double> output_span(results_buffer);

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "serial");
        }

        // Benchmark Serial (using SIMD batch operations)
        auto serial_start = high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "simd");
        }
        auto serial_end = high_resolution_clock::now();
        result.simd_time_us =
            static_cast<double>(duration_cast<microseconds>(serial_end - serial_start).count()) /
            static_cast<double>(TIMING_ITERATIONS);

        // Benchmark True Serial (element by element)
        auto true_serial_start = high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "serial");
        }
        auto true_serial_end = high_resolution_clock::now();
        result.serial_time_us =
            static_cast<double>(
                duration_cast<microseconds>(true_serial_end - true_serial_start).count()) /
            static_cast<double>(TIMING_ITERATIONS);

        // Benchmark Parallel
        auto parallel_start = high_resolution_clock::now();
        for (int i = 0; i < TIMING_ITERATIONS; ++i) {
            performOperation(dist, input_span, output_span, operation, "parallel");
        }
        auto parallel_end = high_resolution_clock::now();
        result.parallel_time_us =
            static_cast<double>(
                duration_cast<microseconds>(parallel_end - parallel_start).count()) /
            static_cast<double>(TIMING_ITERATIONS);

        // Calculate speedups
        result.parallel_speedup = result.simd_time_us / result.parallel_time_us;
        result.simd_speedup = result.serial_time_us / result.simd_time_us;
        result.parallel_beneficial = result.parallel_speedup > 1.0;

        return result;
    }

    template <typename Distribution>
    void performOperation(const Distribution& dist, std::span<const double> input,
                          std::span<double> output, const std::string& operation,
                          const std::string& method) {
        if (method == "serial") {
            // True serial: element by element
            if (operation == "PDF") {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = dist.getProbability(input[i]);
                }
            } else if (operation == "LogPDF") {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = dist.getLogProbability(input[i]);
                }
            } else if (operation == "CDF") {
                for (size_t i = 0; i < input.size(); ++i) {
                    output[i] = dist.getCumulativeProbability(input[i]);
                }
            }
        } else if (method == "simd") {
            // SIMD batch operations using explicit strategy to ensure SIMD benchmarking
            if (operation == "PDF") {
                dist.getProbabilityWithStrategy(input, output,
                                                stats::performance::Strategy::SIMD_BATCH);
            } else if (operation == "LogPDF") {
                dist.getLogProbabilityWithStrategy(input, output,
                                                   stats::performance::Strategy::SIMD_BATCH);
            } else if (operation == "CDF") {
                dist.getCumulativeProbabilityWithStrategy(input, output,
                                                          stats::performance::Strategy::SIMD_BATCH);
            }
        } else if (method == "parallel") {
            // Parallel operations using explicit strategy to ensure parallel benchmarking
            if (operation == "PDF") {
                dist.getProbabilityWithStrategy(input, output,
                                                stats::performance::Strategy::PARALLEL_SIMD);
            } else if (operation == "LogPDF") {
                dist.getLogProbabilityWithStrategy(input, output,
                                                   stats::performance::Strategy::PARALLEL_SIMD);
            } else if (operation == "CDF") {
                dist.getCumulativeProbabilityWithStrategy(
                    input, output, stats::performance::Strategy::PARALLEL_SIMD);
            }
        }
    }

    void analyzeResults() {
        std::cout << "\n=== Analysis Results ===\n";

        // Group results by distribution and operation
        std::map<std::string, std::vector<ToolBenchmarkResult*>> grouped_results;
        for (auto& result : results_) {
            std::string key = result.distribution_type + "_" + result.operation_type;
            grouped_results[key].push_back(&result);
        }

        std::cout << std::left << std::setw(20) << "Dist_Op" << std::setw(10) << "Size"
                  << std::setw(12) << "Serial(μs)" << std::setw(12) << "SIMD(μs)" << std::setw(12)
                  << "Parallel(μs)" << std::setw(12) << "S-Speedup" << std::setw(12) << "P-Speedup"
                  << std::setw(12) << "Beneficial?"
                  << "\n";
        std::cout << std::string(120, '-') << "\n";

        for (const auto& [key, results] : grouped_results) {
            std::size_t beneficial_threshold = SIZE_MAX;

            for (const auto* result : results) {
                std::cout << std::left << std::setw(20) << key << std::setw(10) << result->data_size
                          << std::setw(12) << std::fixed << std::setprecision(1)
                          << result->serial_time_us << std::setw(12) << std::fixed
                          << std::setprecision(1) << result->simd_time_us << std::setw(12)
                          << std::fixed << std::setprecision(1) << result->parallel_time_us
                          << std::setw(12) << std::fixed << std::setprecision(2)
                          << result->simd_speedup << std::setw(12) << std::fixed
                          << std::setprecision(2) << result->parallel_speedup << std::setw(12)
                          << (result->parallel_beneficial ? "YES" : "NO") << "\n";

                if (result->parallel_beneficial && beneficial_threshold == SIZE_MAX) {
                    beneficial_threshold = result->data_size;
                }
            }

            std::cout << "  → Recommended threshold for " << key << ": ";
            if (beneficial_threshold != SIZE_MAX) {
                std::cout << beneficial_threshold << " elements\n";
            } else {
                std::cout << "NEVER (parallel not beneficial)\n";
            }
            std::cout << "\n";
        }

        // Find extreme slowdowns
        std::cout << "\n=== Extreme Slowdowns (Speedup < " << SPEEDUP_SLOWDOWN_THRESHOLD
                  << ") ===\n";
        bool found_extreme = false;
        for (const auto& result : results_) {
            if (result.parallel_speedup < SPEEDUP_SLOWDOWN_THRESHOLD) {
                std::cout << result.distribution_type << " " << result.operation_type << " at size "
                          << result.data_size << ": " << result.parallel_speedup << "x speedup ("
                          << (1.0 / result.parallel_speedup) << "x slowdown)\n";
                found_extreme = true;
            }
        }
        if (!found_extreme) {
            std::cout << "No extreme slowdowns found.\n";
        }
    }

    void saveResults() {
        std::ofstream csv_file(RESULTS_CSV_FILENAME);
        csv_file << "Distribution,Operation,DataSize,SerialTime_us,SIMDTime_us,ParallelTime_us,"
                    "SIMDSpeedup,ParallelSpeedup,ParallelBeneficial\n";

        for (const auto& result : results_) {
            csv_file << result.distribution_type << "," << result.operation_type << ","
                     << result.data_size << "," << result.serial_time_us << ","
                     << result.simd_time_us << "," << result.parallel_time_us << ","
                     << result.simd_speedup << "," << result.parallel_speedup << ","
                     << (result.parallel_beneficial ? "true" : "false") << "\n";
        }

        std::cout << "\n=== Results saved to parallel_threshold_benchmark_results.csv ===\n";
    }
};

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  -l, --large    Include large dataset tests (1M and 2M elements)\n";
    std::cout << "  -h, --help     Show this help message\n";
    std::cout << "\nDefault: Tests up to 524K elements only (faster execution)\n";
    std::cout << "With --large: Tests up to 2M elements (slower but more comprehensive)\n";
}

int main(int argc, char* argv[]) {
    bool include_large = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-l" || arg == "--large") {
            include_large = true;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    try {
        ParallelThresholdBenchmark benchmark(include_large);

        // Display test configuration
        std::cout << "\n=== Test Configuration ===\n";
        std::cout << "Large dataset tests (1M-2M elements): "
                  << (include_large ? "ENABLED" : "DISABLED") << "\n";
        if (!include_large) {
            std::cout << "To enable large tests, use: " << argv[0] << " --large\n";
        }
        std::cout << "\n";

        benchmark.runAllBenchmarks();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}

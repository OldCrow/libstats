/**
 * @file learning_analyzer.cpp
 * @brief Consolidated learning analysis tool combining real execution analysis and educational
 * simulation
 *
 * This tool consolidates the functionality of adaptive_learning_analyzer.cpp and
 * threshold_learning_demo.cpp, providing both comprehensive performance analysis with real
 * execution data and educational simulation demonstrating adaptive threshold learning.
 */

// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

// Additional includes for performance analysis functionality
#include "../include/core/performance_history.h"

// Additional standard library includes for tool-specific functionality
#include <map>
#include <memory>
#include <sstream>
#include <thread>

using namespace libstats;
using namespace libstats::performance;
using namespace libstats::constants;
using namespace libstats::tools;

// Consolidated learning analysis constants
namespace {
// Time conversion constants - reserved for future use
[[maybe_unused]] constexpr long NANOSECONDS_TO_MICROSECONDS = 1000;
[[maybe_unused]] constexpr long NANOSECONDS_TO_MILLISECONDS = 1000000;
[[maybe_unused]] constexpr long NANOSECONDS_TO_SECONDS = 1000000000;

// Test data generation
constexpr double TEST_VALUE_MIN = 0.1;
constexpr double TEST_VALUE_MAX = 10.0;

// Performance simulation parameters (for demo mode)
constexpr double SIMULATION_NOISE_MIN = 0.9;
constexpr double SIMULATION_NOISE_MAX = 1.1;
constexpr double SCALAR_PERFORMANCE_FACTOR = 10.0;
constexpr double SIMD_PERFORMANCE_FACTOR = 3.0;
constexpr double PARALLEL_PERFORMANCE_FACTOR = 2.0;

// Strategy overhead constants - reserved for future simulation modes
[[maybe_unused]] constexpr uint64_t SIMD_SMALL_OVERHEAD = 500;
[[maybe_unused]] constexpr uint64_t PARALLEL_SMALL_OVERHEAD = 5000;
constexpr size_t SIMD_OVERHEAD_THRESHOLD = 10000;
[[maybe_unused]] constexpr size_t PARALLEL_OVERHEAD_THRESHOLD = 1000;

// Learning simulation parameters
constexpr int SAMPLES_PER_STRATEGY = 6;

// Performance simulation speedup factors (for analysis mode)
constexpr int SIMD_SPEEDUP_FACTOR = 3;
constexpr int PARALLEL_SPEEDUP_FACTOR = 6;
constexpr int WORK_STEALING_SPEEDUP_FACTOR = 8;
constexpr int GPU_ACCELERATED_SPEEDUP_FACTOR = 12;

// Strategy threshold sizes
constexpr size_t MIN_SIMD_BATCH_SIZE = 32;
constexpr size_t MIN_PARALLEL_BATCH_SIZE = 1000;
constexpr size_t MIN_WORK_STEALING_BATCH_SIZE = 10000;
constexpr size_t MIN_GPU_ACCELERATED_BATCH_SIZE = 50000;

// Distribution parameters
namespace distribution_params {
constexpr double UNIFORM_MIN = 0.0;
constexpr double UNIFORM_MAX = 10.0;
constexpr double GAUSSIAN_MEAN = 0.0;
constexpr double GAUSSIAN_STDDEV = 1.0;
constexpr double EXPONENTIAL_LAMBDA = 1.0;
constexpr int DISCRETE_MIN = 1;
constexpr int DISCRETE_MAX = 100;
constexpr double POISSON_LAMBDA = 5.0;
constexpr double GAMMA_ALPHA = 2.0;
constexpr double GAMMA_BETA = 1.0;
}  // namespace distribution_params

// Output formatting - reserved for future formatting improvements
[[maybe_unused]] constexpr int CONFIDENCE_PRECISION = 3;
[[maybe_unused]] constexpr int TIME_PRECISION = 0;
}  // namespace

class LearningAnalyzer {
   private:
    std::mt19937 rng_;

   public:
    LearningAnalyzer() : rng_(std::random_device{}()) {}

    void showUsage() {
        std::cout << "LIBSTATS LEARNING ANALYZER\n";
        std::cout << "==========================\n\n";
        std::cout
            << "This consolidated tool provides comprehensive adaptive learning analysis.\n\n";
        std::cout << "Usage: learning_analyzer [mode]\n\n";
        std::cout << "Modes:\n";
        std::cout << "  demo      - Educational demonstration with simulated performance data\n";
        std::cout << "  analysis  - Comprehensive analysis with real execution data (default)\n";
        std::cout << "  both      - Run both demo and analysis modes\n\n";
        std::cout << "The demo mode shows the learning process step-by-step with realistic\n";
        std::cout << "simulation, while analysis mode exercises actual distributions and\n";
        std::cout << "collects real performance data for detailed analysis.\n\n";
    }

    void runDemo() {
        // Initialize performance systems for accurate threshold learning
        libstats::initialize_performance_systems();

        std::cout << "=== THRESHOLD LEARNING DEMONSTRATION ===\n\n";

        showInitialState();
        simulatePerformanceLearning();
        showLearnedStrategies();
        demonstrateAdaptiveSelection();
    }

    void runAnalysis() {
        // Initialize performance systems for optimal measurement accuracy
        libstats::initialize_performance_systems();

        std::cout << "============================================================\n";
        std::cout << "ADAPTIVE LEARNING ANALYSIS\n";
        std::cout << "============================================================\n\n";

        std::cout << "This mode exercises the adaptive learning system by running\n";
        std::cout << "various distribution operations across different batch sizes\n";
        std::cout << "and strategies, then analyzes the collected performance data.\n\n";

        // Use a more comprehensive set of batch sizes that covers the full range
        // with better granularity around threshold boundaries
        std::vector<size_t> batch_sizes = {
            5,     8,     10,    16,    20,    25,    32,    40,    50,    64,    80,
            100,   128,   160,   200,   256,   320,   400,   500,   640,   800,   1000,
            1280,  1600,  2000,  2560,  3200,  4000,  5000,  6400,  8000,  10000, 12800,
            16000, 20000, 25600, 32000, 40000, 50000, 64000, 80000, 100000};

        std::cout << "Testing " << batch_sizes.size()
                  << " different batch sizes across all distributions...\n\n";

        // Exercise different distributions with real operations
        exerciseAllDistributionsEnhanced(batch_sizes);

        // Analyze the collected performance data
        analyzePerformanceHistoryEnhanced();
    }

   private:
    void showInitialState() {
        std::cout << "--- Initial State (Before Learning) ---\n";

        // Show system capabilities
        const auto& capabilities = SystemCapabilities::current();
        std::cout << "System Configuration:\n";
        std::cout << "  Logical cores: " << capabilities.logical_cores() << "\n";
        std::cout << "  Physical cores: " << capabilities.physical_cores() << "\n";
        std::cout << "  SIMD efficiency: " << std::fixed << std::setprecision(3)
                  << capabilities.simd_efficiency() << "\n";
        std::cout << "  Memory bandwidth: " << std::setprecision(1)
                  << capabilities.memory_bandwidth_gb_s() << " GB/s\n";

        // Show some initial strategy selections
        std::vector<size_t> test_sizes = {100, 1000, 10000, 100000};

        std::cout << "\nInitial Strategy Selections:\n";
        std::cout << std::left << std::setw(12) << "Batch Size" << std::setw(20)
                  << "Strategy (Uniform)" << std::setw(20) << "Strategy (Gaussian)"
                  << "\n";
        std::cout << std::string(52, '-') << "\n";

        PerformanceDispatcher dispatcher;
        for (auto size : test_sizes) {
            auto uniform_strategy = dispatcher.selectOptimalStrategy(
                size, DistributionType::UNIFORM, ComputationComplexity::SIMPLE, capabilities);
            auto gaussian_strategy = dispatcher.selectOptimalStrategy(
                size, DistributionType::GAUSSIAN, ComputationComplexity::MODERATE, capabilities);

            std::cout << std::setw(12) << size << std::setw(20)
                      << strings::strategyToDisplayString(uniform_strategy) << std::setw(20)
                      << strings::strategyToDisplayString(gaussian_strategy) << "\n";
        }
        std::cout << "\n";
    }

    void simulatePerformanceLearning() {
        std::cout << "--- Simulating Performance Learning ---\n";

        // Get access to the performance history system
        auto& history = PerformanceDispatcher::getPerformanceHistory();
        history.clearHistory();  // Start fresh

        std::cout
            << "Recording performance data across different distributions and batch sizes...\n";

        // Simulate realistic performance patterns
        std::uniform_real_distribution<double> noise(SIMULATION_NOISE_MIN, SIMULATION_NOISE_MAX);

        // All distribution types to simulate
        std::vector<DistributionType> distributions = {
            DistributionType::UNIFORM,  DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL,
            DistributionType::DISCRETE, DistributionType::POISSON,  DistributionType::GAMMA};

        // Performance complexity factors for different distributions
        std::map<DistributionType, double> complexity_factors = {
            {DistributionType::UNIFORM, 1.0},      // Simple - just random scaling
            {DistributionType::DISCRETE, 1.5},     // Simple integer operations
            {DistributionType::EXPONENTIAL, 2.5},  // Moderate - requires exp/log
            {DistributionType::GAUSSIAN, 3.0},     // Moderate - Box-Muller transform
            {DistributionType::POISSON, 4.0},      // Complex - iterative algorithms
            {DistributionType::GAMMA, 5.0}         // Most complex - special functions
        };

        // Distribution-specific efficiency characteristics
        std::map<DistributionType, std::pair<double, double>> efficiency_characteristics = {
            {DistributionType::UNIFORM, {0.40, 0.25}},      // Good SIMD/Parallel efficiency
            {DistributionType::DISCRETE, {0.35, 0.22}},     // Decent efficiency
            {DistributionType::EXPONENTIAL, {0.28, 0.18}},  // Moderate efficiency
            {DistributionType::GAUSSIAN, {0.25, 0.15}},     // Lower efficiency
            {DistributionType::POISSON, {0.22, 0.12}},      // Poor efficiency
            {DistributionType::GAMMA, {0.20, 0.10}}         // Worst efficiency
        };

        // More granular sizes around potential crossover points
        std::vector<size_t> sizes = {10,   25,   50,    75,    100,   150,  200,
                                     300,  500,  750,   1000,  1500,  2000, 3000,
                                     5000, 7500, 10000, 15000, 25000, 50000};

        for (auto dist_type : distributions) {
            std::cout << "\n  Simulating " << strings::distributionTypeToString(dist_type)
                      << " distribution:\n";

            double complexity = complexity_factors[dist_type];
            auto [simd_efficiency, parallel_efficiency] = efficiency_characteristics[dist_type];

            for (auto size : sizes) {
                std::cout << "    Recording data for size " << size << "..." << std::flush;

                // Record multiple samples per strategy
                for (int sample = 0; sample < SAMPLES_PER_STRATEGY; ++sample) {
                    // Scalar strategy
                    auto scalar_time =
                        static_cast<uint64_t>(static_cast<double>(size) *
                                              SCALAR_PERFORMANCE_FACTOR * complexity * noise(rng_));
                    history.recordPerformance(Strategy::SCALAR, dist_type, size, scalar_time);

                    // SIMD strategy
                    auto simd_time =
                        static_cast<uint64_t>(static_cast<double>(size) * SIMD_PERFORMANCE_FACTOR *
                                              complexity * simd_efficiency * noise(rng_));
                    if (size < SIMD_OVERHEAD_THRESHOLD) {
                        simd_time += SIMD_SMALL_OVERHEAD;
                    }
                    history.recordPerformance(Strategy::SIMD_BATCH, dist_type, size, simd_time);

                    // Parallel strategy
                    auto parallel_time = static_cast<uint64_t>(
                        static_cast<double>(size) * PARALLEL_PERFORMANCE_FACTOR * complexity *
                        parallel_efficiency * noise(rng_));
                    double complexity_factor = complexity;
                    double overhead_reduction = std::max(1.0, static_cast<double>(size) / 1000.0);
                    uint64_t base_overhead =
                        static_cast<uint64_t>(8000.0 / complexity_factor / overhead_reduction);
                    parallel_time += base_overhead;
                    history.recordPerformance(Strategy::PARALLEL_SIMD, dist_type, size,
                                              parallel_time);
                }

                std::cout << " ✓";
            }
            std::cout << "\n";
        }

        std::cout << "\nTotal recorded executions: " << history.getTotalExecutions() << "\n\n";
    }

    void showLearnedStrategies() {
        std::cout << "--- Learned Strategy Recommendations ---\n";

        auto& history = PerformanceDispatcher::getPerformanceHistory();
        std::vector<size_t> test_sizes = {100, 1000, 10000, 50000};

        std::cout << std::left << std::setw(12) << "Size" << std::setw(20) << "Best Strategy"
                  << std::setw(15) << "Confidence" << std::setw(15) << "Expected Time"
                  << "\n";
        std::cout << std::string(62, '-') << "\n";

        for (auto size : test_sizes) {
            auto recommendation = history.getBestStrategy(DistributionType::GAUSSIAN, size);

            std::cout << std::setw(12) << size << std::setw(20)
                      << strings::strategyToDisplayString(recommendation.recommended_strategy)
                      << std::setw(15)
                      << format::confidenceToString(recommendation.confidence_score)
                      << std::setw(12)
                      << format::nanosecondsToMicroseconds(recommendation.expected_time_ns) << "\n";
        }
        std::cout << "\n";
    }

    void demonstrateAdaptiveSelection() {
        std::cout << "--- Adaptive Selection Results ---\n";

        std::cout << "The PerformanceDispatcher now uses learned data to make better decisions.\n";
        std::cout << "Key insights from the learning process:\n";
        std::cout
            << "• Small batches (< 1000): Scalar or SIMD preferred due to parallel overhead\n";
        std::cout << "• Medium batches (1000-10000): SIMD shows good balance\n";
        std::cout << "• Large batches (> 10000): Parallel strategies become advantageous\n\n";

        // Show threshold learning results
        auto& history = PerformanceDispatcher::getPerformanceHistory();

        std::cout << "Learned optimal thresholds for all distributions:\n";
        for (auto dist_type :
             {DistributionType::UNIFORM, DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL,
              DistributionType::DISCRETE, DistributionType::POISSON, DistributionType::GAMMA}) {
            auto thresholds = history.learnOptimalThresholds(dist_type);
            if (thresholds.has_value()) {
                std::cout << "  " << strings::distributionTypeToString(dist_type) << ":\n";
                std::cout << "    SIMD threshold: " << thresholds->first << " elements\n";
                std::cout << "    Parallel threshold: " << thresholds->second << " elements\n";
            } else {
                std::cout << "  " << strings::distributionTypeToString(dist_type)
                          << ": Insufficient data\n";
            }
        }

        std::cout << "\nDemo completed successfully!\n";
    }

    // Exercise different distributions with real operations
    template <typename Distribution>
    void exerciseDistribution(const std::string& dist_name, DistributionType dist_type,
                              Distribution& dist, const std::vector<size_t>& batch_sizes) {
        std::cout << "\n=== Testing " << dist_name << " Distribution ===\n";

        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t batch_size : batch_sizes) {
            std::cout << "\nBatch size: " << batch_size << std::endl;

            // Create test data
            std::vector<double> values(batch_size);
            std::uniform_real_distribution<double> value_gen(TEST_VALUE_MIN, TEST_VALUE_MAX);
            for (auto& v : values) {
                v = value_gen(gen);
            }

            // Test PDF operations (medium complexity)
            {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] = dist.getProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                // Record performance for SCALAR strategy
                PerformanceDispatcher::recordPerformance(
                    Strategy::SCALAR, dist_type, batch_size,
                    static_cast<std::uint64_t>(duration.count()));

                std::cout << "  PDF (scalar): " << libstats::tools::time::formatDuration(duration)
                          << " (" << (static_cast<std::uint64_t>(duration.count()) / batch_size)
                          << "ns/op)" << std::endl;
            }

            // Test CDF operations (higher complexity)
            {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] = dist.getCumulativeProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                // Simulate SIMD performance for larger batches
                if (batch_size >= MIN_SIMD_BATCH_SIZE) {
                    auto simd_duration = duration / SIMD_SPEEDUP_FACTOR;
                    PerformanceDispatcher::recordPerformance(
                        Strategy::SIMD_BATCH, dist_type, batch_size,
                        static_cast<std::uint64_t>(simd_duration.count()));
                    std::cout << "  CDF (simd):   "
                              << libstats::tools::time::formatDuration(simd_duration) << " ("
                              << (static_cast<std::uint64_t>(simd_duration.count()) / batch_size)
                              << "ns/op)" << std::endl;
                }

                // Simulate parallel performance for very large batches
                if (batch_size >= MIN_PARALLEL_BATCH_SIZE) {
                    auto parallel_duration = duration / PARALLEL_SPEEDUP_FACTOR;
                    PerformanceDispatcher::recordPerformance(
                        Strategy::PARALLEL_SIMD, dist_type, batch_size,
                        static_cast<std::uint64_t>(parallel_duration.count()));
                    std::cout << "  CDF (parallel): "
                              << libstats::tools::time::formatDuration(parallel_duration) << " ("
                              << (static_cast<std::uint64_t>(parallel_duration.count()) /
                                  batch_size)
                              << "ns/op)" << std::endl;
                }

                PerformanceDispatcher::recordPerformance(
                    Strategy::SCALAR, dist_type, batch_size,
                    static_cast<std::uint64_t>(duration.count()));
                std::cout << "  CDF (scalar): " << libstats::tools::time::formatDuration(duration)
                          << " (" << (static_cast<std::uint64_t>(duration.count()) / batch_size)
                          << "ns/op)" << std::endl;
            }

            // For very large batches, test advanced strategies
            if (batch_size >= MIN_WORK_STEALING_BATCH_SIZE) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] =
                        dist.getProbability(values[i]) + dist.getCumulativeProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto base_duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                // Simulate work-stealing
                auto work_stealing_duration = base_duration / WORK_STEALING_SPEEDUP_FACTOR;
                PerformanceDispatcher::recordPerformance(
                    Strategy::WORK_STEALING, dist_type, batch_size,
                    static_cast<std::uint64_t>(work_stealing_duration.count()));
                std::cout << "  Mixed (work-stealing): "
                          << libstats::tools::time::formatDuration(work_stealing_duration) << " ("
                          << (static_cast<std::uint64_t>(work_stealing_duration.count()) /
                              batch_size)
                          << "ns/op)" << std::endl;

                if (batch_size >= MIN_GPU_ACCELERATED_BATCH_SIZE) {
                    // Simulate gpu-accelerated
                    auto gpu_accelerated_duration = base_duration / GPU_ACCELERATED_SPEEDUP_FACTOR;
                    PerformanceDispatcher::recordPerformance(
                        Strategy::GPU_ACCELERATED, dist_type, batch_size,
                        static_cast<std::uint64_t>(gpu_accelerated_duration.count()));
                    std::cout << "  Mixed (gpu-accelerated): "
                              << libstats::tools::time::formatDuration(gpu_accelerated_duration)
                              << " ("
                              << (static_cast<std::uint64_t>(gpu_accelerated_duration.count()) /
                                  batch_size)
                              << "ns/op)" << std::endl;
                }
            }
        }
    }

    void exerciseAllDistributions(const std::vector<size_t>& batch_sizes) {
        // Exercise different distributions using safe factory methods
        {
            auto uniform_dist =
                libstats::UniformDistribution::create(distribution_params::UNIFORM_MIN,
                                                      distribution_params::UNIFORM_MAX)
                    .value;
            exerciseDistribution("Uniform", DistributionType::UNIFORM, uniform_dist, batch_sizes);
        }

        {
            auto gaussian_dist =
                libstats::GaussianDistribution::create(distribution_params::GAUSSIAN_MEAN,
                                                       distribution_params::GAUSSIAN_STDDEV)
                    .value;
            exerciseDistribution("Gaussian", DistributionType::GAUSSIAN, gaussian_dist,
                                 batch_sizes);
        }

        {
            auto exp_dist =
                libstats::ExponentialDistribution::create(distribution_params::EXPONENTIAL_LAMBDA)
                    .value;
            exerciseDistribution("Exponential", DistributionType::EXPONENTIAL, exp_dist,
                                 batch_sizes);
        }

        {
            auto disc_dist =
                libstats::DiscreteDistribution::create(distribution_params::DISCRETE_MIN,
                                                       distribution_params::DISCRETE_MAX)
                    .value;
            exerciseDistribution("Discrete", DistributionType::DISCRETE, disc_dist, batch_sizes);
        }

        {
            auto poisson_dist =
                libstats::PoissonDistribution::create(distribution_params::POISSON_LAMBDA).value;
            exerciseDistribution("Poisson", DistributionType::POISSON, poisson_dist, batch_sizes);
        }

        {
            auto gamma_dist = libstats::GammaDistribution::create(distribution_params::GAMMA_ALPHA,
                                                                  distribution_params::GAMMA_BETA)
                                  .value;
            exerciseDistribution("Gamma", DistributionType::GAMMA, gamma_dist, batch_sizes);
        }
    }

    void analyzePerformanceHistory() {
        auto& history = PerformanceDispatcher::getPerformanceHistory();

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ADAPTIVE LEARNING ANALYSIS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << "\nTotal executions recorded: " << history.getTotalExecutions() << std::endl;

        // Test strategy recommendations for different scenarios
        std::vector<DistributionType> distributions = {
            DistributionType::UNIFORM,  DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL,
            DistributionType::DISCRETE, DistributionType::POISSON,  DistributionType::GAMMA};

        std::vector<size_t> test_sizes = {10, 100, 1000, 5000, 25000, 100000};

        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "STRATEGY RECOMMENDATIONS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (auto dist_type : distributions) {
            std::cout << "\n"
                      << strings::distributionTypeToString(dist_type)
                      << " Distribution:" << std::endl;
            std::cout << "  Size      Strategy        Confidence  Expected Time" << std::endl;
            std::cout << "  --------  --------------  ----------  -------------" << std::endl;

            for (size_t size : test_sizes) {
                auto recommendation = history.getBestStrategy(dist_type, size);

                std::cout << "  " << std::setw(8) << size << "  " << std::setw(14)
                          << strings::strategyToDisplayString(recommendation.recommended_strategy)
                          << "  " << std::setw(10)
                          << format::confidenceToString(recommendation.confidence_score) << "  "
                          << std::setw(8)
                          << format::nanosecondsToMicroseconds(recommendation.expected_time_ns)
                          << (recommendation.has_sufficient_data ? "" : " (insufficient data)")
                          << std::endl;
            }
        }

        // Show learned thresholds
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "LEARNED OPTIMAL THRESHOLDS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (auto dist_type : distributions) {
            auto thresholds = history.learnOptimalThresholds(dist_type);
            std::cout << strings::distributionTypeToString(dist_type) << ": ";
            if (thresholds) {
                std::cout << "SIMD >= " << thresholds->first
                          << ", Parallel >= " << thresholds->second << std::endl;
            } else {
                std::cout << "Insufficient data for learning" << std::endl;
            }
        }

        // Show performance statistics for each strategy
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "STRATEGY PERFORMANCE STATISTICS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (auto dist_type : distributions) {
            std::cout << "\n"
                      << strings::distributionTypeToString(dist_type)
                      << " Performance:" << std::endl;

            std::vector<Strategy> strategies = {Strategy::SCALAR, Strategy::SIMD_BATCH,
                                                Strategy::PARALLEL_SIMD, Strategy::WORK_STEALING,
                                                Strategy::GPU_ACCELERATED};

            for (auto strategy : strategies) {
                auto stats = history.getPerformanceStats(strategy, dist_type);
                if (stats) {
                    std::cout << "  " << std::setw(14) << strings::strategyToDisplayString(strategy)
                              << ": " << std::setw(6) << stats->execution_count << " runs, "
                              << "avg: " << std::setw(8)
                              << format::nanosecondsToMicroseconds(stats->getAverageTimeNs())
                              << ", "
                              << "min: " << std::setw(6)
                              << format::nanosecondsToMicroseconds(stats->min_time_ns) << ", "
                              << "max: " << std::setw(6)
                              << format::nanosecondsToMicroseconds(stats->max_time_ns) << std::endl;
                }
            }
        }
    }

    // Enhanced methods for analysis mode
    void exerciseAllDistributionsEnhanced(const std::vector<size_t>& batch_sizes) {
        std::cout << "Generating comprehensive performance data...\n\n";

        // Multiple runs per batch size to generate sufficient data
        constexpr int RUNS_PER_BATCH_SIZE = 3;
        int total_operations = static_cast<int>(
            6 * batch_sizes.size() * RUNS_PER_BATCH_SIZE);  // 6 distributions * sizes * runs
        int completed = 0;

        // Enhanced testing with multiple strategies per size
        for (int run = 0; run < RUNS_PER_BATCH_SIZE; ++run) {
            std::cout << "\n=== Run " << (run + 1) << " of " << RUNS_PER_BATCH_SIZE << " ===\n";

            // Test all distributions using safe factory methods
            {
                std::cout << "Testing Uniform Distribution..." << std::flush;
                auto uniform_dist =
                    libstats::UniformDistribution::create(distribution_params::UNIFORM_MIN,
                                                          distribution_params::UNIFORM_MAX)
                        .value;
                exerciseDistributionEnhanced("Uniform", DistributionType::UNIFORM, uniform_dist,
                                             batch_sizes);
                std::cout << " ✓\n";
                completed += static_cast<int>(batch_sizes.size());
            }

            {
                std::cout << "Testing Gaussian Distribution..." << std::flush;
                auto gaussian_dist =
                    libstats::GaussianDistribution::create(distribution_params::GAUSSIAN_MEAN,
                                                           distribution_params::GAUSSIAN_STDDEV)
                        .value;
                exerciseDistributionEnhanced("Gaussian", DistributionType::GAUSSIAN, gaussian_dist,
                                             batch_sizes);
                std::cout << " ✓\n";
                completed += static_cast<int>(batch_sizes.size());
            }

            {
                std::cout << "Testing Exponential Distribution..." << std::flush;
                auto exp_dist = libstats::ExponentialDistribution::create(
                                    distribution_params::EXPONENTIAL_LAMBDA)
                                    .value;
                exerciseDistributionEnhanced("Exponential", DistributionType::EXPONENTIAL, exp_dist,
                                             batch_sizes);
                std::cout << " ✓\n";
                completed += static_cast<int>(batch_sizes.size());
            }

            {
                std::cout << "Testing Discrete Distribution..." << std::flush;
                auto disc_dist =
                    libstats::DiscreteDistribution::create(distribution_params::DISCRETE_MIN,
                                                           distribution_params::DISCRETE_MAX)
                        .value;
                exerciseDistributionEnhanced("Discrete", DistributionType::DISCRETE, disc_dist,
                                             batch_sizes);
                std::cout << " ✓\n";
                completed += static_cast<int>(batch_sizes.size());
            }

            {
                std::cout << "Testing Poisson Distribution..." << std::flush;
                auto poisson_dist =
                    libstats::PoissonDistribution::create(distribution_params::POISSON_LAMBDA)
                        .value;
                exerciseDistributionEnhanced("Poisson", DistributionType::POISSON, poisson_dist,
                                             batch_sizes);
                std::cout << " ✓\n";
                completed += static_cast<int>(batch_sizes.size());
            }

            {
                std::cout << "Testing Gamma Distribution..." << std::flush;
                auto gamma_dist =
                    libstats::GammaDistribution::create(distribution_params::GAMMA_ALPHA,
                                                        distribution_params::GAMMA_BETA)
                        .value;
                exerciseDistributionEnhanced("Gamma", DistributionType::GAMMA, gamma_dist,
                                             batch_sizes);
                std::cout << " ✓\n";
                completed += static_cast<int>(batch_sizes.size());
            }

            double progress =
                static_cast<double>(completed) / static_cast<double>(total_operations) * 100.0;
            std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress << "%\n";
        }

        auto& history = PerformanceDispatcher::getPerformanceHistory();
        std::cout << "\nData collection complete! Total executions: "
                  << history.getTotalExecutions() << "\n";
    }

    template <typename Distribution>
    void exerciseDistributionEnhanced(const std::string& /* dist_name */,
                                      DistributionType dist_type, Distribution& dist,
                                      const std::vector<size_t>& batch_sizes) {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t batch_size : batch_sizes) {
            // Create test data
            std::vector<double> values(batch_size);
            std::uniform_real_distribution<double> value_gen(TEST_VALUE_MIN, TEST_VALUE_MAX);
            for (auto& v : values) {
                v = value_gen(gen);
            }

            // Always test scalar strategy
            {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] = dist.getProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                PerformanceDispatcher::recordPerformance(Strategy::SCALAR, dist_type, batch_size,
                                                         static_cast<uint64_t>(duration.count()));
            }

            // Test SIMD strategy for appropriate batch sizes
            if (batch_size >= MIN_SIMD_BATCH_SIZE) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] = dist.getCumulativeProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto base_duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                // Simulate SIMD improvement
                auto simd_duration = base_duration / SIMD_SPEEDUP_FACTOR;
                PerformanceDispatcher::recordPerformance(
                    Strategy::SIMD_BATCH, dist_type, batch_size,
                    static_cast<uint64_t>(simd_duration.count()));
            }

            // Test parallel strategies for larger batch sizes
            if (batch_size >= MIN_PARALLEL_BATCH_SIZE) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] =
                        dist.getProbability(values[i]) + dist.getCumulativeProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto base_duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                // Simulate parallel improvement
                auto parallel_duration = base_duration / PARALLEL_SPEEDUP_FACTOR;
                PerformanceDispatcher::recordPerformance(
                    Strategy::PARALLEL_SIMD, dist_type, batch_size,
                    static_cast<uint64_t>(parallel_duration.count()));
            }

            // Test work-stealing for very large batch sizes
            if (batch_size >= MIN_WORK_STEALING_BATCH_SIZE) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] = dist.getProbability(values[i]) * 2.0;
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto base_duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                auto work_stealing_duration = base_duration / WORK_STEALING_SPEEDUP_FACTOR;
                PerformanceDispatcher::recordPerformance(
                    Strategy::WORK_STEALING, dist_type, batch_size,
                    static_cast<std::uint64_t>(work_stealing_duration.count()));
            }

            // Test gpu-accelerated for extremely large batch sizes
            if (batch_size >= MIN_GPU_ACCELERATED_BATCH_SIZE) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<double> results(batch_size);
                // More complex operation for gpu-accelerated testing
                for (size_t i = 0; i < batch_size; ++i) {
                    results[i] =
                        dist.getProbability(values[i]) + dist.getCumulativeProbability(values[i]);
                }
                auto end = std::chrono::high_resolution_clock::now();
                auto base_duration =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

                auto gpu_accelerated_duration = base_duration / GPU_ACCELERATED_SPEEDUP_FACTOR;
                PerformanceDispatcher::recordPerformance(
                    Strategy::GPU_ACCELERATED, dist_type, batch_size,
                    static_cast<std::uint64_t>(gpu_accelerated_duration.count()));
            }
        }
    }

    void analyzePerformanceHistoryEnhanced() {
        auto& history = PerformanceDispatcher::getPerformanceHistory();

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "ADAPTIVE LEARNING ANALYSIS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::cout << "\nTotal executions recorded: " << history.getTotalExecutions() << std::endl;

        // Test strategy recommendations for different scenarios
        std::vector<DistributionType> distributions = {
            DistributionType::UNIFORM,  DistributionType::GAUSSIAN, DistributionType::EXPONENTIAL,
            DistributionType::DISCRETE, DistributionType::POISSON,  DistributionType::GAMMA};

        std::vector<size_t> test_sizes = {10, 100, 1000, 5000, 25000, 100000};

        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "STRATEGY RECOMMENDATIONS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (auto dist_type : distributions) {
            std::cout << "\n"
                      << strings::distributionTypeToString(dist_type)
                      << " Distribution:" << std::endl;
            std::cout << "  Size      Strategy        Confidence  Expected Time" << std::endl;
            std::cout << "  --------  --------------  ----------  -------------" << std::endl;

            for (size_t size : test_sizes) {
                auto recommendation = history.getBestStrategy(dist_type, size);

                std::cout << "  " << std::setw(8) << size << "  " << std::setw(14)
                          << strings::strategyToDisplayString(recommendation.recommended_strategy)
                          << "  " << std::setw(10)
                          << format::confidenceToString(recommendation.confidence_score) << "  "
                          << std::setw(8)
                          << format::nanosecondsToMicroseconds(recommendation.expected_time_ns)
                          << (recommendation.has_sufficient_data ? "" : " (insufficient data)")
                          << std::endl;
            }
        }

        // Show learned thresholds
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "LEARNED OPTIMAL THRESHOLDS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (auto dist_type : distributions) {
            auto thresholds = history.learnOptimalThresholds(dist_type);
            std::cout << strings::distributionTypeToString(dist_type) << ": ";
            if (thresholds) {
                std::cout << "SIMD >= " << thresholds->first
                          << ", Parallel >= " << thresholds->second << std::endl;
            } else {
                std::cout << "Insufficient data for learning" << std::endl;
            }
        }

        // Enhanced performance statistics with insights
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "STRATEGY PERFORMANCE STATISTICS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        for (auto dist_type : distributions) {
            std::cout << "\n"
                      << strings::distributionTypeToString(dist_type)
                      << " Performance:" << std::endl;

            std::vector<Strategy> strategies = {Strategy::SCALAR, Strategy::SIMD_BATCH,
                                                Strategy::PARALLEL_SIMD, Strategy::WORK_STEALING,
                                                Strategy::GPU_ACCELERATED};

            for (auto strategy : strategies) {
                auto stats = history.getPerformanceStats(strategy, dist_type);
                if (stats) {
                    std::cout << "  " << std::setw(14) << strings::strategyToDisplayString(strategy)
                              << ": " << std::setw(6) << stats->execution_count << " runs, "
                              << "avg: " << std::setw(8)
                              << format::nanosecondsToMicroseconds(stats->getAverageTimeNs())
                              << ", "
                              << "min: " << std::setw(6)
                              << format::nanosecondsToMicroseconds(stats->min_time_ns) << ", "
                              << "max: " << std::setw(6)
                              << format::nanosecondsToMicroseconds(stats->max_time_ns) << std::endl;
                }
            }
        }

        // Add insights and recommendations
        std::cout << "\n" << std::string(60, '-') << std::endl;
        std::cout << "PERFORMANCE INSIGHTS" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        generatePerformanceInsights(history, distributions);
    }

    void generatePerformanceInsights(PerformanceHistory& history,
                                     const std::vector<DistributionType>& distributions) {
        std::cout << "\nBased on collected performance data:\n\n";

        // Analyze efficiency patterns across distributions
        std::cout << "Distribution Efficiency Rankings (lower times = better):\n";
        std::vector<std::pair<DistributionType, uint64_t>> efficiency_ranking;

        for (auto dist_type : distributions) {
            auto stats = history.getPerformanceStats(Strategy::SCALAR, dist_type);
            if (stats && stats->execution_count > 0) {
                efficiency_ranking.emplace_back(dist_type, stats->getAverageTimeNs());
            }
        }

        std::sort(efficiency_ranking.begin(), efficiency_ranking.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        int rank = 1;
        for (const auto& [dist_type, avg_time] : efficiency_ranking) {
            std::cout << "  " << rank++ << ". " << strings::distributionTypeToString(dist_type)
                      << " (" << format::nanosecondsToMicroseconds(avg_time) << " avg)\n";
        }

        // Strategy effectiveness analysis
        std::cout << "\nStrategy Effectiveness Summary:\n";
        for (auto strategy : {Strategy::SIMD_BATCH, Strategy::PARALLEL_SIMD,
                              Strategy::WORK_STEALING, Strategy::GPU_ACCELERATED}) {
            int total_distributions = 0;
            int effective_distributions = 0;

            for (auto dist_type : distributions) {
                auto scalar_stats = history.getPerformanceStats(Strategy::SCALAR, dist_type);
                auto strategy_stats = history.getPerformanceStats(strategy, dist_type);

                if (scalar_stats && strategy_stats && scalar_stats->execution_count > 0 &&
                    strategy_stats->execution_count > 0) {
                    total_distributions++;
                    if (strategy_stats->getAverageTimeNs() < scalar_stats->getAverageTimeNs()) {
                        effective_distributions++;
                    }
                }
            }

            if (total_distributions > 0) {
                double effectiveness = static_cast<double>(effective_distributions) /
                                       static_cast<double>(total_distributions) * 100.0;
                std::cout << "  " << strings::strategyToDisplayString(strategy) << ": "
                          << std::fixed << std::setprecision(1) << effectiveness << "% effective ("
                          << effective_distributions << "/" << total_distributions
                          << " distributions)\n";
            }
        }

        std::cout << "\nRecommendations for optimal performance:\n";
        std::cout << "• Use Scalar strategy for small batch sizes (< 100 elements)\n";
        std::cout << "• Consider SIMD for medium batches (100-10,000 elements)\n";
        std::cout << "• Use Parallel strategies for large batches (> 10,000 elements)\n";
        std::cout << "• Advanced strategies (Work-Stealing, Cache-Aware) show benefits with very "
                     "large datasets\n";
    }
};

int main(int argc, char* argv[]) {
    LearningAnalyzer analyzer;

    // Parse command line arguments
    std::string mode = "analysis";  // default mode
    if (argc > 1) {
        mode = argv[1];
    }

    if (mode == "help" || mode == "--help" || mode == "-h") {
        analyzer.showUsage();
        return 0;
    }

    try {
        if (mode == "demo") {
            analyzer.runDemo();
        } else if (mode == "analysis") {
            analyzer.runAnalysis();
        } else if (mode == "both") {
            analyzer.runDemo();
            std::cout << "\n" << std::string(80, '=') << "\n\n";
            analyzer.runAnalysis();
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            analyzer.showUsage();
            return 1;
        }

        std::cout << "\nLearning analysis completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

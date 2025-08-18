#pragma once

#include "../common/platform_common.h"
#include "../libstats/export.h"
#include <iostream>
#include <map>

// Level 1 infrastructure
#include "../core/math_utils.h"

namespace libstats {

/// High-resolution timer for performance measurements
class LIBSTATS_API Timer {
private:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double>;
    
    TimePoint startTime_;
    bool running_;
    
public:
    Timer() : running_(false) {}
    
    /// Start the timer
    void start() {
        startTime_ = Clock::now();
        running_ = true;
    }
    
    /// Stop the timer and return elapsed time in seconds
    /// @return Elapsed time in seconds
    double stop() {
        if (!running_) {
            return 0.0;
        }
        
        const auto endTime = Clock::now();
        const Duration elapsed = endTime - startTime_;
        running_ = false;
        return elapsed.count();
    }
    
    /// Get elapsed time without stopping (if running)
    /// @return Elapsed time in seconds
    double elapsed() const {
        if (!running_) {
            return 0.0;
        }
        
        const auto currentTime = Clock::now();
        const Duration elapsed = currentTime - startTime_;
        return elapsed.count();
    }
    
    /// Check if timer is running
    /// @return True if timer is running
    bool isRunning() const noexcept {
        return running_;
    }
};

/// Statistics for benchmark results
struct LIBSTATS_API BenchmarkStats {
    double mean = 0.0;           ///< Mean execution time
    double median = 0.0;         ///< Median execution time
    double stddev = 0.0;         ///< Standard deviation
    double min = 0.0;            ///< Minimum execution time
    double max = 0.0;            ///< Maximum execution time
    std::size_t samples = 0;     ///< Number of samples
    double throughput = 0.0;     ///< Operations per second (if applicable)
    
    /// Format statistics as string
    /// @return Formatted statistics string
    std::string toString() const;
};

/// Benchmark result for a single test
struct LIBSTATS_API BenchmarkResult {
    std::string name;                    ///< Test name
    BenchmarkStats stats;                ///< Statistical results
    std::vector<double> rawTimes;        ///< Raw timing measurements
    std::map<std::string, double> metrics; ///< Custom metrics
    
    /// Add a custom metric
    /// @param key Metric name
    /// @param value Metric value
    void addMetric(const std::string& key, double value) {
        metrics[key] = value;
    }
    
    /// Get formatted result string
    /// @return Formatted result
    std::string toString() const;
};

/// Benchmark suite for statistical computing performance testing
class LIBSTATS_API Benchmark {
public:
    /// Test function type
    using TestFunction = std::function<void()>;
    
    /// Setup function type (called before each test iteration)
    using SetupFunction = std::function<void()>;
    
    /// Teardown function type (called after each test iteration)
    using TeardownFunction = std::function<void()>;
    
private:
    struct TestCase {
        std::string name;
        TestFunction testFunc;
        SetupFunction setupFunc;
        TeardownFunction teardownFunc;
        std::size_t iterations;
        std::size_t warmupRuns;
        double operationCount; ///< For throughput calculation
    };
    
    std::vector<TestCase> testCases_;
    std::vector<BenchmarkResult> results_;
    bool enableWarmup_;
    std::size_t defaultIterations_;
    std::size_t defaultWarmupRuns_;
    
public:
    /// Constructor
    /// @param enableWarmup Enable warmup runs to stabilize CPU state
    /// @param defaultIterations Default number of iterations per test (0 = auto-detect)
    /// @param defaultWarmupRuns Default number of warmup runs (0 = auto-detect)
    explicit Benchmark(bool enableWarmup = true, 
                      std::size_t defaultIterations = 0,
                      std::size_t defaultWarmupRuns = 0);
    
    /// Add a test case
    /// @param name Test name
    /// @param testFunc Function to benchmark
    /// @param iterations Number of iterations (0 = use default)
    /// @param operationCount Number of operations per test (for throughput)
    /// @param setupFunc Optional setup function
    /// @param teardownFunc Optional teardown function
    /// @param warmupRuns Number of warmup runs (0 = use default)
    void addTest(const std::string& name, 
                 TestFunction testFunc,
                 std::size_t iterations = 0,
                 double operationCount = 1.0,
                 SetupFunction setupFunc = nullptr,
                 TeardownFunction teardownFunc = nullptr,
                 std::size_t warmupRuns = 0);
    
    /// Run all registered benchmarks
    /// @return Vector of benchmark results
    std::vector<BenchmarkResult> runAll();
    
    /// Run a specific benchmark by name
    /// @param testName Name of test to run
    /// @return Benchmark result
    BenchmarkResult run(const std::string& testName);
    
    /// Clear all test cases and results
    void clear();
    
    /// Get results from last run
    /// @return Vector of results
    const std::vector<BenchmarkResult>& getResults() const {
        return results_;
    }
    
    /// Print results to output stream
    /// @param os Output stream
    /// @param showRawTimes Include raw timing data
    void printResults(std::ostream& os = std::cout, bool showRawTimes = false) const;
    
    /// Compare two benchmark results
    /// @param baseline Baseline results
    /// @param comparison Results to compare against baseline
    /// @param os Output stream for comparison
    static void compareResults(const std::vector<BenchmarkResult>& baseline,
                              const std::vector<BenchmarkResult>& comparison,
                              std::ostream& os = std::cout);

private:
    /// Calculate statistics from raw timing data using robust methods
    /// @param times Vector of timing measurements
    /// @param operationCount Number of operations (for throughput)
    /// @return Calculated statistics
    BenchmarkStats calculateStats(const std::vector<double>& times, double operationCount) const;
    
    /// Calculate robust statistics using Level 1 math utilities
    /// @param times Vector of timing measurements
    /// @param operationCount Number of operations (for throughput)
    /// @return Calculated statistics with numerical stability
    BenchmarkStats calculateStatsRobust(const std::vector<double>& times, double operationCount) const;
    
    /// Get optimal benchmark parameters based on CPU characteristics
    /// @return Pair of (iterations, warmup_runs)
    std::pair<std::size_t, std::size_t> getOptimalBenchmarkParams() const;
    
    /// Run warmup iterations
    /// @param testCase Test case to warm up
    void runWarmup(const TestCase& testCase);
    
    /// Execute a single test case
    /// @param testCase Test case to execute
    /// @return Benchmark result
    BenchmarkResult executeTest(const TestCase& testCase);
};

/// Convenience macros for benchmarking
#define LIBSTATS_BENCHMARK_SIMPLE(name, func) \
    benchmark.addTest(name, func)

#define LIBSTATS_BENCHMARK_WITH_SETUP(name, func, setup, teardown) \
    benchmark.addTest(name, func, 0, 1.0, setup, teardown)

#define LIBSTATS_BENCHMARK_ITERATIONS(name, func, iters) \
    benchmark.addTest(name, func, iters)

/// Statistical computing benchmark utilities
class StatsBenchmarkUtils {
public:
    /// Create test data vectors of varying sizes
    /// @param minSize Minimum vector size
    /// @param maxSize Maximum vector size
    /// @param numVectors Number of vectors to generate
    /// @return Vector of test data vectors
    static std::vector<std::vector<double>> createTestVectors(
        std::size_t minSize, std::size_t maxSize, 
        std::size_t numVectors);
    
    /// Create test matrices of varying sizes
    /// @param minRows Minimum number of rows
    /// @param maxRows Maximum number of rows
    /// @param minCols Minimum number of columns
    /// @param maxCols Maximum number of columns
    /// @param numMatrices Number of matrices to generate
    /// @return Vector of test matrices (row-major format)
    static std::vector<std::vector<double>> createTestMatrices(
        std::size_t minRows, std::size_t maxRows,
        std::size_t minCols, std::size_t maxCols,
        std::size_t numMatrices);
    
    /// Benchmark basic statistical operations
    /// @param data Test data vectors
    /// @param benchmark Benchmark object to add tests to
    static void benchmarkBasicStats(
        const std::vector<std::vector<double>>& data,
        Benchmark& benchmark);
    
    /// Benchmark matrix operations
    /// @param matrices Test matrices
    /// @param benchmark Benchmark object to add tests to
    static void benchmarkMatrixOps(
        const std::vector<std::vector<double>>& matrices,
        Benchmark& benchmark);
    
    /// Benchmark SIMD vs scalar implementations
    /// @param data Test data
    /// @param benchmark Benchmark object to add tests to
    static void benchmarkSIMDOperations(
        const std::vector<std::vector<double>>& data,
        Benchmark& benchmark);
};

/// Performance regression testing
class RegressionTester {
private:
    std::string baselineFile_;
    std::vector<BenchmarkResult> baselineResults_;
    double tolerancePercent_;
    
public:
    /// Constructor
    /// @param baselineFile File containing baseline results
    /// @param tolerancePercent Allowed performance degradation percentage
    explicit RegressionTester(const std::string& baselineFile, 
                             double tolerancePercent = 5.0);
    
    /// Load baseline results from file
    /// @return True if loaded successfully
    bool loadBaseline();
    
    /// Save results as new baseline
    /// @param results Results to save
    /// @return True if saved successfully
    bool saveBaseline(const std::vector<BenchmarkResult>& results);
    
    /// Test for performance regressions
    /// @param currentResults Current benchmark results
    /// @return True if no significant regressions detected
    bool checkRegressions(const std::vector<BenchmarkResult>& currentResults,
                         std::ostream& os = std::cout);
    
    /// Set tolerance for regression detection
    /// @param percent Allowed degradation percentage
    void setTolerance(double percent) {
        tolerancePercent_ = percent;
    }
};

} // namespace libstats

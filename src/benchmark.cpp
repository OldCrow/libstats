#include "../include/platform/benchmark.h"
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <random>
#include <cmath>

namespace libstats {

//========== BenchmarkStats Implementation ==========

std::string BenchmarkStats::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "Mean: " << mean << "s, ";
    oss << "Median: " << median << "s, ";
    oss << "StdDev: " << stddev << "s, ";
    oss << "Min: " << min << "s, ";
    oss << "Max: " << max << "s";
    if (throughput > 0.0) {
        oss << ", Throughput: " << std::setprecision(2) << throughput << " ops/s";
    }
    oss << " (n=" << samples << ")";
    return oss.str();
}

//========== BenchmarkResult Implementation ==========

std::string BenchmarkResult::toString() const {
    std::ostringstream oss;
    oss << "=== " << name << " ===\n";
    oss << stats.toString() << "\n";
    
    if (!metrics.empty()) {
        oss << "Custom metrics:\n";
        for (const auto& [key, value] : metrics) {
            oss << "  " << key << ": " << std::fixed << std::setprecision(6) << value << "\n";
        }
    }
    
    return oss.str();
}

//========== Benchmark Implementation ==========

Benchmark::Benchmark(bool enableWarmup, std::size_t defaultIterations, std::size_t defaultWarmupRuns)
    : enableWarmup_(enableWarmup)
    , defaultIterations_(defaultIterations)
    , defaultWarmupRuns_(defaultWarmupRuns) {
    
    // Auto-detect optimal parameters based on CPU characteristics if not specified
    if (defaultIterations_ == 0 || defaultWarmupRuns_ == 0) {
        auto [optimalIterations, optimalWarmupRuns] = getOptimalBenchmarkParams();
        
        if (defaultIterations_ == 0) {
            defaultIterations_ = optimalIterations;
        }
        if (defaultWarmupRuns_ == 0) {
            defaultWarmupRuns_ = optimalWarmupRuns;
        }
    }
}

void Benchmark::addTest(const std::string& name, 
                       TestFunction testFunc,
                       std::size_t iterations,
                       double operationCount,
                       SetupFunction setupFunc,
                       TeardownFunction teardownFunc,
                       std::size_t warmupRuns) {
    
    TestCase testCase;
    testCase.name = name;
    testCase.testFunc = std::move(testFunc);
    testCase.setupFunc = std::move(setupFunc);
    testCase.teardownFunc = std::move(teardownFunc);
    testCase.iterations = (iterations == 0) ? defaultIterations_ : iterations;
    testCase.warmupRuns = (warmupRuns == 0) ? defaultWarmupRuns_ : warmupRuns;
    testCase.operationCount = operationCount;
    
    testCases_.push_back(std::move(testCase));
}

std::vector<BenchmarkResult> Benchmark::runAll() {
    results_.clear();
    results_.reserve(testCases_.size());
    
    for (const auto& testCase : testCases_) {
        std::cout << "Running benchmark: " << testCase.name << " ... " << std::flush;
        
        auto result = executeTest(testCase);
        results_.push_back(std::move(result));
        
        std::cout << "done\n";
    }
    
    return results_;
}

BenchmarkResult Benchmark::run(const std::string& testName) {
    for (const auto& testCase : testCases_) {
        if (testCase.name == testName) {
            return executeTest(testCase);
        }
    }
    
    // Test not found
    BenchmarkResult result;
    result.name = testName + " (NOT FOUND)";
    return result;
}

void Benchmark::clear() {
    testCases_.clear();
    results_.clear();
}

void Benchmark::printResults(std::ostream& os, bool showRawTimes) const {
    os << "\n" << std::string(80, '=') << "\n";
    os << "BENCHMARK RESULTS\n";
    os << std::string(80, '=') << "\n\n";
    
    for (const auto& result : results_) {
        os << result.toString() << "\n";
        
        if (showRawTimes && !result.rawTimes.empty()) {
            os << "Raw times (first 10): ";
            const size_t showCount = std::min(size_t{10}, result.rawTimes.size());
            for (size_t i = 0; i < showCount; ++i) {
                os << std::fixed << std::setprecision(6) << result.rawTimes[i];
                if (i < showCount - 1) os << ", ";
            }
            if (result.rawTimes.size() > 10) {
                os << ", ... (" << result.rawTimes.size() << " total)";
            }
            os << "\n\n";
        }
    }
}

void Benchmark::compareResults(const std::vector<BenchmarkResult>& baseline,
                              const std::vector<BenchmarkResult>& comparison,
                              std::ostream& os) {
    
    os << "\n" << std::string(80, '=') << "\n";
    os << "BENCHMARK COMPARISON\n";
    os << std::string(80, '=') << "\n\n";
    
    // Create maps for easy lookup
    std::map<std::string, const BenchmarkResult*> baselineMap;
    std::map<std::string, const BenchmarkResult*> comparisonMap;
    
    for (const auto& result : baseline) {
        baselineMap[result.name] = &result;
    }
    
    for (const auto& result : comparison) {
        comparisonMap[result.name] = &result;
    }
    
    // Compare common tests
    for (const auto& [name, baseResult] : baselineMap) {
        auto it = comparisonMap.find(name);
        if (it != comparisonMap.end()) {
            const auto& compResult = *it->second;
            
            const double speedup = baseResult->stats.mean / compResult.stats.mean;
            const double percentChange = ((compResult.stats.mean - baseResult->stats.mean) / baseResult->stats.mean) * 100.0;
            
            os << "Test: " << name << "\n";
            os << "  Baseline:   " << std::fixed << std::setprecision(6) << baseResult->stats.mean << "s\n";
            os << "  Comparison: " << std::fixed << std::setprecision(6) << compResult.stats.mean << "s\n";
            os << "  Speedup:    " << std::fixed << std::setprecision(2) << speedup << "x";
            
            if (speedup > 1.05) {
                os << " (FASTER)";
            } else if (speedup < 0.95) {
                os << " (SLOWER)";
            } else {
                os << " (SIMILAR)";
            }
            
            os << " (" << std::showpos << std::fixed << std::setprecision(1) << percentChange << "%)\n\n";
        }
    }
}

BenchmarkStats Benchmark::calculateStats(const std::vector<double>& times, double operationCount) const {
    if (times.empty()) {
        return {};
    }
    
    BenchmarkStats stats;
    stats.samples = times.size();
    
    // Calculate mean
    stats.mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    
    // Calculate median
    std::vector<double> sortedTimes = times;
    std::sort(sortedTimes.begin(), sortedTimes.end());
    
    if (sortedTimes.size() % 2 == 0) {
        const size_t mid = sortedTimes.size() / 2;
        stats.median = (sortedTimes[mid - 1] + sortedTimes[mid]) / 2.0;
    } else {
        stats.median = sortedTimes[sortedTimes.size() / 2];
    }
    
    // Calculate standard deviation
    double sumSquaredDiffs = 0.0;
    for (double time : times) {
        const double diff = time - stats.mean;
        sumSquaredDiffs += diff * diff;
    }
    stats.stddev = std::sqrt(sumSquaredDiffs / times.size());
    
    // Min and max
    stats.min = *std::min_element(times.begin(), times.end());
    stats.max = *std::max_element(times.begin(), times.end());
    
    // Throughput (operations per second)
    if (stats.mean > 0.0 && operationCount > 0.0) {
        stats.throughput = operationCount / stats.mean;
    }
    
    return stats;
}

void Benchmark::runWarmup(const TestCase& testCase) {
    for (std::size_t i = 0; i < testCase.warmupRuns; ++i) {
        if (testCase.setupFunc) {
            testCase.setupFunc();
        }
        
        testCase.testFunc();
        
        if (testCase.teardownFunc) {
            testCase.teardownFunc();
        }
    }
}

BenchmarkResult Benchmark::executeTest(const TestCase& testCase) {
    BenchmarkResult result;
    result.name = testCase.name;
    result.rawTimes.reserve(testCase.iterations);
    
    // Run warmup if enabled
    if (enableWarmup_) {
        runWarmup(testCase);
    }
    
    // Run actual benchmark iterations
    Timer timer;
    for (std::size_t i = 0; i < testCase.iterations; ++i) {
        if (testCase.setupFunc) {
            testCase.setupFunc();
        }
        
        timer.start();
        testCase.testFunc();
        const double elapsed = timer.stop();
        
        if (testCase.teardownFunc) {
            testCase.teardownFunc();
        }
        
        result.rawTimes.push_back(elapsed);
    }
    
    // Calculate statistics using robust methods
    result.stats = calculateStatsRobust(result.rawTimes, testCase.operationCount);
    
    return result;
}

//========== StatsBenchmarkUtils Implementation ==========

std::vector<std::vector<double>> StatsBenchmarkUtils::createTestVectors(
    std::size_t minSize, std::size_t maxSize, std::size_t numVectors) {
    
    std::vector<std::vector<double>> vectors;
    vectors.reserve(numVectors);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> sizeDist(minSize, maxSize);
    
    for (std::size_t i = 0; i < numVectors; ++i) {
        const std::size_t size = sizeDist(gen);
        std::vector<double> vec(size);
        
        for (double& value : vec) {
            value = dist(gen);
        }
        
        vectors.push_back(std::move(vec));
    }
    
    return vectors;
}

std::vector<std::vector<double>> StatsBenchmarkUtils::createTestMatrices(
    std::size_t minRows, std::size_t maxRows,
    std::size_t minCols, std::size_t maxCols,
    std::size_t numMatrices) {
    
    std::vector<std::vector<double>> matrices;
    matrices.reserve(numMatrices);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<std::size_t> rowDist(minRows, maxRows);
    std::uniform_int_distribution<std::size_t> colDist(minCols, maxCols);
    
    for (std::size_t i = 0; i < numMatrices; ++i) {
        const std::size_t rows = rowDist(gen);
        const std::size_t cols = colDist(gen);
        
        std::vector<double> matrix(rows * cols);
        for (double& value : matrix) {
            value = dist(gen);
        }
        
        matrices.push_back(std::move(matrix));
    }
    
    return matrices;
}

void StatsBenchmarkUtils::benchmarkBasicStats(
    const std::vector<std::vector<double>>& data,
    Benchmark& benchmark) {
    
    if (data.empty()) return;
    
    // Mean calculation benchmark
    benchmark.addTest("Mean Calculation", [&data]() {
        for (const auto& vec : data) {
            volatile double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
            [[maybe_unused]] volatile double mean = sum / vec.size();
        }
    }, 0, static_cast<double>(data.size()));

    // Variance calculation benchmark
    benchmark.addTest("Variance Calculation", [&data]() {
        for (const auto& vec : data) {
            const double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
            double sumSq = 0.0;
            for (double val : vec) {
                const double diff = val - mean;
                sumSq += diff * diff;
            }
            [[maybe_unused]] volatile double variance = sumSq / vec.size();
        }
    }, 0, static_cast<double>(data.size()));

    // Sorting benchmark
    benchmark.addTest("Vector Sorting", [&data]() {
        for (const auto& vec : data) {
            std::vector<double> copy = vec;
            std::sort(copy.begin(), copy.end());
        }
    }, 0, static_cast<double>(data.size()));
}

void StatsBenchmarkUtils::benchmarkMatrixOps(
    const std::vector<std::vector<double>>& matrices,
    Benchmark& benchmark) {
    
    if (matrices.empty()) return;
    
    // Matrix sum benchmark
    benchmark.addTest("Matrix Sum", [&matrices]() {
        for (const auto& matrix : matrices) {
            [[maybe_unused]] volatile double sum = std::accumulate(matrix.begin(), matrix.end(), 0.0);
        }
    }, 0, static_cast<double>(matrices.size()));
    
    // Matrix normalization benchmark
    benchmark.addTest("Matrix Normalization", [&matrices]() {
        for (const auto& matrix : matrices) {
            std::vector<double> normalized = matrix;
            const double norm = std::sqrt(std::inner_product(matrix.begin(), matrix.end(), 
                                                            matrix.begin(), 0.0));
            if (norm > 0.0) {
                for (double& val : normalized) {
                    val /= norm;
                }
            }
        }
    }, 0, static_cast<double>(matrices.size()));
}

void StatsBenchmarkUtils::benchmarkSIMDOperations(
    const std::vector<std::vector<double>>& data,
    Benchmark& benchmark) {
    
    if (data.empty()) return;
    
    // Scalar addition benchmark
    benchmark.addTest("Scalar Vector Addition", [&data]() {
        for (const auto& vec : data) {
            std::vector<double> result(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                result[i] = vec[i] + 1.0;
            }
        }
    }, 0, static_cast<double>(data.size()));
    
    // Scalar multiplication benchmark
    benchmark.addTest("Scalar Vector Multiplication", [&data]() {
        for (const auto& vec : data) {
            std::vector<double> result(vec.size());
            for (size_t i = 0; i < vec.size(); ++i) {
                result[i] = vec[i] * 2.0;
            }
        }
    }, 0, static_cast<double>(data.size()));
    
    // Dot product benchmark
    benchmark.addTest("Vector Dot Product", [&data]() {
        for (size_t i = 0; i + 1 < data.size(); ++i) {
            const auto& vec1 = data[i];
            const auto& vec2 = data[i + 1];
            const size_t minSize = std::min(vec1.size(), vec2.size());
            
            double dotProduct = 0.0;
            for (size_t j = 0; j < minSize; ++j) {
                dotProduct += vec1[j] * vec2[j];
            }
            [[maybe_unused]] volatile double result = dotProduct;
        }
    }, 0, data.size() / 2.0);
}

//========== RegressionTester Implementation ==========

RegressionTester::RegressionTester(const std::string& baselineFile, double tolerancePercent)
    : baselineFile_(baselineFile)
    , tolerancePercent_(tolerancePercent) {
}

bool RegressionTester::loadBaseline() {
    std::ifstream file(baselineFile_);
    if (!file.is_open()) {
        return false;
    }
    
    baselineResults_.clear();
    std::string line;
    
    // Simple CSV-like format for baseline storage
    // Format: TestName,Mean,Median,StdDev,Min,Max,Samples,Throughput
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue; // Skip comments
        
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 8) {
            BenchmarkResult result;
            result.name = tokens[0];
            result.stats.mean = std::stod(tokens[1]);
            result.stats.median = std::stod(tokens[2]);
            result.stats.stddev = std::stod(tokens[3]);
            result.stats.min = std::stod(tokens[4]);
            result.stats.max = std::stod(tokens[5]);
            result.stats.samples = std::stoull(tokens[6]);
            result.stats.throughput = std::stod(tokens[7]);
            
            baselineResults_.push_back(std::move(result));
        }
    }
    
    return !baselineResults_.empty();
}

bool RegressionTester::saveBaseline(const std::vector<BenchmarkResult>& results) {
    std::ofstream file(baselineFile_);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# Benchmark Baseline Results\n";
    file << "# Format: TestName,Mean,Median,StdDev,Min,Max,Samples,Throughput\n";
    
    for (const auto& result : results) {
        file << result.name << ","
             << std::fixed << std::setprecision(9) << result.stats.mean << ","
             << result.stats.median << ","
             << result.stats.stddev << ","
             << result.stats.min << ","
             << result.stats.max << ","
             << result.stats.samples << ","
             << result.stats.throughput << "\n";
    }
    
    return true;
}

bool RegressionTester::checkRegressions(const std::vector<BenchmarkResult>& currentResults,
                                       std::ostream& os) {
    if (baselineResults_.empty()) {
        os << "No baseline results loaded for regression testing.\n";
        return false;
    }
    
    // Create map for fast baseline lookup
    std::map<std::string, const BenchmarkResult*> baselineMap;
    for (const auto& result : baselineResults_) {
        baselineMap[result.name] = &result;
    }
    
    bool anyRegressions = false;
    
    os << "\n" << std::string(80, '=') << "\n";
    os << "REGRESSION TEST RESULTS\n";
    os << "Tolerance: " << tolerancePercent_ << "%\n";
    os << std::string(80, '=') << "\n\n";
    
    for (const auto& current : currentResults) {
        auto it = baselineMap.find(current.name);
        if (it == baselineMap.end()) {
            os << "NEW TEST: " << current.name << " (no baseline)\n";
            continue;
        }
        
        const auto& baseline = *it->second;
        const double percentChange = ((current.stats.mean - baseline.stats.mean) / baseline.stats.mean) * 100.0;
        const bool isRegression = percentChange > tolerancePercent_;
        
        if (isRegression) {
            anyRegressions = true;
            os << "REGRESSION: ";
        } else if (percentChange < -tolerancePercent_) {
            os << "IMPROVEMENT: ";
        } else {
            os << "OK: ";
        }
        
        os << current.name << " ("
           << std::showpos << std::fixed << std::setprecision(1) << percentChange << "%"
           << ", " << std::noshowpos << std::setprecision(6) << current.stats.mean << "s vs "
           << baseline.stats.mean << "s)\n";
    }
    
    os << "\nRegression test " << (anyRegressions ? "FAILED" : "PASSED") << "\n\n";
    
    return !anyRegressions;
}

//========== CPU Detection Integration ==========

std::pair<std::size_t, std::size_t> Benchmark::getOptimalBenchmarkParams() const {
    // Use CPU detection to optimize benchmark parameters
    const auto& cpuFeatures = cpu::get_features();
    
    // Base parameters from constants
    std::size_t iterations = constants::benchmark::DEFAULT_ITERATIONS;
    std::size_t warmupRuns = constants::benchmark::DEFAULT_WARMUP_RUNS;
    
    // Adjust based on CPU characteristics
    if (cpuFeatures.topology.physical_cores >= 16) {
        // High-core count systems - more iterations for stable results
        iterations = static_cast<std::size_t>(iterations * 1.5);
        warmupRuns = static_cast<std::size_t>(warmupRuns * 1.2);
    } else if (cpuFeatures.topology.physical_cores <= 4) {
        // Low-core count systems - fewer iterations to save time
        iterations = static_cast<std::size_t>(iterations * 0.8);
        warmupRuns = static_cast<std::size_t>(warmupRuns * 0.8);
    }
    
    // Adjust for cache characteristics
    if (cpuFeatures.l3_cache_size >= 16 * 1024 * 1024) { // 16MB+
        // Large cache - can handle more iterations efficiently
        iterations = static_cast<std::size_t>(iterations * 1.2);
    } else if (cpuFeatures.l3_cache_size <= 4 * 1024 * 1024) { // 4MB or less
        // Small cache - reduce iterations to minimize cache pressure
        iterations = static_cast<std::size_t>(iterations * 0.9);
    }
    
    // Adjust for hyperthreading
    if (cpuFeatures.topology.logical_cores > cpuFeatures.topology.physical_cores) {
        // Hyperthreading enabled - more warmup needed for stable results
        warmupRuns = static_cast<std::size_t>(warmupRuns * 1.3);
    }
    
    // Ensure minimum values
    iterations = std::max(iterations, constants::benchmark::MIN_ITERATIONS);
    warmupRuns = std::max(warmupRuns, constants::benchmark::MIN_WARMUP_RUNS);
    
    return {iterations, warmupRuns};
}

BenchmarkStats Benchmark::calculateStatsRobust(const std::vector<double>& times, double operationCount) const {
    if (times.empty()) {
        return {};
    }
    
    BenchmarkStats stats;
    stats.samples = times.size();
    
    // Use robust statistical methods from math_utils
    
    // Calculate mean using numerically stable method
    double sum = 0.0;
    for (const double time : times) {
        // Use standard checks for finite values
        if (std::isfinite(time) && time >= 0.0) {
            sum += time;
        }
    }
    stats.mean = sum / static_cast<double>(times.size());
    
    // Calculate median using sorted data
    std::vector<double> sortedTimes;
    sortedTimes.reserve(times.size());
    
    // Filter out invalid values
    for (const double time : times) {
        if (std::isfinite(time) && time >= 0.0) {
            sortedTimes.push_back(time);
        }
    }
    
    if (sortedTimes.empty()) {
        return {}; // All values were invalid
    }
    
    std::sort(sortedTimes.begin(), sortedTimes.end());
    
    if (sortedTimes.size() % 2 == 0) {
        const size_t mid = sortedTimes.size() / 2;
        stats.median = (sortedTimes[mid - 1] + sortedTimes[mid]) / 2.0;
    } else {
        stats.median = sortedTimes[sortedTimes.size() / 2];
    }
    
    // Calculate robust standard deviation using math_utils
    double sumSquaredDiffs = 0.0;
    for (const double time : sortedTimes) {
        const double diff = time - stats.mean;
        sumSquaredDiffs += diff * diff;
    }
    
    // Use safe square root from math_utils
    if (sumSquaredDiffs > 0.0 && sortedTimes.size() > 1) {
        stats.stddev = std::sqrt(sumSquaredDiffs / static_cast<double>(sortedTimes.size() - 1));
    } else {
        stats.stddev = 0.0;
    }
    
    // Min and max from filtered data
    stats.min = sortedTimes.front();
    stats.max = sortedTimes.back();
    
    // Calculate throughput safely
    if (std::isfinite(stats.mean) && stats.mean > constants::precision::MACHINE_EPSILON && 
        std::isfinite(operationCount) && operationCount > 0.0) {
        stats.throughput = operationCount / stats.mean;
    } else {
        stats.throughput = 0.0;
    }
    
    return stats;
}

} // namespace libstats

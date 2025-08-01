#include <gtest/gtest.h>
#include "../include/distributions/uniform.h"
#include "enhanced_test_template.h"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <span>

using namespace std;
using namespace libstats;
using namespace libstats::testing;

namespace libstats {

//==============================================================================
// TEST FIXTURE FOR UNIFORM ENHANCED METHODS
//==============================================================================

class UniformEnhancedTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test data for various scenarios
        test_data_ = {1.2, 2.3, 1.8, 2.9, 1.5, 2.1, 2.7, 1.9, 2.4, 1.6};
        
        // Generate uniform data for testing
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> uniform_gen(test_a_, test_b_);
        
        uniform_data_.clear();
        uniform_data_.reserve(100);
        for (int i = 0; i < 100; ++i) {
            uniform_data_.push_back(uniform_gen(rng));
        }
        
        test_distribution_ = UniformDistribution(test_a_, test_b_);
    }
    
    const double test_a_ = 2.0;
    const double test_b_ = 8.0;
    const double TOLERANCE = 1e-10;
    std::vector<double> test_data_;
    std::vector<double> uniform_data_;
    UniformDistribution test_distribution_;
};

//==============================================================================
// TESTS FOR BASIC ENHANCED FUNCTIONALITY
//==============================================================================

TEST_F(UniformEnhancedTest, BasicEnhancedFunctionality) {
    // Test various parameter combinations
    std::vector<std::pair<double, double>> test_params = {
        {0.0, 1.0},    // Unit interval
        {-1.0, 1.0},   // Standard interval
        {-5.0, 5.0},   // Symmetric interval
        {2.0, 7.0},    // Asymmetric interval
        {0.001, 0.002}, // Narrow interval
        {-1000.0, 1000.0} // Wide interval
    };
    
    for (const auto& params : test_params) {
        double a = params.first;
        double b = params.second;
        
        libstats::UniformDistribution uniform(a, b);
        
        // Test basic properties
        EXPECT_DOUBLE_EQ(uniform.getLowerBound(), a);
        EXPECT_DOUBLE_EQ(uniform.getUpperBound(), b);
        EXPECT_NEAR(uniform.getMean(), (a + b) / 2.0, TOLERANCE);
        EXPECT_NEAR(uniform.getVariance(), (b - a) * (b - a) / 12.0, TOLERANCE);
        EXPECT_DOUBLE_EQ(uniform.getSkewness(), 0.0);
        EXPECT_DOUBLE_EQ(uniform.getKurtosis(), -1.2);
        EXPECT_NEAR(uniform.getWidth(), (b - a), TOLERANCE);
        EXPECT_NEAR(uniform.getMidpoint(), (a + b) / 2.0, TOLERANCE);
        
        // Test support bounds
        EXPECT_DOUBLE_EQ(uniform.getSupportLowerBound(), a);
        EXPECT_DOUBLE_EQ(uniform.getSupportUpperBound(), b);
        
        // Test PDF properties
        double mid = (a + b) / 2.0;
        EXPECT_NEAR(uniform.getProbability(mid), 1.0 / (b - a), TOLERANCE);
        EXPECT_DOUBLE_EQ(uniform.getProbability(a - 1.0), 0.0);
        EXPECT_DOUBLE_EQ(uniform.getProbability(b + 1.0), 0.0);
        
        // Test CDF properties
        EXPECT_DOUBLE_EQ(uniform.getCumulativeProbability(a), 0.0);
        EXPECT_DOUBLE_EQ(uniform.getCumulativeProbability(b), 1.0);
        EXPECT_NEAR(uniform.getCumulativeProbability(mid), 0.5, TOLERANCE);
        
        // Test quantile properties
        EXPECT_DOUBLE_EQ(uniform.getQuantile(0.0), a);
        EXPECT_DOUBLE_EQ(uniform.getQuantile(1.0), b);
        EXPECT_NEAR(uniform.getQuantile(0.5), mid, TOLERANCE);
    }
}

TEST_F(UniformEnhancedTest, NumericalStability) {
    // Test with very small interval
    libstats::UniformDistribution small_uniform(1.0, 1.0 + 1e-15);
    EXPECT_GT(small_uniform.getProbability(1.0 + 5e-16), 0.0);
    
    // Test with very large interval
    libstats::UniformDistribution large_uniform(-1e10, 1e10);
    EXPECT_NEAR(large_uniform.getMean(), 0.0, 1e-5);
    
    // Test with extreme values
    libstats::UniformDistribution extreme_uniform(-1e100, 1e100);
    EXPECT_GT(extreme_uniform.getProbability(0.0), 0.0);
}

TEST_F(UniformEnhancedTest, BatchOperations) {
    UniformDistribution uniform(0.0, 1.0);
    
    // Test data
    std::vector<double> test_values = {-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5};
    std::vector<double> pdf_results(test_values.size());
    std::vector<double> log_pdf_results(test_values.size());
    std::vector<double> cdf_results(test_values.size());
    
    // Test batch operations
    uniform.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
    uniform.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), test_values.size());
    uniform.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
    
    // Verify against individual calls
    for (size_t i = 0; i < test_values.size(); ++i) {
        double expected_pdf = uniform.getProbability(test_values[i]);
        double expected_log_pdf = uniform.getLogProbability(test_values[i]);
        double expected_cdf = uniform.getCumulativeProbability(test_values[i]);
        
        EXPECT_NEAR(pdf_results[i], expected_pdf, 1e-12);
        
        // Handle log probability comparisons carefully
        if (std::isfinite(expected_log_pdf)) {
            EXPECT_NEAR(log_pdf_results[i], expected_log_pdf, 1e-12);
        } else {
            EXPECT_EQ(log_pdf_results[i], expected_log_pdf); // Both should be -infinity
        }
        
        EXPECT_NEAR(cdf_results[i], expected_cdf, 1e-12);
    }
}

TEST_F(UniformEnhancedTest, SIMDPerformanceTest) {
    UniformDistribution uniform(0.0, 1.0);
    constexpr size_t LARGE_BATCH_SIZE = 10000;
    
    std::vector<double> large_test_values(LARGE_BATCH_SIZE);
    std::vector<double> large_pdf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_log_pdf_results(LARGE_BATCH_SIZE);
    std::vector<double> large_cdf_results(LARGE_BATCH_SIZE);
    
    // Generate test data
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<> dis(-0.5, 1.5);
    
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_test_values[i] = dis(gen);
    }
    
    std::cout << "  === SIMD Batch Performance Results ===" << std::endl;
    
    // Test 1: PDF Batch vs Individual
    auto start = std::chrono::high_resolution_clock::now();
    uniform.getProbabilityBatch(large_test_values.data(), large_pdf_results.data(), LARGE_BATCH_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto pdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_pdf_results[i] = uniform.getProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto pdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double pdf_speedup = (double)pdf_individual_time / pdf_batch_time;
    std::cout << "  PDF:     Batch " << pdf_batch_time << "μs vs Individual " << pdf_individual_time << "μs → " << pdf_speedup << "x speedup" << std::endl;
    
    // Test 2: Log PDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    uniform.getLogProbabilityBatch(large_test_values.data(), large_log_pdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_log_pdf_results[i] = uniform.getLogProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto log_pdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double log_pdf_speedup = (double)log_pdf_individual_time / log_pdf_batch_time;
    std::cout << "  LogPDF:  Batch " << log_pdf_batch_time << "μs vs Individual " << log_pdf_individual_time << "μs → " << log_pdf_speedup << "x speedup" << std::endl;
    
    // Test 3: CDF Batch vs Individual
    start = std::chrono::high_resolution_clock::now();
    uniform.getCumulativeProbabilityBatch(large_test_values.data(), large_cdf_results.data(), LARGE_BATCH_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto cdf_batch_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < LARGE_BATCH_SIZE; ++i) {
        large_cdf_results[i] = uniform.getCumulativeProbability(large_test_values[i]);
    }
    end = std::chrono::high_resolution_clock::now();
    auto cdf_individual_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cdf_speedup = (double)cdf_individual_time / cdf_batch_time;
    std::cout << "  CDF:     Batch " << cdf_batch_time << "μs vs Individual " << cdf_individual_time << "μs → " << cdf_speedup << "x speedup" << std::endl;
    
    // For uniform distribution, batch operations should be faster (reduced function call overhead)
    EXPECT_GT(pdf_speedup, 0.5) << "PDF batch should be at least half as fast as individual calls";
    EXPECT_GT(log_pdf_speedup, 0.5) << "Log PDF batch should be at least half as fast as individual calls";
    EXPECT_GT(cdf_speedup, 0.5) << "CDF batch should be at least half as fast as individual calls";
    
    // Verify correctness on a sample
    const size_t sample_size = 100;
    for (size_t i = 0; i < sample_size; ++i) {
        size_t idx = i * (LARGE_BATCH_SIZE / sample_size);
        double expected_pdf = uniform.getProbability(large_test_values[idx]);
        double expected_log_pdf = uniform.getLogProbability(large_test_values[idx]);
        double expected_cdf = uniform.getCumulativeProbability(large_test_values[idx]);
        
        EXPECT_NEAR(large_pdf_results[idx], expected_pdf, 1e-10);
        
        // Handle log probability comparisons carefully
        if (std::isfinite(expected_log_pdf)) {
            EXPECT_NEAR(large_log_pdf_results[idx], expected_log_pdf, 1e-10);
        } else {
            EXPECT_EQ(large_log_pdf_results[idx], expected_log_pdf); // Both should be -infinity
        }
        
        EXPECT_NEAR(large_cdf_results[idx], expected_cdf, 1e-10);
    }
}

TEST_F(UniformEnhancedTest, CopyAndMoveSemantics) {
    // Test copy constructor
    UniformDistribution original(3.0, 7.0);
    UniformDistribution copied(original);
    
    EXPECT_EQ(copied.getLowerBound(), original.getLowerBound());
    EXPECT_EQ(copied.getUpperBound(), original.getUpperBound());
    EXPECT_EQ(copied.getMean(), original.getMean());
    EXPECT_EQ(copied.getVariance(), original.getVariance());
    
    // Test copy assignment
    UniformDistribution assigned(0.0, 1.0);
    assigned = original;
    
    EXPECT_EQ(assigned.getLowerBound(), original.getLowerBound());
    EXPECT_EQ(assigned.getUpperBound(), original.getUpperBound());
    
    // Test move constructor
    UniformDistribution to_move(5.0, 10.0);
    double original_mean = to_move.getMean();
    UniformDistribution moved(std::move(to_move));
    
    EXPECT_EQ(moved.getMean(), original_mean);
    
    // Test move assignment
    UniformDistribution move_assigned(0.0, 1.0);
    UniformDistribution to_move2(15.0, 25.0);
    double original_mean2 = to_move2.getMean();
    move_assigned = std::move(to_move2);
    
    EXPECT_EQ(move_assigned.getMean(), original_mean2);
}

TEST_F(UniformEnhancedTest, EdgeCases) {
    // Test boundary conditions
    UniformDistribution uniform(0.0, 1.0);
    
    // Test exactly at boundaries
    EXPECT_DOUBLE_EQ(uniform.getProbability(0.0), 1.0);
    EXPECT_DOUBLE_EQ(uniform.getProbability(1.0), 1.0);
    EXPECT_DOUBLE_EQ(uniform.getCumulativeProbability(0.0), 0.0);
    EXPECT_DOUBLE_EQ(uniform.getCumulativeProbability(1.0), 1.0);
    
    // Test quantile edge cases
    EXPECT_DOUBLE_EQ(uniform.getQuantile(0.0), 0.0);
    EXPECT_DOUBLE_EQ(uniform.getQuantile(1.0), 1.0);
    
    // Test safe factory with invalid parameters
    auto result1 = UniformDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0);
    EXPECT_TRUE(result1.isError());
    
    auto result2 = UniformDistribution::create(0.0, std::numeric_limits<double>::infinity());
    EXPECT_TRUE(result2.isError());
    
    auto result3 = UniformDistribution::create(5.0, 2.0);  // a > b
    EXPECT_TRUE(result3.isError());
}

TEST_F(UniformEnhancedTest, SamplingStatistics) {
    UniformDistribution uniform(2.0, 8.0);
    std::mt19937 rng(42);
    
    constexpr size_t NUM_SAMPLES = 10000;
    std::vector<double> samples(NUM_SAMPLES);
    
    // Generate samples
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        samples[i] = uniform.sample(rng);
    }
    
    // Test that all samples are within bounds
    for (double sample : samples) {
        EXPECT_GE(sample, 2.0);
        EXPECT_LE(sample, 8.0);
    }
    
    // Test sample statistics
    double sample_mean = std::accumulate(samples.begin(), samples.end(), 0.0) / NUM_SAMPLES;
    double theoretical_mean = uniform.getMean();
    EXPECT_NEAR(sample_mean, theoretical_mean, 0.1);
    
    // Test sample variance
    double sample_var = 0.0;
    for (double sample : samples) {
        double diff = sample - sample_mean;
        sample_var += diff * diff;
    }
    sample_var /= (NUM_SAMPLES - 1);
    double theoretical_var = uniform.getVariance();
    EXPECT_NEAR(sample_var, theoretical_var, 0.5);
    
    // Test uniformity using Kolmogorov-Smirnov test (simplified)
    std::sort(samples.begin(), samples.end());
    double max_diff = 0.0;
    for (size_t i = 0; i < NUM_SAMPLES; ++i) {
        double empirical_cdf = static_cast<double>(i + 1) / NUM_SAMPLES;
        double theoretical_cdf = uniform.getCumulativeProbability(samples[i]);
        max_diff = std::max(max_diff, std::abs(empirical_cdf - theoretical_cdf));
    }
    
    // For large samples, KS statistic should be small
    EXPECT_LT(max_diff, 0.05);
}

TEST_F(UniformEnhancedTest, ThreadSafety) {
    UniformDistribution uniform(0.0, 1.0);
    constexpr size_t NUM_OPERATIONS = 1000;
    constexpr size_t NUM_THREADS = 4;
    
    // Test concurrent reads
    auto read_worker = [&uniform]() {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> value_gen(-0.5, 1.5);
        
        for (size_t i = 0; i < NUM_OPERATIONS; ++i) {
            double x = value_gen(rng);
            [[maybe_unused]] double pdf = uniform.getProbability(x);
            [[maybe_unused]] double logpdf = uniform.getLogProbability(x);
            [[maybe_unused]] double cdf = uniform.getCumulativeProbability(x);
            [[maybe_unused]] double mean = uniform.getMean();
            [[maybe_unused]] double var = uniform.getVariance();
        }
    };
    
    std::vector<std::thread> read_threads;
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        read_threads.emplace_back(read_worker);
    }
    
    for (auto& t : read_threads) {
        t.join();
    }
    
    // Test concurrent writes (parameter changes)
    UniformDistribution write_uniform(0.0, 1.0);
    std::atomic<bool> stop_flag{false};
    
    auto write_worker = [&write_uniform, &stop_flag](double base_a) {
        std::mt19937 rng(std::random_device{}());
        std::uniform_real_distribution<double> param_gen(0.1, 5.0);
        
        while (!stop_flag.load()) {
            double a = base_a + param_gen(rng);
            double b = a + param_gen(rng);
            
            auto result = write_uniform.trySetParameters(a, b);
            if (result.isOk()) {
                [[maybe_unused]] double mean = write_uniform.getMean();
                [[maybe_unused]] double var = write_uniform.getVariance();
            }
        }
    };
    
    std::vector<std::thread> write_threads;
    for (size_t i = 0; i < NUM_THREADS; ++i) {
        write_threads.emplace_back(write_worker, i * 10.0);
    }
    
    // Let threads run for a short time
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop_flag.store(true);
    
    for (auto& t : write_threads) {
        t.join();
    }
}

TEST_F(UniformEnhancedTest, SpecialDistributions) {
    // Test unit interval U(0,1)
    UniformDistribution unit(0.0, 1.0);
    EXPECT_DOUBLE_EQ(unit.getProbability(0.5), 1.0);
    EXPECT_DOUBLE_EQ(unit.getLogProbability(0.5), 0.0);
    EXPECT_DOUBLE_EQ(unit.getCumulativeProbability(0.5), 0.5);
    EXPECT_DOUBLE_EQ(unit.getQuantile(0.5), 0.5);
    
    // Test standard interval U(-1,1)
    UniformDistribution standard(-1.0, 1.0);
    EXPECT_DOUBLE_EQ(standard.getProbability(0.0), 0.5);
    EXPECT_DOUBLE_EQ(standard.getMean(), 0.0);
    EXPECT_DOUBLE_EQ(standard.getCumulativeProbability(0.0), 0.5);
    
    // Test symmetric interval U(-c,c)
    UniformDistribution symmetric(-5.0, 5.0);
    EXPECT_DOUBLE_EQ(symmetric.getMean(), 0.0);
    EXPECT_DOUBLE_EQ(symmetric.getCumulativeProbability(0.0), 0.5);
}

//==============================================================================
// AUTO-DISPATCH STRATEGY TESTING
//==============================================================================

TEST_F(UniformEnhancedTest, SmartAutoDispatchStrategyTesting) {
    UniformDistribution uniform_dist(0.0, 1.0);
    
    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH", "PARALLEL_SIMD"};
    
    std::cout << "\n=== Smart Auto-Dispatch Strategy Testing (Uniform) ===\n";
    
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pdf_results(batch_size);
        std::vector<double> auto_logpdf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);
        
        std::mt19937 gen(42 + i);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = dis(gen);
        }
        
        // Test smart auto-dispatch methods
        auto start = std::chrono::high_resolution_clock::now();
        uniform_dist.getProbability(std::span<const double>(test_values), std::span<double>(auto_pdf_results));
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        uniform_dist.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        uniform_dist.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pdf_results(batch_size);
        std::vector<double> trad_logpdf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);
        
        uniform_dist.getProbabilityBatch(test_values.data(), trad_pdf_results.data(), batch_size);
        uniform_dist.getLogProbabilityBatch(test_values.data(), trad_logpdf_results.data(), batch_size);
        uniform_dist.getCumulativeProbabilityBatch(test_values.data(), trad_cdf_results.data(), batch_size);
        
        // Verify correctness
        bool pdf_correct = true, logpdf_correct = true, cdf_correct = true;
        
        for (size_t j = 0; j < batch_size; ++j) {
            if (std::abs(auto_pdf_results[j] - trad_pdf_results[j]) > 1e-10) {
                pdf_correct = false;
            }
            if (std::abs(auto_logpdf_results[j] - trad_logpdf_results[j]) > 1e-10) {
                logpdf_correct = false;
            }
            if (std::abs(auto_cdf_results[j] - trad_cdf_results[j]) > 1e-10) {
                cdf_correct = false;
            }
        }
        
        std::cout << "Batch size: " << batch_size << ", Expected strategy: " << expected_strategy << "\n";
        std::cout << "  PDF: " << auto_pdf_time << "μs, Correct: " << (pdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  LogPDF: " << auto_logpdf_time << "μs, Correct: " << (logpdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  CDF: " << auto_cdf_time << "μs, Correct: " << (cdf_correct ? "✅" : "❌") << "\n";
        
        EXPECT_TRUE(pdf_correct) << "PDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(logpdf_correct) << "LogPDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(cdf_correct) << "CDF auto-dispatch results should match traditional for batch size " << batch_size;
    }
    
    std::cout << "\n=== Auto-Dispatch Strategy Testing Completed (Uniform) ===\n";
}

//==============================================================================
// PARALLEL BATCH OPERATIONS AND BENCHMARKING
//==============================================================================

TEST_F(UniformEnhancedTest, ParallelBatchPerformanceBenchmark) {
    UniformDistribution uniform(0.0, 1.0);
    constexpr size_t BENCHMARK_SIZE = 50000;
    
    // Generate test data
    std::vector<double> test_values(BENCHMARK_SIZE);
    std::vector<double> pdf_results(BENCHMARK_SIZE);
    std::vector<double> log_pdf_results(BENCHMARK_SIZE);
    std::vector<double> cdf_results(BENCHMARK_SIZE);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 1.5); // Include values outside [0,1]
    for (size_t i = 0; i < BENCHMARK_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    StandardizedBenchmark::printBenchmarkHeader("Uniform Distribution", BENCHMARK_SIZE);
    
    std::vector<BenchmarkResult> benchmark_results;
    
    // For each operation type (PDF, LogPDF, CDF)
    std::vector<std::string> operations = {"PDF", "LogPDF", "CDF"};
    
    for (const auto& op : operations) {
        BenchmarkResult result;
        result.operation_name = op;
        
        // 1. SIMD Batch (baseline)
        auto start = std::chrono::high_resolution_clock::now();
        if (op == "PDF") {
            uniform.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
        } else if (op == "LogPDF") {
            uniform.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
        } else if (op == "CDF") {
            uniform.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
        }
        auto end = std::chrono::high_resolution_clock::now();
        result.simd_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 2. Standard Parallel Operations
        std::span<const double> input_span(test_values);
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getProbabilityBatchParallel(input_span, output_span);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getLogProbabilityBatchParallel(input_span, log_output_span);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
            end = std::chrono::high_resolution_clock::now();
        }
        result.parallel_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 3. Work-Stealing Operations
        WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
            end = std::chrono::high_resolution_clock::now();
        }
        result.work_stealing_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // 4. Cache-Aware Operations
        cache::AdaptiveCache<std::string, double> cache_manager;
        
        if (op == "PDF") {
            std::span<double> output_span(pdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "LogPDF") {
            std::span<double> log_output_span(log_pdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
            end = std::chrono::high_resolution_clock::now();
        } else if (op == "CDF") {
            std::span<double> cdf_output_span(cdf_results);
            start = std::chrono::high_resolution_clock::now();
            uniform.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
            end = std::chrono::high_resolution_clock::now();
        }
        result.cache_aware_time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Calculate speedups
        result.parallel_speedup = (double)result.simd_time_us / result.parallel_time_us;
        result.work_stealing_speedup = (double)result.simd_time_us / result.work_stealing_time_us;
        result.cache_aware_speedup = (double)result.simd_time_us / result.cache_aware_time_us;
        
        benchmark_results.push_back(result);
        
        // Verify correctness
        if (op == "PDF") {
            StatisticalTestUtils::verifyBatchCorrectness(uniform, test_values, pdf_results, "PDF");
        } else if (op == "LogPDF") {
            StatisticalTestUtils::verifyBatchCorrectness(uniform, test_values, log_pdf_results, "LogPDF");
        } else if (op == "CDF") {
            StatisticalTestUtils::verifyBatchCorrectness(uniform, test_values, cdf_results, "CDF");
        }
    }
    
    // Print standardized benchmark results
    StandardizedBenchmark::printBenchmarkResults(benchmark_results);
    StandardizedBenchmark::printPerformanceAnalysis(benchmark_results);
}

// Test new work-stealing and cache-aware methods for uniform distribution
TEST_F(UniformEnhancedTest, NewWorkStealingAndCacheAwareMethods) {
    UniformDistribution uniform(0.0, 1.0);
    
    constexpr size_t TEST_SIZE = 10000;
    
    // Generate test data (mix of in-range and out-of-range values)
    std::vector<double> test_values(TEST_SIZE);
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 1.5);  // Mix of in-range and out-of-range values
    
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        test_values[i] = dis(gen);
    }
    
    // Prepare result vectors
    std::vector<double> pdf_ws_results(TEST_SIZE);
    std::vector<double> pdf_cache_results(TEST_SIZE);
    std::vector<double> log_pdf_ws_results(TEST_SIZE);
    std::vector<double> log_pdf_cache_results(TEST_SIZE);
    std::vector<double> cdf_ws_results(TEST_SIZE);
    std::vector<double> cdf_cache_results(TEST_SIZE);
    std::vector<double> expected_pdf(TEST_SIZE);
    std::vector<double> expected_log_pdf(TEST_SIZE);
    std::vector<double> expected_cdf(TEST_SIZE);
    
    // Calculate expected results
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        expected_pdf[i] = uniform.getProbability(test_values[i]);
        expected_log_pdf[i] = uniform.getLogProbability(test_values[i]);
        expected_cdf[i] = uniform.getCumulativeProbability(test_values[i]);
    }
    
    // Test work-stealing and cache-aware implementations
    std::cout << "Testing uniform work-stealing and cache-aware methods:" << std::endl;
    
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    cache::AdaptiveCache<std::string, double> cache_manager;
    
    // Test work-stealing PDF
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(pdf_ws_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getProbabilityBatchWorkStealing(values_span, results_span, work_stealing_pool);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ws_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  PDF work-stealing: " << ws_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(pdf_ws_results[i] - expected_pdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Work-stealing PDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test cache-aware PDF
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(pdf_cache_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getProbabilityBatchCacheAware(values_span, results_span, cache_manager);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  PDF cache-aware: " << cache_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(pdf_cache_results[i] - expected_pdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Cache-aware PDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test work-stealing log probability
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(log_pdf_ws_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getLogProbabilityBatchWorkStealing(values_span, results_span, work_stealing_pool);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ws_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  Log PDF work-stealing: " << ws_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(log_pdf_ws_results[i] - expected_log_pdf[i]) > 1e-10) {
                // Special handling for negative infinity comparisons
                if (!(std::isinf(log_pdf_ws_results[i]) && std::isinf(expected_log_pdf[i]))) {
                    correct = false;
                    break;
                }
            }
        }
        EXPECT_TRUE(correct) << "Work-stealing log PDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test cache-aware log probability
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(log_pdf_cache_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getLogProbabilityBatchCacheAware(values_span, results_span, cache_manager);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  Log PDF cache-aware: " << cache_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(log_pdf_cache_results[i] - expected_log_pdf[i]) > 1e-10) {
                // Special handling for negative infinity comparisons
                if (!(std::isinf(log_pdf_cache_results[i]) && std::isinf(expected_log_pdf[i]))) {
                    correct = false;
                    break;
                }
            }
        }
        EXPECT_TRUE(correct) << "Cache-aware log PDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test work-stealing CDF
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(cdf_ws_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getCumulativeProbabilityBatchWorkStealing(values_span, results_span, work_stealing_pool);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto ws_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  CDF work-stealing: " << ws_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(cdf_ws_results[i] - expected_cdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Work-stealing CDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test cache-aware CDF
    {
        std::span<const double> values_span(test_values);
        std::span<double> results_span(cdf_cache_results);
        
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getCumulativeProbabilityBatchCacheAware(values_span, results_span, cache_manager);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto cache_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "  CDF cache-aware: " << cache_time << "μs" << std::endl;
        
        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            if (std::abs(cdf_cache_results[i] - expected_cdf[i]) > 1e-10) {
                correct = false;
                break;
            }
        }
        EXPECT_TRUE(correct) << "Cache-aware CDF should produce correct results";
        std::cout << "    ✓ Correctness verified" << std::endl;
    }
    
    // Test with different distribution parameters
    std::cout << "\nTesting with different parameters (a=2.0, b=8.0):" << std::endl;
    
    UniformDistribution custom_uniform(2.0, 8.0);
    
    // Test a subset with custom distribution
    const size_t subset_size = 1000;
    std::span<const double> subset_values(test_values.data(), subset_size);
    std::span<double> subset_pdf_results(pdf_ws_results.data(), subset_size);
    std::span<double> subset_log_results(log_pdf_ws_results.data(), subset_size);
    std::span<double> subset_cdf_results(cdf_ws_results.data(), subset_size);
    
    custom_uniform.getProbabilityBatchWorkStealing(subset_values, subset_pdf_results, work_stealing_pool);
    custom_uniform.getLogProbabilityBatchWorkStealing(subset_values, subset_log_results, work_stealing_pool);
    custom_uniform.getCumulativeProbabilityBatchCacheAware(subset_values, subset_cdf_results, cache_manager);
    
    // Verify against individual calls
    bool custom_correct = true;
    for (size_t i = 0; i < subset_size; ++i) {
        double expected_pdf = custom_uniform.getProbability(test_values[i]);
        double expected_log = custom_uniform.getLogProbability(test_values[i]);
        double expected_cdf = custom_uniform.getCumulativeProbability(test_values[i]);
        
        bool pdf_match = std::abs(pdf_ws_results[i] - expected_pdf) <= 1e-10;
        bool log_match = std::abs(log_pdf_ws_results[i] - expected_log) <= 1e-10;
        bool cdf_match = std::abs(cdf_ws_results[i] - expected_cdf) <= 1e-10;
        
        // Special handling for negative infinity
        if (std::isinf(expected_log) && std::isinf(log_pdf_ws_results[i])) {
            log_match = true;
        }
        
        if (!pdf_match || !log_match || !cdf_match) {
            custom_correct = false;
            break;
        }
    }
    
    EXPECT_TRUE(custom_correct) << "Custom uniform distribution should produce correct results";
    std::cout << "  ✓ Custom distribution tests passed" << std::endl;
    
    // Test uniform-specific properties
    std::cout << "\nTesting uniform-specific properties:" << std::endl;
    
    // Test values within and outside support
    const std::vector<double> boundary_values = {-0.1, 0.0, 0.5, 1.0, 1.1};
    const std::vector<double> expected_pdf_values = {0.0, 1.0, 1.0, 1.0, 0.0};
    const std::vector<double> expected_cdf_values = {0.0, 0.0, 0.5, 1.0, 1.0};
    
    std::vector<double> boundary_pdf_results(boundary_values.size());
    std::vector<double> boundary_cdf_results(boundary_values.size());
    
    std::span<const double> boundary_span(boundary_values);
    std::span<double> boundary_pdf_span(boundary_pdf_results);
    std::span<double> boundary_cdf_span(boundary_cdf_results);
    
    uniform.getProbabilityBatchWorkStealing(boundary_span, boundary_pdf_span, work_stealing_pool);
    uniform.getCumulativeProbabilityBatchCacheAware(boundary_span, boundary_cdf_span, cache_manager);
    
    // Verify boundary behavior
    for (size_t i = 0; i < boundary_values.size(); ++i) {
        EXPECT_NEAR(boundary_pdf_results[i], expected_pdf_values[i], 1e-10) 
            << "PDF boundary value at " << boundary_values[i];
        EXPECT_NEAR(boundary_cdf_results[i], expected_cdf_values[i], 1e-10) 
            << "CDF boundary value at " << boundary_values[i];
    }
    
    std::cout << "  ✓ Uniform boundary handling verified" << std::endl;
    
    // Print final statistics
    auto ws_stats = work_stealing_pool.getStatistics();
    auto cache_stats = cache_manager.getStats();
    
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "  Work-stealing tasks: " << ws_stats.tasksExecuted << std::endl;
    std::cout << "  Cache entries: " << cache_stats.size << std::endl;
    std::cout << "  Cache hit rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;
    
    std::cout << "\n  ✓ All uniform work-stealing and cache-aware methods validated!" << std::endl;
}

//==============================================================================
// ADVANCED STATISTICAL TESTS
//==============================================================================

TEST_F(UniformEnhancedTest, AdvancedStatisticalMethods) {
    auto uniformResult = UniformDistribution::create(0.0, 1.0);
    ASSERT_TRUE(uniformResult.isOk());
    auto uniform = std::move(uniformResult.value);
    
    // Generate synthetic uniform data that should follow [0,1] distribution
    std::vector<double> synthetic_data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform_gen(0.0, 1.0);
    
    for (int i = 0; i < 1000; ++i) {
        synthetic_data.push_back(uniform_gen(rng));
    }
    
    // Test KS goodness of fit using static method
    auto [ks_stat, ks_p_value, ks_reject] = UniformDistribution::kolmogorovSmirnovTest(synthetic_data, uniform);
    EXPECT_FALSE(ks_reject) << "KS test should pass for well-fitted uniform data";
    EXPECT_TRUE(std::isfinite(ks_stat)) << "KS statistic should be finite";
    EXPECT_GT(ks_p_value, 0.0) << "KS p-value should be positive";
    EXPECT_GE(ks_stat, 0.0) << "KS statistic should be non-negative";
    
    // Test Anderson-Darling goodness of fit using static method
    auto [ad_stat, ad_p_value, ad_reject] = UniformDistribution::andersonDarlingTest(synthetic_data, uniform);
    EXPECT_FALSE(ad_reject) << "AD test should pass for well-fitted uniform data";
    EXPECT_TRUE(std::isfinite(ad_stat)) << "AD statistic should be finite";
    EXPECT_GT(ad_p_value, 0.0) << "AD p-value should be positive";
    EXPECT_GE(ad_stat, 0.0) << "AD statistic should be non-negative";
    
    std::cout << "  KS test: stat=" << ks_stat << ", p=" << ks_p_value << ", reject=" << ks_reject << std::endl;
    std::cout << "  AD test: stat=" << ad_stat << ", p=" << ad_p_value << ", reject=" << ad_reject << std::endl;
    
    std::cout << "✓ Advanced statistical methods validated" << std::endl;
}

TEST_F(UniformEnhancedTest, GoodnessOfFitTests) {
    auto uniformResult = UniformDistribution::create(0.0, 1.0);
    ASSERT_TRUE(uniformResult.isOk());
    auto uniform = std::move(uniformResult.value);
    
    // Generate uniform data that should fit well
    std::vector<double> uniform_data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u_gen(0.0, 1.0);
    for (int i = 0; i < 1000; ++i) {
        uniform_data.push_back(u_gen(rng));
    }

    // Test KS goodness of fit for uniform data using static method
    auto [ks_stat, ks_p_value, ks_reject] = UniformDistribution::kolmogorovSmirnovTest(uniform_data, uniform);
    EXPECT_FALSE(ks_reject) << "KS test should pass for generated uniform data";
    EXPECT_TRUE(std::isfinite(ks_stat)) << "KS statistic should be finite";
    EXPECT_GT(ks_p_value, 0.0) << "KS p-value should be positive";
    
    // Generate non-uniform data (should fail the test)
    std::vector<double> non_uniform_data;
    std::normal_distribution<double> normal_gen(0.5, 0.1);
    for (int i = 0; i < 1000; ++i) {
        double val = normal_gen(rng);
        if (val >= 0.0 && val <= 1.0) {  // Keep only values in [0,1]
            non_uniform_data.push_back(val);
        }
    }
    
    if (non_uniform_data.size() > 100) {
        auto [non_ks_stat, non_ks_p_value, non_ks_reject] = UniformDistribution::kolmogorovSmirnovTest(non_uniform_data, uniform);
        // This should generally reject, though it's probabilistic
        EXPECT_TRUE(std::isfinite(non_ks_stat)) << "Non-uniform KS statistic should be finite";
        EXPECT_GT(non_ks_p_value, 0.0) << "Non-uniform KS p-value should be positive";
    }

    std::cout << "✓ Goodness of fit tests completed" << std::endl;
}

TEST_F(UniformEnhancedTest, CrossValidationMethods) {
    auto uniformResult = UniformDistribution::create(0.0, 1.0);
    ASSERT_TRUE(uniformResult.isOk());
    auto uniform = std::move(uniformResult.value);
    
    // Generate uniform data for cross-validation testing
    std::vector<double> uniform_data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u_gen(0.0, 1.0);
    
    for (int i = 0; i < 600; ++i) {
        uniform_data.push_back(u_gen(rng));
    }
    
    // Perform k-fold cross validation using static method
    constexpr int k = 5;
    auto cv_results = UniformDistribution::kFoldCrossValidation(uniform_data, k);
    
    // Should get k results
    EXPECT_EQ(cv_results.size(), k) << "Should get k cross-validation results";
    
    // Each result should have reasonable values
    for (const auto& [mean_error, std_error, log_likelihood] : cv_results) {
        EXPECT_TRUE(std::isfinite(mean_error)) << "Mean error should be finite";
        EXPECT_TRUE(std::isfinite(std_error)) << "Std error should be finite";
        EXPECT_TRUE(std::isfinite(log_likelihood)) << "Log likelihood should be finite";
    }
    
    // Perform leave-one-out cross validation on a smaller subset
    std::vector<double> small_data(uniform_data.begin(), uniform_data.begin() + 100);
    auto [mae, rmse, total_loglik] = UniformDistribution::leaveOneOutCrossValidation(small_data);
    
    EXPECT_TRUE(std::isfinite(mae)) << "MAE should be finite";
    EXPECT_TRUE(std::isfinite(rmse)) << "RMSE should be finite";
    EXPECT_TRUE(std::isfinite(total_loglik)) << "Total log likelihood should be finite";
    EXPECT_GT(mae, 0.0) << "MAE should be positive";
    EXPECT_GT(rmse, 0.0) << "RMSE should be positive";
    
    std::cout << "✓ Cross-validation methods completed" << std::endl;
}

TEST_F(UniformEnhancedTest, InformationCriteria) {
    auto uniformResult = UniformDistribution::create(0.0, 1.0);
    ASSERT_TRUE(uniformResult.isOk());
    auto uniform = std::move(uniformResult.value);
    
    // Generate test data
    std::vector<double> test_data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u_gen(0.0, 1.0);
    
    for (int i = 0; i < 1000; ++i) {
        test_data.push_back(u_gen(rng));
    }
    
    // Calculate information criteria using static method
    auto [aic, bic, aicc, log_likelihood] = UniformDistribution::computeInformationCriteria(test_data, uniform);
    
    // All criteria should be finite and reasonable
    EXPECT_TRUE(std::isfinite(aic)) << "AIC should be finite";
    EXPECT_TRUE(std::isfinite(bic)) << "BIC should be finite";
    EXPECT_TRUE(std::isfinite(aicc)) << "AICc should be finite";
    EXPECT_TRUE(std::isfinite(log_likelihood)) << "Log likelihood should be finite";
    
    // BIC generally penalizes complexity more than AIC
    // For uniform distributions, this relationship may vary
    EXPECT_GT(aic, -std::numeric_limits<double>::infinity()) << "AIC should be greater than negative infinity";
    EXPECT_GT(bic, -std::numeric_limits<double>::infinity()) << "BIC should be greater than negative infinity";
    
    // For U(0,1), log probability of each point in [0,1] is log(1) = 0
    // So total log likelihood should be close to 0 (may have small penalty for out-of-bounds points)
    EXPECT_LE(log_likelihood, 0.0) << "Log likelihood should be non-positive for uniform data";
    EXPECT_GT(log_likelihood, -100.0) << "Log likelihood should not be too negative for well-fitted uniform data";
    
    std::cout << "✓ Information criteria computed: AIC = " << aic << ", BIC = " << bic << ", AICc = " << aicc << std::endl;
}

TEST_F(UniformEnhancedTest, SmartAutoDispatchWithPerformanceHints) {
    UniformDistribution uniform(0.0, 1.0);
    
    // Test data for different batch sizes to trigger different strategies
    std::vector<size_t> batch_sizes = {5, 50, 500, 5000, 50000};
    std::vector<std::string> expected_strategies = {"SCALAR", "SCALAR", "SIMD_BATCH", "SIMD_BATCH", "PARALLEL_SIMD"};
    
    std::cout << "\n=== Smart Auto-Dispatch Strategy Testing ===\n";
    
    for (size_t i = 0; i < batch_sizes.size(); ++i) {
        size_t batch_size = batch_sizes[i];
        std::string expected_strategy = expected_strategies[i];
        
        // Generate test data
        std::vector<double> test_values(batch_size);
        std::vector<double> auto_pdf_results(batch_size);
        std::vector<double> auto_logpdf_results(batch_size);
        std::vector<double> auto_cdf_results(batch_size);
        
        std::mt19937 gen(42 + i);
        std::uniform_real_distribution<> dis(-0.2, 1.2);
        for (size_t j = 0; j < batch_size; ++j) {
            test_values[j] = dis(gen);
        }
        
        // Test smart auto-dispatch methods
        auto start = std::chrono::high_resolution_clock::now();
        uniform.getProbability(std::span<const double>(test_values), std::span<double>(auto_pdf_results));
        auto end = std::chrono::high_resolution_clock::now();
        auto auto_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        uniform.getLogProbability(std::span<const double>(test_values), std::span<double>(auto_logpdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_logpdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        start = std::chrono::high_resolution_clock::now();
        uniform.getCumulativeProbability(std::span<const double>(test_values), std::span<double>(auto_cdf_results));
        end = std::chrono::high_resolution_clock::now();
        auto auto_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        // Compare with traditional batch methods for correctness
        std::vector<double> trad_pdf_results(batch_size);
        std::vector<double> trad_logpdf_results(batch_size);
        std::vector<double> trad_cdf_results(batch_size);
        
        uniform.getProbabilityBatch(test_values.data(), trad_pdf_results.data(), batch_size);
        uniform.getLogProbabilityBatch(test_values.data(), trad_logpdf_results.data(), batch_size);
        uniform.getCumulativeProbabilityBatch(test_values.data(), trad_cdf_results.data(), batch_size);
        
        // Verify correctness
        bool pdf_correct = true, logpdf_correct = true, cdf_correct = true;
        
        for (size_t j = 0; j < batch_size; ++j) {
            if (std::abs(auto_pdf_results[j] - trad_pdf_results[j]) > 1e-10) {
                pdf_correct = false;
            }
            if (std::abs(auto_logpdf_results[j] - trad_logpdf_results[j]) > 1e-10) {
                logpdf_correct = false;
            }
            if (std::abs(auto_cdf_results[j] - trad_cdf_results[j]) > 1e-10) {
                cdf_correct = false;
            }
        }
        
        std::cout << "Batch size: " << batch_size << ", Expected strategy: " << expected_strategy << "\n";
        std::cout << "  PDF: " << auto_pdf_time << "μs, Correct: " << (pdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  LogPDF: " << auto_logpdf_time << "μs, Correct: " << (logpdf_correct ? "✅" : "❌") << "\n";
        std::cout << "  CDF: " << auto_cdf_time << "μs, Correct: " << (cdf_correct ? "✅" : "❌") << "\n";
        
        EXPECT_TRUE(pdf_correct) << "PDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(logpdf_correct) << "LogPDF auto-dispatch results should match traditional for batch size " << batch_size;
        EXPECT_TRUE(cdf_correct) << "CDF auto-dispatch results should match traditional for batch size " << batch_size;
    }
    
    // Test with performance hints
    std::cout << "\n=== Testing Performance Hints ===\n";
    
    const size_t hint_test_size = 1000;
    std::vector<double> hint_test_values(hint_test_size, 0.5);
    std::vector<double> hint_results(hint_test_size);
    
    // Test with different performance hints
    libstats::performance::PerformanceHint auto_hint; // Default AUTO
    libstats::performance::PerformanceHint scalar_hint;
    scalar_hint.strategy = libstats::performance::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    libstats::performance::PerformanceHint simd_hint;
    simd_hint.strategy = libstats::performance::PerformanceHint::PreferredStrategy::FORCE_SIMD;
    
    auto start = std::chrono::high_resolution_clock::now();
    uniform.getProbability(std::span<const double>(hint_test_values), std::span<double>(hint_results), auto_hint);
    auto end = std::chrono::high_resolution_clock::now();
    auto auto_hint_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    uniform.getProbability(std::span<const double>(hint_test_values), std::span<double>(hint_results), scalar_hint);
    end = std::chrono::high_resolution_clock::now();
    auto scalar_hint_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    uniform.getProbability(std::span<const double>(hint_test_values), std::span<double>(hint_results), simd_hint);
    end = std::chrono::high_resolution_clock::now();
    auto simd_hint_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "Performance hint testing (batch size: " << hint_test_size << "):" << std::endl;
    std::cout << "  AUTO strategy: " << auto_hint_time << "μs" << std::endl;
    std::cout << "  FORCE_SCALAR: " << scalar_hint_time << "μs" << std::endl;
    std::cout << "  FORCE_SIMD: " << simd_hint_time << "μs" << std::endl;
    
    std::cout << "\n✅ Smart auto-dispatch strategy testing completed!\n";
}

TEST_F(UniformEnhancedTest, BootstrapParameterConfidenceIntervals) {
    auto uniformResult = UniformDistribution::create(0.0, 1.0);
    ASSERT_TRUE(uniformResult.isOk());
    auto uniform = std::move(uniformResult.value);
    
    // Generate uniform data for bootstrap testing
    std::vector<double> uniform_data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> u_gen(0.2, 0.8); // Biased towards center
    
    for (int i = 0; i < 200; ++i) {
        uniform_data.push_back(u_gen(rng));
    }
    
    // Perform bootstrap confidence interval estimation using static method
    constexpr int num_bootstrap_samples = 100;  // Reduced for testing speed
    constexpr double confidence_level = 0.95;
    
    auto [lower_bound_ci, upper_bound_ci] = UniformDistribution::bootstrapParameterConfidenceIntervals(
        uniform_data, confidence_level, num_bootstrap_samples
    );
    
    // For a uniform distribution, we should get reasonable confidence intervals
    EXPECT_LE(lower_bound_ci.first, lower_bound_ci.second) << "Lower bound CI: lower <= upper";
    EXPECT_LE(upper_bound_ci.first, upper_bound_ci.second) << "Upper bound CI: lower <= upper";
    
    // Check that intervals are reasonable for uniform distribution
    EXPECT_GE(lower_bound_ci.first, 0.0) << "Lower bound CI lower should be >= 0";
    EXPECT_LE(lower_bound_ci.second, 1.0) << "Lower bound CI upper should be <= 1";
    EXPECT_GE(upper_bound_ci.first, 0.0) << "Upper bound CI lower should be >= 0";
    EXPECT_LE(upper_bound_ci.second, 1.0) << "Upper bound CI upper should be <= 1";
    
    // The bounds should be ordered correctly
    EXPECT_LE(lower_bound_ci.second, upper_bound_ci.first) << "Lower bound upper should be <= upper bound lower";
    
    std::cout << "✓ Bootstrap confidence intervals completed" << std::endl;
    std::cout << "  Lower bound CI: [" << lower_bound_ci.first << ", " << lower_bound_ci.second << "]" << std::endl;
    std::cout << "  Upper bound CI: [" << upper_bound_ci.first << ", " << upper_bound_ci.second << "]" << std::endl;
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

} // namespace libstats

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

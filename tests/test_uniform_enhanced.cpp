#include <gtest/gtest.h>
#include "../include/uniform.h"
#include "../include/work_stealing_pool.h"
#include "../include/adaptive_cache.h"
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
    
    std::cout << "\n=== Uniform Parallel Batch Operations Performance Benchmark ===" << std::endl;
    std::cout << "Dataset size: " << BENCHMARK_SIZE << " elements" << std::endl;
    
    // 1. Standard SIMD Batch Operations (baseline)
    auto start = std::chrono::high_resolution_clock::now();
    uniform.getProbabilityBatch(test_values.data(), pdf_results.data(), BENCHMARK_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto simd_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    uniform.getLogProbabilityBatch(test_values.data(), log_pdf_results.data(), BENCHMARK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto simd_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    uniform.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), BENCHMARK_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto simd_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::cout << "SIMD Batch (baseline):" << std::endl;
    std::cout << "  PDF:     " << simd_pdf_time << " μs" << std::endl;
    std::cout << "  LogPDF:  " << simd_log_pdf_time << " μs" << std::endl;
    std::cout << "  CDF:     " << simd_cdf_time << " μs" << std::endl;
    
    // 2. Standard Parallel Batch Operations
    std::span<const double> input_span(test_values);
    std::span<double> output_span(pdf_results);
    
    start = std::chrono::high_resolution_clock::now();
    uniform.getProbabilityBatchParallel(input_span, output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::span<double> log_output_span(log_pdf_results);
    start = std::chrono::high_resolution_clock::now();
    uniform.getLogProbabilityBatchParallel(input_span, log_output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    std::span<double> cdf_output_span(cdf_results);
    start = std::chrono::high_resolution_clock::now();
    uniform.getCumulativeProbabilityBatchParallel(input_span, cdf_output_span);
    end = std::chrono::high_resolution_clock::now();
    auto parallel_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double parallel_pdf_speedup = (double)simd_pdf_time / parallel_pdf_time;
    double parallel_log_pdf_speedup = (double)simd_log_pdf_time / parallel_log_pdf_time;
    double parallel_cdf_speedup = (double)simd_cdf_time / parallel_cdf_time;
    
    std::cout << "Standard Parallel:" << std::endl;
    std::cout << "  PDF:     " << parallel_pdf_time << " μs (" << parallel_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << parallel_log_pdf_time << " μs (" << parallel_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << parallel_cdf_time << " μs (" << parallel_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // 3. Work-Stealing Parallel Operations
    WorkStealingPool work_stealing_pool(std::thread::hardware_concurrency());
    
    // PDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    uniform.getProbabilityBatchWorkStealing(input_span, output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // LogPDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    uniform.getLogProbabilityBatchWorkStealing(input_span, log_output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CDF Work-Stealing
    start = std::chrono::high_resolution_clock::now();
    uniform.getCumulativeProbabilityBatchWorkStealing(input_span, cdf_output_span, work_stealing_pool);
    end = std::chrono::high_resolution_clock::now();
    auto work_steal_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double work_steal_pdf_speedup = (double)simd_pdf_time / work_steal_pdf_time;
    double work_steal_log_pdf_speedup = (double)simd_log_pdf_time / work_steal_log_pdf_time;
    double work_steal_cdf_speedup = (double)simd_cdf_time / work_steal_cdf_time;
    
    std::cout << "Work-Stealing Parallel:" << std::endl;
    std::cout << "  PDF:     " << work_steal_pdf_time << " μs (" << work_steal_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << work_steal_log_pdf_time << " μs (" << work_steal_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << work_steal_cdf_time << " μs (" << work_steal_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // 4. Cache-Aware Parallel Operations
    cache::AdaptiveCache<std::string, double> cache_manager;
    
    // PDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    uniform.getProbabilityBatchCacheAware(input_span, output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // LogPDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    uniform.getLogProbabilityBatchCacheAware(input_span, log_output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_log_pdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    // CDF Cache-Aware
    start = std::chrono::high_resolution_clock::now();
    uniform.getCumulativeProbabilityBatchCacheAware(input_span, cdf_output_span, cache_manager);
    end = std::chrono::high_resolution_clock::now();
    auto cache_aware_cdf_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double cache_aware_pdf_speedup = (double)simd_pdf_time / cache_aware_pdf_time;
    double cache_aware_log_pdf_speedup = (double)simd_log_pdf_time / cache_aware_log_pdf_time;
    double cache_aware_cdf_speedup = (double)simd_cdf_time / cache_aware_cdf_time;
    
    std::cout << "Cache-Aware Parallel:" << std::endl;
    std::cout << "  PDF:     " << cache_aware_pdf_time << " μs (" << cache_aware_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  LogPDF:  " << cache_aware_log_pdf_time << " μs (" << cache_aware_log_pdf_speedup << "x vs SIMD)" << std::endl;
    std::cout << "  CDF:     " << cache_aware_cdf_time << " μs (" << cache_aware_cdf_speedup << "x vs SIMD)" << std::endl;
    
    // Performance Analysis
    std::cout << "\nPerformance Analysis:" << std::endl;
    
    if (std::thread::hardware_concurrency() > 2) {
        if (parallel_pdf_speedup > 0.8) {
            std::cout << "  ✓ Standard parallel shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Standard parallel speedup lower than expected" << std::endl;
        }
        
        if (work_steal_pdf_speedup > 0.8) {
            std::cout << "  ✓ Work-stealing shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Work-stealing speedup lower than expected" << std::endl;
        }
        
        if (cache_aware_pdf_speedup > 0.8) {
            std::cout << "  ✓ Cache-aware shows good speedup" << std::endl;
        } else {
            std::cout << "  ⚠ Cache-aware speedup lower than expected" << std::endl;
        }
    } else {
        std::cout << "  ℹ Single/dual-core system - parallel overhead may dominate" << std::endl;
    }
    
    // Verify correctness
    std::cout << "\nCorrectness verification:" << std::endl;
    bool all_correct = true;
    const size_t check_count = 100;
    for (size_t i = 0; i < check_count; ++i) {
        size_t idx = i * (BENCHMARK_SIZE / check_count);
        double expected = uniform.getProbability(test_values[idx]);
        if (std::abs(pdf_results[idx] - expected) > 1e-10) {
            all_correct = false;
            break;
        }
    }
    
    if (all_correct) {
        std::cout << "  ✓ All parallel implementations produce correct results" << std::endl;
    } else {
        std::cout << "  ✗ Correctness check failed" << std::endl;
    }
    
    EXPECT_TRUE(all_correct) << "Parallel batch operations should produce correct results";
    
    // Statistics
    auto ws_stats = work_stealing_pool.getStatistics();
    auto cache_stats = cache_manager.getStats();
    
    std::cout << "\nWork-Stealing Pool Statistics:" << std::endl;
    std::cout << "  Tasks executed: " << ws_stats.tasksExecuted << std::endl;
    std::cout << "  Work steals: " << ws_stats.workSteals << std::endl;
    std::cout << "  Steal success rate: " << (ws_stats.stealSuccessRate * 100) << "%" << std::endl;
    
    std::cout << "\nCache Manager Statistics:" << std::endl;
    std::cout << "  Cache size: " << cache_stats.size << " entries" << std::endl;
    std::cout << "  Hit rate: " << (cache_stats.hit_rate * 100) << "%" << std::endl;
    std::cout << "  Memory usage: " << cache_stats.memory_usage << " bytes" << std::endl;
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
// MAIN FUNCTION
//==============================================================================

} // namespace libstats

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

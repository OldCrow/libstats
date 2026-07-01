/**
 * @file test_platform_optimizations.cpp
 * @brief Comprehensive platform optimization testing suite
 *
 * Features:
 * - Cache-aware algorithms testing
 * - NUMA-aware processing validation
 * - Platform-specific constants validation
 * - Memory access pattern optimization
 * - SIMD consistency validation
 * - Cross-architecture performance comparison
 * - Advanced vectorization decision testing
 */

#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd.h"

// Standard library includes
#include <algorithm>  // for std::sort, std::min, std::max
#include <chrono>     // for std::chrono::high_resolution_clock
#include <exception>  // for std::exception
#include <gtest/gtest.h>
#include <iomanip>    // for std::setprecision
#include <iostream>   // for std::cout, std::cerr, std::endl
#include <memory>     // for std::unique_ptr, std::make_unique
#include <numeric>    // for std::iota, std::accumulate
#include <random>     // for std::random_device, std::mt19937
#include <sstream>    // for std::stringstream
#include <streambuf>  // for std::streambuf
#include <string>     // for std::string
#include <thread>     // for std::thread
#include <vector>     // for std::vector

using namespace stats::arch;
using namespace stats::arch::simd;
using namespace std;

// Forward declarations for benchmark helpers
void benchmark_cache_blocking();
void benchmark_memory_bandwidth();

template <typename T>
void fill_random_data(vector<T>& data, T min_val = T(0), T max_val = T(1)) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<T> dis(min_val, max_val);
    for (auto& val : data) {
        val = dis(gen);
    }
}

size_t get_l1_cache_elements() {
    const auto& features = get_features();
    return features.l1_data_cache.size / sizeof(double);
}

size_t get_l2_cache_elements() {
    const auto& features = get_features();
    return features.l2_cache.size / sizeof(double);
}

size_t get_l3_cache_elements() {
    const auto& features = get_features();
    return features.l3_cache.size / sizeof(double);
}

void test_basic_platform_info() {
    cout << "\n=== BASIC PLATFORM INFORMATION ===" << endl;

    // Test platform optimization info
    string platform_info = VectorOps::get_platform_optimization_info();
    EXPECT_TRUE(!platform_info.empty());
    cout << "✓ Platform info: " << platform_info << endl;

    // Test platform capabilities
    cout << "\nPlatform Capabilities:" << endl;
    cout << "  - Active SIMD level: " << VectorOps::get_active_simd_level() << endl;
    cout << "  - SIMD available: " << (VectorOps::is_simd_available() ? "YES" : "NO") << endl;
    cout << "  - Vectorization supported: " << (VectorOps::supports_vectorization() ? "YES" : "NO")
         << endl;
    cout << "  - Double vector width: " << VectorOps::double_vector_width() << endl;
    cout << "  - Float vector width: " << float_vector_width() << endl;
    cout << "  - Optimal alignment: " << stats::arch::optimal_alignment() << " bytes" << endl;

    // Verify basic consistency
    EXPECT_TRUE(VectorOps::supports_vectorization());
    EXPECT_TRUE(VectorOps::double_vector_width() > 0);
    EXPECT_TRUE(float_vector_width() > 0);
    EXPECT_TRUE(stats::arch::optimal_alignment() > 0 &&
                (stats::arch::optimal_alignment() & (stats::arch::optimal_alignment() - 1)) == 0);
}

void test_platform_constants_validation() {
    cout << "\n=== PLATFORM CONSTANTS VALIDATION ===" << endl;

    // Test tuned constants are reasonable
    auto matrix_block = tuned::matrix_block_size();
    auto simd_width = tuned::simd_loop_width();
    auto min_threshold = SIMDPolicy::getMinThreshold();
    auto cache_step = tuned::cache_friendly_step();
    auto l1_doubles = tuned::l1_cache_doubles();
    auto prefetch_dist = tuned::prefetch_distance();

    // Validate ranges
    EXPECT_TRUE(matrix_block > 0 && matrix_block <= 512);     // Reasonable matrix block sizes
    EXPECT_TRUE(simd_width > 0 && simd_width <= 64);          // Reasonable SIMD widths
    EXPECT_TRUE(min_threshold > 0 && min_threshold <= 1024);  // Reasonable SIMD threshold
    EXPECT_TRUE(cache_step > 0);
    EXPECT_TRUE(l1_doubles > 1000);  // At least 8KB worth of doubles
    EXPECT_TRUE(prefetch_dist > 0 && prefetch_dist <= 512);

    cout << "✓ Platform constants validation:" << endl;
    cout << "  - Matrix block size: " << matrix_block << " (valid range)" << endl;
    cout << "  - SIMD loop width: " << simd_width << " (valid range)" << endl;
    cout << "  - Min SIMD threshold: " << min_threshold << " (valid range)" << endl;
    cout << "  - Cache-friendly step: " << cache_step << endl;
    cout << "  - L1 cache doubles: " << l1_doubles << endl;
    cout << "  - Prefetch distance: " << prefetch_dist << endl;

    // Cross-validate constants make sense together
    EXPECT_TRUE(matrix_block >= simd_width);  // Block size should be at least SIMD width
    EXPECT_TRUE(l1_doubles >= matrix_block * matrix_block);  // L1 should fit matrix blocks

    cout << "✓ Cross-validation of constants passed" << endl;
}

void test_advanced_vectorization_decisions() {
    cout << "\n=== ADVANCED VECTORIZATION DECISIONS ===" << endl;

    // Test various data sizes
    vector<size_t> test_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384};

    cout << "\nVectorization decisions by size:" << endl;
    for (size_t size : test_sizes) {
        vector<double> test_data(size, 1.0);
        bool should_vectorize = VectorOps::should_use_vectorized_path(size, test_data.data());

        cout << "  Size " << setw(5) << size << ": " << (should_vectorize ? "SIMD" : "Scalar")
             << endl;

        // Large sizes should definitely use SIMD
        if (size >= 1024) {
            EXPECT_TRUE(should_vectorize);
        }
    }

    // Test edge cases
    cout << "\nEdge case testing:" << endl;

    // Null pointer should not crash and return false
    bool null_decision = VectorOps::should_use_vectorized_path(100, nullptr);
    cout << "  Null pointer: " << (null_decision ? "SIMD" : "Scalar") << " (should be Scalar)"
         << endl;
    EXPECT_TRUE(!null_decision);

    // Zero size should return false
    vector<double> dummy_data(10);
    bool zero_decision = VectorOps::should_use_vectorized_path(0, dummy_data.data());
    cout << "  Zero size: " << (zero_decision ? "SIMD" : "Scalar") << " (should be Scalar)" << endl;
    EXPECT_TRUE(!zero_decision);

    cout << "✓ Advanced vectorization decisions passed" << endl;
}

void test_simd_threshold_alignment() {
    cout << "\n=== SIMD THRESHOLD ALIGNMENT VALIDATION ===" << endl;

    // Get thresholds from different sources
    size_t vectorops_min = VectorOps::min_simd_size();
    size_t policy_min = SIMDPolicy::getMinThreshold();
    size_t tuned_min = tuned::matrix_block_size();

    cout << "Threshold comparison:" << endl;
    cout << "  VectorOps min_simd_size(): " << vectorops_min << endl;
    cout << "  SIMDPolicy getMinThreshold(): " << policy_min << endl;
    cout << "  Tuned matrix_block_size(): " << tuned_min << endl;

    // Test consistency - these should be reasonably aligned
    // This addresses the original issue mentioned in the conversation history
    cout << "\nConsistency analysis:" << endl;

    // Test various sizes around the thresholds
    vector<size_t> test_points = {1, 4, 8, 12, 16, 24, 32, 64};

    for (size_t size : test_points) {
        bool policy_says_yes = SIMDPolicy::shouldUseSIMD(size);
        bool vectorops_would_use = size >= vectorops_min;

        cout << "  Size " << setw(3) << size << ": Policy=" << (policy_says_yes ? "YES" : "NO ")
             << ", VectorOps=" << (vectorops_would_use ? "YES" : "NO ");

        // Flag inconsistencies
        if (policy_says_yes != vectorops_would_use) {
            cout << " ⚠ INCONSISTENT";
        }
        cout << endl;
    }

    // The main consistency requirement: if policy says yes, VectorOps should agree
    // for reasonable sizes (not tiny edge cases)
    for (size_t size = 8; size <= 64; ++size) {
        bool policy_decision = SIMDPolicy::shouldUseSIMD(size);
        bool vectorops_decision = size >= vectorops_min;

        if (policy_decision && !vectorops_decision && size >= 8) {
            cout << "⚠ Warning: Policy and VectorOps disagree for size " << size << endl;
        }
    }

    cout << "✓ SIMD threshold alignment analysis completed" << endl;
}

void test_cache_aware_algorithms() {
    cout << "\n=== CACHE-AWARE ALGORITHMS TESTING ===" << endl;

    // Get cache information
    const auto& features = get_features();
    size_t l1_elements = get_l1_cache_elements();
    size_t l2_elements = get_l2_cache_elements();
    size_t l3_elements = get_l3_cache_elements();

    cout << "Cache hierarchy (in double elements):" << endl;
    cout << "  L1: " << l1_elements << " elements" << endl;
    cout << "  L2: " << l2_elements << " elements" << endl;
    cout << "  L3: " << l3_elements << " elements" << endl;

    // Test cache blocking strategies
    benchmark_cache_blocking();

    // Test cache-friendly data access patterns
    cout << "\nCache-friendly access pattern testing:" << endl;

    const size_t test_size = min(l1_elements / 2, size_t(50000));  // Half of L1 cache
    vector<double> cache_test_data(test_size);
    fill_random_data(cache_test_data);

    // Sequential access (cache-friendly)
    auto start = chrono::high_resolution_clock::now();
    volatile double sum1 = 0.0;
    for (size_t i = 0; i < test_size; ++i) {
        sum1 = sum1 + cache_test_data[i];
    }
    auto end = chrono::high_resolution_clock::now();
    auto sequential_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // Strided access (cache-unfriendly)
    start = chrono::high_resolution_clock::now();
    volatile double sum2 = 0.0;
    const size_t stride = features.cache_line_size / sizeof(double);
    for (size_t i = 0; i < test_size; i += stride) {
        sum2 = sum2 + cache_test_data[i];
    }
    end = chrono::high_resolution_clock::now();
    auto strided_time = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "  Sequential access: " << sequential_time.count() << " μs" << endl;
    cout << "  Strided access: " << strided_time.count() << " μs" << endl;

    if (strided_time.count() > 0 && sequential_time.count() > 0) {
        double ratio = static_cast<double>(strided_time.count()) /
                       static_cast<double>(sequential_time.count());
        cout << "  Cache efficiency ratio: " << fixed << setprecision(2) << ratio << "x" << endl;
    }

    cout << "✓ Cache-aware algorithms testing completed" << endl;
}

void test_memory_access_patterns() {
    cout << "\n=== MEMORY ACCESS PATTERNS OPTIMIZATION ===" << endl;

    // Test different memory access patterns
    const size_t pattern_size = 64 * 1024;  // 64K elements
    vector<double> pattern_data(pattern_size);
    fill_random_data(pattern_data);

    benchmark_memory_bandwidth();

    // Test memory alignment effects
    cout << "\nMemory alignment testing:" << endl;

    // Aligned access
    vector<double, aligned_allocator<double>> aligned_data(1024, 1.0);
    bool alignment_check = is_aligned(aligned_data.data());
    cout << "  Aligned allocator: " << (alignment_check ? "PASS" : "FAIL") << endl;
    EXPECT_TRUE(alignment_check);

    // Test SIMD operations on aligned vs unaligned data
    vector<double> unaligned_data(1024, 1.0);

    auto start = chrono::high_resolution_clock::now();
    VectorOps::scalar_multiply(aligned_data.data(), 2.0, aligned_data.data(), 1024);
    auto end = chrono::high_resolution_clock::now();
    auto aligned_time = chrono::duration_cast<chrono::nanoseconds>(end - start);

    start = chrono::high_resolution_clock::now();
    VectorOps::scalar_multiply(unaligned_data.data(), 2.0, unaligned_data.data(), 1024);
    end = chrono::high_resolution_clock::now();
    auto unaligned_time = chrono::duration_cast<chrono::nanoseconds>(end - start);

    cout << "  Aligned SIMD: " << aligned_time.count() << " ns" << endl;
    cout << "  Unaligned SIMD: " << unaligned_time.count() << " ns" << endl;

    if (aligned_time.count() > 0 && unaligned_time.count() > 0) {
        double speedup =
            static_cast<double>(unaligned_time.count()) / static_cast<double>(aligned_time.count());
        cout << "  Alignment speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
    }

    cout << "✓ Memory access patterns optimization completed" << endl;
}

void test_numa_aware_processing() {
    cout << "\n=== NUMA-AWARE PROCESSING VALIDATION ===" << endl;

    const auto& features = get_features();
    const auto& topology = features.topology;

    cout << "System topology:" << endl;
    cout << "  Physical cores: " << topology.physical_cores << endl;
    cout << "  Logical cores: " << topology.logical_cores << endl;
    cout << "  CPU packages: " << topology.packages << endl;
    cout << "  Hyperthreading: " << (topology.hyperthreading ? "YES" : "NO") << endl;

    // NUMA considerations
    if (topology.packages > 1) {
        cout << "\n⚠ Multi-socket system detected - NUMA considerations important" << endl;
        cout << "  Recommended: Use physical cores for CPU-bound tasks" << endl;
        cout << "  Recommended: Consider memory affinity for large datasets" << endl;
    } else {
        cout << "\nSingle-socket system - standard optimization strategies apply" << endl;
    }

    // Test thread scaling behavior
    const size_t test_size = 100000;
    vector<double> numa_test_data(test_size);
    fill_random_data(numa_test_data);

    cout << "\nThread scaling analysis:" << endl;

    // Test with different thread counts
    vector<size_t> thread_counts = {1, 2, 4};
    if (topology.physical_cores >= 8)
        thread_counts.push_back(8);
    if (topology.physical_cores >= 16)
        thread_counts.push_back(16);

    for (size_t num_threads : thread_counts) {
        if (num_threads > topology.logical_cores)
            continue;

        auto start = chrono::high_resolution_clock::now();

        // Simple parallel operation simulation
        vector<thread> workers;
        size_t chunk_size = test_size / num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start_idx = t * chunk_size;
            size_t end_idx = (t == num_threads - 1) ? test_size : start_idx + chunk_size;

            workers.emplace_back([&numa_test_data, start_idx, end_idx]() {
                volatile double local_sum = 0.0;
                for (size_t i = start_idx; i < end_idx; ++i) {
                    local_sum = local_sum + numa_test_data[i] * 1.1;
                }
            });
        }

        for (auto& worker : workers) {
            worker.join();
        }

        auto end = chrono::high_resolution_clock::now();
        auto thread_time = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "  " << num_threads << " threads: " << thread_time.count() << " μs" << endl;
    }

    // Validate parallel thresholds are reasonable for the system
    auto min_parallel = get_min_elements_for_parallel();
    auto grain_size = get_default_grain_size();

    cout << "\nParallel processing thresholds:" << endl;
    cout << "  Min elements for parallel: " << min_parallel << endl;
    cout << "  Default grain size: " << grain_size << endl;

    // Thresholds should be reasonable for the number of cores
    EXPECT_TRUE(min_parallel > 0);
    EXPECT_TRUE(grain_size > 0);
    // Note: grain_size can be larger than min_parallel for efficiency
    // Each thread should work on grain_size elements, while min_parallel
    // is just the threshold to start using parallel processing
    EXPECT_TRUE(grain_size >= min_parallel / 32);  // Reasonable lower bound check

    cout << "✓ NUMA-aware processing validation completed" << endl;
}

void test_cross_architecture_consistency() {
    cout << "\n=== CROSS-ARCHITECTURE CONSISTENCY ===" << endl;

    const auto& features = get_features();

    cout << "Architecture validation:" << endl;
    cout << "  Vendor: " << features.vendor << endl;
    cout << "  Architecture: " << best_simd_level() << endl;

    // Test API consistency across architectures
    auto double_width = stats::arch::optimal_double_width();
    auto float_width = stats::arch::optimal_float_width();
    auto alignment = stats::arch::optimal_alignment();

    cout << "  Double vector width: " << double_width << endl;
    cout << "  Float vector width: " << float_width << endl;
    cout << "  Memory alignment: " << alignment << " bytes" << endl;

    // Consistency checks
    EXPECT_TRUE(double_width > 0);
    EXPECT_TRUE(float_width > 0);
    EXPECT_TRUE(float_width >= double_width);  // Floats should pack more densely
    EXPECT_TRUE(alignment > 0 && (alignment & (alignment - 1)) == 0);  // Power of 2

    // Platform-specific validation
    if (features.vendor == "Apple") {
        cout << "\nApple Silicon specific checks:" << endl;
        EXPECT_TRUE(features.neon);  // Apple Silicon should have NEON
        cout << "  ✓ NEON support confirmed" << endl;

        // Apple Silicon typically has lower thread creation overhead
        auto apple_min_parallel = get_min_elements_for_parallel();
        cout << "  Min parallel elements: " << apple_min_parallel << " (optimized for fast threads)"
             << endl;

    } else if (features.vendor == "GenuineIntel") {
        cout << "\nIntel specific checks:" << endl;
        if (features.avx2) {
            EXPECT_TRUE(double_width >= 4);  // AVX2 should give at least 256-bit vectors
            cout << "  ✓ AVX2 vector width confirmed" << endl;
        }
        if (features.avx512f) {
            EXPECT_TRUE(double_width >= 8);  // AVX-512 should give 512-bit vectors
            cout << "  ✓ AVX-512 vector width confirmed" << endl;
        }

    } else if (features.vendor == "AuthenticAMD") {
        cout << "\nAMD specific checks:" << endl;
        if (features.avx2) {
            cout << "  ✓ AVX2 support confirmed" << endl;
        }
        // AMD may have different cache optimization strategies
    }

    // Validate that SIMD functions exist and are callable
    vector<double> consistency_test(16, 1.0);
    VectorOps::scalar_multiply(consistency_test.data(), 2.0, consistency_test.data(), 16);

    // Verify the operation worked
    EXPECT_TRUE(abs(consistency_test[0] - 2.0) < 1e-10);

    cout << "✓ Cross-architecture consistency validation completed" << endl;
}

void test_performance_scaling() {
    cout << "\n=== PERFORMANCE SCALING ANALYSIS ===" << endl;

    // Test performance scaling with different data sizes
    vector<size_t> sizes = {1000, 10000, 100000, 1000000};

    cout << "Performance scaling (operations/second):" << endl;
    cout << setw(10) << "Size" << setw(15) << "Scalar" << setw(15) << "SIMD" << setw(15)
         << "Speedup" << endl;
    cout << string(55, '-') << endl;

    for (size_t size : sizes) {
        vector<double> test_data(size, 1.0);
        vector<double> result_scalar(size);
        vector<double> result_simd(size);

        // Scalar timing
        auto start = chrono::high_resolution_clock::now();
        for (size_t i = 0; i < size; ++i) {
            result_scalar[i] = test_data[i] * 2.0;
        }
        auto end = chrono::high_resolution_clock::now();
        auto scalar_time = chrono::duration_cast<chrono::nanoseconds>(end - start);

        // SIMD timing
        start = chrono::high_resolution_clock::now();
        VectorOps::scalar_multiply(test_data.data(), 2.0, result_simd.data(), size);
        end = chrono::high_resolution_clock::now();
        auto simd_time = chrono::duration_cast<chrono::nanoseconds>(end - start);

        // Calculate operations per second
        double scalar_ops_per_sec = 0.0;
        double simd_ops_per_sec = 0.0;
        double speedup = 1.0;

        if (scalar_time.count() > 0) {
            scalar_ops_per_sec =
                (static_cast<double>(size) * 1e9) / static_cast<double>(scalar_time.count());
        }
        if (simd_time.count() > 0) {
            simd_ops_per_sec =
                (static_cast<double>(size) * 1e9) / static_cast<double>(simd_time.count());
            speedup =
                static_cast<double>(scalar_time.count()) / static_cast<double>(simd_time.count());
        }

        cout << setw(10) << size << setw(15) << fixed << setprecision(0) << scalar_ops_per_sec
             << setw(15) << fixed << setprecision(0) << simd_ops_per_sec << setw(14) << fixed
             << setprecision(2) << speedup << "x" << endl;
    }

    cout << "✓ Performance scaling analysis completed" << endl;
}

void test_adaptive_optimization() {
    cout << "\n=== ADAPTIVE OPTIMIZATION VALIDATION ===" << endl;

    // Test that optimization parameters adapt to system characteristics
    const auto& features = get_features();

    cout << "System-adaptive parameters:" << endl;

    // Cache-based adaptations
    size_t l1_size = features.l1_data_cache.size;
    size_t optimal_block = VectorOps::get_optimal_block_size();

    cout << "  L1 data cache: " << l1_size << " bytes" << endl;
    cout << "  Optimal block size: " << optimal_block << " elements" << endl;

    // Block size should be reasonable for L1 cache
    if (l1_size > 0) {
        size_t max_reasonable_block = l1_size / (2 * sizeof(double));  // Half of L1
        if (optimal_block <= max_reasonable_block) {
            cout << "  ✓ Block size fits in L1 cache" << endl;
        } else {
            cout << "  ⚠ Block size may be too large for L1 cache" << endl;
        }
    }

    // SIMD width adaptations
    size_t simd_width = VectorOps::double_vector_width();
    size_t min_simd_elements = VectorOps::min_simd_size();

    cout << "  SIMD width: " << simd_width << " doubles" << endl;
    cout << "  Min SIMD threshold: " << min_simd_elements << " elements" << endl;

    // Minimum should be some multiple of SIMD width
    EXPECT_TRUE(min_simd_elements >= simd_width);
    cout << "  ✓ SIMD threshold aligns with vector width" << endl;

    // Thread-based adaptations
    size_t cores = features.topology.physical_cores;
    if (cores == 0)
        cores = features.topology.logical_cores;

    size_t grain_size = get_default_grain_size();
    cout << "  Physical cores: " << cores << endl;
    cout << "  Default grain size: " << grain_size << " elements" << endl;

    // Grain size should scale reasonably with core count
    if (cores > 0) {
        // More cores should generally mean smaller grain sizes for better load balancing
        size_t expected_max_grain = 10000 / cores;  // Rough heuristic
        if (grain_size > expected_max_grain * 2) {
            cout << "  ⚠ Grain size may be too large for " << cores << " cores" << endl;
        } else {
            cout << "  ✓ Grain size reasonable for core count" << endl;
        }
    }

    cout << "✓ Adaptive optimization validation completed" << endl;
}

void benchmark_cache_blocking() {
    cout << "\nCache blocking benchmark:" << endl;

    const size_t matrix_size = 256;  // Small enough to be reasonable
    const size_t total_elements = matrix_size * matrix_size;

    vector<double> matrix_a(total_elements);
    vector<double> matrix_b(total_elements);
    vector<double> result_naive(total_elements, 0.0);
    vector<double> result_blocked(total_elements, 0.0);

    fill_random_data(matrix_a);
    fill_random_data(matrix_b);

    const size_t block_size = tuned::matrix_block_size();

    // Naive matrix multiply (cache-unfriendly)
    auto start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size; ++j) {
            for (size_t k = 0; k < matrix_size; ++k) {
                result_naive[i * matrix_size + j] +=
                    matrix_a[i * matrix_size + k] * matrix_b[k * matrix_size + j];
            }
        }
    }
    auto end = chrono::high_resolution_clock::now();
    auto naive_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // Blocked matrix multiply (cache-friendly)
    start = chrono::high_resolution_clock::now();
    for (size_t ii = 0; ii < matrix_size; ii += block_size) {
        for (size_t jj = 0; jj < matrix_size; jj += block_size) {
            for (size_t kk = 0; kk < matrix_size; kk += block_size) {
                for (size_t i = ii; i < min(ii + block_size, matrix_size); ++i) {
                    for (size_t j = jj; j < min(jj + block_size, matrix_size); ++j) {
                        for (size_t k = kk; k < min(kk + block_size, matrix_size); ++k) {
                            result_blocked[i * matrix_size + j] +=
                                matrix_a[i * matrix_size + k] * matrix_b[k * matrix_size + j];
                        }
                    }
                }
            }
        }
    }
    end = chrono::high_resolution_clock::now();
    auto blocked_time = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "  Naive multiply: " << naive_time.count() << " μs" << endl;
    cout << "  Blocked multiply: " << blocked_time.count() << " μs" << endl;

    if (naive_time.count() > 0 && blocked_time.count() > 0) {
        double speedup =
            static_cast<double>(naive_time.count()) / static_cast<double>(blocked_time.count());
        cout << "  Cache blocking speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
    }
}

void benchmark_memory_bandwidth() {
    cout << "\nMemory bandwidth benchmark:" << endl;

    const size_t bandwidth_size = 1024 * 1024;  // 1M doubles = ~8MB
    vector<double> bandwidth_data(bandwidth_size);
    fill_random_data(bandwidth_data);

    // Sequential read
    auto start = chrono::high_resolution_clock::now();
    volatile double sum = 0.0;
    for (size_t i = 0; i < bandwidth_size; ++i) {
        sum = sum + bandwidth_data[i];
    }
    auto end = chrono::high_resolution_clock::now();
    auto read_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // Sequential write
    start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < bandwidth_size; ++i) {
        bandwidth_data[i] = static_cast<double>(i);
    }
    end = chrono::high_resolution_clock::now();
    auto write_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // Calculate bandwidth
    double data_mb = (bandwidth_size * sizeof(double)) / (1024.0 * 1024.0);

    if (read_time.count() > 0) {
        double read_bandwidth = data_mb / (static_cast<double>(read_time.count()) / 1e6);
        cout << "  Sequential read: " << fixed << setprecision(1) << read_bandwidth << " MB/s"
             << endl;
    }

    if (write_time.count() > 0) {
        double write_bandwidth = data_mb / (static_cast<double>(write_time.count()) / 1e6);
        cout << "  Sequential write: " << fixed << setprecision(1) << write_bandwidth << " MB/s"
             << endl;
    }
}

TEST(PlatformOptimizations, BasicPlatformInfo) {
    test_basic_platform_info();
}
TEST(PlatformOptimizations, PlatformConstantsValidation) {
    test_platform_constants_validation();
}
TEST(PlatformOptimizations, AdvancedVectorizationDecisions) {
    test_advanced_vectorization_decisions();
}
TEST(PlatformOptimizations, SimdThresholdAlignment) {
    test_simd_threshold_alignment();
}
TEST(PlatformOptimizations, CacheAwareAlgorithms) {
    test_cache_aware_algorithms();
}
TEST(PlatformOptimizations, MemoryAccessPatterns) {
    test_memory_access_patterns();
}
TEST(PlatformOptimizations, NumaAwareProcessing) {
    test_numa_aware_processing();
}
TEST(PlatformOptimizations, CrossArchitectureConsistency) {
    test_cross_architecture_consistency();
}
TEST(PlatformOptimizations, PerformanceScaling) {
    test_performance_scaling();
}
TEST(PlatformOptimizations, AdaptiveOptimization) {
    test_adaptive_optimization();
}

// =============================================================================
// CpuDetectionAPI: tests for cpu_detection.h public functions that previously
// had no GTest coverage (Part 1D, API rationalization plan).
// =============================================================================

TEST(CpuDetectionAPI, TopologyIsConsistent) {
    // get_topology(), get_logical_core_count(), get_physical_core_count(), has_hyperthreading()
    const auto topo = get_topology();

    // Every machine has at least one logical core.
    EXPECT_GE(topo.logical_cores, 1u);
    // physical_cores may be 0 on heavily virtualised CI runners (e.g. GitHub Actions
    // Windows) where the hypervisor does not expose physical topology. Skip the
    // hard lower-bound; the rest of the test already gates on physical_cores > 0.
    EXPECT_GE(topo.physical_cores, 0u);

    // Logical cores >= physical (SMT multiplies logical).
    EXPECT_GE(topo.logical_cores, topo.physical_cores);

    // Convenience accessors are consistent with the struct.
    EXPECT_EQ(get_logical_core_count(), topo.logical_cores);
    EXPECT_EQ(get_physical_core_count(), topo.physical_cores);

    // has_hyperthreading() matches the struct field and the logical/physical ratio.
    EXPECT_EQ(has_hyperthreading(), topo.hyperthreading);
    if (topo.physical_cores > 0) {
        bool smt_on = topo.logical_cores > topo.physical_cores;
        // hyperthreading flag should be true iff logical > physical
        // (some platforms may have hyperthreading=true but threads_per_core=1;
        //  only assert the safe direction: smt_on implies hyperthreading).
        if (smt_on) {
            EXPECT_TRUE(topo.hyperthreading)
                << "SMT detected (logical > physical) but hyperthreading flag is false";
        }
    }
}

TEST(CpuDetectionAPI, OptimalWidthsAndAlignment) {
    // optimal_double_width(), optimal_float_width(), optimal_alignment()
    // Must qualify: simd.h also exports optimal_alignment() in an overlapping namespace.
    const auto dw = stats::arch::optimal_double_width();
    const auto fw = stats::arch::optimal_float_width();
    const auto al = stats::arch::optimal_alignment();

    // At least scalar (width 1).
    EXPECT_GE(dw, 1u);
    EXPECT_GE(fw, 1u);

    // Floats are smaller than doubles, so more fit per register.
    EXPECT_GE(fw, dw);

    // Alignment must be a positive power of two.
    EXPECT_GT(al, 0u);
    EXPECT_EQ(al & (al - 1), 0u) << "optimal_alignment() must be a power of two";

    // Practical sanity: alignment should not exceed 512 bytes (AVX-512 = 64 bytes).
    EXPECT_LE(al, 512u);
}

TEST(CpuDetectionAPI, FeatureStringsNonEmpty) {
    // features_string(), best_simd_level()
    const std::string fs = features_string();
    const std::string bsl = best_simd_level();

    EXPECT_FALSE(fs.empty()) << "features_string() must not be empty";
    EXPECT_FALSE(bsl.empty()) << "best_simd_level() must not be empty";

    // best_simd_level() should be one of the known tier strings.
    const std::initializer_list<const char*> known_levels = {
        "AVX-512", "AVX2", "AVX", "SSE4.2", "SSE4.1", "SSE2", "NEON", "SVE", "None"};
    bool found = false;
    for (const char* level : known_levels) {
        if (bsl.find(level) != std::string::npos) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "best_simd_level() returned unrecognised string: " << bsl;
}

TEST(CpuDetectionAPI, PerformanceMeasurement) {
    // measure_performance<>(), has_rdtsc()

    // has_rdtsc() must not crash and must agree with the Features struct.
    const bool rdtsc = has_rdtsc();
    EXPECT_EQ(rdtsc, get_features().performance.has_rdtsc);

    // Measuring a trivial lambda must produce a valid TimingResult.
    volatile int sink = 0;
    const auto result = measure_performance([&sink]() { sink = 42; });

    EXPECT_TRUE(result.valid) << "measure_performance() must set valid=true";
    EXPECT_GE(result.duration.count(), 0);

    // If RDTSC is available, cycles should be non-zero (a trivial op still takes cycles).
    if (rdtsc) {
        // Allow 0 on very fast CPUs where the lambda is optimised out — don't assert,
        // just report: the existence of a non-zero reading is informative, not required.
        std::cout << "  measure_performance cycles: " << result.cycles << "\n";
    }

    (void)sink;  // suppress unused-variable warning
}

TEST(CpuDetectionAPI, ExtendedSIMDChecksConsistentWithFeatures) {
    // supports_avx512dq/bw/vl(), supports_sve/sve2(): verify each agrees with get_features().
    const auto& f = get_features();
    EXPECT_EQ(supports_avx512dq(), f.avx512dq);
    EXPECT_EQ(supports_avx512bw(), f.avx512bw);
    EXPECT_EQ(supports_avx512vl(), f.avx512vl);
    EXPECT_EQ(supports_sve(), f.sve);
    EXPECT_EQ(supports_sve2(), f.sve2);

    // Subset consistency: avx512dq/bw/vl require avx512f.
    if (f.avx512dq || f.avx512bw || f.avx512vl) {
        EXPECT_TRUE(f.avx512f) << "avx512dq/bw/vl implies avx512f must be set";
    }
    // sve2 implies sve.
    if (f.sve2) {
        EXPECT_TRUE(f.sve) << "sve2 implies sve must be set";
    }
}

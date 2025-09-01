#include "../include/core/mathematical_constants.h"
#include "../include/core/performance_dispatcher.h"
#include "../include/core/precision_constants.h"
#include "../include/platform/cpu_detection.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

namespace stats {
namespace detail {  // Performance utilities

namespace {
// Benchmark configuration
constexpr size_t BENCHMARK_ITERATIONS = detail::MAX_BISECTION_ITERATIONS;
constexpr size_t BENCHMARK_ARRAY_SIZE = 1024;

// Simple benchmark for SIMD efficiency
double benchmarkSIMDEfficiency() {
    std::vector<double> data(BENCHMARK_ARRAY_SIZE, 1.5);
    std::vector<double> results(BENCHMARK_ARRAY_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        // Simple arithmetic operation
        for (size_t j = 0; j < BENCHMARK_ARRAY_SIZE; ++j) {
            results[j] = data[j] * detail::TWO + detail::ONE;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double time_per_operation =
        static_cast<double>(duration.count()) / (BENCHMARK_ITERATIONS * BENCHMARK_ARRAY_SIZE);

    // Convert to efficiency metric (higher = better)
    // Typical scalar operations are 1-10ns, so we invert and scale to 0-1 range
    return std::min(detail::ONE, 10.0 / std::max(detail::ONE, time_per_operation));
}

// Simple threading overhead benchmark
double benchmarkThreadingOverhead() {
    constexpr size_t num_tests = detail::MAX_NEWTON_ITERATIONS;
    std::vector<std::chrono::nanoseconds> overhead_times;
    overhead_times.reserve(num_tests);

    for (size_t i = 0; i < num_tests; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::thread t([]() {
            // Minimal work
            volatile int x = 42;
            (void)x;
        });
        t.join();

        auto end = std::chrono::high_resolution_clock::now();
        overhead_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
    }

    // Return average overhead
    auto total =
        std::accumulate(overhead_times.begin(), overhead_times.end(), std::chrono::nanoseconds(0));
    return static_cast<double>(total.count()) / num_tests;
}

// Estimate memory bandwidth (simplified)
double estimateMemoryBandwidth() {
    // Use larger arrays and prevent compiler optimizations
    constexpr size_t array_size = 16 * 1024 * 1024;  // 16MB to exceed cache sizes
    constexpr size_t iterations = 10;

    // Create two separate arrays to avoid overlapping memory issues
    std::vector<double> source(array_size, detail::ONE);
    std::vector<double> destination(array_size, detail::ZERO_DOUBLE);

    // Fill source with unique values to prevent optimization
    for (size_t i = 0; i < array_size; ++i) {
        source[i] = static_cast<double>(i % detail::MAX_BISECTION_ITERATIONS) + detail::HALF;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Memory copy operations with volatile to prevent optimization
    volatile double sink = detail::ZERO_DOUBLE;
    for (size_t iter = 0; iter < iterations; ++iter) {
        // Copy source to destination
        std::copy(source.begin(), source.end(), destination.begin());

        // Ensure the operation isn't optimized away
        sink = sink + destination[iter % array_size];
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    // Use the sink to prevent dead code elimination
    (void)sink;

    // Calculate bandwidth: bytes_read + bytes_written per iteration
    double bytes_transferred = detail::TWO * array_size * sizeof(double) * iterations;
    double seconds = static_cast<double>(duration.count()) / 1e9;

    // Avoid division by zero and return a reasonable fallback
    if (seconds <= detail::ZERO_DOUBLE || !std::isfinite(seconds)) {
        return 25.0;  // Reasonable fallback for DDR3-1600 dual channel
    }

    double bandwidth = (bytes_transferred / seconds) / 1e9;  // GB/s

    // Clamp to reasonable bounds to handle measurement errors
    return std::max(detail::ONE, std::min(detail::HUNDRED, bandwidth));
}
}  // namespace

SystemCapabilities::SystemCapabilities() {
    detectCapabilities();
    benchmarkPerformance();
}

const SystemCapabilities& SystemCapabilities::current() {
    static SystemCapabilities instance;
    return instance;
}

void SystemCapabilities::detectCapabilities() {
    // CPU core detection
    logical_cores_ = std::thread::hardware_concurrency();
    physical_cores_ = logical_cores_ / 2;  // Simplified assumption (hyperthreading)

    // Cache sizes (simplified - would need platform-specific detection in production)
    l1_cache_size_ = 32 * 1024;        // 32KB typical L1
    l2_cache_size_ = 256 * 1024;       // 256KB typical L2
    l3_cache_size_ = 8 * 1024 * 1024;  // 8MB typical L3

    // SIMD capability detection using existing CPU detection
    has_sse2_ = stats::arch::supports_sse2();
    has_avx_ = stats::arch::supports_avx();
    has_avx2_ = stats::arch::supports_avx2();
    has_avx512_ = stats::arch::supports_avx512();
    has_neon_ = stats::arch::supports_neon();
}

void SystemCapabilities::benchmarkPerformance() {
    // Benchmark SIMD efficiency
    simd_efficiency_ = benchmarkSIMDEfficiency();

    // Benchmark threading overhead
    threading_overhead_ns_ = benchmarkThreadingOverhead();

    // Estimate memory bandwidth
    memory_bandwidth_gb_s_ = estimateMemoryBandwidth();
}

}  // namespace detail
}  // namespace stats

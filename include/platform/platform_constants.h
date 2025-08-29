#pragma once

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

// Forward declaration for platform-specific tuning
#include "cpu_detection.h"
#include "cpu_vendor_constants.h"  // Phase 3D: CPU vendor-specific constants

/**
 * @file platform/platform_constants.h
 * @brief Platform-dependent optimization constants and runtime tuning functions
 *
 * This header contains all platform-specific constants, SIMD optimization parameters,
 * parallel processing thresholds, and memory optimization constants that depend on
 * the target hardware architecture.
 */

namespace stats {
namespace arch {
/// SIMD optimization parameters and architectural constants
namespace simd {
/// Default SIMD block size for vectorized operations
inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;

/// Minimum problem size to benefit from SIMD
inline constexpr std::size_t MIN_SIMD_SIZE = 4;

/// Maximum block size for cache optimization
inline constexpr std::size_t MAX_BLOCK_SIZE = 64;

/// SIMD alignment requirement (bytes)
/// Platform-specific SIMD alignment constants
/// AVX-512: 64-byte alignment for optimal performance
inline constexpr std::size_t AVX512_ALIGNMENT = 64;

/// AVX/AVX2: 32-byte alignment
inline constexpr std::size_t AVX_ALIGNMENT = 32;

/// SSE: 16-byte alignment
inline constexpr std::size_t SSE_ALIGNMENT = 16;

/// ARM NEON: 16-byte alignment
inline constexpr std::size_t NEON_ALIGNMENT = 16;

/// Generic cache line alignment (64 bytes on most modern systems)
inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;

/// Minimum safe alignment for all platforms
inline constexpr std::size_t MIN_SAFE_ALIGNMENT = 8;

/// Matrix operation block sizes for cache-friendly operations
/// Small matrix block size for L1 cache optimization
inline constexpr std::size_t MATRIX_L1_BLOCK_SIZE = 64;

/// Medium matrix block size for L2 cache optimization
inline constexpr std::size_t MATRIX_L2_BLOCK_SIZE = 256;

/// Large matrix block size for L3 cache optimization
inline constexpr std::size_t MATRIX_L3_BLOCK_SIZE = 1024;

/// Step size for matrix traversal (optimized for cache lines)
inline constexpr std::size_t MATRIX_STEP_SIZE = 8;

/// Panel width for matrix decomposition algorithms
inline constexpr std::size_t MATRIX_PANEL_WIDTH = 64;

/// Minimum matrix size for blocking to be beneficial
inline constexpr std::size_t MATRIX_MIN_BLOCK_SIZE = 32;

/// Maximum practical block size (memory constraint)
inline constexpr std::size_t MATRIX_MAX_BLOCK_SIZE = 2048;

/// Platform-specific SIMD register widths (in number of doubles)
/// AVX-512: 8 doubles per register
inline constexpr std::size_t AVX512_DOUBLES = 8;

/// AVX/AVX2: 4 doubles per register
inline constexpr std::size_t AVX_DOUBLES = 4;
inline constexpr std::size_t AVX2_DOUBLES = 4;

/// SSE2: 2 doubles per register
inline constexpr std::size_t SSE_DOUBLES = 2;

/// ARM NEON: 2 doubles per register
inline constexpr std::size_t NEON_DOUBLES = 2;

/// Scalar: 1 double (no SIMD)
inline constexpr std::size_t SCALAR_DOUBLES = 1;

/// Loop unrolling factors for different architectures
/// Unroll factor for AVX-512 (can handle more parallel operations)
inline constexpr std::size_t AVX512_UNROLL = 4;

/// Unroll factor for AVX/AVX2
inline constexpr std::size_t AVX_UNROLL = 2;

/// Unroll factor for SSE
inline constexpr std::size_t SSE_UNROLL = 2;

/// Unroll factor for ARM NEON
inline constexpr std::size_t NEON_UNROLL = 2;

/// Conservative unroll factor for scalar operations
inline constexpr std::size_t SCALAR_UNROLL = 1;

/// CPU detection and runtime constants
/// Maximum backoff time during CPU feature detection (nanoseconds)
inline constexpr uint64_t CPU_MAX_BACKOFF_NANOSECONDS = 1000;

/// Default cache line size fallback (bytes)
inline constexpr uint32_t CPU_DEFAULT_CACHE_LINE_SIZE = 64;

/// Default TSC frequency measurement duration (milliseconds)
inline constexpr uint32_t CPU_DEFAULT_TSC_SAMPLE_MS = 10;

/// Conversion factor from nanoseconds to Hertz
inline constexpr double CPU_NANOSECONDS_TO_HZ = 1e9;

/// SIMD optimization thresholds and platform-specific constants
/// Medium dataset minimum size for alignment benefits
inline constexpr std::size_t OPT_MEDIUM_DATASET_MIN_SIZE = 32;

/// Minimum threshold for alignment benefit checks
inline constexpr std::size_t OPT_ALIGNMENT_BENEFIT_THRESHOLD = 32;

/// Minimum size for AVX-512 aligned datasets
inline constexpr std::size_t OPT_AVX512_MIN_ALIGNED_SIZE = 8;

/// Aggressive SIMD threshold for Apple Silicon
inline constexpr std::size_t OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD = 6;

/// Minimum size threshold for AVX-512 small benefit
inline constexpr std::size_t OPT_AVX512_SMALL_BENEFIT_THRESHOLD = 4;
}  // namespace simd

/// Parallel processing optimization constants - Architecture-specific tuning
namespace parallel {
/// Architecture-specific parallel thresholds and grain sizes
/// Optimized based on SIMD width, cache hierarchy, and thread overhead characteristics

/// ===== SSE/SSE2 Architecture Constants =====
/// For older x86-64 processors with 128-bit SIMD (2 doubles per vector)
namespace sse {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;            // 64 cache lines
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;    // 32 cache lines
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;  // 128 cache lines
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace sse

/// ===== AVX Architecture Constants =====
/// For Intel Sandy Bridge+ and AMD Bulldozer+ with 256-bit SIMD (4 doubles per vector)
namespace avx {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;  // Higher overhead with wider SIMD
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;            // 128 cache lines, 1KB per thread
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;   // 64 cache lines
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 512;  // 256 cache lines, 2KB per thread
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;
inline constexpr std::size_t MAX_GRAIN_SIZE = 4096;

/// ===== Legacy Intel AVX (Ivy Bridge/Sandy Bridge) Specific Tuning =====
/// Optimized for Intel Core i-series 3rd generation (Ivy Bridge) and similar
/// Family 6, Model 58 - AVX without AVX2/FMA, mobile/desktop CPUs ~2012-2013

// Operation-specific grain sizes optimized for legacy Intel AVX performance
// Reduce operations: benefit from larger grain sizes due to memory bandwidth
inline constexpr std::size_t LEGACY_INTEL_REDUCE_GRAIN_SIZE_LARGE = 32768;  // Expect 8-15x speedup
inline constexpr std::size_t LEGACY_INTEL_REDUCE_GRAIN_SIZE_MEDIUM = 128;   // Expect 4-8x speedup
inline constexpr std::size_t LEGACY_INTEL_REDUCE_GRAIN_SIZE_SMALL = 64;     // Expect 2-4x speedup

// Transform complex: benefit from balanced grain sizes
inline constexpr std::size_t LEGACY_INTEL_TRANSFORM_COMPLEX_GRAIN_SIZE_LARGE =
    32768;  // Expect 3-5x speedup
inline constexpr std::size_t LEGACY_INTEL_TRANSFORM_COMPLEX_GRAIN_SIZE_MEDIUM =
    16384;  // Expect 3-4x speedup
inline constexpr std::size_t LEGACY_INTEL_TRANSFORM_COMPLEX_GRAIN_SIZE_SMALL =
    1024;  // Expect 2-3x speedup

// Count operations: lighter weight, smaller grain sizes work well
inline constexpr std::size_t LEGACY_INTEL_COUNT_IF_GRAIN_SIZE_LARGE = 256;    // Expect 2-4x speedup
inline constexpr std::size_t LEGACY_INTEL_COUNT_IF_GRAIN_SIZE_MEDIUM = 1024;  // Expect 1-2x speedup

// Transform simple: memory-bound, very small grain sizes optimal
inline constexpr std::size_t LEGACY_INTEL_TRANSFORM_SIMPLE_GRAIN_SIZE_LARGE =
    8;  // Expect 1-2x speedup
inline constexpr std::size_t LEGACY_INTEL_TRANSFORM_SIMPLE_GRAIN_SIZE_MEDIUM =
    8192;  // Expect 1-1.5x speedup

// Size thresholds for adaptive grain size selection (in elements)
inline constexpr std::size_t LEGACY_INTEL_SMALL_DATASET_THRESHOLD = 10000;
inline constexpr std::size_t LEGACY_INTEL_MEDIUM_DATASET_THRESHOLD = 100000;
inline constexpr std::size_t LEGACY_INTEL_LARGE_DATASET_THRESHOLD = 1000000;

// Conservative defaults for legacy Intel (already defined above as general AVX defaults)
// DEFAULT_GRAIN_SIZE = 256 already defined
// MONTE_CARLO_GRAIN_SIZE = 64 already defined
inline constexpr std::size_t LEGACY_INTEL_MAX_GRAIN_SIZE =
    32768;  // Upper limit based on cache efficiency

// Distribution-specific parallel thresholds (based on empirical benchmarking)
// These represent sizes where parallel processing becomes beneficial vs serial
// Exponential distribution - very efficient parallel processing
inline constexpr std::size_t EXPONENTIAL_PDF_THRESHOLD = 64;     // Expect 2-4x speedup
inline constexpr std::size_t EXPONENTIAL_CDF_THRESHOLD = 64;     // Expect 2-5x speedup
inline constexpr std::size_t EXPONENTIAL_LOGPDF_THRESHOLD = 64;  // Expect 2-3x speedup

// Gaussian distribution - moderate parallel efficiency
inline constexpr std::size_t GAUSSIAN_PDF_THRESHOLD = 64;      // Expect 1.5-3x speedup
inline constexpr std::size_t GAUSSIAN_CDF_THRESHOLD = 64;      // Expect 2-3x speedup
inline constexpr std::size_t GAUSSIAN_LOGPDF_THRESHOLD = 512;  // Expect 2-3x speedup

// Uniform distribution - simple operations, variable efficiency
inline constexpr std::size_t UNIFORM_PDF_THRESHOLD = 256;    // Expect 1-2x speedup
inline constexpr std::size_t UNIFORM_CDF_THRESHOLD = 64;     // Expect 1-3x speedup
inline constexpr std::size_t UNIFORM_LOGPDF_THRESHOLD = 64;  // Expect 1-2x speedup

// Poisson distribution - complex computations, higher thresholds
inline constexpr std::size_t POISSON_PDF_THRESHOLD = 32768;     // Expect 2-3x speedup
inline constexpr std::size_t POISSON_CDF_THRESHOLD = 2048;      // Expect 2-4x speedup
inline constexpr std::size_t POISSON_LOGPDF_THRESHOLD = 16384;  // Expect 2-3x speedup

// Discrete distribution - complex lookup operations
inline constexpr std::size_t DISCRETE_PDF_THRESHOLD = 524288;   // Expect 1-2x speedup
inline constexpr std::size_t DISCRETE_CDF_THRESHOLD = 32768;    // Expect 2-3x speedup
inline constexpr std::size_t DISCRETE_LOGPDF_THRESHOLD = 4096;  // Expect 2-3x speedup

// Gamma distribution - moderate complexity, similar to Gaussian
inline constexpr std::size_t GAMMA_PDF_THRESHOLD = 128;    // Expect 2-3x speedup
inline constexpr std::size_t GAMMA_CDF_THRESHOLD = 256;    // Expect 2-4x speedup
inline constexpr std::size_t GAMMA_LOGPDF_THRESHOLD = 64;  // Expect 2-3x speedup
}  // namespace avx

/// ===== AVX2 Architecture Constants =====
/// For Intel Haswell+ and AMD Excavator+ with improved 256-bit SIMD + FMA
namespace avx2 {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL =
    1536;  // Better FMA performance
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 512;           // 256 cache lines, 2KB per thread
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 256;  // 128 cache lines
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE =
    1024;  // 512 cache lines, 4KB per thread
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 128;
inline constexpr std::size_t MAX_GRAIN_SIZE = 8192;
}  // namespace avx2

/// ===== AVX-512 Architecture Constants =====
/// For Intel Skylake-X+ and AMD Zen4+ with 512-bit SIMD (8 doubles per vector)
namespace avx512 {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 8192;  // Very high overhead
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 65536;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 1024;          // 512 cache lines, 4KB per thread
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 512;  // 256 cache lines
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE =
    2048;  // 1MB cache lines, 8KB per thread
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 256;
inline constexpr std::size_t MAX_GRAIN_SIZE = 16384;
}  // namespace avx512

/// ===== ARM NEON Architecture Constants =====
/// For ARM Cortex-A series with 128-bit SIMD (2 doubles per vector)
namespace neon {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL =
    1536;  // ARM typically lower thread overhead
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;  // Smaller L1 caches on ARM
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 48;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace neon

/// ===== Fallback Constants for Unknown Architectures =====
namespace fallback {
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 2048;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
}  // namespace fallback

/// ===== Legacy Constants for Backward Compatibility =====
/// NOTE: These constants are defined in platform_constants_impl.cpp to avoid ODR violations
/// For new code, prefer the adaptive:: functions below

/// Minimum dataset size for parallel statistical algorithms
/// Statistical algorithms benefit from parallelization when
/// processing large datasets above this threshold
inline constexpr std::size_t MIN_DATASET_SIZE_FOR_PARALLEL = 1000;

/// Minimum number of bootstrap samples for parallel bootstrap
/// When performing bootstrap resampling, parallelization
/// becomes beneficial above this threshold
inline constexpr std::size_t MIN_BOOTSTRAP_SAMPLES_FOR_PARALLEL = 100;

/// Minimum total work units for parallel Monte Carlo methods
/// Monte Carlo simulations benefit from parallelization when the total
/// computational work exceeds this threshold
inline constexpr std::size_t MIN_TOTAL_WORK_FOR_MONTE_CARLO_PARALLEL = 10000;

/// Minimum work per thread in parallel reductions
/// For parallel sum reductions and similar operations
inline constexpr std::size_t MIN_WORK_PER_THREAD = 100;

/// Batch size for parallel processing of data samples
/// When processing multiple data samples in statistical algorithms
inline constexpr std::size_t SAMPLE_BATCH_SIZE = 16;

/// Minimum matrix size for parallel matrix operations
/// Matrix operations (multiplication, decomposition) benefit from
/// parallelization above this threshold
inline constexpr std::size_t MIN_MATRIX_SIZE_FOR_PARALLEL = 256;

/// Minimum number of iterations for parallel iterative algorithms
/// Iterative algorithms like EM benefit from parallelization
/// when the number of iterations is large
inline constexpr std::size_t MIN_ITERATIONS_FOR_PARALLEL = 10;

/// Parallel processing batch sizes for different operations
namespace batch_sizes {
/// Small batch for lightweight operations
inline constexpr std::size_t SMALL_BATCH = 64;

/// Medium batch for standard operations
inline constexpr std::size_t MEDIUM_BATCH = 256;

/// Large batch for computation-intensive operations
inline constexpr std::size_t LARGE_BATCH = 512;

/// Extra large batch for very intensive operations
inline constexpr std::size_t XLARGE_BATCH = 1024;

/// Maximum batch size (memory constraint)
inline constexpr std::size_t MAX_BATCH = 65536;
}  // namespace batch_sizes

/// Statistical performance tuning constants
namespace tuning {
/// Minimum number of samples required before adaptive tuning kicks in
inline constexpr size_t MIN_SAMPLES_FOR_TUNING = 100;         // Minimum operations before tuning
inline constexpr std::chrono::seconds TUNING_INTERVAL{30};    // How often to consider tuning
inline constexpr double SIGNIFICANT_CHANGE_THRESHOLD = 0.05;  // 5% change triggers re-evaluation
}  // namespace tuning

}  // namespace parallel

/// Runtime adaptive parallel optimization functions (forward declarations)
/// These provide CPU feature-based optimization for parallel processing
/// Implementation is in platform_constants_impl.cpp for PIMPL optimization

/// Get platform-optimized minimum elements for parallel processing
std::size_t get_min_elements_for_parallel();

/// Get platform-optimized minimum elements for distribution parallel processing
std::size_t get_min_elements_for_distribution_parallel();

/// Get platform-optimized minimum elements for simple distribution parallel processing
std::size_t get_min_elements_for_simple_distribution_parallel();

/// Get platform-optimized grain size
std::size_t get_default_grain_size();

/// Get platform-optimized simple operation grain size
std::size_t get_simple_operation_grain_size();

/// Get platform-optimized complex operation grain size
std::size_t get_complex_operation_grain_size();

/// Get platform-optimized Monte Carlo grain size
std::size_t get_monte_carlo_grain_size();

/// Get platform-optimized maximum grain size
std::size_t get_max_grain_size();

/// Memory access and prefetching optimization constants
namespace memory {
/// Platform-specific prefetching distance tuning
namespace prefetch {
// Phase 3C: Flattened from distance:: namespace
/// Conservative prefetch distance for older/low-power CPUs (in cache lines)
inline constexpr std::size_t DISTANCE_CONSERVATIVE = 2;

/// Standard prefetch distance for most modern CPUs (in cache lines)
inline constexpr std::size_t DISTANCE_STANDARD = 4;

/// Aggressive prefetch distance for high-end CPUs with large caches (in cache lines)
inline constexpr std::size_t DISTANCE_AGGRESSIVE = 8;

/// Ultra-aggressive prefetch for specialized workloads (in cache lines)
inline constexpr std::size_t DISTANCE_ULTRA_AGGRESSIVE = 16;

/// Platform-specific prefetch distances are now provided by cpu_vendor_constants.h
/// Access via
/// stats::arch::cpu::{intel,amd,arm,apple_silicon}::{SEQUENTIAL,RANDOM,MATRIX}_PREFETCH_DISTANCE

// Phase 3C: Flattened from strategy:: namespace
/// Sequential access prefetch multipliers
inline constexpr double STRATEGY_SEQUENTIAL_MULTIPLIER = 2.0;  // More aggressive for sequential
inline constexpr double STRATEGY_RANDOM_MULTIPLIER = 0.5;      // Conservative for random
inline constexpr double STRATEGY_STRIDED_MULTIPLIER = 1.5;     // Moderate for strided access

/// Minimum elements before prefetching becomes beneficial
inline constexpr std::size_t STRATEGY_MIN_PREFETCH_SIZE = 32;

/// Maximum practical prefetch distance (memory bandwidth constraint)
inline constexpr std::size_t STRATEGY_MAX_PREFETCH_DISTANCE = 1024;

/// Prefetch granularity (align prefetch to cache line boundaries)
inline constexpr std::size_t STRATEGY_PREFETCH_GRANULARITY =
    8;  // 64-byte cache line / 8-byte double

// Phase 3C: Flattened from timing:: namespace
/// Memory latency estimates for prefetch scheduling (in CPU cycles)
inline constexpr std::size_t TIMING_L1_LATENCY_CYCLES = 4;      // L1 cache hit
inline constexpr std::size_t TIMING_L2_LATENCY_CYCLES = 12;     // L2 cache hit
inline constexpr std::size_t TIMING_L3_LATENCY_CYCLES = 36;     // L3 cache hit
inline constexpr std::size_t TIMING_DRAM_LATENCY_CYCLES = 300;  // Main memory access

/// Prefetch lead time (how far ahead to prefetch based on expected latency)
inline constexpr std::size_t TIMING_L2_PREFETCH_LEAD = 32;     // Elements ahead for L2 prefetch
inline constexpr std::size_t TIMING_L3_PREFETCH_LEAD = 128;    // Elements ahead for L3 prefetch
inline constexpr std::size_t TIMING_DRAM_PREFETCH_LEAD = 512;  // Elements ahead for DRAM prefetch
}  // namespace prefetch

/// Memory access pattern optimization
namespace access {
/// Cache line utilization constants
inline constexpr std::size_t CACHE_LINE_SIZE_BYTES = 64;  // Standard cache line size
inline constexpr std::size_t DOUBLES_PER_CACHE_LINE = 8;  // 64 bytes / 8 bytes per double
inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;   // Alignment requirement

// Phase 3C: Flattened from bandwidth:: namespace
/// Optimal burst sizes for different memory types
inline constexpr std::size_t BANDWIDTH_DDR4_BURST_SIZE = 64;   // Optimal DDR4 burst
inline constexpr std::size_t BANDWIDTH_DDR5_BURST_SIZE = 128;  // Optimal DDR5 burst
inline constexpr std::size_t BANDWIDTH_HBM_BURST_SIZE = 256;   // High Bandwidth Memory burst

/// Memory channel utilization targets
inline constexpr double BANDWIDTH_TARGET_UTILIZATION = 0.8;  // Aim for 80% bandwidth usage
inline constexpr double BANDWIDTH_MAX_UTILIZATION = 0.95;    // Maximum before thrashing

// Phase 3C: Flattened from layout:: namespace
/// Array-of-Structures vs Structure-of-Arrays thresholds
inline constexpr std::size_t LAYOUT_AOS_TO_SOA_THRESHOLD = 1000;  // Switch to SOA for larger sizes

/// Memory pool and alignment settings
inline constexpr std::size_t LAYOUT_MEMORY_POOL_ALIGNMENT = 4096;  // Page-aligned pools
inline constexpr std::size_t LAYOUT_SMALL_ALLOCATION_THRESHOLD =
    256;  // Use pool for smaller allocations
inline constexpr std::size_t LAYOUT_LARGE_PAGE_THRESHOLD = 2097152;  // 2MB huge page threshold

// Phase 3C: Flattened from numa:: namespace
/// NUMA-aware allocation thresholds
inline constexpr std::size_t NUMA_AWARE_THRESHOLD = 1048576;  // 1MB threshold for NUMA awareness

/// Thread affinity and memory locality settings
inline constexpr std::size_t NUMA_LOCAL_THRESHOLD = 65536;  // Prefer local memory below this size
inline constexpr double NUMA_MIGRATION_COST = 0.1;          // Cost factor for NUMA migration
}  // namespace access

/// Memory allocation strategy constants
namespace allocation {
/// Pool-based allocation sizes
inline constexpr std::size_t SMALL_POOL_SIZE = 4096;     // 4KB pools
inline constexpr std::size_t MEDIUM_POOL_SIZE = 65536;   // 64KB pools
inline constexpr std::size_t LARGE_POOL_SIZE = 1048576;  // 1MB pools

/// Allocation alignment requirements
inline constexpr std::size_t MIN_ALLOCATION_ALIGNMENT = 8;      // Minimum 8-byte alignment
inline constexpr std::size_t SIMD_ALLOCATION_ALIGNMENT = 32;    // SIMD-friendly alignment
inline constexpr std::size_t PAGE_ALLOCATION_ALIGNMENT = 4096;  // Page alignment

// Phase 3C: Flattened from growth:: namespace
/// Memory growth strategies
inline constexpr double GROWTH_EXPONENTIAL_FACTOR = 1.5;  // 50% growth per expansion
inline constexpr double GROWTH_LINEAR_FACTOR = 1.2;       // 20% growth for large allocations
inline constexpr std::size_t GROWTH_THRESHOLD = 1048576;  // Switch to linear above 1MB
}  // namespace allocation
}  // namespace memory

/// Platform-specific tuning functions
// Consolidated into detail namespace (was: namespace platform)
/**
 * @brief Get optimized SIMD block size based on detected CPU features
 * @return Optimal block size for SIMD operations
 */
std::size_t get_optimal_simd_block_size();

/**
 * @brief Get optimized memory alignment based on detected CPU features
 * @return Optimal memory alignment in bytes
 */
std::size_t get_optimal_alignment();

/**
 * @brief Get cache line size based on detected CPU vendor
 * @return Platform-specific cache line size in bytes
 */
std::size_t get_cache_line_size();

/**
 * @brief Get optimized minimum size for SIMD operations
 * @return Minimum size threshold for SIMD benefit
 */
std::size_t get_min_simd_size();

/**
 * @brief Get optimized parallel processing thresholds based on CPU features
 * @return Optimal minimum elements for parallel processing
 */
std::size_t get_min_parallel_elements();

/**
 * @brief Get platform-optimized grain size for parallel operations
 * @return Optimal grain size for work distribution
 */
std::size_t get_optimal_grain_size() noexcept;

/**
 * @brief Check if platform supports efficient transcendental functions
 * @return True if CPU has hardware support for fast transcendental operations
 */
bool supports_fast_transcendental();

/**
 * @brief Get cache-optimized thresholds for algorithms
 * @return Structure with GPU-accelerated thresholds
 */
struct CacheThresholds {
    std::size_t l1_optimal_size;  // Optimal size for L1 cache
    std::size_t l2_optimal_size;  // Optimal size for L2 cache
    std::size_t l3_optimal_size;  // Optimal size for L3 cache
    std::size_t blocking_size;    // Optimal blocking size for cache tiling
};

CacheThresholds get_cache_thresholds();
// End of consolidated platform constants
// Phase 3A: Removed unused cache namespaces and constants

}  // namespace arch
}  // namespace stats

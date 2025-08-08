#pragma once

#include <cstddef>
#include <climits>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <algorithm>

// Forward declaration for platform-specific tuning
#include "cpu_detection.h"

namespace libstats {
    namespace cpu { 
        const Features& get_features(); 
        size_t optimal_double_width();
        size_t optimal_alignment();
    }
}

/**
 * @file platform/platform_constants.h
 * @brief Platform-dependent optimization constants and runtime tuning functions
 * 
 * This header contains all platform-specific constants, SIMD optimization parameters,
 * parallel processing thresholds, and memory optimization constants that depend on
 * the target hardware architecture.
 */

namespace libstats {
namespace constants {

/// SIMD optimization parameters and architectural constants
namespace simd {
    /// Default SIMD block size for vectorized operations
    inline constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;
    
    /// Minimum problem size to benefit from SIMD
    inline constexpr std::size_t MIN_SIMD_SIZE = 4;
    
    /// Maximum block size for cache optimization
    inline constexpr std::size_t MAX_BLOCK_SIZE = 64;
    
    /// SIMD alignment requirement (bytes)
    inline constexpr std::size_t SIMD_ALIGNMENT = 32;
    
    /// Platform-specific SIMD alignment constants
    namespace alignment {
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
    }
    
    /// Matrix operation block sizes for cache-friendly operations
    namespace matrix {
        /// Small matrix block size for L1 cache optimization
        inline constexpr std::size_t L1_BLOCK_SIZE = 64;
        
        /// Medium matrix block size for L2 cache optimization  
        inline constexpr std::size_t L2_BLOCK_SIZE = 256;
        
        /// Large matrix block size for L3 cache optimization
        inline constexpr std::size_t L3_BLOCK_SIZE = 1024;
        
        /// Step size for matrix traversal (optimized for cache lines)
        inline constexpr std::size_t STEP_SIZE = 8;
        
        /// Panel width for matrix decomposition algorithms
        inline constexpr std::size_t PANEL_WIDTH = 64;
        
        /// Minimum matrix size for blocking to be beneficial
        inline constexpr std::size_t MIN_BLOCK_SIZE = 32;
        
        /// Maximum practical block size (memory constraint)
        inline constexpr std::size_t MAX_BLOCK_SIZE = 2048;
    }
    
    /// Platform-specific SIMD register widths (in number of doubles)
    namespace registers {
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
    }
    
    /// Loop unrolling factors for different architectures
    namespace unroll {
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
    }
    
    /// CPU detection and runtime constants
    namespace cpu {
        /// Maximum backoff time during CPU feature detection (nanoseconds)
        inline constexpr uint64_t MAX_BACKOFF_NANOSECONDS = 1000;
        
        /// Default cache line size fallback (bytes)
        inline constexpr uint32_t DEFAULT_CACHE_LINE_SIZE = 64;
        
        /// Default TSC frequency measurement duration (milliseconds)
        inline constexpr uint32_t DEFAULT_TSC_SAMPLE_MS = 10;
        
        /// Conversion factor from nanoseconds to Hertz
        inline constexpr double NANOSECONDS_TO_HZ = 1e9;
    }
    
    /// SIMD optimization thresholds and platform-specific constants
    namespace optimization {
        /// Medium dataset minimum size for alignment benefits
        inline constexpr std::size_t MEDIUM_DATASET_MIN_SIZE = 32;
        
        /// Minimum threshold for alignment benefit checks
        inline constexpr std::size_t ALIGNMENT_BENEFIT_THRESHOLD = 32;
        
        /// Minimum size for AVX-512 aligned datasets
        inline constexpr std::size_t AVX512_MIN_ALIGNED_SIZE = 8;
        
        /// Aggressive SIMD threshold for Apple Silicon
        inline constexpr std::size_t APPLE_SILICON_AGGRESSIVE_THRESHOLD = 6;
        
        /// Minimum size threshold for AVX-512 small benefit
        inline constexpr std::size_t AVX512_SMALL_BENEFIT_THRESHOLD = 4;
    }
}

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
        inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;  // 64 cache lines
        inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;   // 32 cache lines
        inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256; // 128 cache lines
        inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 32;
        inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
    }
    
    /// ===== AVX Architecture Constants =====
    /// For Intel Sandy Bridge+ and AMD Bulldozer+ with 256-bit SIMD (4 doubles per vector)
    namespace avx {
        inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;  // Higher overhead with wider SIMD
        inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
        inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
        inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;  // 128 cache lines, 1KB per thread
        inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;  // 64 cache lines
        inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 512; // 256 cache lines, 2KB per thread
        inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;
        inline constexpr std::size_t MAX_GRAIN_SIZE = 4096;
        
        /// ===== Legacy Intel AVX (Ivy Bridge/Sandy Bridge) Specific Tuning =====
        /// Optimized for Intel Core i-series 3rd generation (Ivy Bridge) and similar
        /// Family 6, Model 58 - AVX without AVX2/FMA, mobile/desktop CPUs ~2012-2013
        namespace legacy_intel {
            // Parallel thresholds - confirmed suitable for legacy Intel AVX
            inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
            inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
            inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
            
            // Operation-specific grain sizes optimized for legacy Intel AVX performance
            // Reduce operations: benefit from larger grain sizes due to memory bandwidth
            inline constexpr std::size_t REDUCE_GRAIN_SIZE_LARGE = 32768;    // Expect 8-15x speedup
            inline constexpr std::size_t REDUCE_GRAIN_SIZE_MEDIUM = 128;     // Expect 4-8x speedup
            inline constexpr std::size_t REDUCE_GRAIN_SIZE_SMALL = 64;       // Expect 2-4x speedup
            
            // Transform complex: benefit from balanced grain sizes
            inline constexpr std::size_t TRANSFORM_COMPLEX_GRAIN_SIZE_LARGE = 32768;  // Expect 3-5x speedup
            inline constexpr std::size_t TRANSFORM_COMPLEX_GRAIN_SIZE_MEDIUM = 16384; // Expect 3-4x speedup
            inline constexpr std::size_t TRANSFORM_COMPLEX_GRAIN_SIZE_SMALL = 1024;   // Expect 2-3x speedup
            
            // Count operations: lighter weight, smaller grain sizes work well
            inline constexpr std::size_t COUNT_IF_GRAIN_SIZE_LARGE = 256;     // Expect 2-4x speedup
            inline constexpr std::size_t COUNT_IF_GRAIN_SIZE_MEDIUM = 1024;   // Expect 1-2x speedup
            
            // Transform simple: memory-bound, very small grain sizes optimal
            inline constexpr std::size_t TRANSFORM_SIMPLE_GRAIN_SIZE_LARGE = 8;       // Expect 1-2x speedup
            inline constexpr std::size_t TRANSFORM_SIMPLE_GRAIN_SIZE_MEDIUM = 8192;   // Expect 1-1.5x speedup
            
            // Size thresholds for adaptive grain size selection (in elements)
            inline constexpr std::size_t SMALL_DATASET_THRESHOLD = 10000;
            inline constexpr std::size_t MEDIUM_DATASET_THRESHOLD = 100000;
            inline constexpr std::size_t LARGE_DATASET_THRESHOLD = 1000000;
            
            // Conservative defaults for general use
            inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;     // Balanced for mixed workloads
            inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;  // Conservative for MC simulations
            inline constexpr std::size_t MAX_GRAIN_SIZE = 32768;       // Upper limit based on cache efficiency
            
            // Distribution-specific parallel thresholds (based on empirical benchmarking)
            // These represent sizes where parallel processing becomes beneficial vs serial
            namespace distributions {
                // Exponential distribution - very efficient parallel processing
                inline constexpr std::size_t EXPONENTIAL_PDF_THRESHOLD = 64;     // Expect 2-4x speedup
                inline constexpr std::size_t EXPONENTIAL_CDF_THRESHOLD = 64;     // Expect 2-5x speedup
                inline constexpr std::size_t EXPONENTIAL_LOGPDF_THRESHOLD = 64;  // Expect 2-3x speedup
                
                // Gaussian distribution - moderate parallel efficiency
                inline constexpr std::size_t GAUSSIAN_PDF_THRESHOLD = 64;        // Expect 1.5-3x speedup
                inline constexpr std::size_t GAUSSIAN_CDF_THRESHOLD = 64;        // Expect 2-3x speedup
                inline constexpr std::size_t GAUSSIAN_LOGPDF_THRESHOLD = 512;    // Expect 2-3x speedup
                
                // Uniform distribution - simple operations, variable efficiency
                inline constexpr std::size_t UNIFORM_PDF_THRESHOLD = 256;        // Expect 1-2x speedup
                inline constexpr std::size_t UNIFORM_CDF_THRESHOLD = 64;         // Expect 1-3x speedup
                inline constexpr std::size_t UNIFORM_LOGPDF_THRESHOLD = 64;      // Expect 1-2x speedup
                
                // Poisson distribution - complex computations, higher thresholds
                inline constexpr std::size_t POISSON_PDF_THRESHOLD = 32768;      // Expect 2-3x speedup
                inline constexpr std::size_t POISSON_CDF_THRESHOLD = 2048;       // Expect 2-4x speedup
                inline constexpr std::size_t POISSON_LOGPDF_THRESHOLD = 16384;   // Expect 2-3x speedup
                
                // Discrete distribution - complex lookup operations
                inline constexpr std::size_t DISCRETE_PDF_THRESHOLD = 524288;    // Expect 1-2x speedup
                inline constexpr std::size_t DISCRETE_CDF_THRESHOLD = 32768;     // Expect 2-3x speedup
                inline constexpr std::size_t DISCRETE_LOGPDF_THRESHOLD = 4096;   // Expect 2-3x speedup
                
                // Gamma distribution - moderate complexity, similar to Gaussian
                inline constexpr std::size_t GAMMA_PDF_THRESHOLD = 128;          // Expect 2-3x speedup
                inline constexpr std::size_t GAMMA_CDF_THRESHOLD = 256;          // Expect 2-4x speedup  
                inline constexpr std::size_t GAMMA_LOGPDF_THRESHOLD = 64;        // Expect 2-3x speedup
            }
        }
    }
    
    /// ===== AVX2 Architecture Constants =====
    /// For Intel Haswell+ and AMD Excavator+ with improved 256-bit SIMD + FMA
    namespace avx2 {
        inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
        inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1536;  // Better FMA performance
        inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
        inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 512;  // 256 cache lines, 2KB per thread
        inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 256;  // 128 cache lines
        inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 1024; // 512 cache lines, 4KB per thread
        inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 128;
        inline constexpr std::size_t MAX_GRAIN_SIZE = 8192;
    }
    
    /// ===== AVX-512 Architecture Constants =====
    /// For Intel Skylake-X+ and AMD Zen4+ with 512-bit SIMD (8 doubles per vector)
    namespace avx512 {
        inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 8192;   // Very high overhead
        inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
        inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 65536;
        inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 1024; // 512 cache lines, 4KB per thread
        inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 512;  // 256 cache lines
        inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 2048; // 1MB cache lines, 8KB per thread
        inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 256;
        inline constexpr std::size_t MAX_GRAIN_SIZE = 16384;
    }
    
    /// ===== ARM NEON Architecture Constants =====
    /// For ARM Cortex-A series with 128-bit SIMD (2 doubles per vector)
    namespace neon {
        inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 1536;   // ARM typically lower thread overhead
        inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;
        inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 16384;
        inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;  // Smaller L1 caches on ARM
        inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
        inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
        inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 48;
        inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;
    }
    
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
    }
    
    /// ===== Legacy Constants for Backward Compatibility =====
    /// NOTE: These are fallback constants - for new code, prefer the adaptive:: functions
    inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
    inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;
    inline constexpr std::size_t MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 32768;
    inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;
    inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;
    
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
    }
    
    /// Platform-optimized functions for runtime tuning
    /// These functions provide optimized values based on detected CPU features
    namespace adaptive {
        /// Get platform-optimized minimum elements for parallel processing
        inline std::size_t min_elements_for_parallel() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::MIN_ELEMENTS_FOR_PARALLEL;
            } else if (features.avx2) {
                return avx2::MIN_ELEMENTS_FOR_PARALLEL;
            } else if (cpu::is_sandy_ivy_bridge()) {
                return avx::legacy_intel::MIN_ELEMENTS_FOR_PARALLEL;
            } else if (features.avx) {
                return avx::MIN_ELEMENTS_FOR_PARALLEL;
            } else if (features.sse2) {
                return sse::MIN_ELEMENTS_FOR_PARALLEL;
            } else if (features.neon) {
                return neon::MIN_ELEMENTS_FOR_PARALLEL;
            } else {
                return fallback::MIN_ELEMENTS_FOR_PARALLEL;
            }
        }
        
        /// Get platform-optimized minimum elements for distribution parallel processing
        inline std::size_t min_elements_for_distribution_parallel() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            } else if (features.avx2) {
                return avx2::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            } else if (cpu::is_sandy_ivy_bridge()) {
                return avx::legacy_intel::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            } else if (features.avx) {
                return avx::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            } else if (features.sse2) {
                return sse::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            } else if (features.neon) {
                return neon::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            } else {
                return fallback::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL;
            }
        }
        
        /// Get platform-optimized minimum elements for simple distribution parallel processing
        inline std::size_t min_elements_for_simple_distribution_parallel() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            } else if (features.avx2) {
                return avx2::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            } else if (cpu::is_sandy_ivy_bridge()) {
                return avx::legacy_intel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            } else if (features.avx) {
                return avx::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            } else if (features.sse2) {
                return sse::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            } else if (features.neon) {
                return neon::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            } else {
                return fallback::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL;
            }
        }
        
        /// Get platform-optimized grain size
        inline std::size_t grain_size() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::DEFAULT_GRAIN_SIZE;
            } else if (features.avx2) {
                return avx2::DEFAULT_GRAIN_SIZE;
            } else if (cpu::is_sandy_ivy_bridge()) {
                return avx::legacy_intel::DEFAULT_GRAIN_SIZE;
            } else if (features.avx) {
                return avx::DEFAULT_GRAIN_SIZE;
            } else if (features.sse2) {
                return sse::DEFAULT_GRAIN_SIZE;
            } else if (features.neon) {
                return neon::DEFAULT_GRAIN_SIZE;
            } else {
                return fallback::DEFAULT_GRAIN_SIZE;
            }
        }
        
        /// Get platform-optimized simple operation grain size
        inline std::size_t simple_operation_grain_size() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::SIMPLE_OPERATION_GRAIN_SIZE;
            } else if (features.avx2) {
                return avx2::SIMPLE_OPERATION_GRAIN_SIZE;
            } else if (features.avx) {
                return avx::SIMPLE_OPERATION_GRAIN_SIZE;
            } else if (features.sse2) {
                return sse::SIMPLE_OPERATION_GRAIN_SIZE;
            } else if (features.neon) {
                return neon::SIMPLE_OPERATION_GRAIN_SIZE;
            } else {
                return fallback::SIMPLE_OPERATION_GRAIN_SIZE;
            }
        }
        
        /// Get platform-optimized complex operation grain size
        inline std::size_t complex_operation_grain_size() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::COMPLEX_OPERATION_GRAIN_SIZE;
            } else if (features.avx2) {
                return avx2::COMPLEX_OPERATION_GRAIN_SIZE;
            } else if (features.avx) {
                return avx::COMPLEX_OPERATION_GRAIN_SIZE;
            } else if (features.sse2) {
                return sse::COMPLEX_OPERATION_GRAIN_SIZE;
            } else if (features.neon) {
                return neon::COMPLEX_OPERATION_GRAIN_SIZE;
            } else {
                return fallback::COMPLEX_OPERATION_GRAIN_SIZE;
            }
        }
        
        /// Get platform-optimized Monte Carlo grain size
        inline std::size_t monte_carlo_grain_size() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::MONTE_CARLO_GRAIN_SIZE;
            } else if (features.avx2) {
                return avx2::MONTE_CARLO_GRAIN_SIZE;
            } else if (features.avx) {
                return avx::MONTE_CARLO_GRAIN_SIZE;
            } else if (features.sse2) {
                return sse::MONTE_CARLO_GRAIN_SIZE;
            } else if (features.neon) {
                return neon::MONTE_CARLO_GRAIN_SIZE;
            } else {
                return fallback::MONTE_CARLO_GRAIN_SIZE;
            }
        }
        
        /// Get platform-optimized maximum grain size
        inline std::size_t max_grain_size() {
            const auto& features = cpu::get_features();
            
            if (features.avx512f) {
                return avx512::MAX_GRAIN_SIZE;
            } else if (features.avx2) {
                return avx2::MAX_GRAIN_SIZE;
            } else if (features.avx) {
                return avx::MAX_GRAIN_SIZE;
            } else if (features.sse2) {
                return sse::MAX_GRAIN_SIZE;
            } else if (features.neon) {
                return neon::MAX_GRAIN_SIZE;
            } else {
                return fallback::MAX_GRAIN_SIZE;
            }
        }
    }
    
    /// Statistical performance tuning constants
    namespace tuning {
        /// Minimum number of samples required before adaptive tuning kicks in
        inline constexpr size_t MIN_SAMPLES_FOR_TUNING = 100;      // Minimum operations before tuning
        inline constexpr std::chrono::seconds TUNING_INTERVAL{30}; // How often to consider tuning
        inline constexpr double SIGNIFICANT_CHANGE_THRESHOLD = 0.05; // 5% change triggers re-evaluation
    }
}

/// Memory access and prefetching optimization constants
namespace memory {
    /// Platform-specific prefetching distance tuning
    namespace prefetch {
        /// Base prefetch distance constants (in cache lines)
        namespace distance {
            /// Conservative prefetch distance for older/low-power CPUs
            inline constexpr std::size_t CONSERVATIVE = 2;
            
            /// Standard prefetch distance for most modern CPUs
            inline constexpr std::size_t STANDARD = 4;
            
            /// Aggressive prefetch distance for high-end CPUs with large caches
            inline constexpr std::size_t AGGRESSIVE = 8;
            
            /// Ultra-aggressive prefetch for specialized workloads
            inline constexpr std::size_t ULTRA_AGGRESSIVE = 16;
        }
        
        /// Platform-specific prefetch distances (in elements, not cache lines)
        namespace platform {
            /// Apple Silicon prefetch tuning
            namespace apple_silicon {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 256;  // Elements ahead
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 64;       // Conservative for random access
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 128;      // Matrix operations
                inline constexpr std::size_t PREFETCH_STRIDE = 8;                 // Stride for strided access
            }
            
            /// Intel prefetch tuning (Skylake+)
            namespace intel {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 192;
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 48;
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 96;
                inline constexpr std::size_t PREFETCH_STRIDE = 4;
            }
            
            /// AMD prefetch tuning (Zen+)
            namespace amd {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 128;
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 32;
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 64;
                inline constexpr std::size_t PREFETCH_STRIDE = 4;
            }
            
            /// ARM prefetch tuning
            namespace arm {
                inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 64;
                inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 16;
                inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 32;
                inline constexpr std::size_t PREFETCH_STRIDE = 2;
            }
        }
        
        /// Prefetch strategies based on access patterns
        namespace strategy {
            /// Sequential access prefetch multipliers
            inline constexpr double SEQUENTIAL_MULTIPLIER = 2.0;     // More aggressive for sequential
            inline constexpr double RANDOM_MULTIPLIER = 0.5;         // Conservative for random
            inline constexpr double STRIDED_MULTIPLIER = 1.5;        // Moderate for strided access
            
            /// Minimum elements before prefetching becomes beneficial
            inline constexpr std::size_t MIN_PREFETCH_SIZE = 32;
            
            /// Maximum practical prefetch distance (memory bandwidth constraint)
            inline constexpr std::size_t MAX_PREFETCH_DISTANCE = 1024;
            
            /// Prefetch granularity (align prefetch to cache line boundaries)
            inline constexpr std::size_t PREFETCH_GRANULARITY = 8;   // 64-byte cache line / 8-byte double
        }
        
        /// Software prefetch instruction timing
        namespace timing {
            /// Memory latency estimates for prefetch scheduling (in CPU cycles)
            inline constexpr std::size_t L1_LATENCY_CYCLES = 4;      // L1 cache hit
            inline constexpr std::size_t L2_LATENCY_CYCLES = 12;     // L2 cache hit
            inline constexpr std::size_t L3_LATENCY_CYCLES = 36;     // L3 cache hit
            inline constexpr std::size_t DRAM_LATENCY_CYCLES = 300;  // Main memory access
            
            /// Prefetch lead time (how far ahead to prefetch based on expected latency)
            inline constexpr std::size_t L2_PREFETCH_LEAD = 32;      // Elements ahead for L2 prefetch
            inline constexpr std::size_t L3_PREFETCH_LEAD = 128;     // Elements ahead for L3 prefetch
            inline constexpr std::size_t DRAM_PREFETCH_LEAD = 512;   // Elements ahead for DRAM prefetch
        }
    }
    
    /// Memory access pattern optimization
    namespace access {
        /// Cache line utilization constants
        inline constexpr std::size_t CACHE_LINE_SIZE_BYTES = 64;     // Standard cache line size
        inline constexpr std::size_t DOUBLES_PER_CACHE_LINE = 8;     // 64 bytes / 8 bytes per double
        inline constexpr std::size_t CACHE_LINE_ALIGNMENT = 64;      // Alignment requirement
        
        /// Memory bandwidth optimization
        namespace bandwidth {
            /// Optimal burst sizes for different memory types
            inline constexpr std::size_t DDR4_BURST_SIZE = 64;      // Optimal DDR4 burst
            inline constexpr std::size_t DDR5_BURST_SIZE = 128;     // Optimal DDR5 burst
            inline constexpr std::size_t HBM_BURST_SIZE = 256;      // High Bandwidth Memory burst
            
            /// Memory channel utilization targets
            inline constexpr double TARGET_BANDWIDTH_UTILIZATION = 0.8;  // Aim for 80% bandwidth usage
            inline constexpr double MAX_BANDWIDTH_UTILIZATION = 0.95;    // Maximum before thrashing
        }
        
        /// Memory layout optimization
        namespace layout {
            /// Array-of-Structures vs Structure-of-Arrays thresholds
            inline constexpr std::size_t AOS_TO_SOA_THRESHOLD = 1000;    // Switch to SOA for larger sizes
            
            /// Memory pool and alignment settings
            inline constexpr std::size_t MEMORY_POOL_ALIGNMENT = 4096;   // Page-aligned pools
            inline constexpr std::size_t SMALL_ALLOCATION_THRESHOLD = 256; // Use pool for smaller allocations
            inline constexpr std::size_t LARGE_PAGE_THRESHOLD = 2097152; // 2MB huge page threshold
        }
        
        /// Non-Uniform Memory Access (NUMA) optimization
        namespace numa {
            /// NUMA-aware allocation thresholds
            inline constexpr std::size_t NUMA_AWARE_THRESHOLD = 1048576; // 1MB threshold for NUMA awareness
            
            /// Thread affinity and memory locality settings
            inline constexpr std::size_t NUMA_LOCAL_THRESHOLD = 65536;   // Prefer local memory below this size
            inline constexpr double NUMA_MIGRATION_COST = 0.1;           // Cost factor for NUMA migration
        }
    }
    
    /// Memory allocation strategy constants
    namespace allocation {
        /// Pool-based allocation sizes
        inline constexpr std::size_t SMALL_POOL_SIZE = 4096;        // 4KB pools
        inline constexpr std::size_t MEDIUM_POOL_SIZE = 65536;      // 64KB pools
        inline constexpr std::size_t LARGE_POOL_SIZE = 1048576;     // 1MB pools
        
        /// Allocation alignment requirements
        inline constexpr std::size_t MIN_ALLOCATION_ALIGNMENT = 8;   // Minimum 8-byte alignment
        inline constexpr std::size_t SIMD_ALLOCATION_ALIGNMENT = 32; // SIMD-friendly alignment
        inline constexpr std::size_t PAGE_ALLOCATION_ALIGNMENT = 4096; // Page alignment
        
        /// Memory growth strategies
        namespace growth {
            inline constexpr double EXPONENTIAL_GROWTH_FACTOR = 1.5; // 50% growth per expansion
            inline constexpr double LINEAR_GROWTH_FACTOR = 1.2;      // 20% growth for large allocations
            inline constexpr std::size_t GROWTH_THRESHOLD = 1048576; // Switch to linear above 1MB
        }
    }
}

/// Platform-specific tuning functions
namespace platform {
    /**
     * @brief Get optimized SIMD block size based on detected CPU features
     * @return Optimal block size for SIMD operations
     */
    inline std::size_t get_optimal_simd_block_size() {
        const auto& features = cpu::get_features();
        
        // AVX-512: 8 doubles per register
        if (features.avx512f) {
            return 8;
        }
        // AVX/AVX2: 4 doubles per register
        else if (features.avx || features.avx2) {
            return 4;
        }
        // SSE2: 2 doubles per register
        else if (features.sse2) {
            return 2;
        }
        // ARM NEON: 2 doubles per register
        else if (features.neon) {
            return 2;
        }
        // No SIMD support
        else {
            return 1;
        }
    }
    
    /**
     * @brief Get optimized memory alignment based on detected CPU features
     * @return Optimal memory alignment in bytes
     */
    inline std::size_t get_optimal_alignment() {
        const auto& features = cpu::get_features();
        
        // AVX-512: 64-byte alignment
        if (features.avx512f) {
            return 64;
        }
        // AVX/AVX2: 32-byte alignment
        else if (features.avx || features.avx2) {
            return 32;
        }
        // SSE2: 16-byte alignment
        else if (features.sse2) {
            return 16;
        }
        // ARM NEON: 16-byte alignment
        else if (features.neon) {
            return 16;
        }
        // Default cache line alignment
        else {
            return features.cache_line_size > 0 ? features.cache_line_size : 64;
        }
    }
    
    /**
     * @brief Get optimized minimum size for SIMD operations
     * @return Minimum size threshold for SIMD benefit
     */
    inline std::size_t get_min_simd_size() {
        const auto& features = cpu::get_features();
        
        // Higher-end SIMD can handle smaller datasets efficiently
        if (features.avx512f) {
            return 4;
        }
        else if (features.avx2 || features.fma) {
            return 6;
        }
        else if (features.avx || features.sse4_2) {
            return 8;
        }
        else if (features.sse2 || features.neon) {
            return 12;
        }
        else {
            return 32;  // No SIMD benefit until larger sizes
        }
    }
    
    /**
     * @brief Get optimized parallel processing thresholds based on CPU features
     * @return Optimal minimum elements for parallel processing
     */
    inline std::size_t get_min_parallel_elements() {
        const auto& features = cpu::get_features();
        
        // More powerful SIMD allows for lower parallel thresholds
        if (features.avx512f) {
            return 256;
        }
        else if (features.avx2 || features.fma) {
            return 384;
        }
        else if (features.avx) {
            return 512;
        }
        else if (features.sse4_2) {
            return 768;
        }
        else if (features.sse2 || features.neon) {
            return 1024;
        }
        else {
            return 2048;  // Higher threshold for scalar operations
        }
    }
    
    /**
     * @brief Get platform-optimized grain size for parallel operations
     * @return Optimal grain size for work distribution
     */
    inline std::size_t get_optimal_grain_size() {
        const auto& features = cpu::get_features();
        const std::size_t optimal_block = get_optimal_simd_block_size();
        
        // Grain size should be a multiple of SIMD block size
        // and account for cache line efficiency
        const std::size_t cache_line_elements = features.cache_line_size / sizeof(double);
        const std::size_t base_grain = std::max(optimal_block * 8, cache_line_elements);
        
        // Adjust based on CPU capabilities
        if (features.avx512f) {
            return base_grain * 2;  // Can handle larger chunks efficiently
        }
        else if (features.avx2 || features.fma) {
            return static_cast<std::size_t>(std::round(static_cast<double>(base_grain) * 1.5));
        }
        else {
            return base_grain;
        }
    }
    
    /**
     * @brief Check if platform supports efficient transcendental functions
     * @return True if CPU has hardware support for fast transcendental operations
     */
    inline bool supports_fast_transcendental() {
        const auto& features = cpu::get_features();
        // FMA typically indicates more modern CPU with better transcendental support
        return features.fma || features.avx2 || features.avx512f;
    }
    
    /**
     * @brief Get cache-optimized thresholds for algorithms
     * @return Structure with cache-aware thresholds
     */
    struct CacheThresholds {
        std::size_t l1_optimal_size;    // Optimal size for L1 cache
        std::size_t l2_optimal_size;    // Optimal size for L2 cache
        std::size_t l3_optimal_size;    // Optimal size for L3 cache
        std::size_t blocking_size;      // Optimal blocking size for cache tiling
    };
    
    inline CacheThresholds get_cache_thresholds() {
        const auto& features = cpu::get_features();
        CacheThresholds thresholds{};
        
        // Convert cache sizes from bytes to number of doubles
        thresholds.l1_optimal_size = features.l1_cache_size > 0 ? 
            (features.l1_cache_size / sizeof(double)) / 2 : 4096;  // Use half of L1
        
        thresholds.l2_optimal_size = features.l2_cache_size > 0 ? 
            (features.l2_cache_size / sizeof(double)) / 2 : 32768;
        
        thresholds.l3_optimal_size = features.l3_cache_size > 0 ? 
            (features.l3_cache_size / sizeof(double)) / 4 : 262144;
        
        // Blocking size for cache tiling (typically sqrt of L1 size)
        thresholds.blocking_size = static_cast<std::size_t>(
            std::sqrt(static_cast<double>(thresholds.l1_optimal_size))
        );
        
        return thresholds;
    }
}

/// Cache system optimization constants
namespace cache {
    /// Cache sizing constants
    namespace sizing {
        /// Minimum cache size in bytes
        inline constexpr std::size_t MIN_CACHE_SIZE_BYTES = 64 * 1024;  // 64KB
        
        /// Maximum cache size in bytes
        inline constexpr std::size_t MAX_CACHE_SIZE_BYTES = 64 * 1024 * 1024;  // 64MB
        
        /// Minimum number of cache entries
        inline constexpr std::size_t MIN_ENTRY_COUNT = 32;
        
        /// Maximum number of cache entries
        inline constexpr std::size_t MAX_ENTRY_COUNT = 65536;
        
        /// Estimated bytes per cache entry (for sizing calculations)
        inline constexpr std::size_t BYTES_PER_ENTRY_ESTIMATE = 256;
        
        /// Fraction of L3 cache to use for application caching
        inline constexpr double L3_CACHE_FRACTION = 0.1;  // 10%
        
        /// Fraction of L2 cache to use for application caching (when no L3)
        inline constexpr double L2_CACHE_FRACTION = 0.05;  // 5%
    }
    
    /// Cache tuning parameters
    namespace tuning {
        /// Base TTL for cache entries
        inline constexpr std::chrono::milliseconds BASE_TTL{10000};  // 10 seconds
        
        /// TTL for high frequency CPUs
        inline constexpr std::chrono::milliseconds HIGH_FREQ_TTL{8000};  // 8 seconds
        
        /// TTL for ultra-high frequency CPUs
        inline constexpr std::chrono::milliseconds ULTRA_HIGH_FREQ_TTL{6000};  // 6 seconds
        
        /// CPU frequency thresholds for TTL adjustment
        inline constexpr uint64_t HIGH_FREQ_THRESHOLD_HZ = 3000000000ULL;  // 3 GHz
        inline constexpr uint64_t ULTRA_HIGH_FREQ_THRESHOLD_HZ = 4000000000ULL;  // 4 GHz
        
        /// Prefetch multipliers for different SIMD capabilities
        inline constexpr std::size_t AVX512_PREFETCH_MULTIPLIER = 3;
        inline constexpr std::size_t AVX2_PREFETCH_MULTIPLIER = 2;
        inline constexpr std::size_t SSE_PREFETCH_MULTIPLIER = 1;
    }
    
    /// Platform-specific cache configurations
    namespace platform {
        /// Apple Silicon (M1/M2/M3) cache tuning
        namespace apple_silicon {
            inline constexpr std::size_t DEFAULT_MAX_MEMORY_MB = 8;  // 8MB
            inline constexpr std::size_t DEFAULT_MAX_ENTRIES = 4096;
            inline constexpr std::size_t PREFETCH_QUEUE_SIZE = 64;
            inline constexpr double EVICTION_THRESHOLD = 0.75;
            inline constexpr std::size_t BATCH_EVICTION_SIZE = 16;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{12000};  // 12 seconds
            inline constexpr double HIT_RATE_TARGET = 0.88;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.75;
        }
        
        /// Intel cache tuning
        namespace intel {
            inline constexpr std::size_t DEFAULT_MAX_MEMORY_MB = 6;  // 6MB
            inline constexpr std::size_t DEFAULT_MAX_ENTRIES = 3072;
            inline constexpr std::size_t PREFETCH_QUEUE_SIZE = 48;
            inline constexpr double EVICTION_THRESHOLD = 0.80;
            inline constexpr std::size_t BATCH_EVICTION_SIZE = 12;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{10000};  // 10 seconds
            inline constexpr double HIT_RATE_TARGET = 0.85;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.70;
        }
        
        /// AMD cache tuning
        namespace amd {
            inline constexpr std::size_t DEFAULT_MAX_MEMORY_MB = 5;  // 5MB
            inline constexpr std::size_t DEFAULT_MAX_ENTRIES = 2560;
            inline constexpr std::size_t PREFETCH_QUEUE_SIZE = 40;
            inline constexpr double EVICTION_THRESHOLD = 0.82;
            inline constexpr std::size_t BATCH_EVICTION_SIZE = 10;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{9000};  // 9 seconds
            inline constexpr double HIT_RATE_TARGET = 0.83;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.68;
        }
        
        /// ARM generic cache tuning
        namespace arm {
            inline constexpr std::size_t DEFAULT_MAX_MEMORY_MB = 3;  // 3MB
            inline constexpr std::size_t DEFAULT_MAX_ENTRIES = 1536;
            inline constexpr std::size_t PREFETCH_QUEUE_SIZE = 24;
            inline constexpr double EVICTION_THRESHOLD = 0.85;
            inline constexpr std::size_t BATCH_EVICTION_SIZE = 8;
            inline constexpr std::chrono::milliseconds DEFAULT_TTL{8000};  // 8 seconds
            inline constexpr double HIT_RATE_TARGET = 0.80;
            inline constexpr double MEMORY_EFFICIENCY_TARGET = 0.65;
        }
    }
    
    /// Access pattern analysis constants
    namespace patterns {
        /// Maximum number of access patterns to keep in history
        inline constexpr std::size_t MAX_PATTERN_HISTORY = 1000;
        
        /// Threshold for detecting sequential access patterns
        inline constexpr double SEQUENTIAL_PATTERN_THRESHOLD = 0.7;
        
        /// Threshold for detecting random access patterns
        inline constexpr double RANDOM_PATTERN_THRESHOLD = 0.3;
        
        /// Size multipliers for different access patterns
        inline constexpr double SEQUENTIAL_SIZE_MULTIPLIER = 1.5;
        inline constexpr double RANDOM_SIZE_MULTIPLIER = 0.8;
        inline constexpr double MIXED_SIZE_MULTIPLIER = 1.1;
    }
}

} // namespace constants
} // namespace libstats

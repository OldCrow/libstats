#pragma once

/**
 * @file libstats.h
 * @brief Modern C++20 statistical distributions library
 * 
 * Provides comprehensive statistical interface with:
 * - PDF, CDF, and quantile functions
 * - Random sampling using std:: distributions
 * - Parameter estimation with MLE
 * - Statistical validation and diagnostics
 * - Thread-safe concurrent access
 * - SIMD optimization with runtime detection
 * - C++20 parallel execution with automatic fallback
 * - Advanced adaptive caching with memory management
 * - Zero external dependencies
 * 
 * SIMD USAGE GUIDE:
 * 
 * This library supports both compile-time and runtime SIMD detection:
 * 
 * 1. COMPILE-TIME DETECTION (simd.h):
 *    - Detects what SIMD instructions the compiler can generate
 *    - Provides feature macros: LIBSTATS_HAS_AVX, LIBSTATS_HAS_SSE2, etc.
 *    - Used for conditional compilation of SIMD code paths
 * 
 * 2. RUNTIME DETECTION (cpu_detection.h):
 *    - Detects what SIMD instructions the actual CPU supports
 *    - Provides functions: supports_avx(), supports_sse2(), etc.
 *    - Used to safely execute SIMD code only when CPU supports it
 * 
 * RECOMMENDED USAGE PATTERN:
 * 
 *   #include "libstats.h"  // Includes both simd.h and cpu_detection.h
 * 
 *   // Example: Safe AVX usage
 *   #ifdef LIBSTATS_HAS_AVX  // Compiler can generate AVX
 *   if (libstats::cpu::supports_avx()) {  // CPU actually supports AVX
 *       // Execute AVX-optimized code path
 *       vectorized_computation_avx(data, results, size);
 *   } else {
 *       // Fallback to non-AVX path
 *       vectorized_computation_fallback(data, results, size);
 *   }
 *   #else
 *   // Compiler doesn't support AVX, use fallback
 *   vectorized_computation_fallback(data, results, size);
 *   #endif
 * 
 * INTEGRATION WITH DISTRIBUTION CLASSES:
 * 
 * All distribution classes automatically use the best available SIMD
 * instructions for bulk operations like:
 * - Computing PDF/CDF for arrays of values
 * - Parameter estimation from large datasets
 * - Statistical validation operations
 * 
 * The library handles SIMD selection transparently, but you can query
 * the detected capabilities:
 * 
 *   std::cout << "SIMD level: " << libstats::cpu::best_simd_level() << std::endl;
 *   std::cout << "Vector width: " << libstats::cpu::optimal_double_width() << std::endl;
 * 
 * PARALLEL EXECUTION GUIDE:
 * 
 * This library provides C++20 parallel execution with automatic fallback:
 * 
 * 1. AUTOMATIC PARALLEL ALGORITHMS (parallel_execution.h):
 *    - Uses std::execution policies when available
 *    - Automatic fallback to serial execution when parallel policies unavailable
 *    - CPU-aware threshold detection for optimal performance
 * 
 * 2. RECOMMENDED USAGE PATTERN:
 * 
 *   #include "libstats.h"  // Includes parallel_execution.h
 * 
 *   // Example: Safe parallel transform
 *   std::vector<double> input(10000);
 *   std::vector<double> output(10000);
 *   
 *   // Automatically uses parallel execution if beneficial
 *   libstats::parallel::safe_transform(input.begin(), input.end(), output.begin(),
 *       [](double x) { return x * x; });
 * 
 * 3. AVAILABLE PARALLEL ALGORITHMS:
 *    - safe_fill, safe_transform, safe_reduce, safe_for_each
 *    - safe_sort, safe_partial_sort, safe_find, safe_find_if
 *    - safe_count, safe_count_if, safe_inclusive_scan, safe_exclusive_scan
 * 
 * All parallel algorithms automatically:
 * - Detect CPU capabilities for optimal thresholds
 * - Use C++20 execution policies when available
 * - Fallback to serial execution when parallel processing not beneficial
 * - Include numerical stability checks
 * 
 * ADAPTIVE CACHING GUIDE:
 * 
 * This library provides advanced caching for expensive statistical computations:
 * 
 * 1. AUTOMATIC CACHE MANAGEMENT (adaptive_cache.h):
 *    - Memory-aware eviction policies with configurable limits
 *    - TTL-based expiration to prevent stale data
 *    - Thread-safe concurrent access with optimized locking
 *    - Background optimization and performance metrics
 * 
 * 2. RECOMMENDED USAGE PATTERN:
 * 
 *   #include "libstats.h"  // Includes adaptive_cache.h
 * 
 *   // Example: Create cache for expensive quantile computations
 *   libstats::cache::AdaptiveCacheConfig config;
 *   config.max_memory_bytes = 2 * 1024 * 1024;  // 2MB limit
 *   config.ttl = std::chrono::minutes(5);       // 5 minute expiry
 *   
 *   libstats::cache::AdaptiveCache<double, double> quantile_cache(config);
 * 
 * 3. CACHE FEATURES:
 *    - Automatic memory pressure detection and response
 *    - Access pattern tracking for predictive prefetching
 *    - Comprehensive performance metrics and diagnostics
 *    - Configurable eviction policies (LRU, LFU, TTL, Adaptive)
 * 
 * The distribution classes will automatically use adaptive caching for:
 * - Expensive quantile function evaluations
 * - Complex special function computations (gamma, beta, etc.)
 * - Parameter estimation intermediate results
 * - Batch operation optimizations
 */

// Core framework
#include "core/distribution_base.h"
#include "core/constants.h"

// Performance and SIMD support
#include "platform/simd.h"
#include "platform/cpu_detection.h"

// Parallel execution support
#include "platform/parallel_execution.h"

// Advanced caching infrastructure
#include "platform/adaptive_cache.h"

// Distributions
#include "distributions/gaussian.h"
#include "distributions/exponential.h"
#include "distributions/uniform.h"
#include "distributions/poisson.h"
#include "distributions/gamma.h"
#include "distributions/discrete.h"

// Convenience namespace
namespace libstats {
    // Type aliases for common usage
    using Gaussian = GaussianDistribution;
    using Normal = GaussianDistribution;
    using Exponential = ExponentialDistribution;
    using Uniform = UniformDistribution;
    using Poisson = PoissonDistribution;
    using Gamma = GammaDistribution;
    using Discrete = DiscreteDistribution;
    
// Version information
constexpr int LIBSTATS_VERSION_MAJOR = 0;
constexpr int LIBSTATS_VERSION_MINOR = 7;
constexpr int LIBSTATS_VERSION_PATCH = 1;
    constexpr const char* VERSION_STRING = "0.7.1";
}

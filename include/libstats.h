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
 *   #define LIBSTATS_FULL_INTERFACE  // Enable full functionality
 *   #include "libstats.h"             // Includes adaptive_cache.h
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
 *
 * HEADER OPTIMIZATION GUIDE:
 *
 * This library uses conditional compilation to minimize header inclusion overhead:
 *
 * 1. DEFAULT MODE (forward declarations only):
 *    - By default, libstats.h includes only forward declarations and essential constants
 *    - Provides type information and compile-time features without implementation overhead
 *    - Ideal for header files that only need type information
 *    - Significantly reduces compilation time and preprocessor overhead
 *
 * 2. FULL INTERFACE MODE:
 *    - Define LIBSTATS_FULL_INTERFACE before including libstats.h to get full functionality
 *    - Includes all distribution implementations and platform optimizations
 *    - Required for using distribution objects and calling member functions
 *
 * RECOMMENDED USAGE PATTERNS:
 *
 * Header files (.h):
 *   #include "libstats.h"  // Just forward declarations, minimal overhead
 *   
 *   class MyClass {
 *       libstats::Gaussian* gaussian_;  // Pointer/reference works with forward declaration
 *   };
 *
 * Implementation files (.cpp):
 *   #define LIBSTATS_FULL_INTERFACE  // Enable full functionality
 *   #include "libstats.h"             // Get complete implementation
 *   
 *   void MyClass::compute() {
 *       auto dist = libstats::Gaussian::create(0.0, 1.0);  // Full implementation available
 *   }
 */

// Forward declarations for lightweight header inclusion
#include "core/forward_declarations.h"

// Essential constants that don't pull in heavy dependencies
#include "core/essential_constants.h"

// Light platform detection (compile-time only, no runtime detection yet)
#include "platform/simd.h"

// Conditional includes for full functionality
// These are only included when LIBSTATS_FULL_INTERFACE is defined
#ifdef LIBSTATS_FULL_INTERFACE
    // Core framework - full implementation
    #include "core/distribution_base.h"
    #include "core/constants.h"
    
    // Performance and platform detection (v0.9.1 enhancements)
    #include "platform/cpu_detection.h"
    #include "platform/parallel_execution.h"
    #include "platform/work_stealing_pool.h"
    
    // Advanced caching infrastructure
#include "cache/adaptive_cache.h"
    
    // Performance infrastructure (v0.9.1 additions)
#include "cache/distribution_cache.h"
    #include "core/performance_history.h"
    #include "core/performance_dispatcher.h"
    
    // Distribution implementations
    #include "distributions/gaussian.h"
    #include "distributions/exponential.h"
    #include "distributions/uniform.h"
    #include "distributions/poisson.h"
    #include "distributions/gamma.h"
    #include "distributions/discrete.h"
#endif // LIBSTATS_FULL_INTERFACE

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
constexpr int LIBSTATS_VERSION_MINOR = 9;
constexpr int LIBSTATS_VERSION_PATCH = 1;
    constexpr const char* VERSION_STRING = "0.9.1";
    
    /**
     * @brief Initialize performance systems to eliminate cold-start delays
     * 
     * This function performs one-time initialization of libstats' performance-critical
     * systems to eliminate cold-start latency during first-time batch operation dispatch.
     * 
     * **What gets initialized:**
     * - System capability detection and benchmarking (CPU features, SIMD support)
     * - Performance dispatcher with optimized thresholds
     * - SIMD policy detection and configuration
     * - Performance history singleton
     * - Thread pool infrastructure
     * 
     * **When to call:**
     * - Once at application startup, before using batch operations
     * - In unit tests setup to ensure consistent performance measurements
     * - Before performance-critical code sections to avoid cold-start penalty
     * 
     * **Performance impact:**
     * - First call: ~10-50ms initialization time (system-dependent)
     * - Subsequent calls: ~1-2ns (fast path with static flag)
     * - Eliminates 10-50ms delay from first batch operation call
     * 
     * **Thread safety:**
     * - Thread-safe: safe to call from multiple threads concurrently
     * - Uses static initialization guard for one-time setup
     * 
     * @example Basic usage:
     * @code
     * #include "libstats.h"
     * 
     * int main() {
     *     // Initialize performance systems once at startup
     *     libstats::initialize_performance_systems();
     *     
     *     // Now batch operations will have optimal performance from the start
     *     auto dist = libstats::Gaussian::create(0.0, 1.0);
     *     if (dist.isOk()) {
     *         std::vector<double> values(10000);
     *         std::vector<double> results(10000);
     *         dist.value.getProbability(values, results); // No cold-start delay
     *     }
     *     return 0;
     * }
     * @endcode
     * 
     * @example Unit test usage:
     * @code
     * class StatisticalTestSuite : public ::testing::Test {
     * protected:
     *     static void SetUpTestSuite() {
     *         // Initialize once for all tests in this suite
     *         libstats::initialize_performance_systems();
     *     }
     * };
     * @endcode
     * 
     * @note This function is optional but recommended for performance-critical applications.
     *       All libstats functionality works correctly without calling this function,
     *       but the first batch operation may experience initialization latency.
     * 
     * @since 0.7.2
     */
    void initialize_performance_systems();
}

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
 *   if (stats::cpu::supports_avx()) {  // CPU actually supports AVX
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
 *   std::cout << "SIMD level: " << stats::cpu::best_simd_level() << std::endl;
 *   std::cout << "Vector width: " << stats::cpu::optimal_double_width() << std::endl;
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
 *   stats::arch::safe_transform(input.begin(), input.end(), output.begin(),
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
 * HEADER OPTIMIZATION GUIDE:
 *
 * This library uses conditional compilation to minimize header inclusion overhead:
 *
 * 1. DEFAULT MODE:
 *    - Includes forward_declarations.h, essential_constants.h, and platform/simd.h.
 *    - platform/simd.h transitively pulls in platform_common.h, which includes <thread>,
 *      <mutex>, <shared_mutex>, <functional>, <memory>, <atomic>, <chrono>, and OS-specific
 *      system headers (<mach/mach.h>, <dispatch/dispatch.h>, <sys/sysctl.h> on macOS).
 *    - NOT a zero-overhead include. Suitable for translation units that need distribution
 *      type information (pointers, references) but not method implementations.
 *    - If you need truly minimal includes in your own library headers, include
 *      common/forward_declarations.h directly instead.
 *
 * 2. FULL INTERFACE MODE:
 *    - Define LIBSTATS_FULL_INTERFACE before including libstats.h to get full functionality
 *    - Includes all distribution implementations and platform optimizations
 *    - Required for using distribution objects and calling member functions
 *
 * RECOMMENDED USAGE PATTERNS:
 *
 * Header files (.h) — pointer/reference use only:
 *   #include "libstats.h"  // Forward declarations + simd.h (pulls OS headers)
 *
 *   class MyClass {
 *       stats::Gaussian* gaussian_;  // Pointer/reference works with forward declaration
 *   };
 *
 * Implementation files (.cpp) — full use:
 *   #define LIBSTATS_FULL_INTERFACE  // Enable full functionality
 *   #include "libstats.h"             // Get complete implementation
 *
 *   void MyClass::compute() {
 *       auto dist = stats::Gaussian::create(0.0, 1.0);  // Full implementation available
 *   }
 */

// Forward declarations for lightweight header inclusion
#include "common/forward_declarations.h"

// Essential constants that don't pull in heavy dependencies
#include "core/essential_constants.h"

// Light platform detection (compile-time only, no runtime detection yet)
#include "platform/simd.h"

// Conditional includes for full functionality
// These are only included when LIBSTATS_FULL_INTERFACE is defined
#ifdef LIBSTATS_FULL_INTERFACE
    // Core framework - full implementation
    #include "core/constants.h"
    #include "core/distribution_base.h"

    // Performance and platform detection (v0.9.1 enhancements)
    #include "platform/benchmark.h"
    #include "platform/cpu_detection.h"
    #include "platform/parallel_execution.h"
    #include "platform/thread_pool.h"
    #include "platform/work_stealing_pool.h"

    // Cache infrastructure
    #include "core/distribution_cache.h"  // Distribution parameter caching
    #include "core/performance_dispatcher.h"

    // Distribution implementations
    #include "distributions/beta.h"
    #include "distributions/binomial.h"
    #include "distributions/cauchy.h"
    #include "distributions/chi_squared.h"
    #include "distributions/discrete.h"
    #include "distributions/exponential.h"
    #include "distributions/gamma.h"
    #include "distributions/gaussian.h"
    #include "distributions/geometric.h"
    #include "distributions/laplace.h"
    #include "distributions/lognormal.h"
    #include "distributions/negative_binomial.h"
    #include "distributions/pareto.h"
    #include "distributions/poisson.h"
    #include "distributions/rayleigh.h"
    #include "distributions/student_t.h"
    #include "distributions/uniform.h"
    #include "distributions/von_mises.h"
    #include "distributions/weibull.h"

// Type aliases — inside the guard because they reference types only defined
// when LIBSTATS_FULL_INTERFACE is active.  Using them on incomplete types
// compiles but produces confusing "incomplete type" errors on first use.
namespace stats {
using Gaussian = GaussianDistribution;
using Normal = GaussianDistribution;
using Exponential = ExponentialDistribution;
using Uniform = UniformDistribution;
using Poisson = PoissonDistribution;
using Gamma = GammaDistribution;
using Discrete = DiscreteDistribution;
using ChiSquared = ChiSquaredDistribution;
using StudentT = StudentTDistribution;
using Beta = BetaDistribution;
using LogNormal = LogNormalDistribution;
using Pareto = ParetoDistribution;
using Weibull = WeibullDistribution;
using Rayleigh = RayleighDistribution;
using VonMises = VonMisesDistribution;
using Binomial = BinomialDistribution;
using Geometric = GeometricDistribution;
using Laplace = LaplaceDistribution;
using Cauchy = CauchyDistribution;
using NegativeBinomial = NegativeBinomialDistribution;
}  // namespace stats
#endif  // LIBSTATS_FULL_INTERFACE

// Version constants — generated from CMakeLists.txt project() declaration so
// they always match the CMake version (NEW-BS-1). libstats_version.h is
// produced by configure_file() into the build include shim directory.
#include "libstats_version.h"  // LIBSTATS_VERSION_MAJOR/MINOR/PATCH, VERSION_STRING
namespace stats {

/**
 * @brief Initialize performance systems to eliminate cold-start delays
 *
 * This function performs one-time initialization of libstats' performance-critical
 * systems to eliminate cold-start latency during first-time batch operation dispatch.
 *
 * **What gets initialized:**
 * - System capability detection (CPU features, SIMD support)
 * - SIMD policy detection and configuration
 * - Thread pool singletons (GlobalThreadPool, GlobalWorkStealingPool)
 *
 * **When to call:**
 * - Once at application startup, before using batch operations
 * - In unit tests setup to ensure consistent performance measurements
 * - Before performance-critical code sections to avoid cold-start penalty
 *
 * **Performance impact:**
 * - First call: ~1-5ms (thread pool startup dominates)
 * - Subsequent calls: ~1-2ns (fast path with static flag)
 * - Eliminates thread-launch latency from first parallel batch operation
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
 *     stats::initialize_performance_systems();
 *
 *     // Now batch operations will have optimal performance from the start
 *     auto dist = stats::Gaussian::create(0.0, 1.0);
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
 *         stats::initialize_performance_systems();
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
}  // namespace stats

// Backward compatibility: alias libstats to stats
// This allows existing code using libstats:: to continue working
namespace libstats = stats;

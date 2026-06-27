#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>

// Forward declarations for foundational layer dependencies
namespace stats {
namespace arch {
namespace simd {
class SIMDPolicy;
// We need the actual enum for the interface, so include the header
}  // namespace simd
}  // namespace arch
}  // namespace stats

#include "libstats/platform/simd_policy.h"
// DistributionType is defined in a minimal header with no platform deps (AQ-2).
#include "distribution_type.h"

// Forward declare OperationType so it can be used in selectStrategy
namespace stats {
namespace detail {
enum class OperationType;
}  // namespace detail
}  // namespace stats

/**
 * @file performance_dispatcher.h
 * @brief Intelligent auto-dispatch system for optimal performance strategy selection
 *
 * This system automatically selects the best execution strategy (scalar, SIMD, parallel, etc.)
 * based on problem characteristics, system capabilities, and performance history.
 *
 * Key Benefits:
 * - Eliminates performance decision burden from users
 * - Provides consistent API across all distributions
 * - Adapts to system capabilities and workload characteristics
 * - Maintains backward compatibility while simplifying usage
 */

namespace stats {
namespace detail {  // Performance utilities

/**
 * @brief Execution strategies for batch operations
 *
 * Names reflect what actually happens:
 *   SCALAR       — element-by-element loop, below the SIMD threshold
 *   VECTORIZED   — batch path through VectorOps; SIMD-accelerated for
 *                  distributions with a fixed-step VectorOps pipeline
 *                  (Gaussian, Exponential, Gamma, Uniform, Chi-squared,
 *                  Student's t, Beta, Log-Normal, Pareto, Weibull, Rayleigh).
 *                  Exception: Von Mises uses a scalar loop (no vector_cos)
 *   PARALLEL     — multi-threaded via ParallelUtils::parallelFor; inner
 *                  loops may or may not be vectorized per distribution
 *   WORK_STEALING — work-stealing thread pool for irregular workloads
 *
 * GPU_ACCELERATED was removed: the implementation silently fell back to
 * WORK_STEALING, making the name actively deceptive. If GPU support is
 * added in the future, a new enum value should be introduced at that time.
 */
enum class Strategy {
    SCALAR,        ///< Element-by-element loop (below SIMD threshold)
    VECTORIZED,    ///< Batch path through VectorOps
    PARALLEL,      ///< Multi-threaded via ParallelUtils::parallelFor
    WORK_STEALING  ///< Work-stealing pool for irregular workloads
};

// ComputationComplexity removed in v2.0.0 (Part 4): was the old
// complexity-aware dispatch parameter; never used after the dispatch refactor.
// complexityToString() in tool_utils.h removed alongside it.

/**
 * @brief System capabilities and performance characteristics
 */
class SystemCapabilities {
   public:
    static const SystemCapabilities& current();

    // CPU characteristics
    size_t logical_cores() const noexcept { return logical_cores_; }
    size_t physical_cores() const noexcept { return physical_cores_; }
    size_t l1_cache_size() const noexcept { return l1_cache_size_; }
    size_t l2_cache_size() const noexcept { return l2_cache_size_; }
    size_t l3_cache_size() const noexcept { return l3_cache_size_; }

    // SIMD capabilities
    bool has_sse2() const noexcept { return has_sse2_; }
    bool has_avx() const noexcept { return has_avx_; }
    bool has_avx2() const noexcept { return has_avx2_; }
    bool has_avx512() const noexcept { return has_avx512_; }
    bool has_neon() const noexcept { return has_neon_; }

   private:
    SystemCapabilities();
    void detectCapabilities();

    // Cached capability data
    size_t logical_cores_, physical_cores_;
    size_t l1_cache_size_, l2_cache_size_, l3_cache_size_;
    bool has_sse2_, has_avx_, has_avx2_, has_avx512_, has_neon_;
};

/**
 * @brief Performance decision engine for strategy selection
 */
class PerformanceDispatcher {
   public:
    PerformanceDispatcher();

    /**
     * @brief Select optimal execution strategy using profiling-derived lookup table
     *
     * @param batch_size Number of elements to process
     * @param dist_type Type of distribution
     * @param op_type Operation type (PDF, LOG_PDF, CDF, BATCH_FIT)
     * @param system System capabilities
     * @return Optimal strategy for the given parameters
     */
    Strategy selectStrategy(size_t batch_size, DistributionType dist_type, OperationType op_type,
                            const SystemCapabilities& system) const;

    /**
     * @brief Select multi-threaded strategy (PARALLEL vs WORK_STEALING)
     *
     * The choice depends on the threading backend (GCD vs Windows TP) and
     * whether hyperthreading is present, per four-architecture profiling data.
     * Public so DispatchUtils::mapHintToStrategy can route MAXIMIZE_THROUGHPUT
     * through the same OS-aware selection instead of hardcoding WORK_STEALING.
     */
    static Strategy selectMultiThreadedStrategy(DistributionType dist_type,
                                                const SystemCapabilities& system) noexcept;

   private:
    /// Cached SIMD level for table lookups
    arch::simd::SIMDPolicy::Level simd_level_;
};

/**
 * @brief Optional performance hints for advanced users
 */
struct PerformanceHint {
    enum class PreferredStrategy {
        AUTO,                ///< Let system decide (default)
        FORCE_SCALAR,        ///< Force scalar implementation
        FORCE_VECTORIZED,    ///< Force vectorized batch path
        FORCE_PARALLEL,      ///< Force parallel execution
        MINIMIZE_LATENCY,    ///< Optimize for lowest latency
        MAXIMIZE_THROUGHPUT  ///< Optimize for highest throughput
    };

    PreferredStrategy strategy = PreferredStrategy::AUTO;
    std::optional<size_t> thread_count;  ///< Override thread count (not yet wired into dispatch)

    static PerformanceHint minimal_latency() {
        return {PreferredStrategy::MINIMIZE_LATENCY, std::nullopt};
    }

    static PerformanceHint maximum_throughput() {
        return {PreferredStrategy::MAXIMIZE_THROUGHPUT, std::nullopt};
    }
};

}  // namespace detail
}  // namespace stats

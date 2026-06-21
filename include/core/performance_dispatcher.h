#pragma once

#include <chrono>
#include <functional>
#include <memory>
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

/**
 * @brief Computational complexity levels
 */
enum class ComputationComplexity {
    SIMPLE,    ///< Basic arithmetic operations
    MODERATE,  ///< Transcendental functions (exp, log)
    COMPLEX    ///< Special functions (gamma, erf)
};

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

    // Performance characteristics
    double simd_efficiency() const noexcept { return simd_efficiency_; }
    double threading_overhead_ns() const noexcept { return threading_overhead_ns_; }
    double memory_bandwidth_gb_s() const noexcept { return memory_bandwidth_gb_s_; }

   private:
    SystemCapabilities();
    void detectCapabilities();
    void benchmarkPerformance();

    // Cached capability data
    size_t logical_cores_, physical_cores_;
    size_t l1_cache_size_, l2_cache_size_, l3_cache_size_;
    bool has_sse2_, has_avx_, has_avx2_, has_avx512_, has_neon_;
    double simd_efficiency_, threading_overhead_ns_, memory_bandwidth_gb_s_;
};

/// Forward declaration — defined in performance_history.h.
class PerformanceHistory;

/**
 * @brief Performance decision engine for strategy selection
 */
class PerformanceDispatcher {
   public:
    /**
     * @brief Default constructor - initializes with architecture-aware thresholds
     */
    PerformanceDispatcher();

    /**
     * @brief Constructor with explicit system capabilities
     * @param system System capabilities for threshold initialization
     */
    explicit PerformanceDispatcher(const SystemCapabilities& system);
    /**
     * @brief Decision thresholds for strategy selection.
     * Distribution-specific per-arch thresholds live in dispatch_thresholds.h.
     */
    struct Thresholds {
        size_t simd_min = 8;               ///< Below this, SIMD overhead exceeds benefit
        size_t parallel_min = 1000;        ///< Below this, threading overhead exceeds benefit
        size_t work_stealing_min = 10000;  ///< Minimum batch size where work-stealing helps

        static Thresholds createForSIMDLevel(arch::simd::SIMDPolicy::Level level,
                                             const SystemCapabilities& system);

        static Thresholds getSSE2Profile();
        static Thresholds getAVXProfile();
        static Thresholds getAVX2Profile();
        static Thresholds getAVX512Profile();
        static Thresholds getNEONProfile();
        static Thresholds getScalarProfile();
    };

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
     * @brief Get current decision thresholds
     */
    const Thresholds& getThresholds() const noexcept { return thresholds_; }

    /**
     * @brief Update thresholds based on performance feedback
     */
    void updateThresholds(const Thresholds& new_thresholds);

    /**
     * @brief Record performance data for learning optimization
     *
     * @param strategy The strategy that was used
     * @param distribution_type Type of distribution processed
     * @param batch_size Number of elements processed
     * @param execution_time_ns Actual execution time in nanoseconds
     */
    static void recordPerformance(Strategy strategy, DistributionType distribution_type,
                                  std::size_t batch_size, std::uint64_t execution_time_ns) noexcept;

    /**
     * @brief Get access to the global performance history for advanced users
     * @return Reference to the performance history instance
     */
    static PerformanceHistory& getPerformanceHistory() noexcept;

   private:
    mutable Thresholds thresholds_;

    bool shouldUseWorkStealing(size_t batch_size, DistributionType dist_type) const;

    /**
     * @brief Select multi-threaded strategy (PARALLEL vs WORK_STEALING)
     *
     * The choice depends on the threading backend (GCD vs Windows TP) and
     * whether hyperthreading is present, per four-architecture profiling data.
     */
    static Strategy selectMultiThreadedStrategy(DistributionType dist_type,
                                                const SystemCapabilities& system) noexcept;

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
    bool disable_learning = false;       ///< Don't record performance data
    bool force_strategy = false;         ///< Override all safety checks
    std::optional<size_t> thread_count;  ///< Override thread count

    static PerformanceHint minimal_latency() {
        // Strategy tag drives dispatch; thread_count left unset.
        return {PreferredStrategy::MINIMIZE_LATENCY, false, false, std::nullopt};
    }

    static PerformanceHint maximum_throughput() {
        return {PreferredStrategy::MAXIMIZE_THROUGHPUT, false, false, std::nullopt};
    }
};

}  // namespace detail
}  // namespace stats

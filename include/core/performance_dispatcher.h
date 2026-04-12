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
 * Names reflect what actually happens, not what we might wish happened:
 *   SCALAR       — element-by-element loop, below the SIMD threshold
 *   VECTORIZED   — batch path through VectorOps; SIMD-accelerated for
 *                  distributions that route through VectorOps (currently
 *                  Gaussian), scalar loops for others until Phase 6
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
 * @brief Distribution types for strategy optimization
 */
enum class DistributionType {
    UNIFORM,      ///< Simple uniform distribution
    GAUSSIAN,     ///< Normal distribution (moderate complexity)
    EXPONENTIAL,  ///< Exponential distribution (moderate complexity)
    DISCRETE,     ///< Discrete uniform distribution
    POISSON,      ///< Poisson distribution (complex)
    GAMMA,        ///< Gamma distribution (most complex)
    STUDENT_T,    ///< Student's t distribution (log+transcendental, full real-line domain)
    BETA,         ///< Beta distribution (log-space, bounded support [0,1])
    CHI_SQUARED   ///< Chi-squared distribution (delegates to Gamma; positive real-line support)
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
     * @brief SIMD architecture profiles with optimal thresholds
     */
    enum class SIMDArchitecture {
        NONE,    ///< No SIMD support
        SSE2,    ///< 128-bit SIMD, older x86 processors
        AVX,     ///< 256-bit SIMD, moderate x86 processors
        AVX2,    ///< 256-bit with integer support, modern x86
        AVX512,  ///< 512-bit SIMD, high-end x86 processors
        NEON     ///< 128-bit SIMD, ARM processors
    };

    /**
     * @brief Decision thresholds (architecture-aware with capability refinement)
     */
    struct Thresholds {
        size_t simd_min = 8;               ///< Below this, SIMD overhead exceeds benefit
        size_t parallel_min = 1000;        ///< Below this, threading overhead exceeds benefit
        size_t work_stealing_min = 10000;  ///< Minimum batch size where work-stealing helps

        // Distribution-specific overrides (architecture-dependent)
        size_t uniform_parallel_min = 65536;    ///< Simple operations need higher threshold
        size_t gaussian_parallel_min = 256;     ///< Complex operations benefit earlier
        size_t exponential_parallel_min = 512;  ///< Moderate complexity
        size_t discrete_parallel_min = 1024;    ///< Integer operations
        size_t poisson_parallel_min = 512;      ///< Complex discrete distribution
        size_t gamma_parallel_min = 256;        ///< Most complex distribution
        size_t student_t_parallel_min = 256;    ///< Log+transcendental, matches Gaussian
        size_t beta_parallel_min = 256;  ///< Two log calls, bounded support, matches Gaussian
        size_t chi_squared_parallel_min = 256;  ///< Delegates to Gamma; positive real-line support

        /**
         * @brief Create thresholds based on SIMDPolicy level
         * @param level SIMD level from SIMDPolicy
         * @param system System capabilities for refinement
         * @return Optimized thresholds for the SIMD level
         */
        static Thresholds createForSIMDLevel(arch::simd::SIMDPolicy::Level level,
                                             const SystemCapabilities& system);

        /**
         * @brief Create architecture-specific thresholds (legacy compatibility)
         * @param arch SIMD architecture detected
         * @param system System capabilities for refinement
         * @return Optimized thresholds for the architecture
         */
        static Thresholds createForArchitecture(SIMDArchitecture arch,
                                                const SystemCapabilities& system);

        /**
         * @brief Get SSE2-optimized thresholds (128-bit SIMD)
         */
        static Thresholds getSSE2Profile();

        /**
         * @brief Get AVX-optimized thresholds (256-bit SIMD, limited efficiency)
         */
        static Thresholds getAVXProfile();

        /**
         * @brief Get AVX2-optimized thresholds (256-bit SIMD with integer support)
         */
        static Thresholds getAVX2Profile();

        /**
         * @brief Get AVX-512-optimized thresholds (512-bit SIMD, high efficiency)
         */
        static Thresholds getAVX512Profile();

        /**
         * @brief Get NEON-optimized thresholds (128-bit ARM SIMD)
         */
        static Thresholds getNEONProfile();

        /**
         * @brief Get fallback thresholds (no SIMD support)
         */
        static Thresholds getScalarProfile();

        /**
         * @brief Refine thresholds based on measured system performance
         * @param system System capabilities for fine-tuning
         */
        void refineWithCapabilities(const SystemCapabilities& system);
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
    static class PerformanceHistory& getPerformanceHistory() noexcept;

   private:
    mutable Thresholds thresholds_;

    size_t getDistributionSpecificParallelThreshold(DistributionType dist_type) const;
    bool shouldUseWorkStealing(size_t batch_size, DistributionType dist_type) const;

    /**
     * @brief Detect the highest available SIMD architecture
     */
    static SIMDArchitecture detectSIMDArchitecture(const SystemCapabilities& system) noexcept;

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
        return {PreferredStrategy::MINIMIZE_LATENCY, false, false, 1};
    }

    static PerformanceHint maximum_throughput() {
        return {PreferredStrategy::MAXIMIZE_THROUGHPUT, false, false, std::nullopt};
    }
};

}  // namespace detail
}  // namespace stats

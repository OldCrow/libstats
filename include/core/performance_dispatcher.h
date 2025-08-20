#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>

// Forward declarations for foundational layer dependencies
namespace libstats {
namespace simd {
class SIMDPolicy;
// We need the actual enum for the interface, so include the header
}  // namespace simd
}  // namespace libstats

#include "../platform/simd_policy.h"

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

namespace libstats {
namespace performance {

/**
 * @brief Execution strategies available for batch operations
 */
enum class Strategy {
    SCALAR,          ///< Single element or very small batches
    SIMD_BATCH,      ///< SIMD vectorized for medium batches
    PARALLEL_SIMD,   ///< Parallel + SIMD for large batches
    WORK_STEALING,   ///< Dynamic load balancing for irregular workloads
    GPU_ACCELERATED  ///< GPU-accelerated execution (CPU fallback if GPU unavailable)
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
    GAMMA         ///< Gamma distribution (most complex)
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
        size_t simd_min = 8;                 ///< SIMD overhead threshold
        size_t parallel_min = 1000;          ///< Threading overhead threshold
        size_t work_stealing_min = 10000;    ///< Work-stealing benefit threshold
        size_t gpu_accelerated_min = 50000;  ///< GPU acceleration threshold

        // Distribution-specific overrides (architecture-dependent)
        size_t uniform_parallel_min = 65536;    ///< Simple operations need higher threshold
        size_t gaussian_parallel_min = 256;     ///< Complex operations benefit earlier
        size_t exponential_parallel_min = 512;  ///< Moderate complexity
        size_t discrete_parallel_min = 1024;    ///< Integer operations
        size_t poisson_parallel_min = 512;      ///< Complex discrete distribution
        size_t gamma_parallel_min = 256;        ///< Most complex distribution

        /**
         * @brief Create thresholds based on SIMDPolicy level
         * @param level SIMD level from SIMDPolicy
         * @param system System capabilities for refinement
         * @return Optimized thresholds for the SIMD level
         */
        static Thresholds createForSIMDLevel(simd::SIMDPolicy::Level level,
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
     * @brief Select optimal execution strategy
     *
     * @param batch_size Number of elements to process
     * @param dist_type Type of distribution
     * @param complexity Computational complexity level
     * @param system System capabilities
     * @return Optimal strategy for the given parameters
     */
    Strategy selectOptimalStrategy(size_t batch_size, DistributionType dist_type,
                                   ComputationComplexity complexity,
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
    bool shouldUseGpuAccelerated(size_t batch_size, const SystemCapabilities& system) const;

    /**
     * @brief Detect the highest available SIMD architecture
     * @param system System capabilities for detection
     * @return Detected SIMD architecture
     */
    static SIMDArchitecture detectSIMDArchitecture(const SystemCapabilities& system) noexcept;

    /**
     * @brief Select strategy based on system capabilities and performance metrics
     *
     * Uses measured SIMD efficiency, threading overhead, and memory bandwidth
     * to make adaptive decisions based on actual hardware performance.
     *
     * @param batch_size Number of elements to process
     * @param dist_type Type of distribution for complexity estimation
     * @param system Measured system capabilities
     * @return Optimal strategy for this hardware and workload
     */
    Strategy selectStrategyBasedOnCapabilities(size_t batch_size, DistributionType dist_type,
                                               const SystemCapabilities& system) const;
};

/**
 * @brief Optional performance hints for advanced users
 */
struct PerformanceHint {
    enum class PreferredStrategy {
        AUTO,                ///< Let system decide (default)
        FORCE_SCALAR,        ///< Force scalar implementation
        FORCE_SIMD,          ///< Force SIMD if available
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

}  // namespace performance
}  // namespace libstats

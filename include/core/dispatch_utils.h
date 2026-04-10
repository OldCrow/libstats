#pragma once

#include "libstats/platform/thread_pool.h"  // For ParallelUtils
#include "libstats/platform/work_stealing_pool.h"
#include "performance_dispatcher.h"

#include <functional>
#include <span>

namespace stats {
namespace detail {  // Performance utilities

/**
 * @brief Utility class for templated auto-dispatch of batch operations across distributions
 *
 * ## Call Path
 *
 * Every batch operation (e.g. gaussian.getProbability(span, span)) flows through three layers:
 *
 *   Layer 1 — Validate:
 *     Distribution API method (e.g. GaussianDistribution::getProbabilityBatch)
 *     \u2193 checks spans, handles empty, forwards to DispatchUtils::autoDispatch
 *
 *   Layer 2 — Select strategy:
 *     DispatchUtils::autoDispatch
 *     \u2193 if hint is AUTO: PerformanceDispatcher::selectOptimalStrategy (threshold lookup +
 *                             optional performance history override)
 *       if hint is explicit: DispatchUtils::mapHintToStrategy
 *     \u2193 DispatchUtils::executeStrategy (switches on Strategy enum)
 *
 *   Layer 3 — Execute:
 *     Strategy::SCALAR       \u2192 element-by-element scalar_func loop
 *     Strategy::VECTORIZED   \u2192 batch_func \u2192 *BatchUnsafeImpl \u2192 VectorOps (SIMD or
 * scalar) Strategy::PARALLEL     \u2192 parallel_func \u2192 ParallelUtils::parallelFor
 *     Strategy::WORK_STEALING \u2192 work_stealing_func \u2192 WorkStealingPool::parallelFor
 *
 * The GpuAcceleratedFunc template parameter and gpu_accelerated_func argument are
 * retained for ABI compatibility and will be removed in Phase 6 when all distribution
 * batch implementations are updated.
 */
class DispatchUtils {
   public:
    /**
     * @brief Template function to perform auto-dispatch for batch operations
     *
     * @tparam Distribution The distribution type
     * @tparam ScalarFunc Function type for scalar operations
     * @tparam BatchFunc Function type for SIMD batch operations
     * @tparam ParallelFunc Function type for parallel operations
     * @tparam WorkStealingFunc Function type for work-stealing operations
     * @tparam GpuAcceleratedFunc Function type for GPU-accelerated operations
     *
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param hint Performance hint for strategy selection
     * @param dist_type Distribution type enum for strategy selection
     * @param complexity Computation complexity enum for strategy selection
     * @param scalar_func Function to call for scalar operations
     * @param batch_func Function to call for SIMD batch operations
     * @param parallel_func Function to call for parallel operations
     * @param work_stealing_func Function to call for work-stealing operations
     * @param gpu_accelerated_func Function to call for GPU-accelerated operations
     */
    template <typename Distribution, typename ScalarFunc, typename BatchFunc, typename ParallelFunc,
              typename WorkStealingFunc, typename GpuAcceleratedFunc>
    static void autoDispatch(const Distribution& dist, std::span<const double> values,
                             std::span<double> results, const PerformanceHint& hint,
                             DistributionType dist_type, ComputationComplexity complexity,
                             ScalarFunc&& scalar_func, BatchFunc&& batch_func,
                             ParallelFunc&& parallel_func, WorkStealingFunc&& work_stealing_func,
                             GpuAcceleratedFunc&& gpu_accelerated_func) {
        // Validate input
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }

        const size_t count = values.size();
        if (count == 0)
            return;

        // Handle single-value case efficiently
        if (count == 1) {
            results[0] = scalar_func(dist, values[0]);
            return;
        }

        // Get global dispatcher and system capabilities
        static thread_local PerformanceDispatcher dispatcher;
        const SystemCapabilities& system = SystemCapabilities::current();

        // Smart dispatch based on problem characteristics
        auto strategy = Strategy::SCALAR;

        if (hint.strategy == PerformanceHint::PreferredStrategy::AUTO) {
            strategy = dispatcher.selectOptimalStrategy(count, dist_type, complexity, system);
        } else {
            strategy = mapHintToStrategy(hint.strategy, count);
        }

        // Execute using selected strategy
        executeStrategy(strategy, dist, values, results, count,
                        std::forward<ScalarFunc>(scalar_func), std::forward<BatchFunc>(batch_func),
                        std::forward<ParallelFunc>(parallel_func),
                        std::forward<WorkStealingFunc>(work_stealing_func),
                        std::forward<GpuAcceleratedFunc>(gpu_accelerated_func));
    }

    /**
     * @brief Execute SIMD batch operations with common cache validation pattern
     *
     * @tparam Distribution The distribution type
     * @tparam BatchUnsafeFunc Function type for unsafe SIMD batch implementation
     *
     * @param dist Reference to the distribution instance
     * @param values Input values array
     * @param results Output results array
     * @param count Number of elements to process
     * @param batch_unsafe_func Function to call for SIMD batch processing
     */
    template <typename Distribution, typename BatchUnsafeFunc>
    static void executeBatchSIMD(const Distribution& dist, const double* values, double* results,
                                 std::size_t count, BatchUnsafeFunc&& batch_unsafe_func) {
        if (count == 0)
            return;
        if (!values || !results) {
            return;  // Can't throw in noexcept context for some methods
        }

        // Use the distribution's cache validation logic
        batch_unsafe_func(dist, values, results, count);
    }

    /**
     * @brief Executes a batch operation with explicit strategy selection
     *
     * @tparam Distribution The distribution type
     * @tparam ScalarFunc Function type for scalar operations
     * @tparam BatchFunc Function type for SIMD batch operations
     * @tparam ParallelFunc Function type for parallel operations
     * @tparam WorkStealingFunc Function type for work-stealing operations
     * @tparam GpuAcceleratedFunc Function type for GPU-accelerated operations
     *
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param strategy Explicit strategy to use
     * @param scalar_func Function to call for scalar operations
     * @param batch_func Function to call for SIMD batch operations
     * @param parallel_func Function to call for parallel operations
     * @param work_stealing_func Function to call for work-stealing operations
     * @param gpu_accelerated_func Function to call for GPU-accelerated operations
     */
    template <typename Distribution, typename ScalarFunc, typename BatchFunc, typename ParallelFunc,
              typename WorkStealingFunc, typename GpuAcceleratedFunc>
    static void executeWithStrategy(const Distribution& dist, std::span<const double> values,
                                    std::span<double> results, Strategy strategy,
                                    ScalarFunc&& scalar_func, BatchFunc&& batch_func,
                                    ParallelFunc&& parallel_func,
                                    WorkStealingFunc&& work_stealing_func,
                                    GpuAcceleratedFunc&& gpu_accelerated_func) {
        // Validate input
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }

        const size_t count = values.size();
        if (count == 0)
            return;

        // Execute using provided strategy
        executeStrategy(strategy, dist, values, results, count,
                        std::forward<ScalarFunc>(scalar_func), std::forward<BatchFunc>(batch_func),
                        std::forward<ParallelFunc>(parallel_func),
                        std::forward<WorkStealingFunc>(work_stealing_func),
                        std::forward<GpuAcceleratedFunc>(gpu_accelerated_func));
    }

   private:
    /**
     * @brief Maps performance hints to strategies
     */
    static Strategy mapHintToStrategy(PerformanceHint::PreferredStrategy hint_strategy,
                                      size_t count) {
        switch (hint_strategy) {
            case PerformanceHint::PreferredStrategy::FORCE_SCALAR:
                return Strategy::SCALAR;
            case PerformanceHint::PreferredStrategy::FORCE_VECTORIZED:
                return Strategy::VECTORIZED;
            case PerformanceHint::PreferredStrategy::FORCE_PARALLEL:
                return Strategy::PARALLEL;
            case PerformanceHint::PreferredStrategy::MINIMIZE_LATENCY:
                return (count <= 8) ? Strategy::SCALAR : Strategy::VECTORIZED;
            case PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT:
                return Strategy::PARALLEL;
            default:
                return Strategy::SCALAR;
        }
    }

    /**
     * @brief Executes the selected strategy with appropriate function calls
     */
    template <typename Distribution, typename ScalarFunc, typename BatchFunc, typename ParallelFunc,
              typename WorkStealingFunc, typename GpuAcceleratedFunc>
    static void executeStrategy(Strategy strategy, const Distribution& dist,
                                std::span<const double> values, std::span<double> results,
                                size_t count, ScalarFunc&& scalar_func, BatchFunc&& batch_func,
                                ParallelFunc&& parallel_func, WorkStealingFunc&& work_stealing_func,
                                GpuAcceleratedFunc&& gpu_accelerated_func) {
        switch (strategy) {
            case Strategy::SCALAR:
                // Use simple loop for tiny batches (< 8 elements)
                for (size_t i = 0; i < count; ++i) {
                    results[i] = scalar_func(dist, values[i]);
                }
                break;

            case Strategy::VECTORIZED:
                // Batch path through VectorOps. For Gaussian this uses SIMD intrinsics;
                // for other distributions it currently uses scalar loops until Phase 6.
                batch_func(dist, values.data(), results.data(), count);
                break;

            case Strategy::PARALLEL:
                // Multi-threaded via ParallelUtils::parallelFor.
                parallel_func(dist, values, results);
                break;

            case Strategy::WORK_STEALING: {
                // Work-stealing pool for irregular or variable-cost workloads.
                // Thread explosion note: this pool is thread_local, so each calling
                // thread creates its own WorkStealingPool with hardware_concurrency()
                // workers. N concurrent WORK_STEALING batches spawn N * cores threads.
                // Results are always correct; performance degrades under this condition.
                // Phase 6 will introduce a shared pool to address this.
                static thread_local WorkStealingPool default_pool;
                work_stealing_func(dist, values, results, default_pool);
                break;
            }
        }
        // gpu_accelerated_func is intentionally unused — GPU_ACCELERATED was removed
        // from the Strategy enum. Retained for ABI compatibility until Phase 6.
        (void)gpu_accelerated_func;
    }

    /**
     * @brief Execute parallel batch operations with common pattern
     *
     * @tparam Distribution The distribution type
     * @tparam ComputationFunc Function type for element-wise computation
     *
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param computation_func Function to compute result for each element
     */
    template <typename Distribution, typename ComputationFunc>
    static void executeBatchParallel([[maybe_unused]] const Distribution& dist,
                                     std::span<const double> values, std::span<double> results,
                                     ComputationFunc&& computation_func) {
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }

        const std::size_t count = values.size();
        if (count == 0)
            return;

        // Use ParallelUtils::parallelFor for Level 0-3 integration
        if (arch::should_use_parallel(count)) {
            ParallelUtils::parallelFor(std::size_t{0}, count,
                                       [&](std::size_t i) { computation_func(i); });
        } else {
            // Serial processing for small datasets
            for (std::size_t i = 0; i < count; ++i) {
                computation_func(i);
            }
        }
    }

    /**
     * @brief Execute work-stealing batch operations with common pattern
     *
     * @tparam Distribution The distribution type
     * @tparam ComputationFunc Function type for element-wise computation
     *
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param pool Work-stealing thread pool
     * @param computation_func Function to compute result for each element
     */
    template <typename Distribution, typename ComputationFunc>
    static void executeBatchWorkStealing([[maybe_unused]] const Distribution& dist,
                                         std::span<const double> values, std::span<double> results,
                                         WorkStealingPool& pool,
                                         ComputationFunc&& computation_func) {
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }

        const std::size_t count = values.size();
        if (count == 0)
            return;

        // Use WorkStealingPool for dynamic load balancing
        // Use same threshold as regular parallel operations to avoid inconsistency
        if (stats::shouldUseWorkStealing(
                count, stats::arch::get_min_elements_for_simple_distribution_parallel())) {
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) { computation_func(i); });
            pool.waitForAll();
        } else {
            // Serial processing for small datasets
            for (std::size_t i = 0; i < count; ++i) {
                computation_func(i);
            }
        }
    }

    // executeBatchGpuAccelerated removed — GPU_ACCELERATED strategy removed from enum.
    // Callers should use executeBatchWorkStealing directly.

    /**
     * @brief Common cache validation and parameter extraction pattern
     *
     * @tparam Distribution The distribution type
     * @tparam CacheExtractorFunc Function to extract cached parameters
     * @tparam ExecutionFunc Function to execute with cached parameters
     *
     * @param dist Reference to the distribution instance
     * @param cache_extractor Function that extracts needed cached parameters
     * @param execution_func Function that executes with cached parameters
     */
    template <typename Distribution, typename CacheExtractorFunc, typename ExecutionFunc>
    static void withCachedParameters(const Distribution& dist, CacheExtractorFunc&& cache_extractor,
                                     ExecutionFunc&& execution_func) {
        // This pattern is repeated in every batch method:
        // 1. Ensure cache is valid
        // 2. Extract cached parameters
        // 3. Execute with cached parameters

        // Note: This assumes the distribution has these members and methods
        // In a real implementation, we'd use SFINAE or concepts to ensure compatibility
        auto& cache_mutex = dist.cache_mutex_;
        bool cache_valid = false;

        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex);
            cache_valid = dist.cache_valid_;
            if (!cache_valid) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(cache_mutex);
                if (!dist.cache_valid_) {
                    // Const cast needed for cache updates - this is safe as it's internal state
                    const_cast<Distribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Extract cached parameters while holding the lock
            auto cached_params = cache_extractor(dist);
            lock.unlock();

            // Execute with cached parameters (no lock needed)
            execution_func(cached_params);
        }
    }
};

/**
 * @brief Distribution traits to map distribution types to enums
 *
 * Specializations should be provided for each distribution type to define
 * their corresponding DistributionType and ComputationComplexity values.
 */
template <typename Distribution>
struct DistributionTraits {
    static constexpr DistributionType distType() = delete;
    static constexpr ComputationComplexity complexity() = delete;
};

// Specializations for known distributions
template <>
struct DistributionTraits<class DiscreteDistribution> {
    static constexpr DistributionType distType() { return DistributionType::DISCRETE; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::SIMPLE; }
};

template <>
struct DistributionTraits<class PoissonDistribution> {
    static constexpr DistributionType distType() { return DistributionType::POISSON; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::COMPLEX; }
};

template <>
struct DistributionTraits<class GaussianDistribution> {
    static constexpr DistributionType distType() { return DistributionType::GAUSSIAN; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::MODERATE; }
};

template <>
struct DistributionTraits<class ExponentialDistribution> {
    static constexpr DistributionType distType() { return DistributionType::EXPONENTIAL; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::MODERATE; }
};

template <>
struct DistributionTraits<class UniformDistribution> {
    static constexpr DistributionType distType() { return DistributionType::UNIFORM; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::SIMPLE; }
};

template <>
struct DistributionTraits<class GammaDistribution> {
    static constexpr DistributionType distType() { return DistributionType::GAMMA; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::COMPLEX; }
};

}  // namespace detail
}  // namespace stats

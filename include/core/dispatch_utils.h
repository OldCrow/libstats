#pragma once

#include "dispatch_thresholds.h"
#include "distribution_concepts.h"
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
 *     \u2193 if hint is AUTO: PerformanceDispatcher::selectStrategy (threshold lookup +
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
 * The GPU_ACCELERATED strategy slot has been removed. The GpuAcceleratedFunc
 * template parameter and corresponding lambda are no longer accepted by these
 * templates. See issue #23 for the rationale and prerequisites for any future
 * GPU backend.
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
     */
    template <stats::concepts::AnyDistribution Distribution, typename ScalarFunc, typename BatchFunc,
              typename ParallelFunc, typename WorkStealingFunc>
    static void autoDispatch(const Distribution& dist, std::span<const double> values,
                             std::span<double> results, const PerformanceHint& hint,
                             OperationType op_type,
                             ScalarFunc&& scalar_func, BatchFunc&& batch_func,
                             ParallelFunc&& parallel_func, WorkStealingFunc&& work_stealing_func) {
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
            strategy = dispatcher.selectStrategy(
                count, Distribution::kDistributionType, op_type, system);
        } else {
            strategy = mapHintToStrategy(hint.strategy, count);
        }

        // Execute using selected strategy
        executeStrategy(strategy, dist, values, results, count,
                        std::forward<ScalarFunc>(scalar_func), std::forward<BatchFunc>(batch_func),
                        std::forward<ParallelFunc>(parallel_func),
                        std::forward<WorkStealingFunc>(work_stealing_func));
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
     *
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param strategy Explicit strategy to use
     * @param scalar_func Function to call for scalar operations
     * @param batch_func Function to call for SIMD batch operations
     * @param parallel_func Function to call for parallel operations
     * @param work_stealing_func Function to call for work-stealing operations
     */
    template <typename Distribution, typename ScalarFunc, typename BatchFunc, typename ParallelFunc,
              typename WorkStealingFunc>
    static void executeWithStrategy(const Distribution& dist, std::span<const double> values,
                                    std::span<double> results, Strategy strategy,
                                    ScalarFunc&& scalar_func, BatchFunc&& batch_func,
                                    ParallelFunc&& parallel_func,
                                    WorkStealingFunc&& work_stealing_func) {
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
                        std::forward<WorkStealingFunc>(work_stealing_func));
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
                // MAXIMIZE_THROUGHPUT routes to WORK_STEALING, not PARALLEL.
                // Work-stealing is better for throughput on irregular workloads;
                // PARALLEL is adequate for uniform ones. A FORCE_PARALLEL hint
                // exists for callers that specifically need the thread-pool path.
                return Strategy::WORK_STEALING;
            default:
                return Strategy::SCALAR;
        }
    }

    /**
     * @brief Executes the selected strategy with appropriate function calls
     */
    template <typename Distribution, typename ScalarFunc, typename BatchFunc, typename ParallelFunc,
              typename WorkStealingFunc>
    static void executeStrategy(Strategy strategy, const Distribution& dist,
                                std::span<const double> values, std::span<double> results,
                                size_t count, ScalarFunc&& scalar_func, BatchFunc&& batch_func,
                                ParallelFunc&& parallel_func,
                                WorkStealingFunc&& work_stealing_func) {
        switch (strategy) {
            case Strategy::SCALAR:
                // Use simple loop for tiny batches (< 8 elements)
                for (size_t i = 0; i < count; ++i) {
                    results[i] = scalar_func(dist, values[i]);
                }
                break;

            case Strategy::VECTORIZED:
                // Batch path through VectorOps. For Gaussian this uses SIMD intrinsics;
                // for other distributions it currently uses scalar loops.
                batch_func(dist, values.data(), results.data(), count);
                break;

            case Strategy::PARALLEL:
                // Multi-threaded via ParallelUtils::parallelFor.
                parallel_func(dist, values, results);
                break;

            case Strategy::WORK_STEALING: {
                // Work-stealing pool for irregular or variable-cost workloads.
                // Use the shared GlobalWorkStealingPool singleton instead of
                // a thread_local WorkStealingPool. The thread_local approach spawned
                // N * hardware_concurrency threads for N concurrent callers, risking
                // thread explosion under load. The global singleton amortises thread
                // creation across all callers on all threads.
                work_stealing_func(dist, values, results, GlobalWorkStealingPool::getInstance());
                break;
            }
        }
    }

};

// DistributionTraits<> removed in v2.0.0.
// Use D::kDistributionType and D::kIsDiscrete directly.
// Use stats::concepts::AnyDistribution<D>, stats::concepts::ContinuousDistribution<D>,
// or stats::concepts::DiscreteDistribution<D> to constrain template parameters.

}  // namespace detail
}  // namespace stats

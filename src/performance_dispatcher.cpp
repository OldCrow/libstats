#include "libstats/core/performance_dispatcher.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/platform/simd_policy.h"

namespace stats {
namespace detail {  // Performance utilities

PerformanceDispatcher::PerformanceDispatcher()
    : simd_level_(arch::simd::SIMDPolicy::getBestLevel()) {}

// ── Profiling-derived dispatch ───────────────────────────────────────────────────

Strategy PerformanceDispatcher::selectStrategy(size_t batch_size, DistributionType dist_type,
                                               OperationType op_type,
                                               const SystemCapabilities& system) const {
    // 1. Below SIMD threshold → SCALAR
    const size_t simd_min = arch::simd::SIMDPolicy::getMinThreshold();
    if (batch_size < simd_min) {
        return Strategy::SCALAR;
    }

    // 2. Below parallel threshold → VECTORIZED
    const size_t parallel_threshold = getParallelThreshold(simd_level_, dist_type, op_type);
    if (batch_size < parallel_threshold) {
        return Strategy::VECTORIZED;
    }

    // 3. At or above parallel threshold → the multi-threaded strategy (PARALLEL)
    return selectMultiThreadedStrategy(dist_type, system);
}

Strategy PerformanceDispatcher::selectMultiThreadedStrategy(
    [[maybe_unused]] DistributionType dist_type,
    [[maybe_unused]] const SystemCapabilities& system) noexcept {
    // PARALLEL on every platform (decided 2026-11-25). Work stealing was selected on
    // hyper-threaded macOS on profiling that showed it up to 7:1 ahead there. Those captures ran
    // with its idle workers polling, which held the cores' clock up, and with twice the threads
    // (logical against physical cores); neither is a property of stealing. The pool now sleeps
    // when idle (#333). Strategy::WORK_STEALING and WorkStealingPool remain, but no hint and no
    // AUTO decision selects them.
    return Strategy::PARALLEL;
}

}  // namespace detail
}  // namespace stats

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

    // 3. At or above parallel threshold → PARALLEL or WORK_STEALING
    return selectMultiThreadedStrategy(dist_type, system);
}

Strategy PerformanceDispatcher::selectMultiThreadedStrategy(
    [[maybe_unused]] DistributionType dist_type, const SystemCapabilities& system) noexcept {
    // Four-architecture profiling shows the threading backend is the dominant
    // factor in P-vs-WS selection:
    //   macOS/GCD + HT:       WORK_STEALING wins (up to 7:1)
    //   macOS/GCD + no HT:    roughly even, slight PARALLEL preference
    //   Windows/Thread Pool:   PARALLEL wins (3.3:1)

#if defined(_WIN32)
    // Windows Thread Pool: PARALLEL dominates across distributions.
    return Strategy::PARALLEL;
#elif defined(__APPLE__)
    // macOS/GCD: prefer WORK_STEALING when hyperthreading is present.
    if (system.logical_cores() > system.physical_cores()) {
        return Strategy::WORK_STEALING;
    }
    return Strategy::PARALLEL;
#else
    // Linux/other: default to PARALLEL (conservative; no profiling data yet).
    (void)system;
    return Strategy::PARALLEL;
#endif
}

}  // namespace detail
}  // namespace stats

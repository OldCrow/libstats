#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/performance_dispatcher.h"

#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/math_constants.h"
#include "libstats/core/performance_constants.h"
#include "libstats/core/performance_history.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd_policy.h"

#include <iostream>
#include <mutex>

namespace stats {
namespace detail {  // Performance utilities

PerformanceDispatcher::PerformanceDispatcher()
    : PerformanceDispatcher(SystemCapabilities::current()) {}

PerformanceDispatcher::PerformanceDispatcher(const SystemCapabilities& system)
    : simd_level_(arch::simd::SIMDPolicy::getBestLevel()) {
    thresholds_ = Thresholds::createForSIMDLevel(simd_level_, system);
}

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

bool PerformanceDispatcher::shouldUseWorkStealing(
    size_t batch_size, [[maybe_unused]] DistributionType dist_type) const {
    return batch_size >= thresholds_.work_stealing_min;
}

void PerformanceDispatcher::updateThresholds(const Thresholds& new_thresholds) {
    thresholds_ = new_thresholds;
}

void PerformanceDispatcher::recordPerformance(Strategy strategy, DistributionType distribution_type,
                                              std::size_t batch_size,
                                              std::uint64_t execution_time_ns) noexcept {
    // Use the shared global instance
    getPerformanceHistory().recordPerformance(strategy, distribution_type, batch_size,
                                              execution_time_ns);
}

PerformanceHistory& PerformanceDispatcher::getPerformanceHistory() noexcept {
    static PerformanceHistory global_performance_history;
    return global_performance_history;
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::createForSIMDLevel(
    arch::simd::SIMDPolicy::Level level, [[maybe_unused]] const SystemCapabilities& system) {
    Thresholds thresholds;

    // Use SIMDPolicy's thresholds as foundation
    thresholds.simd_min = arch::simd::SIMDPolicy::getMinThreshold();

    // Set base parallel thresholds based on SIMD level capability
    // AVX-512's wider registers process more elements per cycle, so VECTORIZED remains
    // faster than PARALLEL up to higher batch sizes than narrower SIMD levels.
    switch (level) {
        case arch::simd::SIMDPolicy::Level::AVX512:
            thresholds.parallel_min = 5000;
            thresholds.work_stealing_min = 50000;
            break;
        case arch::simd::SIMDPolicy::Level::AVX2:
            thresholds.parallel_min = detail::MAX_BISECTION_ITERATIONS;
            thresholds.work_stealing_min = 10000;
            break;
        case arch::simd::SIMDPolicy::Level::AVX:
            thresholds.parallel_min = detail::MAX_DATA_POINTS_FOR_SW_TEST;
            thresholds.work_stealing_min = 50000;
            break;
        case arch::simd::SIMDPolicy::Level::SSE2:
            thresholds.parallel_min = 2000;
            thresholds.work_stealing_min = 20000;
            break;
        case arch::simd::SIMDPolicy::Level::NEON:
            thresholds.parallel_min = 1500;
            thresholds.work_stealing_min = 15000;
            break;
        case arch::simd::SIMDPolicy::Level::None:
        default:
            thresholds.simd_min = SIZE_MAX;  // Disable SIMD entirely
            thresholds.parallel_min = 500;
            thresholds.work_stealing_min = detail::MAX_DATA_POINTS_FOR_SW_TEST;
            break;
    }

    return thresholds;
}

// Legacy profile methods delegate to SIMDPolicy-based logic

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getSSE2Profile() {
    return createForSIMDLevel(arch::simd::SIMDPolicy::Level::SSE2, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getAVXProfile() {
    return createForSIMDLevel(arch::simd::SIMDPolicy::Level::AVX, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getAVX2Profile() {
    return createForSIMDLevel(arch::simd::SIMDPolicy::Level::AVX2, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getAVX512Profile() {
    return createForSIMDLevel(arch::simd::SIMDPolicy::Level::AVX512, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getNEONProfile() {
    return createForSIMDLevel(arch::simd::SIMDPolicy::Level::NEON, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getScalarProfile() {
    return createForSIMDLevel(arch::simd::SIMDPolicy::Level::None, SystemCapabilities::current());
}

}  // namespace detail
}  // namespace stats

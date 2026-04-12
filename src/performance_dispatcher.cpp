#include "libstats/core/performance_dispatcher.h"

#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/math_constants.h"
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

PerformanceDispatcher::SIMDArchitecture PerformanceDispatcher::detectSIMDArchitecture(
    [[maybe_unused]] const SystemCapabilities& system) noexcept {
    // Delegate to SIMDPolicy instead of duplicating detection logic
    auto level = arch::simd::SIMDPolicy::getBestLevel();

    switch (level) {
        case arch::simd::SIMDPolicy::Level::AVX512:
            return SIMDArchitecture::AVX512;
        case arch::simd::SIMDPolicy::Level::AVX2:
            return SIMDArchitecture::AVX2;
        case arch::simd::SIMDPolicy::Level::AVX:
            return SIMDArchitecture::AVX;
        case arch::simd::SIMDPolicy::Level::NEON:
            return SIMDArchitecture::NEON;
        case arch::simd::SIMDPolicy::Level::SSE2:
            return SIMDArchitecture::SSE2;
        case arch::simd::SIMDPolicy::Level::None:
        default:
            return SIMDArchitecture::NONE;
    }
}

// ── New profiling-derived dispatch ──────────────────────────────────────────

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

size_t PerformanceDispatcher::getDistributionSpecificParallelThreshold(
    DistributionType dist_type) const {
    switch (dist_type) {
        case DistributionType::UNIFORM:
            return thresholds_.uniform_parallel_min;
        case DistributionType::GAUSSIAN:
            return thresholds_.gaussian_parallel_min;
        case DistributionType::EXPONENTIAL:
            return thresholds_.exponential_parallel_min;
        case DistributionType::DISCRETE:
            return thresholds_.discrete_parallel_min;
        case DistributionType::POISSON:
            return thresholds_.poisson_parallel_min;
        case DistributionType::GAMMA:
            return thresholds_.gamma_parallel_min;
        case DistributionType::STUDENT_T:
            return thresholds_.student_t_parallel_min;
        case DistributionType::BETA:
            return thresholds_.beta_parallel_min;
        case DistributionType::CHI_SQUARED:
            return thresholds_.chi_squared_parallel_min;
        default:
            return thresholds_.parallel_min;
    }
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
    arch::simd::SIMDPolicy::Level level, const SystemCapabilities& system) {
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

    // Distribution-specific thresholds are now handled by the constexpr lookup
    // table in dispatch_thresholds.h. The Thresholds struct members below are
    // populated with reasonable defaults for backward compatibility only.
    thresholds.uniform_parallel_min = thresholds.parallel_min * 2;
    thresholds.gaussian_parallel_min = thresholds.parallel_min;
    thresholds.exponential_parallel_min = thresholds.parallel_min;
    thresholds.discrete_parallel_min = thresholds.parallel_min * 2;
    thresholds.poisson_parallel_min = thresholds.parallel_min;
    thresholds.gamma_parallel_min = thresholds.parallel_min;
    thresholds.student_t_parallel_min = thresholds.parallel_min;
    thresholds.beta_parallel_min = SIZE_MAX;  // Beta: never parallel
    thresholds.chi_squared_parallel_min = thresholds.parallel_min;

    return thresholds;
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::createForArchitecture(
    SIMDArchitecture arch, const SystemCapabilities& system) {
    // Convert to SIMDPolicy level and delegate
    arch::simd::SIMDPolicy::Level level;
    switch (arch) {
        case SIMDArchitecture::AVX512:
            level = arch::simd::SIMDPolicy::Level::AVX512;
            break;
        case SIMDArchitecture::AVX2:
            level = arch::simd::SIMDPolicy::Level::AVX2;
            break;
        case SIMDArchitecture::AVX:
            level = arch::simd::SIMDPolicy::Level::AVX;
            break;
        case SIMDArchitecture::SSE2:
            level = arch::simd::SIMDPolicy::Level::SSE2;
            break;
        case SIMDArchitecture::NEON:
            level = arch::simd::SIMDPolicy::Level::NEON;
            break;
        case SIMDArchitecture::NONE:
        default:
            level = arch::simd::SIMDPolicy::Level::None;
            break;
    }

    return createForSIMDLevel(level, system);
}

// Legacy profile methods kept for backward compatibility but now delegate to SIMDPolicy-based logic

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

void PerformanceDispatcher::Thresholds::refineWithCapabilities(const SystemCapabilities& system) {
    // Adjust thresholds based on measured system performance

    auto simd_efficiency = system.simd_efficiency();
    auto threading_overhead = system.threading_overhead_ns();
    // memory_bandwidth_gb_s() was used for GPU acceleration threshold adjustment.
    // GPU_ACCELERATED strategy was removed in Phase 2, so this is no longer needed.
    auto logical_cores = system.logical_cores();

    // Refine SIMD thresholds based on efficiency
    if (simd_efficiency < detail::LARGE_EFFECT) {
        // SIMD is inefficient, raise thresholds
        simd_min = static_cast<size_t>(static_cast<double>(simd_min) * (1.5 / simd_efficiency));

        // Also raise distribution-specific parallel thresholds
        uniform_parallel_min = static_cast<size_t>(static_cast<double>(uniform_parallel_min) * 1.5);
        gaussian_parallel_min =
            static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * 1.5);
        exponential_parallel_min =
            static_cast<size_t>(static_cast<double>(exponential_parallel_min) * 1.5);
        discrete_parallel_min =
            static_cast<size_t>(static_cast<double>(discrete_parallel_min) * 1.5);
        poisson_parallel_min = static_cast<size_t>(static_cast<double>(poisson_parallel_min) * 1.5);
        gamma_parallel_min = static_cast<size_t>(static_cast<double>(gamma_parallel_min) * 1.5);
    } else if (simd_efficiency > 1.5) {
        // SIMD is very efficient, lower thresholds
        simd_min = static_cast<size_t>(static_cast<double>(simd_min) * detail::STRONG_CORRELATION);

        // Lower distribution-specific thresholds
        uniform_parallel_min =
            static_cast<size_t>(static_cast<double>(uniform_parallel_min) * detail::LARGE_EFFECT);
        gaussian_parallel_min =
            static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * detail::LARGE_EFFECT);
        exponential_parallel_min = static_cast<size_t>(
            static_cast<double>(exponential_parallel_min) * detail::LARGE_EFFECT);
        discrete_parallel_min =
            static_cast<size_t>(static_cast<double>(discrete_parallel_min) * detail::LARGE_EFFECT);
        poisson_parallel_min =
            static_cast<size_t>(static_cast<double>(poisson_parallel_min) * detail::LARGE_EFFECT);
        gamma_parallel_min =
            static_cast<size_t>(static_cast<double>(gamma_parallel_min) * detail::LARGE_EFFECT);
    }

    // Refine parallel thresholds based on threading overhead
    if (threading_overhead > 100000.0) {  // > 100μs overhead
        // High threading overhead, raise parallel thresholds
        double multiplier = std::min(detail::THREE, threading_overhead / 50000.0);
        parallel_min = static_cast<size_t>(static_cast<double>(parallel_min) * multiplier);
        work_stealing_min =
            static_cast<size_t>(static_cast<double>(work_stealing_min) * multiplier);

        // Raise distribution-specific thresholds
        uniform_parallel_min =
            static_cast<size_t>(static_cast<double>(uniform_parallel_min) * multiplier);
        gaussian_parallel_min =
            static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * multiplier);
        exponential_parallel_min =
            static_cast<size_t>(static_cast<double>(exponential_parallel_min) * multiplier);
        discrete_parallel_min =
            static_cast<size_t>(static_cast<double>(discrete_parallel_min) * multiplier);
        poisson_parallel_min =
            static_cast<size_t>(static_cast<double>(poisson_parallel_min) * multiplier);
        gamma_parallel_min =
            static_cast<size_t>(static_cast<double>(gamma_parallel_min) * multiplier);
    } else if (threading_overhead < 10000.0) {  // < 10μs overhead
        // Low threading overhead, lower parallel thresholds
        double multiplier = std::max(detail::HALF, threading_overhead / 20000.0);
        parallel_min = static_cast<size_t>(static_cast<double>(parallel_min) * multiplier);
        work_stealing_min =
            static_cast<size_t>(static_cast<double>(work_stealing_min) * multiplier);

        // Lower distribution-specific thresholds
        uniform_parallel_min =
            static_cast<size_t>(static_cast<double>(uniform_parallel_min) * multiplier);
        gaussian_parallel_min =
            static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * multiplier);
        exponential_parallel_min =
            static_cast<size_t>(static_cast<double>(exponential_parallel_min) * multiplier);
        discrete_parallel_min =
            static_cast<size_t>(static_cast<double>(discrete_parallel_min) * multiplier);
        poisson_parallel_min =
            static_cast<size_t>(static_cast<double>(poisson_parallel_min) * multiplier);
        gamma_parallel_min =
            static_cast<size_t>(static_cast<double>(gamma_parallel_min) * multiplier);
    }

    // Adjust work-stealing based on core count
    if (logical_cores <= 2) {
        // Few cores, raise work-stealing threshold significantly
        work_stealing_min =
            static_cast<size_t>(static_cast<double>(work_stealing_min) * detail::TWO);
    } else if (logical_cores >= 16) {
        // Many cores, lower work-stealing threshold
        work_stealing_min =
            static_cast<size_t>(static_cast<double>(work_stealing_min) * detail::LARGE_EFFECT);
    }

    // Ensure minimums
    simd_min = std::max(simd_min, static_cast<size_t>(4));
    parallel_min = std::max(parallel_min, static_cast<size_t>(detail::MAX_NEWTON_ITERATIONS));
    work_stealing_min =
        std::max(work_stealing_min, static_cast<size_t>(detail::MAX_BISECTION_ITERATIONS));

    // Ensure distribution-specific thresholds don't drop below parallel_min.
    // Simple distributions (Uniform, Discrete) must stay at or above the base;
    // complex ones are allowed lower thresholds but still have a floor.
    uniform_parallel_min = std::max(uniform_parallel_min, parallel_min * 2);
    discrete_parallel_min = std::max(discrete_parallel_min, parallel_min * 2);
    gaussian_parallel_min = std::max(gaussian_parallel_min, parallel_min / 2);
    exponential_parallel_min = std::max(exponential_parallel_min, parallel_min / 2);
    student_t_parallel_min = std::max(student_t_parallel_min, parallel_min / 2);
    beta_parallel_min = std::max(beta_parallel_min, parallel_min / 2);
    poisson_parallel_min = std::max(poisson_parallel_min, parallel_min / 4);
    gamma_parallel_min = std::max(gamma_parallel_min, parallel_min / 4);
    chi_squared_parallel_min = std::max(chi_squared_parallel_min, parallel_min / 4);
}

}  // namespace detail
}  // namespace stats

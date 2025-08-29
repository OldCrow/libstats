#include "../include/core/performance_dispatcher.h"

#include "../include/core/distribution_characteristics.h"
#include "../include/core/performance_history.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd_policy.h"

#include <iostream>
#include <mutex>

namespace stats {
namespace detail {  // Performance utilities

PerformanceDispatcher::PerformanceDispatcher()
    : PerformanceDispatcher(SystemCapabilities::current()) {}

PerformanceDispatcher::PerformanceDispatcher(const SystemCapabilities& system) {
    // Use SIMDPolicy to get the best SIMD level and initialize thresholds accordingly
    auto simd_level = arch::simd::SIMDPolicy::getBestLevel();
    thresholds_ = Thresholds::createForSIMDLevel(simd_level, system);
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

// GPU fallback logging function
static void logGPUFallback() noexcept {
    // Only log once per application run to avoid spam
    static std::once_flag logged;
    std::call_once(logged, []() {
        std::cerr << "INFO: GPU acceleration requested but not yet implemented. "
                  << "Using optimal CPU strategy." << std::endl;
    });
}

Strategy PerformanceDispatcher::selectOptimalStrategy(
    size_t batch_size, DistributionType dist_type,
    [[maybe_unused]] ComputationComplexity complexity, const SystemCapabilities& system) const {
    // Use the shared global instance
    auto& performance_history = getPerformanceHistory();

    // Retrieve the best strategy recommendation based on performance history
    auto recommendation = performance_history.getBestStrategy(dist_type, batch_size);

    // Check if GPU acceleration was requested (this would come from explicit strategy requests)
    if (recommendation.has_sufficient_data &&
        recommendation.recommended_strategy == Strategy::GPU_ACCELERATED) {
        // GPU acceleration not yet implemented - select best CPU strategy
        logGPUFallback();

        // Use same selection logic as auto-dispatch
        auto parallel_threshold = getDistributionSpecificParallelThreshold(dist_type);
        if (batch_size >= parallel_threshold) {
            // For large batches, prefer work-stealing for optimal CPU performance
            if (batch_size >= thresholds_.work_stealing_min && system.logical_cores() > 2) {
                return Strategy::WORK_STEALING;
            } else {
                return Strategy::PARALLEL_SIMD;
            }
        } else if (batch_size >= thresholds_.simd_min) {
            return Strategy::SIMD_BATCH;
        } else {
            return Strategy::SCALAR;
        }
    }

    // Use historical data if we have high confidence
    if (recommendation.has_sufficient_data && recommendation.confidence_score > 0.8) {
        return recommendation.recommended_strategy;
    }

    // Fallback to adaptive logic based on system capabilities
    return selectStrategyBasedOnCapabilities(batch_size, dist_type, system);
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
        default:
            return thresholds_.parallel_min;
    }
}

bool PerformanceDispatcher::shouldUseWorkStealing(
    size_t batch_size, [[maybe_unused]] DistributionType dist_type) const {
    return batch_size >= thresholds_.work_stealing_min;
}

bool PerformanceDispatcher::shouldUseGpuAccelerated(size_t batch_size,
                                                    const SystemCapabilities& system) const {
    return batch_size >= thresholds_.gpu_accelerated_min && system.memory_bandwidth_gb_s() >= 50.0;
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

Strategy PerformanceDispatcher::selectStrategyBasedOnCapabilities(
    size_t batch_size, DistributionType dist_type, const SystemCapabilities& system) const {
    // Get empirical characteristics for this distribution
    using namespace detail;
    const auto& dist_chars = getCharacteristics(dist_type);

    // Extract system capabilities
    auto simd_efficiency = system.simd_efficiency();
    auto threading_overhead = system.threading_overhead_ns();
    auto memory_bandwidth = system.memory_bandwidth_gb_s();

    // Adjust system SIMD efficiency based on distribution characteristics
    double effective_simd_efficiency = simd_efficiency * dist_chars.vectorization_efficiency;

    // Get distribution-specific threshold
    auto parallel_threshold = getDistributionSpecificParallelThreshold(dist_type);
    auto simd_threshold = std::max(dist_chars.min_simd_threshold, thresholds_.simd_min);

    // Decision logic based on empirical characteristics and system capabilities

    // For very small batches, use scalar regardless of distribution
    if (batch_size <= simd_threshold) {
        return Strategy::SCALAR;
    }

    // Use empirical characteristics to guide strategy selection

    // If effective SIMD efficiency is poor for this distribution, avoid SIMD strategies
    if (effective_simd_efficiency < 0.3) {
        // SIMD performs poorly for this distribution, prefer parallel or scalar
        if (batch_size >= parallel_threshold &&
            threading_overhead < (1000000.0 * dist_chars.base_complexity)) {
            // Complex distributions justify higher threading overhead
            return (dist_chars.parallelization_efficiency > 0.6) ? Strategy::PARALLEL_SIMD
                                                                 : Strategy::SCALAR;
        }
        return Strategy::SCALAR;
    }

    // For medium batches, consider SIMD based on distribution characteristics
    if (batch_size < parallel_threshold) {
        // Use SIMD if the distribution vectorizes well and system supports it
        if (effective_simd_efficiency > 0.5 && dist_chars.vectorization_efficiency > 0.6) {
            return Strategy::SIMD_BATCH;
        }
        // For distributions with poor vectorization, stick with scalar until parallel threshold
        return Strategy::SCALAR;
    }

    // For large batches, select parallel strategy based on distribution complexity

    // High threading overhead limits parallel strategies for simple distributions
    double acceptable_overhead = 200000.0 * dist_chars.base_complexity;  // Scale by complexity
    if (threading_overhead > acceptable_overhead && dist_chars.base_complexity < 2.0) {
        // Simple distributions with high threading overhead: prefer SIMD
        if (effective_simd_efficiency > 0.4) {
            return Strategy::SIMD_BATCH;
        }
        return Strategy::SCALAR;
    }

    // Memory-intensive operations would benefit from GPU acceleration
    // For now, return work-stealing as GPU is not yet implemented
    bool use_gpu_equivalent = (batch_size >= thresholds_.gpu_accelerated_min) &&
                              (memory_bandwidth >= 50.0) &&
                              (dist_chars.memory_access_pattern > 0.8);

    if (use_gpu_equivalent) {
        // GPU acceleration not implemented - use best CPU alternative
        return Strategy::WORK_STEALING;
    }

    // Work stealing benefits distributions with variable execution time
    bool use_work_stealing =
        (batch_size >= thresholds_.work_stealing_min) && (system.logical_cores() > 2) &&
        (threading_overhead < acceptable_overhead) &&
        (dist_chars.branch_prediction_cost > 1.2);  // High branching variability

    if (use_work_stealing) {
        return Strategy::WORK_STEALING;
    }

    // Default to parallel SIMD for large batches with good parallel efficiency
    if (dist_chars.parallelization_efficiency > 0.6) {
        return Strategy::PARALLEL_SIMD;
    }

    // Fall back to SIMD for distributions with poor parallelization but good vectorization
    if (effective_simd_efficiency > 0.4) {
        return Strategy::SIMD_BATCH;
    }

    // Last resort: scalar
    return Strategy::SCALAR;
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::createForSIMDLevel(
    arch::simd::SIMDPolicy::Level level, const SystemCapabilities& system) {
    Thresholds thresholds;

    // Use SIMDPolicy's thresholds as foundation
    thresholds.simd_min = arch::simd::SIMDPolicy::getMinThreshold();

    // Set base parallel thresholds based on SIMD level capability
    switch (level) {
        case arch::simd::SIMDPolicy::Level::AVX512:
            thresholds.parallel_min = 500;  // Powerful SIMD reduces parallel threshold
            thresholds.work_stealing_min = 8000;
            thresholds.gpu_accelerated_min = 32000;
            break;
        case arch::simd::SIMDPolicy::Level::AVX2:
            thresholds.parallel_min = 1000;  // Good SIMD efficiency
            thresholds.work_stealing_min = 10000;
            thresholds.gpu_accelerated_min = 50000;
            break;
        case arch::simd::SIMDPolicy::Level::AVX:
            thresholds.parallel_min = 5000;  // AVX often has limited efficiency
            thresholds.work_stealing_min = 50000;
            thresholds.gpu_accelerated_min = 200000;
            break;
        case arch::simd::SIMDPolicy::Level::SSE2:
            thresholds.parallel_min = 2000;  // Older architecture, conservative
            thresholds.work_stealing_min = 20000;
            thresholds.gpu_accelerated_min = 100000;
            break;
        case arch::simd::SIMDPolicy::Level::NEON:
            thresholds.parallel_min = 1500;  // ARM characteristics
            thresholds.work_stealing_min = 15000;
            thresholds.gpu_accelerated_min = 75000;
            break;
        case arch::simd::SIMDPolicy::Level::None:
        default:
            thresholds.simd_min = SIZE_MAX;  // Disable SIMD entirely
            thresholds.parallel_min = 500;   // Lower threshold since SIMD unavailable
            thresholds.work_stealing_min = 5000;
            thresholds.gpu_accelerated_min = 25000;
            break;
    }

    // Set distribution-specific thresholds based on empirical characteristics
    using namespace detail;

    // Calculate SIMD and parallel thresholds using empirical data
    for (size_t i = 0; i < 6; ++i) {
        const auto& chars = DISTRIBUTION_CHARACTERISTICS[i];

        // Scale base thresholds by complexity - more complex operations need lower thresholds
        // to benefit from parallelization due to higher computation-to-overhead ratios
        double complexity_scaling = 1.0 / std::max(1.0, chars.base_complexity / 2.0);

        // Use empirical minimum thresholds, scaled by system characteristics
        size_t empirical_parallel_threshold = static_cast<size_t>(
            static_cast<double>(chars.min_parallel_threshold) * complexity_scaling);

        // Assign to distribution-specific thresholds
        switch (i) {
            case 0:  // UNIFORM
                thresholds.uniform_parallel_min =
                    std::max(empirical_parallel_threshold, thresholds.parallel_min);
                break;
            case 1:  // GAUSSIAN
                thresholds.gaussian_parallel_min =
                    std::max(empirical_parallel_threshold, thresholds.parallel_min / 2);
                break;
            case 2:  // EXPONENTIAL
                thresholds.exponential_parallel_min =
                    std::max(empirical_parallel_threshold, thresholds.parallel_min / 2);
                break;
            case 3:  // DISCRETE
                thresholds.discrete_parallel_min =
                    std::max(empirical_parallel_threshold, thresholds.parallel_min);
                break;
            case 4:  // POISSON
                thresholds.poisson_parallel_min =
                    std::max(empirical_parallel_threshold, thresholds.parallel_min / 4);
                break;
            case 5:  // GAMMA
                thresholds.gamma_parallel_min =
                    std::max(empirical_parallel_threshold, thresholds.parallel_min / 4);
                break;
        }
    }

    // Refine with measured system capabilities
    thresholds.refineWithCapabilities(system);

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
    auto memory_bandwidth = system.memory_bandwidth_gb_s();
    auto logical_cores = system.logical_cores();

    // Refine SIMD thresholds based on efficiency
    if (simd_efficiency < 0.8) {
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
        simd_min = static_cast<size_t>(static_cast<double>(simd_min) * 0.7);

        // Lower distribution-specific thresholds
        uniform_parallel_min = static_cast<size_t>(static_cast<double>(uniform_parallel_min) * 0.8);
        gaussian_parallel_min =
            static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * 0.8);
        exponential_parallel_min =
            static_cast<size_t>(static_cast<double>(exponential_parallel_min) * 0.8);
        discrete_parallel_min =
            static_cast<size_t>(static_cast<double>(discrete_parallel_min) * 0.8);
        poisson_parallel_min = static_cast<size_t>(static_cast<double>(poisson_parallel_min) * 0.8);
        gamma_parallel_min = static_cast<size_t>(static_cast<double>(gamma_parallel_min) * 0.8);
    }

    // Refine parallel thresholds based on threading overhead
    if (threading_overhead > 100000.0) {  // > 100μs overhead
        // High threading overhead, raise parallel thresholds
        double multiplier = std::min(3.0, threading_overhead / 50000.0);
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
        double multiplier = std::max(0.5, threading_overhead / 20000.0);
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

    // Refine GPU acceleration thresholds based on memory bandwidth
    if (memory_bandwidth < 20.0) {
        // Low memory bandwidth, raise GPU acceleration threshold
        gpu_accelerated_min = static_cast<size_t>(static_cast<double>(gpu_accelerated_min) * 1.5);
    } else if (memory_bandwidth > 100.0) {
        // High memory bandwidth, lower GPU acceleration threshold
        gpu_accelerated_min = static_cast<size_t>(static_cast<double>(gpu_accelerated_min) * 0.7);
    }

    // Adjust work-stealing based on core count
    if (logical_cores <= 2) {
        // Few cores, raise work-stealing threshold significantly
        work_stealing_min = static_cast<size_t>(static_cast<double>(work_stealing_min) * 2.0);
    } else if (logical_cores >= 16) {
        // Many cores, lower work-stealing threshold
        work_stealing_min = static_cast<size_t>(static_cast<double>(work_stealing_min) * 0.8);
    }

    // Ensure minimums
    simd_min = std::max(simd_min, static_cast<size_t>(4));
    parallel_min = std::max(parallel_min, static_cast<size_t>(100));
    work_stealing_min = std::max(work_stealing_min, static_cast<size_t>(1000));
    gpu_accelerated_min = std::max(gpu_accelerated_min, static_cast<size_t>(10000));
}

}  // namespace detail
}  // namespace stats

#include "../include/core/performance_dispatcher.h"
#include "../include/platform/simd_policy.h"
#include "../include/platform/cpu_detection.h"
#include "../include/core/performance_history.h"

namespace libstats {
namespace performance {

PerformanceDispatcher::PerformanceDispatcher() 
    : PerformanceDispatcher(SystemCapabilities::current()) {
}

PerformanceDispatcher::PerformanceDispatcher(const SystemCapabilities& system) {
    // Use SIMDPolicy to get the best SIMD level and initialize thresholds accordingly
    auto simd_level = simd::SIMDPolicy::getBestLevel();
    thresholds_ = Thresholds::createForSIMDLevel(simd_level, system);
}

PerformanceDispatcher::SIMDArchitecture PerformanceDispatcher::detectSIMDArchitecture(
    [[maybe_unused]] const SystemCapabilities& system) noexcept {
    
    // Delegate to SIMDPolicy instead of duplicating detection logic
    auto level = simd::SIMDPolicy::getBestLevel();
    
    switch (level) {
        case simd::SIMDPolicy::Level::AVX512:
            return SIMDArchitecture::AVX512;
        case simd::SIMDPolicy::Level::AVX2:
            return SIMDArchitecture::AVX2;
        case simd::SIMDPolicy::Level::AVX:
            return SIMDArchitecture::AVX;
        case simd::SIMDPolicy::Level::NEON:
            return SIMDArchitecture::NEON;
        case simd::SIMDPolicy::Level::SSE2:
            return SIMDArchitecture::SSE2;
        case simd::SIMDPolicy::Level::None:
        default:
            return SIMDArchitecture::NONE;
    }
}

Strategy PerformanceDispatcher::selectOptimalStrategy(
    size_t batch_size,
    DistributionType dist_type,
    [[maybe_unused]] ComputationComplexity complexity,
    const SystemCapabilities& system
) const {
    // Use the shared global instance
    auto& performance_history = getPerformanceHistory();
     
    // Retrieve the best strategy recommendation based on performance history
    auto recommendation = performance_history.getBestStrategy(dist_type, batch_size);
    
    // Use historical data if we have high confidence
    if (recommendation.has_sufficient_data && recommendation.confidence_score > 0.8) {
        return recommendation.recommended_strategy;
    }
    
    // Fallback to adaptive logic based on system capabilities
    return selectStrategyBasedOnCapabilities(batch_size, dist_type, system);
}

size_t PerformanceDispatcher::getDistributionSpecificParallelThreshold(DistributionType dist_type) const {
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

bool PerformanceDispatcher::shouldUseWorkStealing(size_t batch_size, [[maybe_unused]] DistributionType dist_type) const {
    return batch_size >= thresholds_.work_stealing_min;
}

bool PerformanceDispatcher::shouldUseCacheAware(size_t batch_size, const SystemCapabilities& system) const {
    return batch_size >= thresholds_.cache_aware_min && 
           system.memory_bandwidth_gb_s() >= 50.0;
}

void PerformanceDispatcher::updateThresholds(const Thresholds& new_thresholds) {
    thresholds_ = new_thresholds;
}

void PerformanceDispatcher::recordPerformance(
    Strategy strategy,
    DistributionType distribution_type,
    std::size_t batch_size,
    std::uint64_t execution_time_ns
) noexcept {
    // Use the shared global instance
    getPerformanceHistory().recordPerformance(strategy, distribution_type, batch_size, execution_time_ns);
}

PerformanceHistory& PerformanceDispatcher::getPerformanceHistory() noexcept {
    static PerformanceHistory global_performance_history;
    return global_performance_history;
}

Strategy PerformanceDispatcher::selectStrategyBasedOnCapabilities(
    size_t batch_size,
    DistributionType dist_type,
    const SystemCapabilities& system
) const {
    // Extract system capabilities
    auto simd_efficiency = system.simd_efficiency();
    auto threading_overhead = system.threading_overhead_ns();
    auto memory_bandwidth = system.memory_bandwidth_gb_s();
    
    // Get distribution-specific threshold
    auto parallel_threshold = getDistributionSpecificParallelThreshold(dist_type);
    
    // Decision logic based on actual system performance characteristics
    
    // If SIMD is inefficient on this system, avoid SIMD strategies
    if (simd_efficiency < 0.5) {
        // SIMD performs poorly, use scalar or basic parallel
        if (batch_size >= parallel_threshold && threading_overhead < 1000000.0) {
            return Strategy::PARALLEL_SIMD; // Use parallel without heavy SIMD reliance
        }
        return Strategy::SCALAR;
    }
    
    // For very small batches, always use scalar
    if (batch_size <= thresholds_.simd_min) {
        return Strategy::SCALAR;
    }
    
    // For medium batches, consider SIMD if it's efficient
    if (batch_size < parallel_threshold) {
        if (simd_efficiency > 0.7) {
            return Strategy::SIMD_BATCH;
        }
        return Strategy::SCALAR;
    }
    
    // For large batches, consider parallel strategies
    // But only if threading overhead is reasonable
    if (threading_overhead > 1000000.0) {
        // Threading overhead is too high, stick with SIMD
        return Strategy::SIMD_BATCH;
    }
    
    // Choose between different parallel strategies based on size and capabilities
    if (batch_size >= thresholds_.cache_aware_min && memory_bandwidth >= 50.0) {
        return Strategy::CACHE_AWARE;
    }
    
    if (batch_size >= thresholds_.work_stealing_min) {
        // Only use work stealing if we have multiple cores and reasonable overhead
        if (system.logical_cores() > 2 && threading_overhead < 500000.0) {
            return Strategy::WORK_STEALING;
        }
    }
    
    // Default to parallel SIMD for large batches
    return Strategy::PARALLEL_SIMD;
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::createForSIMDLevel(
    simd::SIMDPolicy::Level level, const SystemCapabilities& system) {
    
    Thresholds thresholds;
    
    // Use SIMDPolicy's thresholds as foundation
    thresholds.simd_min = simd::SIMDPolicy::getMinThreshold();
    
    // Set base parallel thresholds based on SIMD level capability
    switch (level) {
        case simd::SIMDPolicy::Level::AVX512:
            thresholds.parallel_min = 500;         // Powerful SIMD reduces parallel threshold
            thresholds.work_stealing_min = 8000;
            thresholds.cache_aware_min = 32000;
            break;
        case simd::SIMDPolicy::Level::AVX2:
            thresholds.parallel_min = 1000;        // Good SIMD efficiency
            thresholds.work_stealing_min = 10000;
            thresholds.cache_aware_min = 50000;
            break;
        case simd::SIMDPolicy::Level::AVX:
            thresholds.parallel_min = 5000;        // AVX often has limited efficiency
            thresholds.work_stealing_min = 50000;
            thresholds.cache_aware_min = 200000;
            break;
        case simd::SIMDPolicy::Level::SSE2:
            thresholds.parallel_min = 2000;        // Older architecture, conservative
            thresholds.work_stealing_min = 20000;
            thresholds.cache_aware_min = 100000;
            break;
        case simd::SIMDPolicy::Level::NEON:
            thresholds.parallel_min = 1500;        // ARM characteristics
            thresholds.work_stealing_min = 15000;
            thresholds.cache_aware_min = 75000;
            break;
        case simd::SIMDPolicy::Level::None:
        default:
            thresholds.simd_min = SIZE_MAX;         // Disable SIMD entirely
            thresholds.parallel_min = 500;         // Lower threshold since SIMD unavailable
            thresholds.work_stealing_min = 5000;
            thresholds.cache_aware_min = 25000;
            break;
    }
    
    // Set distribution-specific thresholds based on computational complexity
    // These are relative to the base parallel threshold
    auto base = thresholds.parallel_min;
    thresholds.uniform_parallel_min = base * 8;         // Simple operations need higher threshold
    thresholds.gaussian_parallel_min = base / 4;        // Complex operations benefit earlier
    thresholds.exponential_parallel_min = base / 2;     // Moderate complexity
    thresholds.discrete_parallel_min = base * 2;        // Integer operations
    thresholds.poisson_parallel_min = base / 2;         // Complex discrete distribution
    thresholds.gamma_parallel_min = base / 4;           // Most complex distribution
    
    // Refine with measured system capabilities
    thresholds.refineWithCapabilities(system);
    
    return thresholds;
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::createForArchitecture(
    SIMDArchitecture arch, const SystemCapabilities& system) {
    
    // Convert to SIMDPolicy level and delegate
    simd::SIMDPolicy::Level level;
    switch (arch) {
        case SIMDArchitecture::AVX512:
            level = simd::SIMDPolicy::Level::AVX512;
            break;
        case SIMDArchitecture::AVX2:
            level = simd::SIMDPolicy::Level::AVX2;
            break;
        case SIMDArchitecture::AVX:
            level = simd::SIMDPolicy::Level::AVX;
            break;
        case SIMDArchitecture::SSE2:
            level = simd::SIMDPolicy::Level::SSE2;
            break;
        case SIMDArchitecture::NEON:
            level = simd::SIMDPolicy::Level::NEON;
            break;
        case SIMDArchitecture::NONE:
        default:
            level = simd::SIMDPolicy::Level::None;
            break;
    }
    
    return createForSIMDLevel(level, system);
}

// Legacy profile methods kept for backward compatibility but now delegate to SIMDPolicy-based logic

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getSSE2Profile() {
    return createForSIMDLevel(simd::SIMDPolicy::Level::SSE2, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getAVXProfile() {
    return createForSIMDLevel(simd::SIMDPolicy::Level::AVX, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getAVX2Profile() {
    return createForSIMDLevel(simd::SIMDPolicy::Level::AVX2, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getAVX512Profile() {
    return createForSIMDLevel(simd::SIMDPolicy::Level::AVX512, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getNEONProfile() {
    return createForSIMDLevel(simd::SIMDPolicy::Level::NEON, SystemCapabilities::current());
}

PerformanceDispatcher::Thresholds PerformanceDispatcher::Thresholds::getScalarProfile() {
    return createForSIMDLevel(simd::SIMDPolicy::Level::None, SystemCapabilities::current());
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
        gaussian_parallel_min = static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * 1.5);
        exponential_parallel_min = static_cast<size_t>(static_cast<double>(exponential_parallel_min) * 1.5);
        discrete_parallel_min = static_cast<size_t>(static_cast<double>(discrete_parallel_min) * 1.5);
        poisson_parallel_min = static_cast<size_t>(static_cast<double>(poisson_parallel_min) * 1.5);
        gamma_parallel_min = static_cast<size_t>(static_cast<double>(gamma_parallel_min) * 1.5);
    } else if (simd_efficiency > 1.5) {
        // SIMD is very efficient, lower thresholds
        simd_min = static_cast<size_t>(static_cast<double>(simd_min) * 0.7);
        
        // Lower distribution-specific thresholds
        uniform_parallel_min = static_cast<size_t>(static_cast<double>(uniform_parallel_min) * 0.8);
        gaussian_parallel_min = static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * 0.8);
        exponential_parallel_min = static_cast<size_t>(static_cast<double>(exponential_parallel_min) * 0.8);
        discrete_parallel_min = static_cast<size_t>(static_cast<double>(discrete_parallel_min) * 0.8);
        poisson_parallel_min = static_cast<size_t>(static_cast<double>(poisson_parallel_min) * 0.8);
        gamma_parallel_min = static_cast<size_t>(static_cast<double>(gamma_parallel_min) * 0.8);
    }
    
    // Refine parallel thresholds based on threading overhead
    if (threading_overhead > 100000.0) {  // > 100μs overhead
        // High threading overhead, raise parallel thresholds
        double multiplier = std::min(3.0, threading_overhead / 50000.0);
        parallel_min = static_cast<size_t>(static_cast<double>(parallel_min) * multiplier);
        work_stealing_min = static_cast<size_t>(static_cast<double>(work_stealing_min) * multiplier);
        
        // Raise distribution-specific thresholds
        uniform_parallel_min = static_cast<size_t>(static_cast<double>(uniform_parallel_min) * multiplier);
        gaussian_parallel_min = static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * multiplier);
        exponential_parallel_min = static_cast<size_t>(static_cast<double>(exponential_parallel_min) * multiplier);
        discrete_parallel_min = static_cast<size_t>(static_cast<double>(discrete_parallel_min) * multiplier);
        poisson_parallel_min = static_cast<size_t>(static_cast<double>(poisson_parallel_min) * multiplier);
        gamma_parallel_min = static_cast<size_t>(static_cast<double>(gamma_parallel_min) * multiplier);
    } else if (threading_overhead < 10000.0) {  // < 10μs overhead
        // Low threading overhead, lower parallel thresholds
        double multiplier = std::max(0.5, threading_overhead / 20000.0);
        parallel_min = static_cast<size_t>(static_cast<double>(parallel_min) * multiplier);
        work_stealing_min = static_cast<size_t>(static_cast<double>(work_stealing_min) * multiplier);
        
        // Lower distribution-specific thresholds
        uniform_parallel_min = static_cast<size_t>(static_cast<double>(uniform_parallel_min) * multiplier);
        gaussian_parallel_min = static_cast<size_t>(static_cast<double>(gaussian_parallel_min) * multiplier);
        exponential_parallel_min = static_cast<size_t>(static_cast<double>(exponential_parallel_min) * multiplier);
        discrete_parallel_min = static_cast<size_t>(static_cast<double>(discrete_parallel_min) * multiplier);
        poisson_parallel_min = static_cast<size_t>(static_cast<double>(poisson_parallel_min) * multiplier);
        gamma_parallel_min = static_cast<size_t>(static_cast<double>(gamma_parallel_min) * multiplier);
    }
    
    // Refine cache-aware thresholds based on memory bandwidth
    if (memory_bandwidth < 20.0) {
        // Low memory bandwidth, raise cache-aware threshold
        cache_aware_min = static_cast<size_t>(static_cast<double>(cache_aware_min) * 1.5);
    } else if (memory_bandwidth > 100.0) {
        // High memory bandwidth, lower cache-aware threshold
        cache_aware_min = static_cast<size_t>(static_cast<double>(cache_aware_min) * 0.7);
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
    cache_aware_min = std::max(cache_aware_min, static_cast<size_t>(10000));
}

} // namespace performance
} // namespace libstats

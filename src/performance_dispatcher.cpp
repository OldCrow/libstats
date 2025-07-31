#include "../include/core/performance_dispatcher.h"
#include "../include/platform/simd_policy.h"

namespace libstats {
namespace performance {

Strategy PerformanceDispatcher::selectOptimalStrategy(
    size_t batch_size,
    DistributionType dist_type,
    ComputationComplexity complexity,
    const SystemCapabilities& system
) const {
    // Determine the proper strategy based on size and complexity
    auto best_threshold = getDistributionSpecificParallelThreshold(dist_type);
    
    if (batch_size >= best_threshold) {
        if (shouldUseWorkStealing(batch_size, dist_type)) {
            return Strategy::WORK_STEALING;
        } else if (shouldUseCacheAware(batch_size, system)) {
            return Strategy::CACHE_AWARE;
        } 
        return Strategy::PARALLEL_SIMD;
    } 
    if (simd::SIMDPolicy::shouldUseSIMD(batch_size)) {
        return Strategy::SIMD_BATCH;
    } 
    return Strategy::SCALAR;
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

bool PerformanceDispatcher::shouldUseWorkStealing(size_t batch_size, DistributionType dist_type) const {
    return batch_size >= thresholds_.work_stealing_min;
}

bool PerformanceDispatcher::shouldUseCacheAware(size_t batch_size, const SystemCapabilities& system) const {
    return batch_size >= thresholds_.cache_aware_min && 
           system.memory_bandwidth_gb_s() >= 50.0;
}

void PerformanceDispatcher::updateThresholds(const Thresholds& new_thresholds) {
    thresholds_ = new_thresholds;
}

} // namespace performance
} // namespace libstats

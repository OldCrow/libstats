#include "../include/core/performance_dispatcher.h"
#include "../include/platform/simd_policy.h"
#include "../include/core/performance_history.h"

namespace libstats {
namespace performance {

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
    
    // Fallback to default logic without sufficient history
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

} // namespace performance
} // namespace libstats

#pragma once

#include <span>
#include <functional>
#include "performance_dispatcher.h"
#include "../platform/work_stealing_pool.h"
#include "../platform/adaptive_cache.h"
#include "../platform/thread_pool.h" // For ParallelUtils

namespace libstats {
namespace performance {

/**
 * @brief Utility class for templated auto-dispatch of batch operations across distributions
 * 
 * This utility eliminates code duplication in the auto-dispatch methods by providing
 * a common templated implementation that can be specialized for different distributions
 * and operations (getProbability, getLogProbability, getCumulativeProbability).
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
     * @tparam CacheAwareFunc Function type for cache-aware operations
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
     * @param cache_aware_func Function to call for cache-aware operations
     */
    template<
        typename Distribution,
        typename ScalarFunc,
        typename BatchFunc,
        typename ParallelFunc,
        typename WorkStealingFunc,
        typename CacheAwareFunc
    >
    static void autoDispatch(
        const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        const PerformanceHint& hint,
        DistributionType dist_type,
        ComputationComplexity complexity,
        ScalarFunc&& scalar_func,
        BatchFunc&& batch_func,
        ParallelFunc&& parallel_func,
        WorkStealingFunc&& work_stealing_func,
        CacheAwareFunc&& cache_aware_func
    ) {
        // Validate input
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }
        
        const size_t count = values.size();
        if (count == 0) return;
        
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
            strategy = dispatcher.selectOptimalStrategy(
                count,
                dist_type,
                complexity,
                system
            );
        } else {
            strategy = mapHintToStrategy(hint.strategy, count);
        }
        
        // Execute using selected strategy
        executeStrategy(
            strategy,
            dist,
            values,
            results,
            count,
            std::forward<ScalarFunc>(scalar_func),
            std::forward<BatchFunc>(batch_func),
            std::forward<ParallelFunc>(parallel_func),
            std::forward<WorkStealingFunc>(work_stealing_func),
            std::forward<CacheAwareFunc>(cache_aware_func)
        );
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
    template<typename Distribution, typename BatchUnsafeFunc>
    static void executeBatchSIMD(
        const Distribution& dist,
        const double* values,
        double* results,
        std::size_t count,
        BatchUnsafeFunc&& batch_unsafe_func
    ) {
        if (count == 0) return;
        if (!values || !results) {
            return; // Can't throw in noexcept context for some methods
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
     * @tparam CacheAwareFunc Function type for cache-aware operations
     * 
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param strategy Explicit strategy to use
     * @param scalar_func Function to call for scalar operations
     * @param batch_func Function to call for SIMD batch operations
     * @param parallel_func Function to call for parallel operations  
     * @param work_stealing_func Function to call for work-stealing operations
     * @param cache_aware_func Function to call for cache-aware operations
     */
    template<
        typename Distribution,
        typename ScalarFunc,
        typename BatchFunc,
        typename ParallelFunc,
        typename WorkStealingFunc,
        typename CacheAwareFunc
    >
    static void executeWithStrategy(
        const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        Strategy strategy,
        ScalarFunc&& scalar_func,
        BatchFunc&& batch_func,
        ParallelFunc&& parallel_func,
        WorkStealingFunc&& work_stealing_func,
        CacheAwareFunc&& cache_aware_func
    ) {
        // Validate input
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }
        
        const size_t count = values.size();
        if (count == 0) return;
        
        // Execute using provided strategy
        executeStrategy(
            strategy,
            dist,
            values,
            results,
            count,
            std::forward<ScalarFunc>(scalar_func),
            std::forward<BatchFunc>(batch_func),
            std::forward<ParallelFunc>(parallel_func),
            std::forward<WorkStealingFunc>(work_stealing_func),
            std::forward<CacheAwareFunc>(cache_aware_func)
        );
    }
    
private:
    /**
     * @brief Maps performance hints to strategies
     */
    static Strategy mapHintToStrategy(PerformanceHint::PreferredStrategy hint_strategy, size_t count) {
        switch (hint_strategy) {
            case PerformanceHint::PreferredStrategy::FORCE_SCALAR:
                return Strategy::SCALAR;
            case PerformanceHint::PreferredStrategy::FORCE_SIMD:
                return Strategy::SIMD_BATCH;
            case PerformanceHint::PreferredStrategy::FORCE_PARALLEL:
                return Strategy::PARALLEL_SIMD;
            case PerformanceHint::PreferredStrategy::MINIMIZE_LATENCY:
                return (count <= 8) ? Strategy::SCALAR : Strategy::SIMD_BATCH;
            case PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT:
                return Strategy::PARALLEL_SIMD;
            default:
                return Strategy::SCALAR;
        }
    }
    
    /**
     * @brief Executes the selected strategy with appropriate function calls
     */
    template<
        typename Distribution,
        typename ScalarFunc,
        typename BatchFunc,
        typename ParallelFunc,
        typename WorkStealingFunc,
        typename CacheAwareFunc
    >
    static void executeStrategy(
        Strategy strategy,
        const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        size_t count,
        ScalarFunc&& scalar_func,
        BatchFunc&& batch_func,
        ParallelFunc&& parallel_func,
        WorkStealingFunc&& work_stealing_func,
        CacheAwareFunc&& cache_aware_func
    ) {
        
        switch (strategy) {
            case Strategy::SCALAR:
                // Use simple loop for tiny batches (< 8 elements)
                for (size_t i = 0; i < count; ++i) {
                    results[i] = scalar_func(dist, values[i]);
                }
                break;
                
            case Strategy::SIMD_BATCH:
                // Use existing SIMD implementation
                batch_func(dist, values.data(), results.data(), count);
                break;
                
            case Strategy::PARALLEL_SIMD:
                // Use existing parallel implementation
                parallel_func(dist, values, results);
                break;
                
            case Strategy::WORK_STEALING: {
                // Use work-stealing pool for load balancing
                static thread_local WorkStealingPool default_pool;
                work_stealing_func(dist, values, results, default_pool);
                break;
            }
                
            case Strategy::CACHE_AWARE: {
                // Use cache-aware implementation
                static thread_local cache::AdaptiveCache<std::string, double> default_cache;
                cache_aware_func(dist, values, results, default_cache);
                break;
            }
        }
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
    template<typename Distribution, typename ComputationFunc>
    static void executeBatchParallel(
        [[maybe_unused]] const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        ComputationFunc&& computation_func
    ) {
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }
        
        const std::size_t count = values.size();
        if (count == 0) return;
        
        // Use ParallelUtils::parallelFor for Level 0-3 integration
        if (parallel::should_use_parallel(count)) {
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                computation_func(i);
            });
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
    template<typename Distribution, typename ComputationFunc>
    static void executeBatchWorkStealing(
        [[maybe_unused]] const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        WorkStealingPool& pool,
        ComputationFunc&& computation_func
    ) {
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }
        
        const std::size_t count = values.size();
        if (count == 0) return;
        
        // Use WorkStealingPool for dynamic load balancing
        // Use same threshold as regular parallel operations to avoid inconsistency
        if (WorkStealingUtils::shouldUseWorkStealing(count, constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL)) {
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                computation_func(i);
            });
            pool.waitForAll();
        } else {
            // Serial processing for small datasets
            for (std::size_t i = 0; i < count; ++i) {
                computation_func(i);
            }
        }
    }
    
    /**
     * @brief Execute cache-aware batch operations with common pattern
     * 
     * @tparam Distribution The distribution type
     * @tparam ComputationFunc Function type for element-wise computation
     * 
     * @param dist Reference to the distribution instance
     * @param values Input values span
     * @param results Output results span
     * @param cache_manager Adaptive cache manager
     * @param operation_name Name of the operation for cache key generation
     * @param computation_func Function to compute result for each element
     */
    template<typename Distribution, typename ComputationFunc>
    static void executeBatchCacheAware(
        [[maybe_unused]] const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        cache::AdaptiveCache<std::string, double>& cache_manager,
        const std::string& operation_name,
        ComputationFunc&& computation_func
    ) {
        if (values.size() != results.size()) {
            throw std::invalid_argument("Input and output spans must have the same size");
        }
        
        const std::size_t count = values.size();
        if (count == 0) return;
        
        // Integrate with Level 0-3 adaptive cache system
        const std::string cache_key = operation_name + "_batch_" + std::to_string(count);
        
        auto cached_params = cache_manager.getCachedComputationParams(cache_key);
        if (cached_params.has_value()) {
            // Future: Use cached performance metrics for optimization
        }
        
        // Determine optimal batch size based on cache behavior
        const std::size_t optimal_grain_size = cache_manager.getOptimalGrainSize(count, operation_name);
        
        // Use cache-aware parallel processing
        if (parallel::should_use_parallel(count)) {
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                computation_func(i);
            }, optimal_grain_size);
        } else {
            // Serial processing for small datasets
            for (std::size_t i = 0; i < count; ++i) {
                computation_func(i);
            }
        }
        
        // Update cache manager with performance metrics
        cache_manager.recordBatchPerformance(cache_key, count, optimal_grain_size);
    }
    
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
    template<typename Distribution, typename CacheExtractorFunc, typename ExecutionFunc>
    static void withCachedParameters(
        const Distribution& dist,
        CacheExtractorFunc&& cache_extractor,
        ExecutionFunc&& execution_func
    ) {
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
template<typename Distribution>
struct DistributionTraits {
    static constexpr DistributionType distType() = delete;
    static constexpr ComputationComplexity complexity() = delete;
};

// Specializations for known distributions
template<>
struct DistributionTraits<class DiscreteDistribution> {
    static constexpr DistributionType distType() { return DistributionType::DISCRETE; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::SIMPLE; }
};

template<>
struct DistributionTraits<class PoissonDistribution> {
    static constexpr DistributionType distType() { return DistributionType::POISSON; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::COMPLEX; }
};

template<>
struct DistributionTraits<class GaussianDistribution> {
    static constexpr DistributionType distType() { return DistributionType::GAUSSIAN; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::MODERATE; }
};

template<>
struct DistributionTraits<class ExponentialDistribution> {
    static constexpr DistributionType distType() { return DistributionType::EXPONENTIAL; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::MODERATE; }
};

template<>
struct DistributionTraits<class UniformDistribution> {
    static constexpr DistributionType distType() { return DistributionType::UNIFORM; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::SIMPLE; }
};

template<>
struct DistributionTraits<class GammaDistribution> {
    static constexpr DistributionType distType() { return DistributionType::GAMMA; }
    static constexpr ComputationComplexity complexity() { return ComputationComplexity::COMPLEX; }
};

} // namespace performance
} // namespace libstats

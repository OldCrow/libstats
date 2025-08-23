#include "../include/core/performance_dispatcher.h"
#include "../include/libstats.h"
#include "../include/platform/simd_policy.h"
#include "../include/platform/thread_pool.h"

namespace stats {

void initialize_performance_systems() {
    // Thread-safe one-time initialization using static local variable
    static bool initialized = false;
    static std::mutex init_mutex;

    // Fast path: if already initialized, return immediately
    if (initialized) {
        return;
    }

    // Slow path: acquire mutex and initialize
    std::lock_guard<std::mutex> lock(init_mutex);

    // Double-check pattern: another thread might have initialized while we waited
    if (initialized) {
        return;
    }

    try {
        // 1. Initialize system capabilities (triggers CPU detection and benchmarking)
        // This is the most expensive operation (~10-30ms)
        [[maybe_unused]] const auto& system_capabilities =
            performance::SystemCapabilities::current();

        // 2. Initialize SIMD policy (triggers SIMD feature detection)
        // Moderate cost (~1-5ms)
        [[maybe_unused]] auto simd_level = simd::SIMDPolicy::getBestLevel();

        // 3. Initialize performance dispatcher with detected capabilities
        // Creates optimized thresholds based on system characteristics (~1-2ms)
        [[maybe_unused]] performance::PerformanceDispatcher dispatcher(system_capabilities);

        // 4. Initialize performance history singleton
        // Creates the global performance history instance (~0.1ms)
        [[maybe_unused]] auto& perf_history =
            performance::PerformanceDispatcher::getPerformanceHistory();

        // 5. Initialize thread pool infrastructure
        // Creates optimal thread pool for parallel operations (~1-3ms)
        [[maybe_unused]] auto optimal_threads = ThreadPool::getOptimalThreadCount();

        // Mark as initialized
        initialized = true;

    } catch (...) {
        // If initialization fails, don't mark as initialized
        // This allows retry on next call
        // In practice, initialization should rarely fail as it only does capability detection
        throw;
    }
}

}  // namespace stats

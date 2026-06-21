#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/performance_dispatcher.h"
#include "libstats/libstats.h"
#include "libstats/platform/simd_policy.h"
#include "libstats/platform/thread_pool.h"

#include <atomic>
#include <mutex>

namespace stats {

void initialize_performance_systems() {
    // Thread-safe one-time initialization using double-checked locking.
    // The atomic load on the fast path avoids a data race under the C++ memory model.
    static std::atomic<bool> initialized{false};
    static std::mutex init_mutex;

    // Fast path: if already initialized, return immediately
    if (initialized.load(std::memory_order_acquire)) {
        return;
    }

    // Slow path: acquire mutex and initialize
    std::lock_guard<std::mutex> lock(init_mutex);

    // Double-check: another thread might have initialized while we waited
    if (initialized.load(std::memory_order_acquire)) {
        return;
    }

    try {
        // 1. Initialize system capabilities (triggers CPU detection and benchmarking)
        // This is the most expensive operation (~10-30ms)
        [[maybe_unused]] const auto& system_capabilities = detail::SystemCapabilities::current();

        // 2. Initialize SIMD policy (triggers SIMD feature detection)
        // Moderate cost (~1-5ms)
        [[maybe_unused]] auto simd_level = arch::simd::SIMDPolicy::getBestLevel();

        // 3. Initialize performance dispatcher with detected capabilities
        // Creates optimized thresholds based on system characteristics (~1-2ms)
        [[maybe_unused]] detail::PerformanceDispatcher dispatcher(system_capabilities);

        // 4. Initialize performance history singleton
        // Creates the global performance history instance (~0.1ms)
        [[maybe_unused]] auto& perf_history =
            detail::PerformanceDispatcher::getPerformanceHistory();

        // 5. Initialize thread pool infrastructure
        // Creates optimal thread pool for parallel operations (~1-3ms)
        [[maybe_unused]] auto optimal_threads = ThreadPool::getOptimalThreadCount();

        // Mark as initialized
        initialized.store(true, std::memory_order_release);

    } catch (...) {
        // If initialization fails, don't mark as initialized
        // This allows retry on next call
        // In practice, initialization should rarely fail as it only does capability detection
        throw;
    }
}

}  // namespace stats

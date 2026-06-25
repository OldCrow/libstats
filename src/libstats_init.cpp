#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/performance_dispatcher.h"
#include "libstats/libstats.h"
#include "libstats/platform/simd_policy.h"
#include "libstats/platform/thread_pool.h"
#include "libstats/platform/work_stealing_pool.h"

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
        // 1. Warm up system capabilities (CPUID + core detection; ~0.1ms after benchmark removal)
        [[maybe_unused]] const auto& system_capabilities = detail::SystemCapabilities::current();

        // 2. Warm up SIMD policy (amortises CPUID detection on first batch call)
        [[maybe_unused]] auto simd_level = arch::simd::SIMDPolicy::getBestLevel();

        // 3. Warm up thread pool singletons (amortises thread-launch cost on first
        // batch call). getOptimalThreadCount() only returns a number; the singletons
        // must be touched explicitly to trigger thread creation.
        [[maybe_unused]] auto& thread_pool = ParallelUtils::getGlobalThreadPool();
        [[maybe_unused]] auto& ws_pool = GlobalWorkStealingPool::getInstance();

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

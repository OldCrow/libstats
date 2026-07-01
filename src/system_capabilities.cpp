#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/math_constants.h"
#include "libstats/core/performance_dispatcher.h"
#include "libstats/platform/cpu_detection.h"

#include <thread>

namespace stats {
namespace detail {  // Performance utilities

SystemCapabilities::SystemCapabilities() {
    detectCapabilities();
    // benchmarkPerformance() removed: the three microbenchmarks (SIMD efficiency,
    // threading overhead, memory bandwidth) ran at every first-call cold-start and
    // allocated up to 256MB of scratch memory, but their results were never read by
    // selectStrategy() or selectMultiThreadedStrategy(). detectCapabilities() is
    // the only load-bearing initialisation step; it sets SIMD flags and core counts.
}

const SystemCapabilities& SystemCapabilities::current() {
    static SystemCapabilities instance;
    return instance;
}

void SystemCapabilities::detectCapabilities() {
    // CPU core detection
    logical_cores_ = std::thread::hardware_concurrency();
    physical_cores_ = logical_cores_ / 2;  // Simplified assumption (hyperthreading)

    // Cache sizes (simplified - would need platform-specific detection in production)
    l1_cache_size_ = 32 * 1024;        // 32KB typical L1
    l2_cache_size_ = 256 * 1024;       // 256KB typical L2
    l3_cache_size_ = 8 * 1024 * 1024;  // 8MB typical L3

    // SIMD capability detection using existing CPU detection
    has_sse2_ = stats::arch::supports_sse2();
    has_avx_ = stats::arch::supports_avx();
    has_avx2_ = stats::arch::supports_avx2();
    has_avx512_ = stats::arch::supports_avx512();
    has_neon_ = stats::arch::supports_neon();
}

}  // namespace detail
}  // namespace stats

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)  // Suppress MSVC static analysis VRC003 warnings for GTest
#endif

// Use focused header for system capabilities testing
#include "../include/core/performance_dispatcher.h"

#include <chrono>
#include <gtest/gtest.h>
#include <thread>

using namespace stats::performance;

class SystemCapabilitiesIntegrationTest : public ::testing::Test {
   protected:
    const SystemCapabilities& capabilities = SystemCapabilities::current();
};

TEST_F(SystemCapabilitiesIntegrationTest, SingletonBehavior) {
    // Test that SystemCapabilities is a proper singleton
    const SystemCapabilities& caps1 = SystemCapabilities::current();
    const SystemCapabilities& caps2 = SystemCapabilities::current();

    // Should be the same instance
    EXPECT_EQ(&caps1, &caps2);

    // Values should be identical
    EXPECT_EQ(caps1.logical_cores(), caps2.logical_cores());
    EXPECT_EQ(caps1.physical_cores(), caps2.physical_cores());
    EXPECT_EQ(caps1.l1_cache_size(), caps2.l1_cache_size());
}

TEST_F(SystemCapabilitiesIntegrationTest, ReasonableSystemValues) {
    // Test that detected values are within reasonable ranges

    // Core counts
    EXPECT_GE(capabilities.logical_cores(), static_cast<std::size_t>(1));
    EXPECT_LE(capabilities.logical_cores(),
              static_cast<std::size_t>(256));  // Reasonable upper bound
    EXPECT_GE(capabilities.physical_cores(), 1);
    EXPECT_LE(capabilities.physical_cores(), capabilities.logical_cores());

    // Cache sizes (in bytes)
    EXPECT_GE(capabilities.l1_cache_size(), 1024);         // At least 1KB
    EXPECT_LE(capabilities.l1_cache_size(), 1024 * 1024);  // At most 1MB (generous)

    if (capabilities.l2_cache_size() > 0) {
        EXPECT_GE(capabilities.l2_cache_size(), capabilities.l1_cache_size());
        EXPECT_LE(capabilities.l2_cache_size(), 64 * 1024 * 1024);  // At most 64MB
    }

    if (capabilities.l3_cache_size() > 0) {
        EXPECT_GE(capabilities.l3_cache_size(), capabilities.l2_cache_size());
        EXPECT_LE(capabilities.l3_cache_size(), 1024 * 1024 * 1024);  // At most 1GB
    }

    // Performance characteristics
    EXPECT_GE(capabilities.simd_efficiency(), 0.0);
    EXPECT_LE(capabilities.simd_efficiency(), 10.0);  // Very generous upper bound

    EXPECT_GE(capabilities.threading_overhead_ns(), 0.0);
    EXPECT_LE(capabilities.threading_overhead_ns(), 1000000.0);  // 1ms max overhead

    EXPECT_GE(capabilities.memory_bandwidth_gb_s(), 0.0);
    EXPECT_LE(capabilities.memory_bandwidth_gb_s(),
              100.0);  // 100GB/s upper bound (clamped in benchmark)
}

TEST_F(SystemCapabilitiesIntegrationTest, SIMDCapabilityConsistency) {
    // Test SIMD capability hierarchy (if AVX2 is available, AVX should be too, etc.)
    if (capabilities.has_avx512()) {
        EXPECT_TRUE(capabilities.has_avx2());
        EXPECT_TRUE(capabilities.has_avx());
        EXPECT_TRUE(capabilities.has_sse2());
    }

    if (capabilities.has_avx2()) {
        EXPECT_TRUE(capabilities.has_avx());
        EXPECT_TRUE(capabilities.has_sse2());
    }

    if (capabilities.has_avx()) {
        EXPECT_TRUE(capabilities.has_sse2());
    }

    // ARM NEON is independent of x86 SIMD
    // (A system shouldn't have both, but we don't enforce that)
}

TEST_F(SystemCapabilitiesIntegrationTest, ThreadSafety) {
    // Test that multiple threads can safely access system capabilities
    constexpr std::size_t num_threads = 8;
    constexpr std::size_t accesses_per_thread = 1000;

    std::vector<std::thread> threads;
    std::vector<bool> success(static_cast<std::size_t>(num_threads), false);

    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            bool thread_success = true;

            for (std::size_t i = 0; i < accesses_per_thread && thread_success; ++i) {
                const SystemCapabilities& caps = SystemCapabilities::current();

                // Verify consistency
                if (caps.logical_cores() == 0)
                    thread_success = false;
                if (caps.physical_cores() == 0)
                    thread_success = false;
                if (caps.physical_cores() > caps.logical_cores())
                    thread_success = false;
                if (caps.l1_cache_size() == 0)
                    thread_success = false;

                // Verify SIMD consistency
                if (caps.has_avx2() && !caps.has_avx())
                    thread_success = false;
                if (caps.has_avx() && !caps.has_sse2())
                    thread_success = false;

                // Small delay to increase chance of race conditions
                if (i % 100 == 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }

            success[t] = thread_success;
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // All threads should have succeeded
    for (std::size_t t = 0; t < num_threads; ++t) {
        EXPECT_TRUE(success[t]) << "Thread " << t << " failed consistency checks";
    }
}

TEST_F(SystemCapabilitiesIntegrationTest, PerformanceCharacteristicsRealistic) {
    // Test that performance characteristics are within realistic bounds

    // SIMD efficiency should be reasonable
    // (1.0 = perfect efficiency, 0.5 = half efficiency due to overhead)
    if (capabilities.has_sse2() || capabilities.has_avx() || capabilities.has_neon()) {
        EXPECT_GE(capabilities.simd_efficiency(), 0.01);  // Some benefit (reduced from 0.1)
        EXPECT_LE(capabilities.simd_efficiency(), 4.0);   // Not impossibly good
    }

    // Threading overhead should be measurable but not excessive
    if (capabilities.physical_cores() > 1) {
        EXPECT_GE(capabilities.threading_overhead_ns(), 10.0);      // At least 10ns
        EXPECT_LE(capabilities.threading_overhead_ns(), 100000.0);  // At most 100μs
    }

    // Memory bandwidth should be realistic for the era
    if (capabilities.memory_bandwidth_gb_s() > 0.0) {
        EXPECT_GE(capabilities.memory_bandwidth_gb_s(), 0.1);    // At least 100MB/s
        EXPECT_LE(capabilities.memory_bandwidth_gb_s(), 100.0);  // At most 100GB/s (clamped)
    }
}

TEST_F(SystemCapabilitiesIntegrationTest, IntegrationWithDispatcher) {
    // Test that SystemCapabilities integrates properly with PerformanceDispatcher
    PerformanceDispatcher dispatcher;

    // The dispatcher should be able to use the capabilities
    auto strategy = dispatcher.selectOptimalStrategy(1000, DistributionType::GAUSSIAN,
                                                     ComputationComplexity::MODERATE, capabilities);

    // Should return a valid strategy
    EXPECT_TRUE(strategy >= Strategy::SCALAR && strategy <= Strategy::GPU_ACCELERATED);

    // Test with different parameters
    auto small_strategy = dispatcher.selectOptimalStrategy(
        10, DistributionType::UNIFORM, ComputationComplexity::SIMPLE, capabilities);
    // Accept either SCALAR or SIMD_BATCH for small batches (depends on SIMD policy)
    EXPECT_TRUE(small_strategy == Strategy::SCALAR || small_strategy == Strategy::SIMD_BATCH);

    // Large batch should consider parallel strategies (if we have multiple cores)
    if (capabilities.physical_cores() > 1) {
        auto large_strategy = dispatcher.selectOptimalStrategy(
            100000, DistributionType::GAMMA, ComputationComplexity::COMPLEX, capabilities);
        EXPECT_TRUE(large_strategy == Strategy::PARALLEL_SIMD ||
                    large_strategy == Strategy::WORK_STEALING ||
                    large_strategy == Strategy::GPU_ACCELERATED);
    }
}

TEST_F(SystemCapabilitiesIntegrationTest, CacheHierarchyLogical) {
    // Test that cache hierarchy makes sense
    EXPECT_GT(capabilities.l1_cache_size(), 0);

    if (capabilities.l2_cache_size() > 0) {
        EXPECT_GE(capabilities.l2_cache_size(), capabilities.l1_cache_size());
    }

    if (capabilities.l3_cache_size() > 0) {
        // L3 should be larger than L2 (if L2 exists)
        if (capabilities.l2_cache_size() > 0) {
            EXPECT_GE(capabilities.l3_cache_size(), capabilities.l2_cache_size());
        }
        // L3 should definitely be larger than L1
        EXPECT_GE(capabilities.l3_cache_size(), capabilities.l1_cache_size());
    }
}

TEST_F(SystemCapabilitiesIntegrationTest, PlatformSpecificFeatures) {
    // Test platform-specific feature detection

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    // On x86/x64 platforms, should not have NEON
    EXPECT_FALSE(capabilities.has_neon());

    // Should have at least SSE2 on modern x86_64
    #if defined(__x86_64__) || defined(_M_X64)
        // Most modern x86_64 systems have SSE2
        // (but we don't enforce this for compatibility)
    #endif

#elif defined(__aarch64__) || defined(_M_ARM64)
    // On ARM64 platforms, should not have x86 SIMD
    EXPECT_FALSE(capabilities.has_sse2());
    EXPECT_FALSE(capabilities.has_avx());
    EXPECT_FALSE(capabilities.has_avx2());
    EXPECT_FALSE(capabilities.has_avx512());

    // Should have NEON on ARM64
    EXPECT_TRUE(capabilities.has_neon());

#endif
}

TEST_F(SystemCapabilitiesIntegrationTest, ConsistentResults) {
    // Test that multiple calls return consistent results
    const SystemCapabilities& caps1 = SystemCapabilities::current();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const SystemCapabilities& caps2 = SystemCapabilities::current();

    // All values should be identical
    EXPECT_EQ(caps1.logical_cores(), caps2.logical_cores());
    EXPECT_EQ(caps1.physical_cores(), caps2.physical_cores());
    EXPECT_EQ(caps1.l1_cache_size(), caps2.l1_cache_size());
    EXPECT_EQ(caps1.l2_cache_size(), caps2.l2_cache_size());
    EXPECT_EQ(caps1.l3_cache_size(), caps2.l3_cache_size());

    EXPECT_EQ(caps1.has_sse2(), caps2.has_sse2());
    EXPECT_EQ(caps1.has_avx(), caps2.has_avx());
    EXPECT_EQ(caps1.has_avx2(), caps2.has_avx2());
    EXPECT_EQ(caps1.has_avx512(), caps2.has_avx512());
    EXPECT_EQ(caps1.has_neon(), caps2.has_neon());

    EXPECT_DOUBLE_EQ(caps1.simd_efficiency(), caps2.simd_efficiency());
    EXPECT_DOUBLE_EQ(caps1.threading_overhead_ns(), caps2.threading_overhead_ns());
    EXPECT_DOUBLE_EQ(caps1.memory_bandwidth_gb_s(), caps2.memory_bandwidth_gb_s());
}

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

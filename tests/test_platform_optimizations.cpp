// Use focused header for platform optimization testing
#include "../include/platform/simd.h"

#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using namespace stats::simd;

    std::cout << "=== PLATFORM OPTIMIZATION TEST ===" << std::endl;

    // Test 1: Platform optimization info should return non-empty string
    std::string platform_info = VectorOps::get_platform_optimization_info();
    assert(!platform_info.empty());
    std::cout << "✓ Platform info: " << platform_info << std::endl;

    // Test 2: Platform-specific constants should be reasonable values
    assert(tuned::matrix_block_size() > 0 && tuned::matrix_block_size() <= 128);
    assert(tuned::simd_loop_width() > 0 && tuned::simd_loop_width() <= 16);
    assert(SIMDPolicy::getMinThreshold() > 0);
    assert(tuned::cache_friendly_step() > 0);
    assert(tuned::l1_cache_doubles() > 1000);  // Should be reasonable cache size

    std::cout << "✓ Platform constants are within reasonable ranges" << std::endl;
    std::cout << "  - Matrix block size: " << tuned::matrix_block_size() << std::endl;
    std::cout << "  - SIMD loop width: " << tuned::simd_loop_width() << std::endl;
    std::cout << "  - Min states for SIMD: " << SIMDPolicy::getMinThreshold() << std::endl;
    std::cout << "  - Cache-friendly step: " << tuned::cache_friendly_step() << std::endl;
    std::cout << "  - L1 cache doubles: " << tuned::l1_cache_doubles() << std::endl;
    std::cout << "  - Prefetch distance: " << tuned::prefetch_distance() << std::endl;

    // Test 3: Smart vectorization decision function
    std::vector<double> test_data_tiny(2, 1.0);
    std::vector<double> test_data_small(8, 1.0);
    std::vector<double> test_data_medium(64, 1.0);
    std::vector<double> test_data_large(2000, 1.0);

    // For very small data, decision should be conservative
    bool tiny_decision =
        VectorOps::should_use_vectorized_path(test_data_tiny.size(), test_data_tiny.data());

    // For medium and large data, should generally use vectorization
    bool medium_decision =
        VectorOps::should_use_vectorized_path(test_data_medium.size(), test_data_medium.data());
    bool large_decision =
        VectorOps::should_use_vectorized_path(test_data_large.size(), test_data_large.data());

    // Large data should always use vectorization
    assert(large_decision);

    std::cout << "✓ Smart vectorization decisions:" << std::endl;
    std::cout << "  - Tiny data (2 elements): " << (tiny_decision ? "SIMD" : "Scalar") << std::endl;
    std::cout << "  - Small data (8 elements): "
              << (VectorOps::should_use_vectorized_path(test_data_small.size(),
                                                        test_data_small.data())
                      ? "SIMD"
                      : "Scalar")
              << std::endl;
    std::cout << "  - Medium data (64 elements): " << (medium_decision ? "SIMD" : "Scalar")
              << std::endl;
    std::cout << "  - Large data (2000 elements): " << (large_decision ? "SIMD" : "Scalar")
              << std::endl;

    // Test 4: Verify platform-adaptive thresholds make sense
    std::size_t min_simd = VectorOps::min_simd_size();
    std::size_t block_size = VectorOps::get_optimal_block_size();

    assert(min_simd > 0);
    assert(block_size > 0);
    assert(block_size >= VectorOps::double_vector_width());

    std::cout << "✓ Adaptive thresholds:" << std::endl;
    std::cout << "  - Min SIMD size: " << min_simd << std::endl;
    std::cout << "  - Optimal block size: " << block_size << std::endl;
    std::cout << "  - Double vector width: " << VectorOps::double_vector_width() << std::endl;

    // Test 5: Memory alignment checking (basic functionality test)
    std::vector<double, aligned_allocator<double>> aligned_data(32, 1.0);

    // The aligned allocator should produce properly aligned memory
    bool alignment_ok = is_aligned(aligned_data.data());
    std::cout << "✓ Aligned allocator: " << (alignment_ok ? "PASS" : "FAIL") << std::endl;

// Test 6: Verify Apple Silicon detection (if applicable)
#if defined(LIBSTATS_APPLE_SILICON)
    std::cout << "✓ Apple Silicon optimizations: ENABLED" << std::endl;
    // Apple Silicon should have aggressive SIMD thresholds
    assert(tuned::min_parallel_work() <= 128);  // Should be low for fast thread creation
#else
    std::cout << "✓ Apple Silicon optimizations: not applicable" << std::endl;
#endif

    // Test 7: Platform consistency check
    std::string active_simd = VectorOps::get_active_simd_level();
    bool simd_available = VectorOps::is_simd_available();
    bool supports_vec = VectorOps::supports_vectorization();

    assert(supports_vec);  // Should always support some form of vectorization

    if (active_simd != "Scalar") {
        assert(simd_available);
    }

    std::cout << "✓ Platform consistency:" << std::endl;
    std::cout << "  - Active SIMD level: " << active_simd << std::endl;
    std::cout << "  - SIMD available: " << (simd_available ? "YES" : "NO") << std::endl;
    std::cout << "  - Vectorization supported: " << (supports_vec ? "YES" : "NO") << std::endl;

    std::cout << std::endl;
    std::cout << "=== ALL PLATFORM OPTIMIZATION TESTS PASSED ===" << std::endl;

    return 0;
}

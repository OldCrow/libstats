#include "../include/core/constants.h"

#include <cassert>
#include <iostream>

using namespace stats::detail;

void test_simd_constants() {
    std::cout << "Testing SIMD architectural constants..." << std::endl;

    // Test SIMD alignment constants
    assert(arch::simd::AVX512_ALIGNMENT == 64);
    assert(arch::simd::AVX_ALIGNMENT == 32);
    assert(arch::simd::SSE_ALIGNMENT == 16);
    assert(arch::simd::NEON_ALIGNMENT == 16);
    assert(arch::simd::CACHE_LINE_ALIGNMENT == 64);

    // Test matrix block sizes
    assert(arch::simd::matrix::L1_BLOCK_SIZE == 64);
    assert(arch::simd::matrix::L2_BLOCK_SIZE == 256);
    assert(arch::simd::matrix::L3_BLOCK_SIZE == 1024);
    assert(arch::simd::matrix::STEP_SIZE == 8);
    assert(arch::simd::matrix::PANEL_WIDTH == 64);

    // Test SIMD register widths
    assert(arch::simd::registers::AVX512_DOUBLES == 8);
    assert(arch::simd::registers::AVX_DOUBLES == 4);
    assert(arch::simd::registers::SSE_DOUBLES == 2);
    assert(arch::simd::registers::NEON_DOUBLES == 2);
    assert(arch::simd::registers::SCALAR_DOUBLES == 1);

    // Test unroll factors
    assert(arch::simd::unroll::AVX512_UNROLL == 4);
    assert(arch::simd::unroll::AVX_UNROLL == 2);
    assert(arch::simd::unroll::SSE_UNROLL == 2);
    assert(arch::simd::unroll::NEON_UNROLL == 2);
    assert(arch::simd::unroll::SCALAR_UNROLL == 1);

    std::cout << "✓ SIMD architectural constants test passed" << std::endl;
}

void test_memory_constants() {
    std::cout << "Testing memory access and prefetching constants..." << std::endl;

    // Test prefetch distance constants (Phase 3C: flattened namespaces)
    assert(memory::prefetch::DISTANCE_CONSERVATIVE == 2);
    assert(memory::prefetch::DISTANCE_STANDARD == 4);
    assert(memory::prefetch::DISTANCE_AGGRESSIVE == 8);
    assert(memory::prefetch::DISTANCE_ULTRA_AGGRESSIVE == 16);

    // Test platform-specific prefetch distances
    assert(memory::prefetch::platform::apple_silicon::SEQUENTIAL_PREFETCH_DISTANCE == 256);
    assert(memory::prefetch::platform::intel::SEQUENTIAL_PREFETCH_DISTANCE == 192);
    assert(memory::prefetch::platform::amd::SEQUENTIAL_PREFETCH_DISTANCE == 128);
    assert(memory::prefetch::platform::arm::SEQUENTIAL_PREFETCH_DISTANCE == 64);

    // Test prefetch strategy constants (Phase 3C: flattened namespaces)
    assert(memory::prefetch::STRATEGY_SEQUENTIAL_MULTIPLIER == 2.0);
    assert(memory::prefetch::STRATEGY_RANDOM_MULTIPLIER == 0.5);
    assert(memory::prefetch::STRATEGY_STRIDED_MULTIPLIER == 1.5);
    assert(memory::prefetch::STRATEGY_MIN_PREFETCH_SIZE == 32);
    assert(memory::prefetch::STRATEGY_MAX_PREFETCH_DISTANCE == 1024);
    assert(memory::prefetch::STRATEGY_PREFETCH_GRANULARITY == 8);

    // Test memory access constants
    assert(memory::access::CACHE_LINE_SIZE_BYTES == 64);
    assert(memory::access::DOUBLES_PER_CACHE_LINE == 8);
    assert(memory::access::CACHE_LINE_ALIGNMENT == 64);

    // Test bandwidth optimization (Phase 3C: flattened namespaces)
    assert(memory::access::BANDWIDTH_DDR4_BURST_SIZE == 64);
    assert(memory::access::BANDWIDTH_DDR5_BURST_SIZE == 128);
    assert(memory::access::BANDWIDTH_HBM_BURST_SIZE == 256);
    assert(memory::access::BANDWIDTH_TARGET_UTILIZATION == 0.8);
    assert(memory::access::BANDWIDTH_MAX_UTILIZATION == 0.95);

    // Test memory layout constants (Phase 3C: flattened namespaces)
    assert(memory::access::LAYOUT_AOS_TO_SOA_THRESHOLD == 1000);
    assert(memory::access::LAYOUT_MEMORY_POOL_ALIGNMENT == 4096);
    assert(memory::access::LAYOUT_SMALL_ALLOCATION_THRESHOLD == 256);
    assert(memory::access::LAYOUT_LARGE_PAGE_THRESHOLD == 2097152);

    // Test NUMA constants (Phase 3C: flattened namespaces)
    assert(memory::access::NUMA_AWARE_THRESHOLD == 1048576);
    assert(memory::access::NUMA_LOCAL_THRESHOLD == 65536);
    assert(memory::access::NUMA_MIGRATION_COST == 0.1);

    // Test memory allocation constants
    assert(memory::allocation::SMALL_POOL_SIZE == 4096);
    assert(memory::allocation::MEDIUM_POOL_SIZE == 65536);
    assert(memory::allocation::LARGE_POOL_SIZE == 1048576);
    assert(memory::allocation::MIN_ALLOCATION_ALIGNMENT == 8);
    assert(memory::allocation::SIMD_ALLOCATION_ALIGNMENT == 32);
    assert(memory::allocation::PAGE_ALLOCATION_ALIGNMENT == 4096);

    // Test growth factors (Phase 3C: flattened namespaces)
    assert(memory::allocation::GROWTH_EXPONENTIAL_FACTOR == 1.5);
    assert(memory::allocation::GROWTH_LINEAR_FACTOR == 1.2);
    assert(memory::allocation::GROWTH_THRESHOLD == 1048576);

    std::cout << "✓ Memory access and prefetching constants test passed" << std::endl;
}

void test_constant_relationships() {
    std::cout << "Testing architectural constant relationships..." << std::endl;

    // Test alignment hierarchy
    assert(arch::simd::AVX512_ALIGNMENT >= arch::simd::AVX_ALIGNMENT);
    assert(arch::simd::AVX_ALIGNMENT >= arch::simd::SSE_ALIGNMENT);
    assert(arch::simd::SSE_ALIGNMENT == arch::simd::NEON_ALIGNMENT);

    // Test matrix block size hierarchy
    assert(arch::simd::matrix::L3_BLOCK_SIZE >= arch::simd::matrix::L2_BLOCK_SIZE);
    assert(arch::simd::matrix::L2_BLOCK_SIZE >= arch::simd::matrix::L1_BLOCK_SIZE);
    assert(arch::simd::matrix::L1_BLOCK_SIZE >= arch::simd::matrix::MIN_BLOCK_SIZE);
    assert(arch::simd::matrix::MAX_BLOCK_SIZE >= arch::simd::matrix::L3_BLOCK_SIZE);

    // Test SIMD register width hierarchy
    assert(arch::simd::registers::AVX512_DOUBLES >= arch::simd::registers::AVX_DOUBLES);
    assert(arch::simd::registers::AVX_DOUBLES >= arch::simd::registers::SSE_DOUBLES);
    assert(arch::simd::registers::SSE_DOUBLES == arch::simd::registers::NEON_DOUBLES);
    assert(arch::simd::registers::NEON_DOUBLES >= arch::simd::registers::SCALAR_DOUBLES);

    // Test prefetch distance hierarchy (Phase 3C: flattened namespaces)
    assert(memory::prefetch::DISTANCE_ULTRA_AGGRESSIVE >= memory::prefetch::DISTANCE_AGGRESSIVE);
    assert(memory::prefetch::DISTANCE_AGGRESSIVE >= memory::prefetch::DISTANCE_STANDARD);
    assert(memory::prefetch::DISTANCE_STANDARD >= memory::prefetch::DISTANCE_CONSERVATIVE);

    // Test prefetch platform hierarchy (Apple Silicon should be most aggressive)
    assert(memory::prefetch::platform::apple_silicon::SEQUENTIAL_PREFETCH_DISTANCE >=
           memory::prefetch::platform::intel::SEQUENTIAL_PREFETCH_DISTANCE);
    assert(memory::prefetch::platform::intel::SEQUENTIAL_PREFETCH_DISTANCE >=
           memory::prefetch::platform::amd::SEQUENTIAL_PREFETCH_DISTANCE);
    assert(memory::prefetch::platform::amd::SEQUENTIAL_PREFETCH_DISTANCE >=
           memory::prefetch::platform::arm::SEQUENTIAL_PREFETCH_DISTANCE);

    // Test allocation size hierarchy
    assert(memory::allocation::LARGE_POOL_SIZE >= memory::allocation::MEDIUM_POOL_SIZE);
    assert(memory::allocation::MEDIUM_POOL_SIZE >= memory::allocation::SMALL_POOL_SIZE);

    // Test memory bandwidth hierarchy (Phase 3C: flattened namespaces)
    assert(memory::access::BANDWIDTH_HBM_BURST_SIZE >= memory::access::BANDWIDTH_DDR5_BURST_SIZE);
    assert(memory::access::BANDWIDTH_DDR5_BURST_SIZE >= memory::access::BANDWIDTH_DDR4_BURST_SIZE);

    std::cout << "✓ Architectural constant relationships test passed" << std::endl;
}

void print_summary() {
    std::cout << "\n=== Architectural Constants Summary ===" << std::endl;

    std::cout << "SIMD Alignments:" << std::endl;
    std::cout << "  AVX-512: " << arch::simd::AVX512_ALIGNMENT << " bytes" << std::endl;
    std::cout << "  AVX/AVX2: " << arch::simd::AVX_ALIGNMENT << " bytes" << std::endl;
    std::cout << "  SSE: " << arch::simd::SSE_ALIGNMENT << " bytes" << std::endl;
    std::cout << "  ARM NEON: " << arch::simd::NEON_ALIGNMENT << " bytes" << std::endl;

    std::cout << "\nMatrix Block Sizes:" << std::endl;
    std::cout << "  L1 optimized: " << arch::simd::matrix::L1_BLOCK_SIZE << std::endl;
    std::cout << "  L2 optimized: " << arch::simd::matrix::L2_BLOCK_SIZE << std::endl;
    std::cout << "  L3 optimized: " << arch::simd::matrix::L3_BLOCK_SIZE << std::endl;
    std::cout << "  Step size: " << arch::simd::matrix::STEP_SIZE << std::endl;

    std::cout << "\nPrefetch Distances (elements):" << std::endl;
    std::cout << "  Apple Silicon: "
              << memory::prefetch::platform::apple_silicon::SEQUENTIAL_PREFETCH_DISTANCE
              << std::endl;
    std::cout << "  Intel: " << memory::prefetch::platform::intel::SEQUENTIAL_PREFETCH_DISTANCE
              << std::endl;
    std::cout << "  AMD: " << memory::prefetch::platform::amd::SEQUENTIAL_PREFETCH_DISTANCE
              << std::endl;
    std::cout << "  ARM: " << memory::prefetch::platform::arm::SEQUENTIAL_PREFETCH_DISTANCE
              << std::endl;

    std::cout << "\nMemory Access:" << std::endl;
    std::cout << "  Cache line size: " << memory::access::CACHE_LINE_SIZE_BYTES << " bytes"
              << std::endl;
    std::cout << "  Doubles per cache line: " << memory::access::DOUBLES_PER_CACHE_LINE
              << std::endl;
    std::cout << "  NUMA aware threshold: " << (memory::access::NUMA_AWARE_THRESHOLD / 1024 / 1024)
              << " MB" << std::endl;
}

int main() {
    std::cout << "=== Testing New Architectural Constants ===" << std::endl;
    std::cout << std::endl;

    try {
        test_simd_constants();
        std::cout << std::endl;

        test_memory_constants();
        std::cout << std::endl;

        test_constant_relationships();
        std::cout << std::endl;

        print_summary();
        std::cout << std::endl;

        std::cout << "🎉 ALL NEW ARCHITECTURAL CONSTANTS TESTS PASSED! 🎉" << std::endl;
        std::cout
            << "📊 SIMD alignment, matrix blocking, and memory prefetching constants verified! 📊"
            << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }

    return 0;
}

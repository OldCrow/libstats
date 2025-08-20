#include "../include/core/constants.h"

#include <cassert>
#include <iostream>

using namespace libstats::constants;

void test_simd_constants() {
    std::cout << "Testing SIMD architectural constants..." << std::endl;

    // Test SIMD alignment constants
    assert(simd::alignment::AVX512_ALIGNMENT == 64);
    assert(simd::alignment::AVX_ALIGNMENT == 32);
    assert(simd::alignment::SSE_ALIGNMENT == 16);
    assert(simd::alignment::NEON_ALIGNMENT == 16);
    assert(simd::alignment::CACHE_LINE_ALIGNMENT == 64);

    // Test matrix block sizes
    assert(simd::matrix::L1_BLOCK_SIZE == 64);
    assert(simd::matrix::L2_BLOCK_SIZE == 256);
    assert(simd::matrix::L3_BLOCK_SIZE == 1024);
    assert(simd::matrix::STEP_SIZE == 8);
    assert(simd::matrix::PANEL_WIDTH == 64);

    // Test SIMD register widths
    assert(simd::registers::AVX512_DOUBLES == 8);
    assert(simd::registers::AVX_DOUBLES == 4);
    assert(simd::registers::SSE_DOUBLES == 2);
    assert(simd::registers::NEON_DOUBLES == 2);
    assert(simd::registers::SCALAR_DOUBLES == 1);

    // Test unroll factors
    assert(simd::unroll::AVX512_UNROLL == 4);
    assert(simd::unroll::AVX_UNROLL == 2);
    assert(simd::unroll::SSE_UNROLL == 2);
    assert(simd::unroll::NEON_UNROLL == 2);
    assert(simd::unroll::SCALAR_UNROLL == 1);

    std::cout << "✓ SIMD architectural constants test passed" << std::endl;
}

void test_memory_constants() {
    std::cout << "Testing memory access and prefetching constants..." << std::endl;

    // Test prefetch distance constants
    assert(memory::prefetch::distance::CONSERVATIVE == 2);
    assert(memory::prefetch::distance::STANDARD == 4);
    assert(memory::prefetch::distance::AGGRESSIVE == 8);
    assert(memory::prefetch::distance::ULTRA_AGGRESSIVE == 16);

    // Test platform-specific prefetch distances
    assert(memory::prefetch::platform::apple_silicon::SEQUENTIAL_PREFETCH_DISTANCE == 256);
    assert(memory::prefetch::platform::intel::SEQUENTIAL_PREFETCH_DISTANCE == 192);
    assert(memory::prefetch::platform::amd::SEQUENTIAL_PREFETCH_DISTANCE == 128);
    assert(memory::prefetch::platform::arm::SEQUENTIAL_PREFETCH_DISTANCE == 64);

    // Test prefetch strategy constants
    assert(memory::prefetch::strategy::SEQUENTIAL_MULTIPLIER == 2.0);
    assert(memory::prefetch::strategy::RANDOM_MULTIPLIER == 0.5);
    assert(memory::prefetch::strategy::STRIDED_MULTIPLIER == 1.5);
    assert(memory::prefetch::strategy::MIN_PREFETCH_SIZE == 32);
    assert(memory::prefetch::strategy::MAX_PREFETCH_DISTANCE == 1024);
    assert(memory::prefetch::strategy::PREFETCH_GRANULARITY == 8);

    // Test memory access constants
    assert(memory::access::CACHE_LINE_SIZE_BYTES == 64);
    assert(memory::access::DOUBLES_PER_CACHE_LINE == 8);
    assert(memory::access::CACHE_LINE_ALIGNMENT == 64);

    // Test bandwidth optimization
    assert(memory::access::bandwidth::DDR4_BURST_SIZE == 64);
    assert(memory::access::bandwidth::DDR5_BURST_SIZE == 128);
    assert(memory::access::bandwidth::HBM_BURST_SIZE == 256);
    assert(memory::access::bandwidth::TARGET_BANDWIDTH_UTILIZATION == 0.8);
    assert(memory::access::bandwidth::MAX_BANDWIDTH_UTILIZATION == 0.95);

    // Test memory layout constants
    assert(memory::access::layout::AOS_TO_SOA_THRESHOLD == 1000);
    assert(memory::access::layout::MEMORY_POOL_ALIGNMENT == 4096);
    assert(memory::access::layout::SMALL_ALLOCATION_THRESHOLD == 256);
    assert(memory::access::layout::LARGE_PAGE_THRESHOLD == 2097152);

    // Test NUMA constants
    assert(memory::access::numa::NUMA_AWARE_THRESHOLD == 1048576);
    assert(memory::access::numa::NUMA_LOCAL_THRESHOLD == 65536);
    assert(memory::access::numa::NUMA_MIGRATION_COST == 0.1);

    // Test memory allocation constants
    assert(memory::allocation::SMALL_POOL_SIZE == 4096);
    assert(memory::allocation::MEDIUM_POOL_SIZE == 65536);
    assert(memory::allocation::LARGE_POOL_SIZE == 1048576);
    assert(memory::allocation::MIN_ALLOCATION_ALIGNMENT == 8);
    assert(memory::allocation::SIMD_ALLOCATION_ALIGNMENT == 32);
    assert(memory::allocation::PAGE_ALLOCATION_ALIGNMENT == 4096);

    // Test growth factors
    assert(memory::allocation::growth::EXPONENTIAL_GROWTH_FACTOR == 1.5);
    assert(memory::allocation::growth::LINEAR_GROWTH_FACTOR == 1.2);
    assert(memory::allocation::growth::GROWTH_THRESHOLD == 1048576);

    std::cout << "✓ Memory access and prefetching constants test passed" << std::endl;
}

void test_constant_relationships() {
    std::cout << "Testing architectural constant relationships..." << std::endl;

    // Test alignment hierarchy
    assert(simd::alignment::AVX512_ALIGNMENT >= simd::alignment::AVX_ALIGNMENT);
    assert(simd::alignment::AVX_ALIGNMENT >= simd::alignment::SSE_ALIGNMENT);
    assert(simd::alignment::SSE_ALIGNMENT == simd::alignment::NEON_ALIGNMENT);

    // Test matrix block size hierarchy
    assert(simd::matrix::L3_BLOCK_SIZE >= simd::matrix::L2_BLOCK_SIZE);
    assert(simd::matrix::L2_BLOCK_SIZE >= simd::matrix::L1_BLOCK_SIZE);
    assert(simd::matrix::L1_BLOCK_SIZE >= simd::matrix::MIN_BLOCK_SIZE);
    assert(simd::matrix::MAX_BLOCK_SIZE >= simd::matrix::L3_BLOCK_SIZE);

    // Test SIMD register width hierarchy
    assert(simd::registers::AVX512_DOUBLES >= simd::registers::AVX_DOUBLES);
    assert(simd::registers::AVX_DOUBLES >= simd::registers::SSE_DOUBLES);
    assert(simd::registers::SSE_DOUBLES == simd::registers::NEON_DOUBLES);
    assert(simd::registers::NEON_DOUBLES >= simd::registers::SCALAR_DOUBLES);

    // Test prefetch distance hierarchy
    assert(memory::prefetch::distance::ULTRA_AGGRESSIVE >= memory::prefetch::distance::AGGRESSIVE);
    assert(memory::prefetch::distance::AGGRESSIVE >= memory::prefetch::distance::STANDARD);
    assert(memory::prefetch::distance::STANDARD >= memory::prefetch::distance::CONSERVATIVE);

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

    // Test memory bandwidth hierarchy
    assert(memory::access::bandwidth::HBM_BURST_SIZE >= memory::access::bandwidth::DDR5_BURST_SIZE);
    assert(memory::access::bandwidth::DDR5_BURST_SIZE >=
           memory::access::bandwidth::DDR4_BURST_SIZE);

    std::cout << "✓ Architectural constant relationships test passed" << std::endl;
}

void print_summary() {
    std::cout << "\n=== Architectural Constants Summary ===" << std::endl;

    std::cout << "SIMD Alignments:" << std::endl;
    std::cout << "  AVX-512: " << simd::alignment::AVX512_ALIGNMENT << " bytes" << std::endl;
    std::cout << "  AVX/AVX2: " << simd::alignment::AVX_ALIGNMENT << " bytes" << std::endl;
    std::cout << "  SSE: " << simd::alignment::SSE_ALIGNMENT << " bytes" << std::endl;
    std::cout << "  ARM NEON: " << simd::alignment::NEON_ALIGNMENT << " bytes" << std::endl;

    std::cout << "\nMatrix Block Sizes:" << std::endl;
    std::cout << "  L1 optimized: " << simd::matrix::L1_BLOCK_SIZE << std::endl;
    std::cout << "  L2 optimized: " << simd::matrix::L2_BLOCK_SIZE << std::endl;
    std::cout << "  L3 optimized: " << simd::matrix::L3_BLOCK_SIZE << std::endl;
    std::cout << "  Step size: " << simd::matrix::STEP_SIZE << std::endl;

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
    std::cout << "  NUMA aware threshold: "
              << (memory::access::numa::NUMA_AWARE_THRESHOLD / 1024 / 1024) << " MB" << std::endl;
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

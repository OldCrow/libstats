// Use focused header for CPU detection testing
#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_thresholds.h"

#include <iostream>

int main() {
    try {
        std::cout << "=== Comprehensive CPU Detection Test ===" << std::endl;

        // Get detected features
        const auto& features = stats::arch::get_features();

        std::cout << "\n=== CPU Identification ===" << std::endl;
        std::cout << "CPU Vendor: " << features.vendor << std::endl;
        if (!features.brand.empty()) {
            std::cout << "CPU Brand: " << features.brand << std::endl;
        }
        std::cout << "Family: " << features.family << ", Model: " << features.model
                  << ", Stepping: " << features.stepping << std::endl;

        std::cout << "\n=== Basic SIMD Features ===" << std::endl;
        std::cout << "  SSE2: " << (features.sse2 ? "Yes" : "No") << std::endl;
        std::cout << "  SSE3: " << (features.sse3 ? "Yes" : "No") << std::endl;
        std::cout << "  SSSE3: " << (features.ssse3 ? "Yes" : "No") << std::endl;
        std::cout << "  SSE4.1: " << (features.sse4_1 ? "Yes" : "No") << std::endl;
        std::cout << "  SSE4.2: " << (features.sse4_2 ? "Yes" : "No") << std::endl;
        std::cout << "  AVX: " << (features.avx ? "Yes" : "No") << std::endl;
        std::cout << "  AVX2: " << (features.avx2 ? "Yes" : "No") << std::endl;
        std::cout << "  FMA: " << (features.fma ? "Yes" : "No") << std::endl;

        std::cout << "\n=== AVX-512 Features ===" << std::endl;
        std::cout << "  AVX512F: " << (features.avx512f ? "Yes" : "No") << std::endl;
        std::cout << "  AVX512DQ: " << (features.avx512dq ? "Yes" : "No") << std::endl;
        std::cout << "  AVX512CD: " << (features.avx512cd ? "Yes" : "No") << std::endl;
        std::cout << "  AVX512BW: " << (features.avx512bw ? "Yes" : "No") << std::endl;
        std::cout << "  AVX512VL: " << (features.avx512vl ? "Yes" : "No") << std::endl;
        std::cout << "  AVX512VNNI: " << (features.avx512vnni ? "Yes" : "No") << std::endl;
        std::cout << "  AVX512BF16: " << (features.avx512bf16 ? "Yes" : "No") << std::endl;

        std::cout << "\n=== ARM Features ===" << std::endl;
        std::cout << "  NEON: " << (features.neon ? "Yes" : "No") << std::endl;
        std::cout << "  SVE: " << (features.sve ? "Yes" : "No") << std::endl;
        std::cout << "  SVE2: " << (features.sve2 ? "Yes" : "No") << std::endl;
        std::cout << "  Crypto: " << (features.crypto ? "Yes" : "No") << std::endl;
        std::cout << "  CRC32: " << (features.crc32 ? "Yes" : "No") << std::endl;

        std::cout << "\n=== Legacy Cache Information ===" << std::endl;
        std::cout << "  L1 Cache Size: " << features.l1_cache_size << " bytes" << std::endl;
        std::cout << "  L2 Cache Size: " << features.l2_cache_size << " bytes" << std::endl;
        std::cout << "  L3 Cache Size: " << features.l3_cache_size << " bytes" << std::endl;
        std::cout << "  Cache Line Size: " << features.cache_line_size << " bytes" << std::endl;

        std::cout << "\n=== Enhanced Cache Information ===" << std::endl;
        std::cout << "  L1 Data Cache: " << features.l1_data_cache.size << " bytes" << std::endl;
        std::cout << "  L1 Instruction Cache: " << features.l1_instruction_cache.size << " bytes"
                  << std::endl;
        std::cout << "  L2 Cache: " << features.l2_cache.size << " bytes" << std::endl;
        std::cout << "  L3 Cache: " << features.l3_cache.size << " bytes" << std::endl;

        std::cout << "\n=== CPU Topology ===" << std::endl;
        std::cout << "  Logical Cores: " << features.topology.logical_cores << std::endl;
        std::cout << "  Physical Cores: " << features.topology.physical_cores << std::endl;
        std::cout << "  CPU Packages: " << features.topology.packages << std::endl;
        std::cout << "  Threads per Core: " << features.topology.threads_per_core << std::endl;
        std::cout << "  Hyperthreading: " << (features.topology.hyperthreading ? "Yes" : "No")
                  << std::endl;

        std::cout << "\n=== Performance Monitoring ===" << std::endl;
        std::cout << "  Has Performance Counters: "
                  << (features.performance.has_perf_counters ? "Yes" : "No") << std::endl;
        std::cout << "  Has RDTSC: " << (features.performance.has_rdtsc ? "Yes" : "No")
                  << std::endl;
        std::cout << "  Has Invariant TSC: "
                  << (features.performance.has_invariant_tsc ? "Yes" : "No") << std::endl;
        std::cout << "  TSC Frequency: " << features.performance.tsc_frequency << " Hz"
                  << std::endl;

        std::cout << "\n=== Summary Information ===" << std::endl;
        std::cout << "Feature Summary: " << stats::arch::features_string() << std::endl;
        std::cout << "Best SIMD Level: " << stats::arch::best_simd_level() << std::endl;

        std::cout << "\n=== Optimal Configuration ===" << std::endl;
        std::cout << "  Double Vector Width: " << stats::arch::optimal_double_width() << std::endl;
        std::cout << "  Float Vector Width: " << stats::arch::optimal_float_width() << std::endl;
        std::cout << "  Memory Alignment: " << stats::arch::optimal_alignment() << " bytes"
                  << std::endl;

        std::cout << "\n=== CPU Generation Detection ===" << std::endl;
        std::cout << "  is_sandy_ivy_bridge(): "
                  << (stats::arch::is_sandy_ivy_bridge() ? "YES" : "NO") << std::endl;
        std::cout << "  is_haswell_broadwell(): "
                  << (stats::arch::is_haswell_broadwell() ? "YES" : "NO") << std::endl;
        std::cout << "  is_skylake_generation(): "
                  << (stats::arch::is_skylake_generation() ? "YES" : "NO") << std::endl;
        std::cout << "  is_kaby_coffee_lake(): "
                  << (stats::arch::is_kaby_coffee_lake() ? "YES" : "NO") << std::endl;
        std::cout << "  is_modern_intel(): " << (stats::arch::is_modern_intel() ? "YES" : "NO")
                  << std::endl;

        std::cout << "\n=== Adaptive Parallel Constants ===" << std::endl;
        std::cout << "  min_elements_for_parallel(): "
                  << stats::arch::parallel::detail::min_elements_for_parallel() << std::endl;
        std::cout << "  min_elements_for_distribution_parallel(): "
                  << stats::arch::parallel::detail::min_elements_for_distribution_parallel()
                  << std::endl;
        std::cout << "  min_elements_for_simple_distribution_parallel(): "
                  << stats::arch::parallel::detail::min_elements_for_simple_distribution_parallel()
                  << std::endl;
        std::cout << "  grain_size(): " << stats::arch::parallel::detail::grain_size() << std::endl;

        std::cout << "\n=== Direct Constants Comparison ===" << std::endl;
        std::cout << "  avx::MIN_ELEMENTS_FOR_PARALLEL: "
                  << stats::arch::parallel::avx::MIN_ELEMENTS_FOR_PARALLEL << std::endl;
        std::cout << "  avx::LEGACY_INTEL_MAX_GRAIN_SIZE: "
                  << stats::arch::parallel::avx::LEGACY_INTEL_MAX_GRAIN_SIZE << std::endl;
        std::cout << "  fallback::MIN_ELEMENTS_FOR_PARALLEL: "
                  << stats::arch::parallel::fallback::MIN_ELEMENTS_FOR_PARALLEL << std::endl;
        std::cout << "  Legacy constants (backward compat): "
                  << stats::arch::parallel::detail::min_elements_for_parallel() << std::endl;

        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "✓ All CPU detection tests completed successfully!" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

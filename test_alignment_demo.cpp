#include "include/common/platform_constants_fwd.h"
#include "include/platform/cpu_detection.h"
#include "include/platform/simd.h"

#include <iostream>

int main() {
    std::cout << "=== Alignment Functions Comparison ===\n\n";

    // Show CPU SIMD register alignment requirements
    std::cout << "CPU SIMD Register Alignment Requirements:\n";
    std::cout << "  cpu::optimal_alignment(): " << stats::arch::optimal_alignment() << " bytes\n";
    std::cout << "  └─ This is the MINIMUM alignment required for SIMD registers\n\n";

    // Show platform-optimized cache-aware alignment
    std::cout << "Platform-Optimized Cache-Aware Alignment:\n";
    std::cout << "  arch::get_optimal_alignment(): " << stats::arch::get_optimal_alignment()
              << " bytes\n";
    std::cout << "  └─ This considers BOTH SIMD requirements AND cache line optimization\n\n";

    // Show runtime vendor-aware cache line size
    std::cout << "Vendor-Aware Cache Line Size:\n";
    std::cout << "  arch::get_cache_line_size(): " << stats::arch::get_cache_line_size()
              << " bytes\n";
    std::cout << "  └─ This is the detected cache line size for this CPU vendor\n\n";

    // Show compile-time cache line constants
    std::cout << "Compile-Time Cache Line Constants:\n";
    std::cout << "  stats::simd::utils::CACHE_LINE_SIZE: " << stats::simd::utils::CACHE_LINE_SIZE
              << " bytes\n";
    std::cout << "  └─ This is determined at compile time based on target architecture\n\n";

    // Show SIMD constants
    std::cout << "SIMD Alignment Constants:\n";
    std::cout << "  stats::simd::utils::SIMD_ALIGNMENT: " << stats::simd::utils::SIMD_ALIGNMENT
              << " bytes\n";
    std::cout << "  └─ This uses the compile-time optimal alignment function\n\n";

    // Explain the differences
    std::cout << "=== Key Differences ===\n";
    std::cout << "1. cpu::optimal_alignment():\n";
    std::cout << "   - Pure SIMD register alignment requirement\n";
    std::cout << "   - Based solely on instruction set capabilities\n\n";

    std::cout << "2. arch::get_optimal_alignment():\n";
    std::cout << "   - Considers BOTH SIMD and cache performance\n";
    std::cout << "   - Uses vendor-specific cache line sizes when beneficial\n";
    std::cout << "   - Recommended for general memory allocation\n\n";

    std::cout << "3. arch::get_cache_line_size():\n";
    std::cout << "   - Pure cache line size for the detected CPU vendor\n";
    std::cout << "   - Useful for cache-conscious algorithm design\n\n";

    // Show platform info
    const auto& features = stats::arch::get_features();
    std::cout << "=== Current Platform ===\n";
    std::cout << "CPU: " << features.vendor;
    if (!features.brand.empty()) {
        std::cout << " (" << features.brand << ")";
    }
    std::cout << "\n";
    std::cout << "Best SIMD: " << stats::arch::best_simd_level() << "\n";

    return 0;
}

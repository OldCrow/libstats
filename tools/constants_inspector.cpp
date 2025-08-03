/**
 * @file test_adaptive_constants.cpp
 * @brief Test program to verify architecture-specific constant selection
 */

#include <iostream>
#include <iomanip>

#include "../include/libstats.h"
#include "../include/core/performance_dispatcher.h"

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    // _setmode(_fileno(stdout), _O_U8TEXT); // Removed for std::cout compatibility
#endif

    std::cout << "=== Architecture-Specific Constants Test ===" << std::endl;
    
    // Display detected CPU features
    const auto& features = libstats::cpu::get_features();
    std::cout << "\n--- CPU Features Detected ---" << std::endl;
    std::cout << "AVX-512: " << (features.avx512f ? "Yes" : "No") << std::endl;
    std::cout << "AVX2: " << (features.avx2 ? "Yes" : "No") << std::endl;
    std::cout << "AVX: " << (features.avx ? "Yes" : "No") << std::endl;
    std::cout << "SSE2: " << (features.sse2 ? "Yes" : "No") << std::endl;
    std::cout << "NEON: " << (features.neon ? "Yes" : "No") << std::endl;
    std::cout << "FMA: " << (features.fma ? "Yes" : "No") << std::endl;
    
    // Display cache information
    std::cout << "\n--- Cache Information ---" << std::endl;
    std::cout << "L1 Cache: " << features.l1_cache_size / 1024 << " KB" << std::endl;
    std::cout << "L2 Cache: " << features.l2_cache_size / 1024 << " KB" << std::endl;
    std::cout << "L3 Cache: " << features.l3_cache_size / 1024 << " KB" << std::endl;
    std::cout << "Cache Line Size: " << features.cache_line_size << " bytes" << std::endl;
    
    // Display selected architecture
    std::cout << "\n--- Selected Architecture ---" << std::endl;
    
    // Determine active architecture based on CPU features
    std::string active_arch;
    if (features.avx512f) {
        active_arch = "AVX-512";
    } else if (features.avx2) {
        active_arch = "AVX2";
    } else if (features.avx) {
        active_arch = "AVX";
    } else if (features.sse2) {
        active_arch = "SSE2";
    } else if (features.neon) {
        active_arch = "NEON";
    } else {
        active_arch = "Fallback";
    }
    
    std::cout << "Active Architecture: " << active_arch << std::endl;
    
    // TODO: Display SystemCapabilities information when API is fully implemented
    // std::cout << "\n--- System Capabilities (Phase 3) ---" << std::endl;
    // Enable once the correct API is available
    
    // Display adaptive constants
    std::cout << "\n--- Adaptive Parallel Constants ---" << std::endl;
    std::cout << "Min Elements for Parallel: " 
              << libstats::constants::parallel::adaptive::min_elements_for_parallel() << std::endl;
    std::cout << "Min Elements for Distribution Parallel: " 
              << libstats::constants::parallel::adaptive::min_elements_for_distribution_parallel() << std::endl;
    std::cout << "Min Elements for Simple Distribution Parallel: " 
              << libstats::constants::parallel::adaptive::min_elements_for_simple_distribution_parallel() << std::endl;
    std::cout << "Default Grain Size: " 
              << libstats::constants::parallel::adaptive::grain_size() << std::endl;
    std::cout << "Simple Operation Grain Size: " 
              << libstats::constants::parallel::adaptive::simple_operation_grain_size() << std::endl;
    std::cout << "Complex Operation Grain Size: " 
              << libstats::constants::parallel::adaptive::complex_operation_grain_size() << std::endl;
    std::cout << "Monte Carlo Grain Size: " 
              << libstats::constants::parallel::adaptive::monte_carlo_grain_size() << std::endl;
    std::cout << "Max Grain Size: " 
              << libstats::constants::parallel::adaptive::max_grain_size() << std::endl;
    
    // Display platform-specific constants
    std::cout << "\n--- Platform Constants ---" << std::endl;
    std::cout << "SIMD Block Size: " 
              << libstats::constants::platform::get_optimal_simd_block_size() << " doubles" << std::endl;
    std::cout << "Memory Alignment: " 
              << libstats::constants::platform::get_optimal_alignment() << " bytes" << std::endl;
    std::cout << "Min SIMD Size: " 
              << libstats::constants::platform::get_min_simd_size() << " elements" << std::endl;
    std::cout << "Optimal Grain Size (platform): " 
              << libstats::constants::platform::get_optimal_grain_size() << " elements" << std::endl;
    std::cout << "Fast Transcendental Support: " 
              << (libstats::constants::platform::supports_fast_transcendental() ? "Yes" : "No") << std::endl;
    
    // Display cache thresholds
    std::cout << "\n--- Cache Thresholds ---" << std::endl;
    auto cache_thresholds = libstats::constants::platform::get_cache_thresholds();
    std::cout << "L1 Optimal Size: " << cache_thresholds.l1_optimal_size << " doubles" << std::endl;
    std::cout << "L2 Optimal Size: " << cache_thresholds.l2_optimal_size << " doubles" << std::endl;
    std::cout << "L3 Optimal Size: " << cache_thresholds.l3_optimal_size << " doubles" << std::endl;
    std::cout << "Blocking Size: " << cache_thresholds.blocking_size << " doubles" << std::endl;
    
    // Test architecture-specific namespace access
    std::cout << "\n--- Architecture-Specific Constants Comparison ---" << std::endl;
    std::cout << std::left << std::setw(25) << "Architecture" 
              << std::setw(15) << "Min Parallel" 
              << std::setw(15) << "Grain Size" 
              << std::setw(15) << "Simple Grain" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    std::cout << std::setw(25) << "SSE" 
              << std::setw(15) << libstats::constants::parallel::sse::MIN_ELEMENTS_FOR_PARALLEL
              << std::setw(15) << libstats::constants::parallel::sse::DEFAULT_GRAIN_SIZE
              << std::setw(15) << libstats::constants::parallel::sse::SIMPLE_OPERATION_GRAIN_SIZE << std::endl;
              
    std::cout << std::setw(25) << "AVX" 
              << std::setw(15) << libstats::constants::parallel::avx::MIN_ELEMENTS_FOR_PARALLEL
              << std::setw(15) << libstats::constants::parallel::avx::DEFAULT_GRAIN_SIZE
              << std::setw(15) << libstats::constants::parallel::avx::SIMPLE_OPERATION_GRAIN_SIZE << std::endl;
              
    std::cout << std::setw(25) << "AVX2" 
              << std::setw(15) << libstats::constants::parallel::avx2::MIN_ELEMENTS_FOR_PARALLEL
              << std::setw(15) << libstats::constants::parallel::avx2::DEFAULT_GRAIN_SIZE
              << std::setw(15) << libstats::constants::parallel::avx2::SIMPLE_OPERATION_GRAIN_SIZE << std::endl;
              
    std::cout << std::setw(25) << "AVX-512" 
              << std::setw(15) << libstats::constants::parallel::avx512::MIN_ELEMENTS_FOR_PARALLEL
              << std::setw(15) << libstats::constants::parallel::avx512::DEFAULT_GRAIN_SIZE
              << std::setw(15) << libstats::constants::parallel::avx512::SIMPLE_OPERATION_GRAIN_SIZE << std::endl;
              
    std::cout << std::setw(25) << "NEON" 
              << std::setw(15) << libstats::constants::parallel::neon::MIN_ELEMENTS_FOR_PARALLEL
              << std::setw(15) << libstats::constants::parallel::neon::DEFAULT_GRAIN_SIZE
              << std::setw(15) << libstats::constants::parallel::neon::SIMPLE_OPERATION_GRAIN_SIZE << std::endl;
              
    std::cout << std::setw(25) << "Fallback" 
              << std::setw(15) << libstats::constants::parallel::fallback::MIN_ELEMENTS_FOR_PARALLEL
              << std::setw(15) << libstats::constants::parallel::fallback::DEFAULT_GRAIN_SIZE
              << std::setw(15) << libstats::constants::parallel::fallback::SIMPLE_OPERATION_GRAIN_SIZE << std::endl;
    
    std::cout << std::string(70, '-') << std::endl;
    std::cout << std::setw(25) << "SELECTED (adaptive)" 
              << std::setw(15) << libstats::constants::parallel::adaptive::min_elements_for_parallel()
              << std::setw(15) << libstats::constants::parallel::adaptive::grain_size()
              << std::setw(15) << libstats::constants::parallel::adaptive::simple_operation_grain_size() << std::endl;
    
    return 0;
}

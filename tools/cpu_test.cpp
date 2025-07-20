#include <iostream>
#include <string>
#include <vector>
#include <exception>

// Include CPU detection for runtime capability testing
#include "../include/cpu_detection.h"

/**
 * @brief Standalone CPU capability tester
 * 
 * This utility can be compiled and run independently to test
 * CPU capabilities without running the full library build.
 * Useful for CI/CD systems and cross-compilation validation.
 */

int main(int argc, char* argv[]) {
    try {
        std::cout << "LibStats CPU Capability Test\n";
        std::cout << "============================\n\n";
        
        // Get detected features
        const auto& features = libstats::cpu::get_features();
        
        // Basic CPU info
        std::cout << "CPU Information:\n";
        std::cout << "  Vendor: " << features.vendor << "\n";
        if (!features.brand.empty()) {
            std::cout << "  Brand: " << features.brand << "\n";
        }
        std::cout << "  Family: " << features.family << "\n";
        std::cout << "  Model: " << features.model << "\n";
        std::cout << "  Stepping: " << features.stepping << "\n";
        std::cout << "\n";
        
        // SIMD capabilities
        std::cout << "SIMD Capabilities:\n";
        std::cout << "  SSE2:     " << (features.sse2 ? "YES" : "NO") << "\n";
        std::cout << "  SSE3:     " << (features.sse3 ? "YES" : "NO") << "\n";
        std::cout << "  SSSE3:    " << (features.ssse3 ? "YES" : "NO") << "\n";
        std::cout << "  SSE4.1:   " << (features.sse4_1 ? "YES" : "NO") << "\n";
        std::cout << "  SSE4.2:   " << (features.sse4_2 ? "YES" : "NO") << "\n";
        std::cout << "  AVX:      " << (features.avx ? "YES" : "NO") << "\n";
        std::cout << "  FMA:      " << (features.fma ? "YES" : "NO") << "\n";
        std::cout << "  AVX2:     " << (features.avx2 ? "YES" : "NO") << "\n";
        std::cout << "  AVX-512F: " << (features.avx512f ? "YES" : "NO") << "\n";
        std::cout << "  NEON:     " << (features.neon ? "YES" : "NO") << "\n";
        std::cout << "\n";
        
        // Optimal SIMD settings
        std::cout << "Optimal SIMD Configuration:\n";
        std::cout << "  Best Level: " << libstats::cpu::best_simd_level() << "\n";
        std::cout << "  Double Width: " << libstats::cpu::optimal_double_width() << " elements\n";
        std::cout << "  Float Width: " << libstats::cpu::optimal_float_width() << " elements\n";
        std::cout << "  Alignment: " << libstats::cpu::optimal_alignment() << " bytes\n";
        std::cout << "\n";
        
        // Cache information
        std::cout << "Cache Information:\n";
        if (auto l1d = libstats::cpu::get_l1_data_cache()) {
            std::cout << "  L1 Data: " << (l1d->size / 1024) << " KB\n";
        }
        if (auto l1i = libstats::cpu::get_l1_instruction_cache()) {
            std::cout << "  L1 Instruction: " << (l1i->size / 1024) << " KB\n";
        }
        if (auto l2 = libstats::cpu::get_l2_cache()) {
            std::cout << "  L2: " << (l2->size / 1024) << " KB\n";
        }
        if (auto l3 = libstats::cpu::get_l3_cache()) {
            std::cout << "  L3: " << (l3->size / 1024) << " KB\n";
        }
        std::cout << "\n";
        
        // CPU topology
        auto topology = libstats::cpu::get_topology();
        std::cout << "CPU Topology:\n";
        std::cout << "  Logical Cores: " << topology.logical_cores << "\n";
        std::cout << "  Physical Cores: " << topology.physical_cores << "\n";
        std::cout << "  Hyperthreading: " << (topology.hyperthreading ? "YES" : "NO") << "\n";
        std::cout << "\n";
        
        // Feature consistency validation
        bool consistent = libstats::cpu::validate_feature_consistency();
        std::cout << "Feature Consistency: " << (consistent ? "VALID" : "INVALID") << "\n";
        if (!consistent) {
            std::cout << "WARNING: Detected features are inconsistent!\n";
            return 1;
        }
        
        // If user provided specific tests
        if (argc > 1) {
            std::cout << "\nSpecific Feature Tests:\n";
            for (int i = 1; i < argc; ++i) {
                std::string test = argv[i];
                bool supported = false;
                
                if (test == "sse2") supported = libstats::cpu::supports_sse2();
                else if (test == "sse4.1") supported = libstats::cpu::supports_sse4_1();
                else if (test == "avx") supported = libstats::cpu::supports_avx();
                else if (test == "avx2") supported = libstats::cpu::supports_avx2();
                else if (test == "avx512") supported = libstats::cpu::supports_avx512();
                else if (test == "neon") supported = libstats::cpu::supports_neon();
                else {
                    std::cout << "  " << test << ": UNKNOWN TEST\n";
                    continue;
                }
                
                std::cout << "  " << test << ": " << (supported ? "SUPPORTED" : "NOT SUPPORTED") << "\n";
            }
        }
        
        std::cout << "\nCPU capability test completed successfully.\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}

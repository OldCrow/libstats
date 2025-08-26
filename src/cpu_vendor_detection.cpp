#include "../include/platform/cpu_detection.h"
#include "../include/platform/cpu_vendor_constants.h"

namespace stats {
namespace arch {
namespace cpu {

bool is_intel_cpu() noexcept {
    const auto& features = get_features();
    // Check for "GenuineIntel" vendor string
    return features.vendor == "GenuineIntel" || features.vendor == "Intel" ||
           (features.brand.find("Intel") != std::string::npos);
}

bool is_amd_cpu() noexcept {
    const auto& features = get_features();
    // Check for "AuthenticAMD" vendor string
    return features.vendor == "AuthenticAMD" || features.vendor == "AMD" ||
           (features.brand.find("AMD") != std::string::npos) ||
           (features.brand.find("Ryzen") != std::string::npos);
}

bool is_arm_cpu() noexcept {
    const auto& features = get_features();
    // Check for ARM indicators
    return features.vendor == "ARM" || features.vendor == "Apple" ||  // Apple Silicon is ARM-based
           features.neon ||                                           // NEON is ARM-specific
           (features.brand.find("ARM") != std::string::npos) ||
           (features.brand.find("Cortex") != std::string::npos);
}

bool is_apple_silicon() noexcept {
    const auto& features = get_features();
    // Check for Apple Silicon specifically
    return features.vendor == "Apple" || (features.brand.find("Apple") != std::string::npos) ||
           (features.brand.find("M1") != std::string::npos) ||
           (features.brand.find("M2") != std::string::npos) ||
           (features.brand.find("M3") != std::string::npos) ||
           (features.brand.find("M4") != std::string::npos);
}

std::size_t get_vendor_cache_line_size() noexcept {
    if (is_apple_silicon()) {
        return apple_silicon::CACHE_LINE_SIZE;  // 128 bytes
    } else if (is_intel_cpu()) {
        return intel::CACHE_LINE_SIZE;  // 64 bytes
    } else if (is_amd_cpu()) {
        return amd::CACHE_LINE_SIZE;  // 64 bytes
    } else if (is_arm_cpu()) {
        return arm::CACHE_LINE_SIZE;  // 64 bytes
    }
    return 64;  // Default to 64 bytes
}

std::size_t get_vendor_l1_cache_size() noexcept {
    if (is_apple_silicon()) {
        // Return P-core size as the optimistic case
        return apple_silicon::L1_CACHE_SIZE_PERF;
    } else if (is_intel_cpu()) {
        return intel::L1_CACHE_SIZE;
    } else if (is_amd_cpu()) {
        return amd::L1_CACHE_SIZE;
    } else if (is_arm_cpu()) {
        return arm::L1_CACHE_SIZE;
    }
    return 32768;  // Default to 32KB
}

std::size_t get_vendor_optimal_simd_block() noexcept {
    if (is_apple_silicon()) {
        return apple_silicon::OPTIMAL_SIMD_BLOCK;  // 48
    } else if (is_intel_cpu()) {
        return intel::OPTIMAL_SIMD_BLOCK;  // 32
    } else if (is_amd_cpu()) {
        return amd::OPTIMAL_SIMD_BLOCK;  // 32
    } else if (is_arm_cpu()) {
        return arm::OPTIMAL_SIMD_BLOCK;  // 16
    }
    return 16;  // Conservative default
}

}  // namespace cpu
}  // namespace arch
}  // namespace stats

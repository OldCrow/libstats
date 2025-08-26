#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @file cpu_vendor_constants.h
 * @brief CPU vendor-specific tuning parameters and optimization constants
 *
 * Phase 3D: CPU vendor-specific namespace organization
 * This file centralizes all CPU vendor-specific constants under stats::arch::cpu::
 */

namespace stats {
namespace arch {
namespace cpu {

//==============================================================================
// Intel CPU-specific constants
//==============================================================================
namespace intel {

// Cache hierarchy characteristics
inline constexpr std::size_t L1_CACHE_SIZE = 32768;    // 32KB typical L1D
inline constexpr std::size_t L2_CACHE_SIZE = 262144;   // 256KB typical L2
inline constexpr std::size_t L3_CACHE_SIZE = 8388608;  // 8MB typical L3
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

// Prefetch tuning (Skylake+ optimized)
inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 192;
inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 48;
inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 96;
inline constexpr std::size_t PREFETCH_STRIDE = 4;

// SIMD optimization thresholds
inline constexpr std::size_t MIN_SIMD_SIZE = 8;
inline constexpr std::size_t OPTIMAL_SIMD_BLOCK = 32;

// Legacy Intel CPU-specific tuning (Sandy Bridge/Ivy Bridge)
namespace legacy {
// Sandy Bridge/Ivy Bridge specific (AVX without AVX2/FMA)
inline constexpr std::size_t REDUCE_GRAIN_SIZE_LARGE = 32768;
inline constexpr std::size_t REDUCE_GRAIN_SIZE_MEDIUM = 128;
inline constexpr std::size_t REDUCE_GRAIN_SIZE_SMALL = 64;

inline constexpr std::size_t TRANSFORM_COMPLEX_GRAIN_SIZE_LARGE = 32768;
inline constexpr std::size_t TRANSFORM_COMPLEX_GRAIN_SIZE_MEDIUM = 16384;
inline constexpr std::size_t TRANSFORM_COMPLEX_GRAIN_SIZE_SMALL = 1024;

inline constexpr std::size_t COUNT_IF_GRAIN_SIZE_LARGE = 256;
inline constexpr std::size_t COUNT_IF_GRAIN_SIZE_MEDIUM = 1024;

inline constexpr std::size_t TRANSFORM_SIMPLE_GRAIN_SIZE_LARGE = 8;
inline constexpr std::size_t TRANSFORM_SIMPLE_GRAIN_SIZE_MEDIUM = 8192;

// Dataset size thresholds
inline constexpr std::size_t SMALL_DATASET_THRESHOLD = 10000;
inline constexpr std::size_t MEDIUM_DATASET_THRESHOLD = 100000;
inline constexpr std::size_t LARGE_DATASET_THRESHOLD = 1000000;

inline constexpr std::size_t MAX_GRAIN_SIZE = 32768;
}  // namespace legacy

// Modern Intel CPU tuning (Haswell+)
namespace modern {
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 512;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 1024;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 128;
inline constexpr std::size_t MAX_GRAIN_SIZE = 8192;
}  // namespace modern

// Intel-specific parallel thresholds
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1536;

}  // namespace intel

//==============================================================================
// AMD CPU-specific constants
//==============================================================================
namespace amd {

// Cache hierarchy characteristics (Zen architecture)
inline constexpr std::size_t L1_CACHE_SIZE = 32768;     // 32KB L1D
inline constexpr std::size_t L2_CACHE_SIZE = 524288;    // 512KB L2
inline constexpr std::size_t L3_CACHE_SIZE = 33554432;  // 32MB L3 (CCX shared)
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

// Prefetch tuning (Zen+ optimized)
inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 128;
inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 32;
inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 64;
inline constexpr std::size_t PREFETCH_STRIDE = 4;

// SIMD optimization thresholds
inline constexpr std::size_t MIN_SIMD_SIZE = 8;
inline constexpr std::size_t OPTIMAL_SIMD_BLOCK = 32;

// Ryzen-specific optimizations
namespace ryzen {
// Zen/Zen+ optimizations
inline constexpr std::size_t CCX_SIZE = 4;           // Cores per CCX
inline constexpr std::size_t CROSS_CCX_PENALTY = 2;  // Latency multiplier

// Zen 2/3 optimizations
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 512;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;
inline constexpr std::size_t MAX_GRAIN_SIZE = 4096;

// Infinity Fabric optimizations
inline constexpr std::size_t INFINITY_FABRIC_OPTIMAL_SIZE = 64;
inline constexpr bool PREFER_LOCAL_MEMORY = true;
}  // namespace ryzen

// AMD-specific parallel thresholds
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 4096;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 2048;

}  // namespace amd

//==============================================================================
// ARM CPU-specific constants
//==============================================================================
namespace arm {

// Cache hierarchy characteristics (Generic ARM64)
inline constexpr std::size_t L1_CACHE_SIZE = 32768;    // 32KB typical L1D
inline constexpr std::size_t L2_CACHE_SIZE = 262144;   // 256KB typical L2
inline constexpr std::size_t L3_CACHE_SIZE = 2097152;  // 2MB typical L3
inline constexpr std::size_t CACHE_LINE_SIZE = 64;

// Prefetch tuning (Generic ARM)
inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 64;
inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 16;
inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 32;
inline constexpr std::size_t PREFETCH_STRIDE = 2;

// SIMD optimization thresholds (NEON)
inline constexpr std::size_t MIN_SIMD_SIZE = 4;
inline constexpr std::size_t OPTIMAL_SIMD_BLOCK = 16;

// ARM-specific grain sizes
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 128;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 64;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 256;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 48;
inline constexpr std::size_t MAX_GRAIN_SIZE = 2048;

// ARM-specific parallel thresholds
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 1536;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 1024;

// Cortex-specific tuning
namespace cortex {
// Cortex-A76/A78 optimizations
inline constexpr std::size_t OUT_OF_ORDER_WINDOW = 128;
inline constexpr std::size_t BRANCH_PREDICTOR_SIZE = 8192;

// Cortex-X series optimizations
inline constexpr std::size_t X_SERIES_L2_SIZE = 1048576;  // 1MB L2
inline constexpr std::size_t X_SERIES_PREFETCH_DEPTH = 8;
}  // namespace cortex

}  // namespace arm

//==============================================================================
// Apple Silicon-specific constants
//==============================================================================
namespace apple_silicon {

// Cache hierarchy characteristics (M-series)
inline constexpr std::size_t L1_CACHE_SIZE_PERF = 196608;    // 192KB L1D (P-cores)
inline constexpr std::size_t L1_CACHE_SIZE_EFF = 131072;     // 128KB L1D (E-cores)
inline constexpr std::size_t L2_CACHE_SIZE_PERF = 12582912;  // 12MB L2 (P-cores)
inline constexpr std::size_t L2_CACHE_SIZE_EFF = 4194304;    // 4MB L2 (E-cores)
inline constexpr std::size_t CACHE_LINE_SIZE = 128;          // Apple uses 128-byte cache lines

// Prefetch tuning (M-series optimized)
inline constexpr std::size_t SEQUENTIAL_PREFETCH_DISTANCE = 256;
inline constexpr std::size_t RANDOM_PREFETCH_DISTANCE = 64;
inline constexpr std::size_t MATRIX_PREFETCH_DISTANCE = 128;
inline constexpr std::size_t PREFETCH_STRIDE = 8;

// SIMD optimization thresholds (NEON with Apple enhancements)
inline constexpr std::size_t MIN_SIMD_SIZE = 4;
inline constexpr std::size_t OPTIMAL_SIMD_BLOCK = 48;
inline constexpr std::size_t AGGRESSIVE_SIMD_THRESHOLD = 6;

// M-series specific optimizations
namespace m_series {
// M1/M2/M3 unified memory architecture
inline constexpr bool UNIFIED_MEMORY = true;
inline constexpr std::size_t MEMORY_BANDWIDTH_GBPS = 400;  // Up to 400GB/s

// Performance/Efficiency core scheduling
inline constexpr std::size_t P_CORES = 8;                     // Performance cores (varies by model)
inline constexpr std::size_t E_CORES = 4;                     // Efficiency cores (varies by model)
inline constexpr std::size_t PREFER_P_CORE_THRESHOLD = 1024;  // Use P-cores above this

// Neural Engine integration possibilities
inline constexpr bool HAS_NEURAL_ENGINE = true;
inline constexpr std::size_t NEURAL_ENGINE_OPS_PER_SEC = 15800000000;  // 15.8 TOPS

// AMX (Apple Matrix Extension) support
inline constexpr bool HAS_AMX = true;
inline constexpr std::size_t AMX_TILE_SIZE = 64;  // 64x64 tiles
}  // namespace m_series

// Apple Silicon grain sizes (optimized for wide execution)
inline constexpr std::size_t DEFAULT_GRAIN_SIZE = 256;
inline constexpr std::size_t SIMPLE_OPERATION_GRAIN_SIZE = 128;
inline constexpr std::size_t COMPLEX_OPERATION_GRAIN_SIZE = 512;
inline constexpr std::size_t MONTE_CARLO_GRAIN_SIZE = 64;
inline constexpr std::size_t MAX_GRAIN_SIZE = 4096;

// Apple Silicon parallel thresholds (excellent thread creation)
inline constexpr std::size_t MIN_ELEMENTS_FOR_PARALLEL = 1024;
inline constexpr std::size_t MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 512;

// Quality of Service (QoS) hints
namespace qos {
inline constexpr uint8_t USER_INTERACTIVE = 0x21;  // Highest priority
inline constexpr uint8_t USER_INITIATED = 0x19;    // High priority
inline constexpr uint8_t DEFAULT = 0x15;           // Default priority
inline constexpr uint8_t UTILITY = 0x11;           // Low priority
inline constexpr uint8_t BACKGROUND = 0x09;        // Lowest priority
}  // namespace qos

}  // namespace apple_silicon

//==============================================================================
// CPU vendor detection helpers
//==============================================================================

/**
 * @brief Determine if we're on Intel CPU
 */
bool is_intel_cpu() noexcept;

/**
 * @brief Determine if we're on AMD CPU
 */
bool is_amd_cpu() noexcept;

/**
 * @brief Determine if we're on ARM CPU
 */
bool is_arm_cpu() noexcept;

/**
 * @brief Determine if we're on Apple Silicon
 */
bool is_apple_silicon() noexcept;

/**
 * @brief Get vendor-specific cache line size
 */
std::size_t get_vendor_cache_line_size() noexcept;

/**
 * @brief Get vendor-specific L1 cache size
 */
std::size_t get_vendor_l1_cache_size() noexcept;

/**
 * @brief Get vendor-specific optimal SIMD block size
 */
std::size_t get_vendor_optimal_simd_block() noexcept;

}  // namespace cpu
}  // namespace arch
}  // namespace stats

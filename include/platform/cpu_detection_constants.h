#pragma once

/**
 * @file platform/cpu_detection_constants.h
 * @brief Constants specific to CPU feature detection and CPUID operations
 *
 * This header contains all the constants used in CPU detection, including
 * CPUID leaf numbers, bit masks, and architectural constants.
 */

#include <cstdint>

namespace libstats {
namespace constants {
namespace cpu_detection {

// CPUID leaf numbers (EAX values)
namespace cpuid_leaf {
inline constexpr uint32_t VENDOR_STRING = 0x0;
inline constexpr uint32_t BASIC_INFO = 0x1;
inline constexpr uint32_t CACHE_DESCRIPTORS = 0x2;
inline constexpr uint32_t SERIAL_NUMBER = 0x3;
inline constexpr uint32_t CACHE_PARAMETERS = 0x4;
inline constexpr uint32_t MONITOR_MWAIT = 0x5;
inline constexpr uint32_t THERMAL_POWER = 0x6;
inline constexpr uint32_t EXTENDED_FEATURES = 0x7;
inline constexpr uint32_t PERFORMANCE_MONITORING = 0xA;
inline constexpr uint32_t EXTENDED_TOPOLOGY = 0xB;
inline constexpr uint32_t EXTENDED_STATE = 0xD;
inline constexpr uint32_t EXTENDED_FUNCTION_BASE = 0x80000000;
inline constexpr uint32_t EXTENDED_PROCESSOR_INFO = 0x80000001;
inline constexpr uint32_t EXTENDED_BRAND_STRING_1 = 0x80000002;
inline constexpr uint32_t EXTENDED_BRAND_STRING_2 = 0x80000003;
inline constexpr uint32_t EXTENDED_BRAND_STRING_3 = 0x80000004;
inline constexpr uint32_t EXTENDED_ADVANCED_POWER = 0x80000007;
}  // namespace cpuid_leaf

// CPUID cache type values
namespace cache_type {
inline constexpr uint32_t NO_MORE_CACHES = 0;
inline constexpr uint32_t DATA_CACHE = 1;
inline constexpr uint32_t INSTRUCTION_CACHE = 2;
inline constexpr uint32_t UNIFIED_CACHE = 3;
}  // namespace cache_type

// CPUID cache level values
namespace cache_level {
inline constexpr uint32_t L1 = 1;
inline constexpr uint32_t L2 = 2;
inline constexpr uint32_t L3 = 3;
}  // namespace cache_level

// CPUID topology level types
namespace topology_level {
inline constexpr uint32_t INVALID = 0;
inline constexpr uint32_t SMT = 1;   // Thread level
inline constexpr uint32_t CORE = 2;  // Core level
}  // namespace topology_level

// Bit positions for feature detection
namespace feature_bits {
// EDX register bits from CPUID leaf 1
inline constexpr uint32_t RDTSC_BIT = 4;
inline constexpr uint32_t SSE2_BIT = 26;
inline constexpr uint32_t HTT_BIT = 28;  // Hyperthreading

// ECX register bits from CPUID leaf 1
inline constexpr uint32_t SSE3_BIT = 0;
inline constexpr uint32_t SSSE3_BIT = 9;
inline constexpr uint32_t FMA_BIT = 12;
inline constexpr uint32_t SSE4_1_BIT = 19;
inline constexpr uint32_t SSE41_BIT = 19;  // Alias for compatibility
inline constexpr uint32_t SSE4_2_BIT = 20;
inline constexpr uint32_t SSE42_BIT = 20;  // Alias for compatibility
inline constexpr uint32_t OSXSAVE_BIT = 27;
inline constexpr uint32_t AVX_BIT = 28;

// EBX register bits from CPUID leaf 7
inline constexpr uint32_t AVX2_BIT = 5;
inline constexpr uint32_t AVX512F_BIT = 16;
inline constexpr uint32_t AVX512DQ_BIT = 17;
inline constexpr uint32_t AVX512CD_BIT = 28;
inline constexpr uint32_t AVX512BW_BIT = 30;
inline constexpr uint32_t AVX512VL_BIT = 31;

// EDX register bits from CPUID leaf 0x80000007
inline constexpr uint32_t INVARIANT_TSC_BIT = 8;
}  // namespace feature_bits

// Bit masks for extracting values
namespace bit_masks {
inline constexpr uint32_t CACHE_TYPE_MASK = 0x1F;
inline constexpr uint32_t CACHE_LEVEL_SHIFT = 5;
inline constexpr uint32_t CACHE_LEVEL_MASK = 0x7;
inline constexpr uint32_t CACHE_LINE_SIZE_MASK = 0xFFF;
inline constexpr uint32_t CACHE_PARTITIONS_SHIFT = 12;
inline constexpr uint32_t CACHE_PARTITIONS_MASK = 0x3FF;
inline constexpr uint32_t CACHE_ASSOCIATIVITY_SHIFT = 22;
inline constexpr uint32_t CACHE_ASSOCIATIVITY_MASK = 0x3FF;
inline constexpr uint32_t LOGICAL_PROCESSORS_SHIFT = 16;
inline constexpr uint32_t LOGICAL_PROCESSORS_MASK = 0xFF;
inline constexpr uint32_t TOPOLOGY_LEVEL_SHIFT = 8;
inline constexpr uint32_t TOPOLOGY_LEVEL_MASK = 0xFF;
inline constexpr uint32_t TOPOLOGY_MASK_WIDTH_MASK = 0x1F;
inline constexpr uint32_t TOPOLOGY_PROCESSOR_COUNT_MASK = 0xFFFF;
inline constexpr uint32_t FAMILY_BASE_MASK = 0xF;
inline constexpr uint32_t FAMILY_BASE_SHIFT = 8;
inline constexpr uint32_t FAMILY_EXTENDED_MASK = 0xFF;
inline constexpr uint32_t FAMILY_EXTENDED_SHIFT = 20;
inline constexpr uint32_t MODEL_BASE_MASK = 0xF;
inline constexpr uint32_t MODEL_BASE_SHIFT = 4;
inline constexpr uint32_t MODEL_EXTENDED_MASK = 0xF0;
inline constexpr uint32_t MODEL_EXTENDED_SHIFT = 12;
inline constexpr uint32_t STEPPING_MASK = 0xF;
inline constexpr uint32_t PERFORMANCE_VERSION_MASK = 0xFF;
}  // namespace bit_masks

// XCR0 (Extended Control Register) masks
namespace xcr0 {
inline constexpr uint64_t X87_STATE = 0x1;
inline constexpr uint64_t SSE_STATE = 0x2;
inline constexpr uint64_t AVX_STATE = 0x4;
inline constexpr uint64_t AVX_REQUIRED = 0x6;  // Both SSE and AVX state
}  // namespace xcr0

// Add XCR0 mask to bit_masks namespace for consistency
namespace bit_masks {
inline constexpr uint64_t XCR0_SSE_AVX_MASK = 0x6;  // Bits 1 and 2 for SSE and AVX state
}  // namespace bit_masks

// Other constants
inline constexpr uint32_t MAX_CACHE_LEVELS = 32;
inline constexpr uint32_t VENDOR_STRING_LENGTH = 13;
inline constexpr uint32_t BRAND_STRING_LENGTH = 49;
inline constexpr uint32_t BRAND_STRING_SIZE = 49;  // Alias for compatibility
inline constexpr uint32_t VENDOR_REGISTER_SIZE = 4;
inline constexpr uint32_t BRAND_STRING_REGISTERS = 4;
inline constexpr uint32_t CPUID_REGISTER_SIZE = 4;  // Size of each CPUID register in bytes
inline constexpr uint32_t REGISTERS_PER_LEAF = 4;   // Number of registers returned per CPUID leaf

// Numeric constants for calculations
inline constexpr uint32_t INCREMENT_ONE = 1;
inline constexpr uint32_t DIVISOR_TWO = 2;

}  // namespace cpu_detection
}  // namespace constants
}  // namespace libstats

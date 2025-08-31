#pragma once

#include <chrono>
#include <cstddef>  // for size_t
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/**
 * @file cpu_detection.h
 * @brief Runtime CPU feature detection for SIMD capabilities
 *
 * This header provides runtime detection of CPU features to ensure that
 * SIMD instructions are only used when actually supported by the hardware,
 * not just when the compiler can generate them.
 *
 * This solves the problem where:
 * - Compiler capability != CPU capability
 * - CMake's check_cxx_source_compiles only tests compilation, not execution
 * - -march=native generates instructions for build machine, not target machine
 */

namespace stats {
namespace arch {

/**
 * @brief Cache hierarchy information
 */
struct CacheInfo {
    uint32_t size = 0;           ///< Cache size in bytes
    uint32_t line_size = 64;     ///< Cache line size in bytes
    uint32_t associativity = 0;  ///< Cache associativity (0 if unknown)
    uint32_t sets = 0;           ///< Number of cache sets
    bool is_unified = false;     ///< True if unified instruction/data cache
};

/**
 * @brief CPU topology information
 */
struct TopologyInfo {
    uint32_t logical_cores = 0;     ///< Number of logical CPU cores
    uint32_t physical_cores = 0;    ///< Number of physical CPU cores
    uint32_t packages = 0;          ///< Number of CPU packages/sockets
    uint32_t threads_per_core = 0;  ///< Threads per physical core (SMT)
    bool hyperthreading = false;    ///< Hyperthreading/SMT enabled
};

/**
 * @brief Performance monitoring capabilities
 */
struct PerformanceInfo {
    bool has_perf_counters = false;          ///< Hardware performance counters available
    bool has_rdtsc = false;                  ///< RDTSC instruction available
    bool has_invariant_tsc = false;          ///< TSC frequency is invariant
    uint64_t tsc_frequency = 0;              ///< TSC frequency in Hz (0 if unknown)
    std::vector<std::string> counter_names;  ///< Available performance counter names
};

/**
 * @brief Structure holding detected CPU features
 */
struct Features {
    // x86/x64 features
    bool sse2 = false;
    bool sse3 = false;
    bool ssse3 = false;
    bool sse4_1 = false;
    bool sse4_2 = false;
    bool avx = false;
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512dq = false;    ///< AVX-512 Doubleword and Quadword Instructions
    bool avx512cd = false;    ///< AVX-512 Conflict Detection Instructions
    bool avx512bw = false;    ///< AVX-512 Byte and Word Instructions
    bool avx512vl = false;    ///< AVX-512 Vector Length Extensions
    bool avx512vnni = false;  ///< AVX-512 Vector Neural Network Instructions
    bool avx512bf16 = false;  ///< AVX-512 BFLOAT16 Instructions

    // ARM features
    bool neon = false;
    bool sve = false;     ///< ARM Scalable Vector Extensions
    bool sve2 = false;    ///< ARM Scalable Vector Extensions 2
    bool crypto = false;  ///< ARM Cryptography Extensions
    bool crc32 = false;   ///< ARM CRC32 Instructions

    // CPU identification
    std::string vendor;
    std::string brand;
    uint32_t family = 0;
    uint32_t model = 0;
    uint32_t stepping = 0;

    // Enhanced cache information
    CacheInfo l1_data_cache;
    CacheInfo l1_instruction_cache;
    CacheInfo l2_cache;
    CacheInfo l3_cache;

    // CPU topology
    TopologyInfo topology;

    // Performance monitoring
    PerformanceInfo performance;

    // Legacy cache info for backwards compatibility
    uint32_t l1_cache_size = 0;
    uint32_t l2_cache_size = 0;
    uint32_t l3_cache_size = 0;
    uint32_t cache_line_size = 64;
};

/**
 * @brief Detect CPU features at runtime
 * @return Structure containing detected CPU capabilities
 */
Features detect_features();

/**
 * @brief Get the detected CPU features (cached after first call)
 * @return Reference to cached CPU features
 */
const Features& get_features();

/**
 * @brief Check if a specific SIMD instruction set is supported
 */
bool supports_sse2();
bool supports_sse4_1();
bool supports_avx();
bool supports_avx2();
bool supports_fma();
bool supports_avx512();
bool supports_neon();

/**
 * @brief Intel CPU generation detection functions
 * These functions identify specific Intel CPU generations for optimized constants
 */
bool is_sandy_ivy_bridge();    // Sandy Bridge (2011) / Ivy Bridge (2012) - AVX, no AVX2
bool is_haswell_broadwell();   // Haswell (2013) / Broadwell (2014) - AVX2, FMA
bool is_skylake_generation();  // Skylake (2015) and derivatives - improved AVX2
bool is_kaby_coffee_lake();    // Kaby Lake (2016) / Coffee Lake (2017-2018) - optimized Skylake
bool is_modern_intel();        // Ice Lake (2019+) and newer - AVX-512 or latest optimizations

/**
 * @brief Get a human-readable string of supported features
 * @return String describing detected CPU features
 */
std::string features_string();

/**
 * @brief Get the best available SIMD instruction set
 * @return String identifier of the highest supported SIMD level
 */
std::string best_simd_level();

/**
 * @brief Get the optimal vector width for double-precision operations
 * @return Number of doubles that fit in the best available SIMD register
 */
size_t optimal_double_width();

/**
 * @brief Get the optimal vector width for single-precision operations
 * @return Number of floats that fit in the best available SIMD register
 */
size_t optimal_float_width();

/**
 * @brief Get the optimal memory alignment for SIMD operations
 * @return Alignment in bytes for best SIMD performance
 */
size_t optimal_alignment();

/**
 * @brief Enhanced CPU feature detection functions
 */
bool supports_avx512dq();
bool supports_avx512bw();
bool supports_avx512vl();
bool supports_sve();
bool supports_sve2();

/**
 * @brief Cache information queries
 */
std::optional<CacheInfo> get_l1_data_cache();
std::optional<CacheInfo> get_l1_instruction_cache();
std::optional<CacheInfo> get_l2_cache();
std::optional<CacheInfo> get_l3_cache();

/**
 * @brief CPU topology queries
 */
TopologyInfo get_topology();
uint32_t get_logical_core_count();
uint32_t get_physical_core_count();
bool has_hyperthreading();

/**
 * @brief Performance monitoring utilities
 */
PerformanceInfo get_performance_info();
bool has_rdtsc();
bool has_invariant_tsc();
std::optional<uint64_t> get_tsc_frequency();

/**
 * @brief Performance measurement utilities
 */
struct TimingResult {
    uint64_t cycles = 0;
    std::chrono::nanoseconds duration{0};
    bool valid = false;
};

/**
 * @brief Measure execution time and cycles for a function
 * @param func Function to measure
 * @return Timing measurements
 */
template <typename Func>
TimingResult measure_performance(Func&& func);

// Forward declaration
uint64_t read_tsc();

// Template implementation
template <typename Func>
TimingResult measure_performance(Func&& func) {
    TimingResult result;

    if (!has_rdtsc()) {
        // Fallback to time-only measurement
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        result.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        result.valid = true;
        return result;
    }

    // Use TSC for cycle counting
    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t start_cycles = read_tsc();

    func();

    uint64_t end_cycles = read_tsc();
    auto end_time = std::chrono::high_resolution_clock::now();

    result.cycles = end_cycles - start_cycles;
    result.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    result.valid = true;

    return result;
}

/**
 * @brief Get current timestamp counter value
 * @return TSC value, or 0 if not supported
 */
uint64_t read_tsc();

/**
 * @brief Estimate CPU frequency using TSC
 * @param duration_ms Duration to measure in milliseconds
 * @return Estimated frequency in Hz, or 0 if measurement failed
 */
std::optional<uint64_t> estimate_cpu_frequency(uint32_t duration_ms = 100);

/**
 * @brief Get detailed CPU information string
 * @return Multi-line string with comprehensive CPU information
 */
std::string detailed_cpu_info();

/**
 * @brief Validate CPU feature consistency
 * @return True if detected features are consistent, false otherwise
 */
bool validate_feature_consistency();

}  // namespace arch
}  // namespace stats

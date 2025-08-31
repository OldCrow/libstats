// CRITICAL: Ensure CPU detection code uses NO advanced SIMD instructions
// This file detects CPU features and must not use the features it's detecting!
// Use compiler-specific approaches to disable SIMD in CPU detection code
#if defined(__clang__)
    // Clang: use function attributes to disable SIMD per function
    #define CPU_DETECTION_NO_SIMD                                                                  \
        __attribute__((                                                                            \
            target("no-avx512f,no-avx512cd,no-avx512bw,no-avx512dq,no-avx512vl,no-avx2,no-avx,no-" \
                   "sse4.2,no-sse4.1,no-ssse3,no-sse3")))
#elif defined(__GNUC__)
    // GCC: use pragma to disable SIMD globally for this file
    #pragma GCC push_options
    #pragma GCC target(                                                                            \
        "no-avx512f,no-avx512cd,no-avx512bw,no-avx512dq,no-avx512vl,no-avx2,no-avx,no-sse4.2,no-sse4.1,no-ssse3,no-sse3")
    #define CPU_DETECTION_NO_SIMD
#else
    #define CPU_DETECTION_NO_SIMD
#endif

#include "../include/platform/cpu_detection.h"

#include "../include/platform/platform_constants.h"
#include "core/mathematical_constants.h"

#include <atomic>
#include <chrono>
#include <cstring>  // For memcpy
#include <optional>
#include <string>
#include <thread>
#include <version>  // for __cpp_lib_atomic...

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #if defined(_MSC_VER)
        #include <intrin.h>  // MSVC intrinsics
    #else
        #include <cpuid.h>
    #endif
    #include <immintrin.h>
    #define LIBSTATS_X86_FAMILY
    #if defined(__APPLE__)
        #include <sys/sysctl.h>
    #elif defined(_WIN32)
    // clang-format off
        #include <windows.h>     // Must be included before sysinfoapi.h
        #include <sysinfoapi.h>
    // clang-format on
    #endif
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define LIBSTATS_ARM64_FAMILY
    #if defined(__APPLE__)
        #include <sys/sysctl.h>
    #elif defined(__linux__)
        #include <asm/hwcap.h>
        #include <sys/auxv.h>
    #endif
#endif

namespace stats {
namespace arch {

namespace {
// Thread-safe singleton for cached features with C++20 atomic enhancements
struct FeaturesSingleton {
    std::atomic<Features*> ptr{nullptr};
    std::atomic<bool> initializing{false};

    ~FeaturesSingleton() {
        Features* features = ptr.load(std::memory_order_relaxed);
        delete features;
    }

    const Features& get() {
        Features* features = ptr.load(std::memory_order_acquire);
        if (features == nullptr) {
            // C++20 atomic wait/notify pattern for efficient initialization
            bool expected = false;
            if (initializing.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                // We won the race - initialize
                Features* new_features = new Features(detect_features());
                ptr.store(new_features, std::memory_order_release);
                initializing.store(false, std::memory_order_release);

// C++20 feature: notify waiting threads
#if __cplusplus >= 202002L && defined(__cpp_lib_atomic_wait)
                ptr.notify_all();
#endif

                features = new_features;
            } else {
// Another thread is initializing - wait for completion
#if __cplusplus >= 202002L && defined(__cpp_lib_atomic_wait)
                ptr.wait(nullptr, std::memory_order_acquire);
                features = ptr.load(std::memory_order_acquire);
#else
                // Fallback: busy wait with exponential backoff
                int backoff = 1;
                while ((features = ptr.load(std::memory_order_acquire)) == nullptr) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(backoff));
                    backoff =
                        std::min(backoff * detail::TWO_INT,
                                 static_cast<int>(
                                     arch::simd::CPU_MAX_BACKOFF_NANOSECONDS));  // Max 1μs backoff
                }
#endif
            }
        }
        return *features;
    }
};

static FeaturesSingleton g_features_manager;

#ifdef LIBSTATS_X86_FAMILY

    #if defined(__APPLE__)
// Forward declaration for macOS-specific function used in x86 path
void detect_macos_topology(Features& features);
    #endif

/**
 * @brief Execute CPUID instruction safely
 */
void safe_cpuid(uint32_t eax, uint32_t ecx, uint32_t& out_eax, uint32_t& out_ebx, uint32_t& out_ecx,
                uint32_t& out_edx) {
    #if defined(__GNUC__) || defined(__clang__)
    __cpuid_count(eax, ecx, out_eax, out_ebx, out_ecx, out_edx);
    #elif defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, eax, ecx);
    out_eax = regs[0];
    out_ebx = regs[1];
    out_ecx = regs[2];
    out_edx = regs[3];
    #else
    // Fallback for unknown compilers
    out_eax = out_ebx = out_ecx = out_edx = 0;
    #endif
}

    #if !defined(__APPLE__)
/**
 * @brief Detect cache information using CPUID
 */
void detect_cache_info(Features& features) {
    uint32_t eax, ebx, ecx, edx;

    // Try deterministic cache parameters (CPUID leaf 4)
    safe_cpuid(0, 0, eax, ebx, ecx, edx);
    if (eax >= 4) {
        for (uint32_t i = 0; i < 32; ++i) {
            safe_cpuid(4, i, eax, ebx, ecx, edx);
            uint32_t cache_type = eax & 0x1F;

            if (cache_type == 0)
                break;  // No more cache levels

            uint32_t cache_level = (eax >> 5) & 0x7;
            uint32_t line_size = (ebx & 0xFFF) + detail::ONE_INT;
            uint32_t partitions = ((ebx >> 12) & 0x3FF) + detail::ONE_INT;
            uint32_t associativity = ((ebx >> 22) & 0x3FF) + detail::ONE_INT;
            uint32_t sets = ecx + detail::ONE_INT;

            uint32_t cache_size = line_size * partitions * associativity * sets;

            if (cache_level == 1) {
                if (cache_type == 1) {  // Data cache
                    features.l1_data_cache.size = cache_size;
                    features.l1_data_cache.line_size = line_size;
                    features.l1_data_cache.associativity = associativity;
                    features.l1_data_cache.sets = sets;
                    features.l1_cache_size = cache_size;  // Legacy
                } else if (cache_type == 2) {             // Instruction cache
                    features.l1_instruction_cache.size = cache_size;
                    features.l1_instruction_cache.line_size = line_size;
                    features.l1_instruction_cache.associativity = associativity;
                    features.l1_instruction_cache.sets = sets;
                }
            } else if (cache_level == 2) {
                features.l2_cache.size = cache_size;
                features.l2_cache.line_size = line_size;
                features.l2_cache.associativity = associativity;
                features.l2_cache.sets = sets;
                features.l2_cache.is_unified = (cache_type == 3);
                features.l2_cache_size = cache_size;  // Legacy
            } else if (cache_level == 3) {
                features.l3_cache.size = cache_size;
                features.l3_cache.line_size = line_size;
                features.l3_cache.associativity = associativity;
                features.l3_cache.sets = sets;
                features.l3_cache.is_unified = (cache_type == 3);
                features.l3_cache_size = cache_size;  // Legacy
            }
        }
    }

    // Set default cache line size if not detected
    if (features.l1_data_cache.line_size == 0) {
        features.cache_line_size = arch::simd::CPU_DEFAULT_CACHE_LINE_SIZE;  // Common default
    } else {
        features.cache_line_size = features.l1_data_cache.line_size;
    }
}

/**
 * @brief Detect CPU topology using CPUID
 */
void detect_topology_info(Features& features) {
    uint32_t eax, ebx, ecx, edx;

    // Get logical processor count from CPUID leaf 1
    safe_cpuid(1, 0, eax, ebx, ecx, edx);
    uint32_t logical_processors = (ebx >> 16) & 0xFF;

    features.topology.logical_cores = logical_processors;

    // Try extended topology enumeration (CPUID leaf 0xB)
    safe_cpuid(0, 0, eax, ebx, ecx, edx);
    if (eax >= 0xB) {
        [[maybe_unused]] uint32_t smt_mask_width = 0;
        [[maybe_unused]] uint32_t core_mask_width = 0;

        // Level 0: SMT level
        safe_cpuid(0xB, 0, eax, ebx, ecx, edx);
        if (((ecx >> 8) & 0xFF) == 1) {  // SMT level
            smt_mask_width = eax & 0x1F;
            features.topology.threads_per_core = ebx & 0xFFFF;
        }

        // Level 1: Core level
        safe_cpuid(0xB, 1, eax, ebx, ecx, edx);
        if (((ecx >> 8) & 0xFF) == 2) {  // Core level
            core_mask_width = eax & 0x1F;
            uint32_t cores_per_package = ebx & 0xFFFF;
            features.topology.physical_cores = cores_per_package;
        }

        features.topology.packages = 1;  // Assume single package
        features.topology.hyperthreading = (features.topology.threads_per_core > 1);
    } else {
        // Fallback: use basic detection
        features.topology.physical_cores = logical_processors;
        features.topology.threads_per_core = detail::ONE_INT;
        features.topology.packages = detail::ONE_INT;
        features.topology.hyperthreading = false;

        // Check if hyperthreading is supported
        if (edx & (detail::ONE_INT << 28)) {  // HTT bit
            features.topology.hyperthreading = true;
            features.topology.physical_cores = logical_processors / detail::TWO_INT;
            features.topology.threads_per_core = detail::TWO_INT;
        }
    }
}
    #endif  // !defined(__APPLE__)

/**
 * @brief Estimate CPU frequency using TSC without circular dependency
 */
uint64_t estimate_tsc_frequency_internal(uint32_t duration_ms) {
    #if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
        #if defined(__GNUC__) || defined(__clang__)
    uint64_t start_tsc = __rdtsc();
    auto start_time = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

    uint64_t end_tsc = __rdtsc();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    uint64_t cycles = end_tsc - start_tsc;

    if (duration.count() > 0) {
        // Calculate frequency: cycles per nanosecond * conversion factor = Hz
        double freq = static_cast<double>(cycles) / static_cast<double>(duration.count()) *
                      arch::simd::CPU_NANOSECONDS_TO_HZ;
        return static_cast<uint64_t>(freq);
    }
        #elif defined(_MSC_VER)
    uint64_t start_tsc = __rdtsc();
    auto start_time = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

    uint64_t end_tsc = __rdtsc();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    uint64_t cycles = end_tsc - start_tsc;

    if (duration.count() > 0) {
        // Calculate frequency: cycles per nanosecond * conversion factor = Hz
        double freq =
            static_cast<double>(cycles) / duration.count() * arch::simd::CPU_NANOSECONDS_TO_HZ;
        return static_cast<uint64_t>(freq);
    }
        #endif
    #endif
    return 0;
}

/**
 * @brief Detect performance monitoring capabilities
 */
void detect_performance_info(Features& features) {
    uint32_t eax, ebx, ecx, edx;

    // Check for RDTSC support
    safe_cpuid(1, 0, eax, ebx, ecx, edx);
    features.performance.has_rdtsc = (edx & (1 << 4)) != 0;

    // Check for invariant TSC
    safe_cpuid(0x80000000, 0, eax, ebx, ecx, edx);
    if (eax >= 0x80000007) {
        safe_cpuid(0x80000007, 0, eax, ebx, ecx, edx);
        features.performance.has_invariant_tsc = (edx & (1 << 8)) != 0;
    }

    // Check for performance monitoring capabilities
    safe_cpuid(0, 0, eax, ebx, ecx, edx);
    if (eax >= 0xA) {
        safe_cpuid(0xA, 0, eax, ebx, ecx, edx);
        features.performance.has_perf_counters = (eax & 0xFF) > 0;
    }

    // Estimate TSC frequency if RDTSC is available
    if (features.performance.has_rdtsc) {
        features.performance.tsc_frequency =
            estimate_tsc_frequency_internal(arch::simd::CPU_DEFAULT_TSC_SAMPLE_MS);  // Quick sample
    } else {
        features.performance.tsc_frequency = 0;
    }
}

/**
 * @brief Detect x86/x64 CPU features using CPUID
 */
Features detect_x86_features() {
    Features features;

    uint32_t eax, ebx, ecx, edx;

    // Check if CPUID is supported
    safe_cpuid(0, 0, eax, ebx, ecx, edx);
    uint32_t max_cpuid = eax;

    if (max_cpuid < 1) {
        return features;  // Very old CPU, no features
    }

    // Get vendor string
    char vendor[13] = {0};
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + detail::FOUR_INT, &edx, detail::FOUR_INT);
    memcpy(vendor + 8, &ecx, 4);
    features.vendor = vendor;

    // Get basic CPU info and feature flags
    safe_cpuid(1, 0, eax, ebx, ecx, edx);

    // Extract family, model, stepping
    features.family = ((eax >> 8) & 0xF) + ((eax >> 20) & 0xFF);
    features.model = ((eax >> 4) & 0xF) | ((eax >> 12) & 0xF0);
    features.stepping = eax & 0xF;

    // Feature detection from CPUID leaf 1
    features.sse2 = (edx & (1 << 26)) != 0;
    features.sse3 = (ecx & (1 << 0)) != 0;
    features.ssse3 = (ecx & (1 << 9)) != 0;
    features.sse4_1 = (ecx & (1 << 19)) != 0;
    features.sse4_2 = (ecx & (1 << 20)) != 0;

    // Check for AVX support
    bool osxsave = (ecx & (1 << 27)) != 0;
    bool avx_cpuid = (ecx & (1 << 28)) != 0;

    // AVX requires both CPUID support AND OS support (OSXSAVE)
    if (osxsave && avx_cpuid) {
        // Check if OS saves AVX registers
        uint64_t xcr0 = 0;
    #if defined(__GNUC__) || defined(__clang__)
        asm("xgetbv" : "=a"(xcr0) : "c"(0) : "edx");
    #elif defined(_MSC_VER)
        xcr0 = _xgetbv(0);
    #endif

        if ((xcr0 & 0x6) == 0x6) {  // Check bits 1 and 2
            features.avx = true;
            features.fma = (ecx & (1 << 12)) != 0;
        }
    }

    // Check for AVX2 support (requires CPUID leaf 7)
    if (features.avx && max_cpuid >= 7) {
        safe_cpuid(7, 0, eax, ebx, ecx, edx);
        features.avx2 = (ebx & (1 << 5)) != 0;
        features.avx512f = (ebx & (1 << 16)) != 0;
    }

    // Get brand string if available
    safe_cpuid(0x80000000, 0, eax, ebx, ecx, edx);
    if (eax >= 0x80000004) {
        char brand[49] = {0};
        safe_cpuid(0x80000002, 0, eax, ebx, ecx, edx);
        memcpy(brand, &eax, 16);
        safe_cpuid(0x80000003, 0, eax, ebx, ecx, edx);
        memcpy(brand + 16, &eax, 16);
        safe_cpuid(0x80000004, 0, eax, ebx, ecx, edx);
        memcpy(brand + 32, &eax, 16);
        features.brand = brand;
    }

    // Additional AVX-512 detection
    if (features.avx512f) {
        features.avx512dq = (ebx & (1 << 17)) != 0;
        features.avx512cd = (ebx & (1 << 28)) != 0;
        features.avx512bw = (ebx & (1 << 30)) != 0;
        features.avx512vl = (ebx & (1U << 31)) != 0;
    }

    #if defined(__APPLE__)
    // On macOS, prefer sysctl over CPUID for topology and cache info
    // sysctl is more reliable and consistent across different Intel generations
    detect_macos_topology(features);
    #else
    // Use CPUID for cache and topology detection on non-Apple systems
    detect_cache_info(features);
    detect_topology_info(features);
    #endif

    // Detect performance monitoring capabilities
    detect_performance_info(features);

    return features;
}
#endif  // LIBSTATS_X86_FAMILY

#if defined(__APPLE__)
// Forward declaration for macOS-specific function
void detect_macos_topology(Features& features);

/**
 * @brief Detect macOS CPU topology using sysctl (works for both Intel and ARM)
 */
void detect_macos_topology(Features& features) {
    size_t size = sizeof(int);
    int value = 0;

    // Get logical CPU count
    if (sysctlbyname("hw.logicalcpu", &value, &size, NULL, 0) == 0) {
        features.topology.logical_cores = static_cast<uint32_t>(value);
    }

    // Get physical CPU count
    if (sysctlbyname("hw.physicalcpu", &value, &size, NULL, 0) == 0) {
        features.topology.physical_cores = static_cast<uint32_t>(value);
    }

    // Calculate threads per core
    if (features.topology.physical_cores > 0 && features.topology.logical_cores > 0) {
        features.topology.threads_per_core =
            features.topology.logical_cores / features.topology.physical_cores;
        features.topology.hyperthreading = (features.topology.threads_per_core > 1);
    }

    // Assume single package (most common case for consumer Macs)
    features.topology.packages = 1;

    // Try to get cache information
    size_t cache_size = sizeof(uint64_t);
    uint64_t cache_value = 0;

    // L1 data cache
    if (sysctlbyname("hw.l1dcachesize", &cache_value, &cache_size, NULL, 0) == 0) {
        features.l1_data_cache.size = static_cast<uint32_t>(cache_value);
        features.l1_cache_size = features.l1_data_cache.size;  // Legacy compatibility
    }

    // L1 instruction cache
    if (sysctlbyname("hw.l1icachesize", &cache_value, &cache_size, NULL, 0) == 0) {
        features.l1_instruction_cache.size = static_cast<uint32_t>(cache_value);
    }

    // L2 cache
    if (sysctlbyname("hw.l2cachesize", &cache_value, &cache_size, NULL, 0) == 0) {
        features.l2_cache.size = static_cast<uint32_t>(cache_value);
        features.l2_cache_size = features.l2_cache.size;  // Legacy compatibility
    }

    // L3 cache
    if (sysctlbyname("hw.l3cachesize", &cache_value, &cache_size, NULL, 0) == 0) {
        features.l3_cache.size = static_cast<uint32_t>(cache_value);
        features.l3_cache_size = features.l3_cache.size;  // Legacy compatibility
    }

    // Cache line size
    int cache_line_size = 0;
    size = sizeof(int);
    if (sysctlbyname("hw.cachelinesize", &cache_line_size, &size, NULL, 0) == 0) {
        features.cache_line_size = static_cast<uint32_t>(cache_line_size);
        features.l1_data_cache.line_size = features.cache_line_size;
        features.l1_instruction_cache.line_size = features.cache_line_size;
        features.l2_cache.line_size = features.cache_line_size;
        features.l3_cache.line_size = features.cache_line_size;
    }
}
#endif  // __APPLE__

#ifdef LIBSTATS_ARM64_FAMILY

/**
 * @brief Detect ARM64 CPU features
 */
Features detect_arm_features() {
    Features features;

    #if defined(__APPLE__)
    features.vendor = "Apple";

    // macOS/iOS detection using sysctl
    size_t size = sizeof(int);
    int value = 0;

    // Check for NEON (should be available on all Apple Silicon)
    if (sysctlbyname("hw.optional.neon", &value, &size, NULL, 0) == 0 && value) {
        features.neon = true;
    }

    // Try to get CPU brand
    size = 0;
    if (sysctlbyname("machdep.cpu.brand_string", NULL, &size, NULL, 0) == 0) {
        char* brand = new char[size];
        if (sysctlbyname("machdep.cpu.brand_string", brand, &size, NULL, 0) == 0) {
            features.brand = brand;
        }
        delete[] brand;
    }

    // Detect topology and cache using macOS sysctl
    detect_macos_topology(features);

    #elif defined(__linux__)
    features.vendor = "ARM";

    // Linux detection using auxv
    unsigned long hwcap = getauxval(AT_HWCAP);
    features.neon = (hwcap & HWCAP_ASIMD) != 0;  // Advanced SIMD (NEON)

        // Check for SVE if available
        #ifdef HWCAP_SVE
    features.sve = (hwcap & HWCAP_SVE) != 0;
        #endif

    // For Linux ARM, use standard methods for topology detection
    // (could be expanded with /proc/cpuinfo parsing if needed)
    features.topology.logical_cores = std::thread::hardware_concurrency();
    features.topology.physical_cores = features.topology.logical_cores;  // Conservative estimate

    #else
    features.vendor = "ARM";
    // Assume NEON is available on AArch64 (it's mandatory in the spec)
    features.neon = true;
    features.topology.logical_cores = std::thread::hardware_concurrency();
    features.topology.physical_cores = features.topology.logical_cores;
    #endif

    return features;
}
#endif
}  // namespace

Features detect_features() {
    Features features;

#ifdef LIBSTATS_X86_FAMILY
    features = detect_x86_features();
#elif defined(LIBSTATS_ARM64_FAMILY)
    features = detect_arm_features();
#else
    // Unknown architecture - no SIMD features detected
    features.vendor = "Unknown";
#endif

    return features;
}

const Features& get_features() {
    return g_features_manager.get();
}

bool supports_sse2() {
    return get_features().sse2;
}

bool supports_sse4_1() {
    return get_features().sse4_1;
}

bool supports_avx() {
    return get_features().avx;
}

bool supports_avx2() {
    return get_features().avx2;
}

bool supports_fma() {
    return get_features().fma;
}

bool supports_avx512() {
    return get_features().avx512f;
}

bool supports_neon() {
    return get_features().neon;
}

std::string features_string() {
    const Features& f = get_features();
    std::string result;

    if (!f.vendor.empty()) {
        result += f.vendor;
        if (!f.brand.empty()) {
            result += " (" + f.brand + ")";
        }
        result += ": ";
    }

    bool first = true;
    auto add_feature = [&](const std::string& name) {
        if (!first)
            result += ", ";
        result += name;
        first = false;
    };

    if (f.sse2)
        add_feature("SSE2");
    if (f.sse3)
        add_feature("SSE3");
    if (f.ssse3)
        add_feature("SSSE3");
    if (f.sse4_1)
        add_feature("SSE4.1");
    if (f.sse4_2)
        add_feature("SSE4.2");
    if (f.avx)
        add_feature("AVX");
    if (f.fma)
        add_feature("FMA");
    if (f.avx2)
        add_feature("AVX2");
    if (f.avx512f)
        add_feature("AVX512F");
    if (f.neon)
        add_feature("NEON");
    if (f.sve)
        add_feature("SVE");

    if (first) {
        result += "No SIMD features detected";
    }

    return result;
}

std::string best_simd_level() {
    const Features& f = get_features();

    if (f.avx512f)
        return "AVX512";
    if (f.avx2)
        return "AVX2";
    if (f.avx)
        return "AVX";
    if (f.sse4_1)
        return "SSE4.1";
    if (f.sse2)
        return "SSE2";
    if (f.neon)
        return "NEON";
    return "Scalar";
}

size_t optimal_double_width() {
    const Features& f = get_features();

    if (f.avx512f)
        return 8;  // 512 bits / 64 bits per double
    if (f.avx || f.avx2)
        return 4;  // 256 bits / 64 bits per double
    if (f.sse2)
        return 2;  // 128 bits / 64 bits per double
    if (f.neon)
        return 2;  // 128 bits / 64 bits per double
    return 1;      // Scalar
}

size_t optimal_float_width() {
    const Features& f = get_features();

    if (f.avx512f)
        return 16;  // 512 bits / 32 bits per float
    if (f.avx || f.avx2)
        return 8;  // 256 bits / 32 bits per float
    if (f.sse2)
        return 4;  // 128 bits / 32 bits per float
    if (f.neon)
        return 4;  // 128 bits / 32 bits per float
    return 1;      // Scalar
}

size_t optimal_alignment() {
    const Features& f = get_features();

    if (f.avx512f)
        return 64;  // 512 bits = 64 bytes
    if (f.avx || f.avx2)
        return 32;  // 256 bits = 32 bytes
    if (f.sse2 || f.neon)
        return 16;  // 128 bits = 16 bytes
    return 8;       // Basic double alignment
}

// Enhanced CPU feature detection functions
bool supports_avx512dq() {
    return get_features().avx512dq;
}

bool supports_avx512bw() {
    return get_features().avx512bw;
}

bool supports_avx512vl() {
    return get_features().avx512vl;
}

bool supports_sve() {
    return get_features().sve;
}

bool supports_sve2() {
    return get_features().sve2;
}

// Cache information queries
std::optional<CacheInfo> get_l1_data_cache() {
    const Features& f = get_features();
    if (f.l1_data_cache.size > 0) {
        return f.l1_data_cache;
    }
    return std::nullopt;
}

std::optional<CacheInfo> get_l1_instruction_cache() {
    const Features& f = get_features();
    if (f.l1_instruction_cache.size > 0) {
        return f.l1_instruction_cache;
    }
    return std::nullopt;
}

std::optional<CacheInfo> get_l2_cache() {
    const Features& f = get_features();
    if (f.l2_cache.size > 0) {
        return f.l2_cache;
    }
    return std::nullopt;
}

std::optional<CacheInfo> get_l3_cache() {
    const Features& f = get_features();
    if (f.l3_cache.size > 0) {
        return f.l3_cache;
    }
    return std::nullopt;
}

// CPU topology queries
TopologyInfo get_topology() {
    return get_features().topology;
}

uint32_t get_logical_core_count() {
    return get_features().topology.logical_cores;
}

uint32_t get_physical_core_count() {
    return get_features().topology.physical_cores;
}

bool has_hyperthreading() {
    return get_features().topology.hyperthreading;
}

// Performance monitoring utilities
PerformanceInfo get_performance_info() {
    return get_features().performance;
}

bool has_rdtsc() {
    return get_features().performance.has_rdtsc;
}

bool has_invariant_tsc() {
    return get_features().performance.has_invariant_tsc;
}

std::optional<uint64_t> get_tsc_frequency() {
    const Features& f = get_features();
    if (f.performance.tsc_frequency > 0) {
        return f.performance.tsc_frequency;
    }
    return std::nullopt;
}

// Performance measurement utilities
uint64_t read_tsc() {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #if defined(__GNUC__) || defined(__clang__)
    return __rdtsc();
    #elif defined(_MSC_VER)
    return __rdtsc();
    #else
    return 0;
    #endif
#else
    return 0;
#endif
}

std::optional<uint64_t> estimate_cpu_frequency(uint32_t duration_ms) {
    if (!has_rdtsc()) {
        return std::nullopt;
    }

    uint64_t start_tsc = read_tsc();
    auto start_time = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));

    uint64_t end_tsc = read_tsc();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    uint64_t cycles = end_tsc - start_tsc;

    if (duration.count() > 0) {
        // Calculate frequency: cycles per nanosecond * conversion factor = Hz
        double freq = static_cast<double>(cycles) / static_cast<double>(duration.count()) *
                      arch::simd::CPU_NANOSECONDS_TO_HZ;
        return static_cast<uint64_t>(freq);
    }

    return std::nullopt;
}

std::string detailed_cpu_info() {
    const Features& f = get_features();
    std::string result;

    result += "CPU Details:\n";
    result += "  Vendor: " + f.vendor + "\n";
    if (!f.brand.empty()) {
        result += "  Brand: " + f.brand + "\n";
    }
    result += "  Family: " + std::to_string(f.family) + "\n";
    result += "  Model: " + std::to_string(f.model) + "\n";
    result += "  Stepping: " + std::to_string(f.stepping) + "\n";

    result += "\nSIMD Features: " + features_string() + "\n";
    result += "Best SIMD Level: " + best_simd_level() + "\n";

    result += "\nOptimal Vector Widths:\n";
    result += "  Double: " + std::to_string(optimal_double_width()) + "\n";
    result += "  Float: " + std::to_string(optimal_float_width()) + "\n";
    result += "  Alignment: " + std::to_string(optimal_alignment()) + " bytes\n";

    result += "\nCache Information:\n";
    result += "  L1 Data: " + std::to_string(f.l1_data_cache.size) + " bytes\n";
    result += "  L1 Instruction: " + std::to_string(f.l1_instruction_cache.size) + " bytes\n";
    result += "  L2: " + std::to_string(f.l2_cache.size) + " bytes\n";
    result += "  L3: " + std::to_string(f.l3_cache.size) + " bytes\n";

    result += "\nTopology:\n";
    result += "  Logical Cores: " + std::to_string(f.topology.logical_cores) + "\n";
    result += "  Physical Cores: " + std::to_string(f.topology.physical_cores) + "\n";
    result += "  Hyperthreading: " +
              (f.topology.hyperthreading ? std::string("Yes") : std::string("No")) + "\n";

    return result;
}

bool validate_feature_consistency() {
    const Features& f = get_features();

    // Check AVX hierarchy consistency
    if (f.avx2 && !f.avx)
        return false;
    if (f.avx && !f.sse2)
        return false;
    if (f.avx512f && !f.avx2)
        return false;

    // Check SSE hierarchy consistency
    if (f.sse4_2 && !f.sse4_1)
        return false;
    if (f.sse4_1 && !f.ssse3)
        return false;
    if (f.ssse3 && !f.sse3)
        return false;
    if (f.sse3 && !f.sse2)
        return false;

    // Check AVX-512 sub-features require AVX-512F
    if ((f.avx512dq || f.avx512cd || f.avx512bw || f.avx512vl) && !f.avx512f)
        return false;

    // Check ARM hierarchy consistency
    if (f.sve2 && !f.sve)
        return false;

    return true;
}

// Intel CPU generation detection functions
bool is_sandy_ivy_bridge() {
    const Features& features = get_features();
    return features.vendor == "GenuineIntel" && features.family == 6 &&
           (features.model == 42 || features.model == 58);  // Sandy Bridge: 42, Ivy Bridge: 58
}

bool is_haswell_broadwell() {
    const Features& features = get_features();
    return features.vendor == "GenuineIntel" && features.family == 6 &&
           (features.model == 60 || features.model == 61     // Haswell: 60, Broadwell: 61
            || features.model == 69 || features.model == 70  // Haswell-ULT: 69, Haswell-GT3e: 70
            || features.model == 71);                        // Broadwell-GT3e: 71
}

bool is_skylake_generation() {
    const Features& features = get_features();
    return features.vendor == "GenuineIntel" && features.family == 6 &&
           (features.model == 78 || features.model == 94);  // Skylake-U/Y: 78, Skylake-S/H: 94
}

bool is_kaby_coffee_lake() {
    const Features& features = get_features();
    return features.vendor == "GenuineIntel" && features.family == 6 &&
           (features.model == 142 ||
            features.model == 158  // Kaby Lake-U/Y: 142, Coffee Lake-S: 158
            || features.model == 165 ||
            features.model == 166);  // Coffee Lake-H: 165, Cannon Lake: 166
}

bool is_modern_intel() {
    const Features& features = get_features();
    // Modern Intel includes Ice Lake (2019+) with AVX-512 or newer architectures
    return features.vendor == "GenuineIntel" &&
           (features.avx512f                                      // Any CPU with AVX-512 is modern
            || (features.family == 6 && features.model >= 125));  // Ice Lake and newer models
}

}  // namespace arch
}  // namespace stats

// Restore original compiler SIMD settings (only needed for GCC)
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC pop_options
#endif

#pragma once

/**
 * @file common/platform_common.h
 * @brief Common includes and utilities for all platform headers
 *
 * This header consolidates the standard library and core project headers that are
 * commonly needed by all platform implementation headers. Platform headers should
 * include this instead of duplicating these common includes.
 */

// Standard library includes commonly needed by all platform implementations
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <vector>

// System and platform detection includes
#include <cassert>
#include <climits>
#include <cstring>

// Platform-specific system includes (conditional)
#if defined(_WIN32)
    #include <malloc.h>
    #include <windows.h>
#elif defined(__APPLE__)
    #include <cerrno>
    #include <cstdlib>
    #include <mach/mach.h>
    #include <sys/sysctl.h>
#elif defined(__linux__)
    #include <cerrno>
    #include <cstdlib>
    #include <sys/sysinfo.h>
    #include <unistd.h>
#else
    #include <cerrno>
    #include <cstdlib>
#endif

// Core libstats headers needed by platform implementations
#include "../core/constants.h"
#include "../core/error_handling.h"
#include "../core/safety.h"

// Platform constants that are shared across platform modules
#include "../platform/platform_constants.h"

// Utility using declarations to avoid repetition across platform headers
namespace libstats {
namespace platform_utils {

/**
 * @brief Get optimal alignment for the current platform
 * @return Alignment in bytes for optimal performance
 */
constexpr std::size_t get_optimal_alignment() noexcept {
#if defined(__AVX512F__)
    return 64;  // AVX-512 benefits from 64-byte alignment
#elif defined(__AVX__) || defined(__AVX2__)
    return 32;  // AVX requires 32-byte alignment
#elif defined(__SSE2__) || defined(__ARM_NEON)
    return 16;  // SSE2/NEON benefits from 16-byte alignment
#else
    return 8;  // Basic double alignment
#endif
}

/**
 * @brief Get platform-specific cache line size
 * @return Cache line size in bytes for the current platform
 */
constexpr std::size_t get_cache_line_size() noexcept {
#if defined(__APPLE__) && defined(__aarch64__)
    return 128;  // Apple Silicon has 128-byte cache lines
#elif defined(__x86_64__) || defined(__i386__)
    return 64;  // Most x86/x64 processors have 64-byte cache lines
#elif defined(__ARM_NEON)
    return 64;  // ARM processors typically have 64-byte cache lines
#else
    return 64;  // Safe default for most modern processors
#endif
}

/**
 * @brief Check if the platform supports high-resolution timing
 * @return true if std::chrono::high_resolution_clock is steady
 */
constexpr bool has_steady_high_resolution_clock() noexcept {
    return std::chrono::high_resolution_clock::is_steady;
}

}  // namespace platform_utils
}  // namespace libstats

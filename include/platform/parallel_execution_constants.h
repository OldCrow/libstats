#pragma once

/**
 * @file platform/parallel_execution_constants.h
 * @brief Constants specific to parallel execution components
 *
 * This header contains all the constants used in parallel execution,
 * including thread pool settings, work stealing parameters, and
 * parallel threshold values.
 */

#include <cstddef>
#include <cstdint>

namespace libstats {
namespace constants {
namespace parallel_execution {

// Thread pool and work stealing configuration
namespace threading {
// Thread initialization timeouts
inline constexpr uint32_t THREAD_INIT_TIMEOUT_MS =
    5000;  // 5 second max wait for thread initialization

// Work stealing backoff parameters
inline constexpr uint32_t YIELD_BACKOFF_LIMIT = 10;          // Number of yields before sleeping
inline constexpr uint32_t SLEEP_BACKOFF_MICROSECONDS = 100;  // Sleep duration after yield limit

// Default thread counts
inline constexpr std::size_t FALLBACK_THREAD_COUNT = 4;  // When hardware_concurrency() returns 0
inline constexpr std::size_t MIN_WORK_STEALING_THREADS = 2;  // Minimum for meaningful work stealing

// QoS relative priority range for macOS work stealing
inline constexpr int WORK_STEALING_QOS_MIN_PRIORITY = -2;
inline constexpr int WORK_STEALING_QOS_MAX_PRIORITY = 2;
inline constexpr int WORK_STEALING_QOS_PRIORITY_RANGE = 4;  // Total range (-2 to +2)
}  // namespace threading

// Architecture profile constants
namespace architecture {
// Thread creation costs (microseconds)
inline constexpr uint32_t APPLE_SILICON_THREAD_COST_US = 2;
inline constexpr uint32_t HIGH_END_X86_THREAD_COST_US = 5;
inline constexpr uint32_t STANDARD_X86_THREAD_COST_US = 8;
inline constexpr uint32_t DEFAULT_THREAD_COST_US = 10;

// SIMD width (elements)
inline constexpr uint32_t APPLE_SILICON_SIMD_WIDTH = 2;  // NEON 128-bit
inline constexpr uint32_t AVX2_SIMD_WIDTH = 4;           // AVX2 256-bit / 4 doubles
inline constexpr uint32_t SSE_SIMD_WIDTH = 2;            // SSE 128-bit / 2 doubles
inline constexpr uint32_t NO_SIMD_WIDTH = 1;             // No SIMD

// Thread efficiency factors
inline constexpr double APPLE_SILICON_EFFICIENCY = 0.95;
inline constexpr double HIGH_END_X86_EFFICIENCY = 0.85;
inline constexpr double STANDARD_X86_EFFICIENCY = 0.75;
inline constexpr double DEFAULT_EFFICIENCY = 0.7;

// Base parallel thresholds
inline constexpr std::size_t APPLE_SILICON_BASE_THRESHOLD = 1024;
inline constexpr std::size_t HIGH_END_X86_BASE_THRESHOLD = 2048;
inline constexpr std::size_t STANDARD_X86_BASE_THRESHOLD = 4096;
inline constexpr std::size_t DEFAULT_BASE_THRESHOLD = 8192;

// Default L3 cache size (in elements, when detection fails)
inline constexpr std::size_t DEFAULT_L3_CACHE_ELEMENTS = 2 * 1024 * 1024;  // 2MB worth of doubles
}  // namespace architecture

// Distribution-specific parallel thresholds
namespace thresholds {
// Uniform distribution thresholds
namespace uniform {
inline constexpr std::size_t PDF_THRESHOLD = 16384;
inline constexpr std::size_t LOGPDF_THRESHOLD = 64;
inline constexpr std::size_t CDF_THRESHOLD = 16384;
inline constexpr std::size_t BATCH_FIT_THRESHOLD = 64;
inline constexpr std::size_t DEFAULT_THRESHOLD = 8192;
}  // namespace uniform

// Discrete distribution thresholds
namespace discrete {
inline constexpr std::size_t PDF_THRESHOLD = 1048576;
inline constexpr std::size_t LOGPDF_THRESHOLD = 32768;
inline constexpr std::size_t CDF_THRESHOLD = 65536;
inline constexpr std::size_t BATCH_FIT_THRESHOLD = 64;
inline constexpr std::size_t DEFAULT_THRESHOLD = 32768;
}  // namespace discrete

// Exponential distribution thresholds
namespace exponential {
inline constexpr std::size_t PDF_THRESHOLD = 64;
inline constexpr std::size_t LOGPDF_THRESHOLD = 128;
inline constexpr std::size_t CDF_THRESHOLD = 64;
inline constexpr std::size_t BATCH_FIT_THRESHOLD = 32;
inline constexpr std::size_t DEFAULT_THRESHOLD = 64;
}  // namespace exponential

// Gaussian/Normal distribution thresholds
namespace gaussian {
inline constexpr std::size_t PDF_THRESHOLD = 64;
inline constexpr std::size_t LOGPDF_THRESHOLD = 256;
inline constexpr std::size_t CDF_THRESHOLD = 64;
inline constexpr std::size_t BATCH_FIT_THRESHOLD = 32;
inline constexpr std::size_t DEFAULT_THRESHOLD = 256;
}  // namespace gaussian

// Poisson distribution thresholds
namespace poisson {
inline constexpr std::size_t PDF_THRESHOLD = 4096;
inline constexpr std::size_t LOGPDF_THRESHOLD = 8192;
inline constexpr std::size_t CDF_THRESHOLD = 512;
inline constexpr std::size_t BATCH_FIT_THRESHOLD = 64;
inline constexpr std::size_t DEFAULT_THRESHOLD = 4096;
}  // namespace poisson

// Gamma distribution thresholds
namespace gamma {
inline constexpr std::size_t PDF_THRESHOLD = 256;
inline constexpr std::size_t LOGPDF_THRESHOLD = 512;
inline constexpr std::size_t CDF_THRESHOLD = 128;
inline constexpr std::size_t BATCH_FIT_THRESHOLD = 64;
inline constexpr std::size_t DEFAULT_THRESHOLD = 256;
}  // namespace gamma

// Generic operation thresholds
namespace generic {
inline constexpr std::size_t FILL_TRANSFORM_THRESHOLD = 8192;
inline constexpr std::size_t SORT_THRESHOLD = 4096;
inline constexpr std::size_t SCAN_THRESHOLD = 16384;
inline constexpr std::size_t SEARCH_COUNT_THRESHOLD = 8192;
inline constexpr std::size_t DEFAULT_THRESHOLD = 8192;
}  // namespace generic
}  // namespace thresholds

// Memory and cache constants
namespace memory {
inline constexpr std::size_t CACHE_LINE_SIZE = 64;             // Common cache line size in bytes
inline constexpr std::size_t L1_CACHE_SIZE = 32 * 1024;        // 32KB typical L1 cache
inline constexpr std::size_t L2_CACHE_SIZE = 256 * 1024;       // 256KB typical L2 cache
inline constexpr std::size_t L3_CACHE_SIZE = 8 * 1024 * 1024;  // 8MB typical L3 cache
}  // namespace memory

}  // namespace parallel_execution
}  // namespace constants
}  // namespace libstats

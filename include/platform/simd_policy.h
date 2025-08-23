#pragma once

#include <cstddef>
#include <string>

/**
 * @file simd_policy.h
 * @brief Centralized, platform-independent SIMD policy abstraction
 *
 * This header provides the SIMDPolicy class that centralizes all SIMD-related
 * decision making. It abstracts over x86 (SSE, AVX) and ARM (NEON) instruction
 * sets to provide a single, consistent interface.
 *
 * Key benefits of this abstraction:
 * - Single source of truth for SIMD decisions
 * - Consistent thresholds and detection logic for all architectures
 * - Easier testing and maintenance
 * - Reduced code duplication and platform-specific logic
 * - Centralized performance tuning
 */

namespace stats {
namespace simd {

/**
 * @brief Centralized SIMD policy class for consistent usage decisions
 *
 * This class provides a clean abstraction over complex SIMD detection,
 * supporting both x86 (SSE, AVX) and ARM (NEON) instruction sets.
 * It encapsulates both compile-time and runtime capabilities to make
 * optimal decisions about when to use vectorized code paths.
 *
 * Usage examples:
 * @code
 * // Simple SIMD decision
 * if (SIMDPolicy::shouldUseSIMD(data.size())) {
 *     // This will work on both an Intel CPU with AVX2 and an Apple M2 with NEON
 *     processWithSIMD(data);
 * } else {
 *     processScalar(data);
 * }
 *
 * // Advanced usage with specific level
 * auto level = SIMDPolicy::getBestLevel();
 * switch (level) {
 *     case SIMDPolicy::Level::AVX512:
 *         processWithAVX512(data);
 *         break;
 *     case SIMDPolicy::Level::AVX2:
 *         processWithAVX2(data);
 *         break;
 *     case SIMDPolicy::Level::NEON:
 *         processWithNEON(data);
 *         break;
 *     // ... etc
 * }
 * @endcode
 */
class SIMDPolicy {
   public:
    /**
     * @brief SIMD instruction set levels in order of general capability
     */
    enum class Level {
        None,   ///< No SIMD support - use scalar implementation
        NEON,   ///< ARM NEON - 128-bit vectors
        SSE2,   ///< SSE2 - 128-bit vectors, 2 doubles
        AVX,    ///< AVX - 256-bit vectors, 4 doubles
        AVX2,   ///< AVX2 + FMA - enhanced 256-bit vectors
        AVX512  ///< AVX-512 - 512-bit vectors, 8 doubles
    };

    /**
     * @brief Determine if SIMD should be used for a given data size
     *
     * This method encapsulates the complete logic for deciding whether to use
     * SIMD operations. It considers both the data size (to ensure SIMD overhead
     * is justified) and the runtime CPU capabilities (for AVX, NEON, etc.).
     *
     * @param count Number of elements to process
     * @return true if SIMD should be used, false for scalar processing
     */
    static bool shouldUseSIMD(std::size_t count) noexcept;

    /**
     * @brief Get the best available SIMD instruction set level
     *
     * This method determines the highest SIMD instruction set supported by both
     * the compiler and the runtime CPU hardware.
     *
     * @return The best available SIMD level (e.g., Level::AVX2 or Level::NEON)
     */
    static Level getBestLevel() noexcept;

    /**
     * @brief Get the minimum data size threshold for the current SIMD level
     *
     * Returns the minimum number of elements required to justify the overhead
     * of SIMD operations. This threshold is tuned for the best available
     * SIMD instruction set (e.g., AVX2 might have a higher threshold than NEON).
     *
     * @return Minimum element count for beneficial SIMD usage
     */
    static std::size_t getMinThreshold() noexcept;

    /**
     * @brief Get the optimal SIMD block size for the detected instruction set
     *
     * Returns the number of elements to process in each SIMD iteration
     * based on the vector width (e.g., 8 for AVX-512, 4 for AVX2, 2 for NEON/SSE2).
     *
     * @return Optimal block size (in elements) for SIMD operations
     */
    static std::size_t getOptimalBlockSize() noexcept;

    /**
     * @brief Get the optimal memory alignment for SIMD operations
     *
     * Returns the memory alignment requirement for optimal performance
     * based on the detected instruction set (e.g., 64 bytes for AVX-512).
     *
     * @return Memory alignment in bytes
     */
    static std::size_t getOptimalAlignment() noexcept;

    /**
     * @brief Get a human-readable string describing the detected SIMD level
     *
     * @return String description of the SIMD level (e.g., "AVX2", "NEON", "None")
     */
    static std::string getLevelString() noexcept;

    /**
     * @brief Get a human-readable string describing SIMD capabilities
     *
     * Provides detailed information about the detected SIMD capabilities,
     * including vector widths and thresholds. Useful for diagnostics.
     *
     * @return Detailed capability string for diagnostics
     */
    static std::string getCapabilityString() noexcept;

    /**
     * @brief Force refresh of cached SIMD detection results
     *
     * Forces a re-detection of CPU capabilities. Useful for testing.
     * This method is thread-safe but has a performance cost.
     */
    static void refreshCache() noexcept;

   private:
    /**
     * @brief Detects the best available SIMD level by checking runtime CPU features
     * in order from most to least powerful.
     */
    static Level detectBestLevel() noexcept;

    /**
     * @brief Computes the optimal threshold based on the detected SIMD level.
     */
    static std::size_t computeOptimalThreshold(Level level) noexcept;

    /**
     * @brief Computes the optimal block size (element count) based on the SIMD level.
     */
    static std::size_t computeOptimalBlockSize(Level level) noexcept;

    /**
     * @brief Computes the optimal memory alignment (in bytes) based on the SIMD level.
     */
    static std::size_t computeOptimalAlignment(Level level) noexcept;

    /**
     * @brief Converts a SIMD level enum to its string representation.
     */
    static std::string levelToString(Level level) noexcept;
};

}  // namespace simd
}  // namespace stats

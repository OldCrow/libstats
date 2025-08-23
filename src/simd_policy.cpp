#include "../include/platform/simd_policy.h"

#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd.h"

#include <atomic>
#include <mutex>

/**
 * @file simd_policy.cpp
 * @brief Implementation of the centralized SIMDPolicy class
 */

namespace stats {
namespace simd {

namespace {
// Thread-safe, cached state for SIMD policy decisions
struct SIMDState {
    std::atomic<SIMDPolicy::Level> best_level{SIMDPolicy::Level::None};
    std::atomic<bool> initialized{false};
    std::once_flag init_flag;

    void initialize() {
        // Detect best level inline since detectBestLevel is private
        SIMDPolicy::Level detected_level = SIMDPolicy::Level::None;

#if defined(LIBSTATS_HAS_AVX512)
        if (cpu::supports_avx512()) {
            detected_level = SIMDPolicy::Level::AVX512;
        } else
#endif
#if defined(LIBSTATS_HAS_AVX2)
            if (cpu::supports_avx2()) {
            detected_level = SIMDPolicy::Level::AVX2;
        } else
#endif
#if defined(LIBSTATS_HAS_AVX)
            if (cpu::supports_avx()) {
            detected_level = SIMDPolicy::Level::AVX;
        } else
#endif
#if defined(LIBSTATS_HAS_SSE2)
            if (cpu::supports_sse2()) {
            detected_level = SIMDPolicy::Level::SSE2;
        } else
#endif
#if defined(LIBSTATS_HAS_NEON)
            if (cpu::supports_neon()) {
            detected_level = SIMDPolicy::Level::NEON;
        } else
#endif
        {
            detected_level = SIMDPolicy::Level::None;
        }

        best_level.store(detected_level, std::memory_order_release);
        initialized.store(true, std::memory_order_release);
    }

    SIMDPolicy::Level get_level() {
        if (!initialized.load(std::memory_order_acquire)) {
            std::call_once(init_flag, [this] { this->initialize(); });
        }
        return best_level.load(std::memory_order_acquire);
    }
};

// Global singleton for the SIMD state
SIMDState g_simd_state;

}  // anonymous namespace

// Public Methods

SIMDPolicy::Level SIMDPolicy::getBestLevel() noexcept {
    return g_simd_state.get_level();
}

void SIMDPolicy::refreshCache() noexcept {
    g_simd_state.initialized.store(false, std::memory_order_release);
}

std::size_t SIMDPolicy::getMinThreshold() noexcept {
    return computeOptimalThreshold(getBestLevel());
}

std::size_t SIMDPolicy::getOptimalBlockSize() noexcept {
    return computeOptimalBlockSize(getBestLevel());
}

std::size_t SIMDPolicy::getOptimalAlignment() noexcept {
    return computeOptimalAlignment(getBestLevel());
}

bool SIMDPolicy::shouldUseSIMD(std::size_t count) noexcept {
    const auto level = getBestLevel();
    if (level == SIMDPolicy::Level::None) {
        return false;
    }
    return count >= getMinThreshold();
}

std::string SIMDPolicy::getLevelString() noexcept {
    return levelToString(getBestLevel());
}

std::string SIMDPolicy::getCapabilityString() noexcept {
    const auto level = getBestLevel();
    if (level == SIMDPolicy::Level::None) {
        return "No SIMD support available.";
    }
    return "Best level: " + levelToString(level) +
           ", Threshold: " + std::to_string(getMinThreshold()) +
           ", Block Size: " + std::to_string(getOptimalBlockSize()) +
           ", Alignment: " + std::to_string(getOptimalAlignment()) + " bytes";
}

// Private Implementation Methods

SIMDPolicy::Level SIMDPolicy::detectBestLevel() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    if (cpu::supports_avx512())
        return SIMDPolicy::Level::AVX512;
#endif
#if defined(LIBSTATS_HAS_AVX2)
    if (cpu::supports_avx2())
        return SIMDPolicy::Level::AVX2;
#endif
#if defined(LIBSTATS_HAS_AVX)
    if (cpu::supports_avx())
        return SIMDPolicy::Level::AVX;
#endif
#if defined(LIBSTATS_HAS_SSE2)
    if (cpu::supports_sse2())
        return SIMDPolicy::Level::SSE2;
#endif
#if defined(LIBSTATS_HAS_NEON)
    if (cpu::supports_neon())
        return SIMDPolicy::Level::NEON;
#endif
    return SIMDPolicy::Level::None;
}

std::size_t SIMDPolicy::computeOptimalThreshold(SIMDPolicy::Level level) noexcept {
    switch (level) {
        case SIMDPolicy::Level::AVX512:
            return 16;
        case SIMDPolicy::Level::AVX2:
            return 8;
        case SIMDPolicy::Level::AVX:
            return 8;
        case SIMDPolicy::Level::SSE2:
            return 4;
        case SIMDPolicy::Level::NEON:
            return 4;
        default:
            return SIZE_MAX;
    }
}

std::size_t SIMDPolicy::computeOptimalBlockSize(SIMDPolicy::Level level) noexcept {
    switch (level) {
        case SIMDPolicy::Level::AVX512:
            return 8;  // 8 doubles
        case SIMDPolicy::Level::AVX2:
            return 4;  // 4 doubles
        case SIMDPolicy::Level::AVX:
            return 4;  // 4 doubles
        case SIMDPolicy::Level::SSE2:
            return 2;  // 2 doubles
        case SIMDPolicy::Level::NEON:
            return 2;  // 2 doubles
        default:
            return 1;  // Scalar
    }
}

std::size_t SIMDPolicy::computeOptimalAlignment(SIMDPolicy::Level level) noexcept {
    switch (level) {
        case SIMDPolicy::Level::AVX512:
            return 64;
        case SIMDPolicy::Level::AVX2:
            return 32;
        case SIMDPolicy::Level::AVX:
            return 32;
        case SIMDPolicy::Level::SSE2:
            return 16;
        case SIMDPolicy::Level::NEON:
            return 16;
        default:
            return 8;  // Default alignment for double
    }
}

std::string SIMDPolicy::levelToString(SIMDPolicy::Level level) noexcept {
    switch (level) {
        case SIMDPolicy::Level::AVX512:
            return "AVX-512";
        case SIMDPolicy::Level::AVX2:
            return "AVX2";
        case SIMDPolicy::Level::AVX:
            return "AVX";
        case SIMDPolicy::Level::SSE2:
            return "SSE2";
        case SIMDPolicy::Level::NEON:
            return "NEON";
        default:
            return "None";
    }
}

}  // namespace simd
}  // namespace stats

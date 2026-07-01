#include "libstats/platform/internal/cpu_tier.h"

#include "libstats/platform/cpu_detection.h"

namespace stats::arch {

CpuTier cpu_tier() noexcept {
    // Apple Silicon is identified at compile time — no runtime vendor lookup needed.
#if defined(__APPLE__) && defined(__aarch64__)
    return CpuTier::Apple_Silicon;

#elif defined(__x86_64__) || defined(_M_X64)
    /**
     * Sandy Bridge (2011): Intel family 6, model 42
     * Ivy Bridge (2012):   Intel family 6, model 58
     * These are the only Intel generations with AVX but not AVX2 or FMA.
     */
    auto is_legacy_intel = [](const Features& f) noexcept -> bool {
        return f.vendor == "GenuineIntel" && f.family == 6 && (f.model == 42 || f.model == 58);
    };

    const auto& f = get_features();

    if (f.vendor == "GenuineIntel") {
        // Sandy/Ivy Bridge: AVX without FMA or AVX2 — use intel::legacy constants.
        // Every other Intel generation (Haswell through Sapphire Rapids) shares
        // intel::modern constants even when AVX-512 is present, because the vendor
        // constant tables have no intel::avx512 sub-namespace.
        return is_legacy_intel(f) ? CpuTier::Intel_Legacy : CpuTier::Intel_Modern;
    }

    if (f.vendor == "AuthenticAMD") {
        return CpuTier::AMD_Zen;
    }

    // Unknown x86 vendor (VIA, Hygon, etc.).  Callers fall back to SIMD-capability
    // selection for this tier.
    return (f.avx || f.avx2 || f.avx512f || f.sse2) ? CpuTier::x86_Generic : CpuTier::Scalar_Only;

#else
    // Non-Apple ARM or other architecture.
    const auto& f = get_features();
    return f.neon ? CpuTier::ARM_Generic : CpuTier::Scalar_Only;
#endif
}

}  // namespace stats::arch

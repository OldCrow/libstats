#pragma once

/**
 * @file platform/internal/cpu_tier.h
 * @brief Single-classification CPU tier for vendor-specific constant selection.
 *
 * Encapsulates the Intel microarchitecture cascade and vendor string comparisons
 * previously scattered across platform_constants_impl.cpp (24 vendor-string
 * comparisons + 4 is_sandy_ivy_bridge() calls, now reduced to one cpu_tier()
 * call per function).
 *
 * Internal to the platform layer — not part of the public libstats API.
 * Do not include this header from public-facing headers.
 */

namespace stats::arch {

/**
 * @brief Coarse CPU tier for selecting vendor-tuned grain-size constants.
 *
 * Two Intel tiers (not three): both Haswell+ and AVX-512 Intel CPUs share
 * intel::modern constants — there is no intel::avx512 constant sub-namespace.
 */
enum class CpuTier {
    Intel_Legacy,   ///< Sandy Bridge (model 42) / Ivy Bridge (model 58): AVX, no FMA or AVX2
    Intel_Modern,   ///< Haswell+ (AVX2+FMA); also covers Ice Lake+ (AVX-512) Intel CPUs
    AMD_Zen,        ///< AMD Ryzen any generation: AVX2
    Apple_Silicon,  ///< Apple M-series: NEON, 128-byte cache lines, GCD scheduler
    ARM_Generic,    ///< Other NEON-capable ARM (non-Apple)
    x86_Generic,    ///< Unknown x86 vendor; SIMD-capability fallback applied per-function
    Scalar_Only     ///< No SIMD detected
};

/**
 * @brief Classify the current CPU into a CpuTier.
 *
 * Calls get_features() (already cached after first use) and encodes the
 * classification logic that was previously duplicated across platform_constants_impl.cpp.
 * is_sandy_ivy_bridge() is absorbed here and no longer needs to be public.
 */
[[nodiscard]] CpuTier cpu_tier() noexcept;

}  // namespace stats::arch

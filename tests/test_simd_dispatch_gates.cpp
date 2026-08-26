/**
 * @file test_simd_dispatch_gates.cpp
 * @brief Guards for #117 — the CPUID feature gates that decide which SIMD tier
 *        the dispatcher selects.
 *
 * The AVX-512 kernels in simd_avx512.cpp need AVX-512DQ (_mm512_cvtepi64_pd,
 * _mm512_andnot_pd, ...) and an OS that saves opmask/ZMM state; the AVX2 kernels
 * need FMA. Before #117 the dispatcher asked for neither, so a redistributed
 * binary could reach an F-only or state-disabled CPU and SIGILL.
 *
 * These are build/dispatch plumbing guards rather than distribution tests, which
 * is why they live in their own binary — and the file is deliberately unlabelled
 * so it runs in the standard correctness suite (`ctest -LE "timing|benchmark"`).
 *
 * Every assertion is two-sided: this file reads CPUID leaf 7 EBX and XCR0 for
 * itself and then requires the library's dispatch decision to agree with the
 * conclusion it reached independently. A one-sided "if AVX-512 then all is well"
 * check cannot fail on any machine.
 *
 * The third test pins SIMDPolicy's tier report to VectorOps' — the same gates
 * live in two translation units, and a mismatch there is silent.
 */

#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd.h"
#include "libstats/platform/simd_policy.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
    #define LIBSTATS_TEST_X86_FAMILY
    #if defined(_MSC_VER)
        #include <intrin.h>
    #else
        #include <cpuid.h>
        #include <immintrin.h>
    #endif
#endif

namespace {

#if defined(LIBSTATS_TEST_X86_FAMILY)

struct CpuidRegs {
    std::uint32_t eax = 0;
    std::uint32_t ebx = 0;
    std::uint32_t ecx = 0;
    std::uint32_t edx = 0;
};

/// Raw CPUID, deliberately independent of src/cpu_detection.cpp.
CpuidRegs raw_cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
    CpuidRegs r;
    #if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<std::uint32_t>(regs[0]);
    r.ebx = static_cast<std::uint32_t>(regs[1]);
    r.ecx = static_cast<std::uint32_t>(regs[2]);
    r.edx = static_cast<std::uint32_t>(regs[3]);
    #else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
    #endif
    return r;
}

/// Raw XGETBV, deliberately independent of src/cpu_detection.cpp.
std::uint64_t raw_xgetbv(std::uint32_t index) noexcept {
    #if defined(_MSC_VER)
    return _xgetbv(index);
    #else
    std::uint32_t lo = 0;
    std::uint32_t hi = 0;
    asm volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(index));
    return (static_cast<std::uint64_t>(hi) << 32) | lo;
    #endif
}

/// What this file concludes about the CPU, without asking libstats anything.
struct IndependentView {
    bool osxsave = false;
    bool avx = false;        ///< leaf 1 ECX[28]
    bool fma = false;        ///< leaf 1 ECX[12]
    bool avx2 = false;       ///< leaf 7 EBX[5]
    bool avx512f = false;    ///< leaf 7 EBX[16]
    bool avx512dq = false;   ///< leaf 7 EBX[17]
    bool ymm_state = false;  ///< XCR0[2:1] == 11b
    bool zmm_state = false;  ///< XCR0[7:5] == 111b
};

IndependentView read_cpu() noexcept {
    IndependentView v;
    const CpuidRegs leaf0 = raw_cpuid(0, 0);
    const CpuidRegs leaf1 = raw_cpuid(1, 0);
    v.osxsave = (leaf1.ecx & (1U << 27)) != 0U;
    v.avx = (leaf1.ecx & (1U << 28)) != 0U;
    v.fma = (leaf1.ecx & (1U << 12)) != 0U;

    if (leaf0.eax >= 7U) {
        const CpuidRegs leaf7 = raw_cpuid(7, 0);
        v.avx2 = (leaf7.ebx & (1U << 5)) != 0U;
        v.avx512f = (leaf7.ebx & (1U << 16)) != 0U;
        v.avx512dq = (leaf7.ebx & (1U << 17)) != 0U;
    }

    if (v.osxsave) {
        const std::uint64_t xcr0 = raw_xgetbv(0);
        v.ymm_state = (xcr0 & 0x6U) == 0x6U;
        v.zmm_state = (xcr0 & 0xE0U) == 0xE0U;
    }
    return v;
}

#endif  // LIBSTATS_TEST_X86_FAMILY

std::string active_tier() {
    return stats::simd::ops::VectorOps::get_active_simd_level();
}

/// SIMDPolicy::levelToString() and VectorOps::get_active_simd_level() agree on every
/// tier name except the no-SIMD case, which one spells "None" and the other "Scalar".
std::string normalize_tier(const std::string& tier) {
    return tier == "None" ? "Scalar" : tier;
}

std::string policy_tier() {
    using stats::arch::simd::SIMDPolicy;
    switch (SIMDPolicy::getBestLevel()) {
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
        case SIMDPolicy::Level::None:
            break;
    }
    return "Scalar";
}

}  // namespace

TEST(SimdDispatchGates, Avx512TierImpliesDqAndZmmState) {
#if !defined(LIBSTATS_TEST_X86_FAMILY)
    GTEST_SKIP() << "x86-only gate; nothing to check on this ISA";
#else
    const IndependentView cpu = read_cpu();
    const std::string tier = active_tier();

    if (tier == "AVX-512") {
        EXPECT_TRUE(cpu.avx512dq)
            << "dispatch selected the AVX-512 tier on a CPU without AVX-512DQ (leaf 7 EBX[17]); "
               "simd_avx512.cpp executes _mm512_cvtepi64_pd and would SIGILL here (#117)";
        EXPECT_TRUE(cpu.zmm_state)
            << "dispatch selected the AVX-512 tier while XCR0[7:5] != 111b, so the OS is not "
               "saving opmask/ZMM_Hi256/Hi16_ZMM state (Intel SDM Vol. 1 §15.2) (#117)";
        EXPECT_TRUE(cpu.ymm_state)
            << "dispatch selected the AVX-512 tier while XCR0[2:1] != 11b (#117)";
    }

    #if defined(LIBSTATS_HAS_AVX512)
    // The other side of the guard: with the AVX-512 kernels compiled in, hardware
    // that really does have F+DQ and full ZMM state must have been selected. Without
    // this half, dropping the tier entirely would still pass the block above.
    if (cpu.avx512f && cpu.avx512dq && cpu.zmm_state && cpu.ymm_state) {
        EXPECT_EQ(tier, "AVX-512")
            << "this CPU reports AVX-512F+DQ with full ZMM state and the AVX-512 kernels are "
               "compiled in, but dispatch did not select them";
    }
    #endif

    // And the library's own feature view must match what this file read directly.
    // supports_avx512() is the XCR0 half of #117: CPUID's F bit alone used to set it.
    const bool expect_avx512f = cpu.avx && cpu.ymm_state && cpu.zmm_state && cpu.avx512f;
    EXPECT_EQ(stats::arch::supports_avx512(), expect_avx512f);
    EXPECT_EQ(stats::arch::supports_avx512dq(), expect_avx512f && cpu.avx512dq);
#endif
}

TEST(SimdDispatchGates, Avx2TierImpliesFma) {
#if !defined(LIBSTATS_TEST_X86_FAMILY)
    GTEST_SKIP() << "x86-only gate; nothing to check on this ISA";
#else
    const IndependentView cpu = read_cpu();
    const std::string tier = active_tier();

    if (tier == "AVX2") {
        EXPECT_TRUE(cpu.fma)
            << "dispatch selected the AVX2 tier on a CPU without FMA (leaf 1 ECX[12]); "
               "simd_avx2.cpp is compiled with FMA enabled (#117)";
    }

    #if defined(LIBSTATS_HAS_AVX2)
    // Two-sided: AVX2+FMA hardware must land on AVX2 or better, never below it.
    if (cpu.avx2 && cpu.fma && cpu.ymm_state) {
        EXPECT_TRUE(tier == "AVX2" || tier == "AVX-512")
            << "this CPU reports AVX2+FMA and the AVX2 kernels are compiled in, but dispatch "
               "selected "
            << tier;
    }
    #endif

    EXPECT_EQ(stats::arch::supports_fma(), cpu.avx && cpu.ymm_state && cpu.fma);
    EXPECT_EQ(stats::arch::supports_avx2(), cpu.avx && cpu.ymm_state && cpu.avx2);
#endif
}

TEST(SimdDispatchGates, PolicyAndDispatchAgreeOnTier) {
    // SIMDPolicy and VectorOps gate their tier ladders on the same predicates but from
    // separate translation units. They drifted apart once already (#117 fixed the AVX-512
    // DQ and AVX2 FMA gates in simd_dispatch.cpp before simd_policy.cpp), and a mismatch
    // is silent: SIMDPolicy only feeds block size, threshold and alignment, so it mistunes
    // rather than crashes. Pin the two reporting surfaces together.
    const std::string dispatch = active_tier();
    const std::string policy = policy_tier();

    EXPECT_EQ(policy, normalize_tier(dispatch))
        << "SIMDPolicy::getBestLevel() and VectorOps::get_active_simd_level() disagree; "
           "block size/threshold/alignment would be tuned for a tier the dispatcher is not "
           "running (#117)";

    // getLevelString() is the string surface of the same enum and must not diverge from it.
    EXPECT_EQ(normalize_tier(stats::arch::simd::SIMDPolicy::getLevelString()), policy);
}

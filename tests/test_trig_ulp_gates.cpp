// tests/test_trig_ulp_gates.cpp
//
// ULP accuracy gate for VectorOps::vector_cos and vector_sin (issue #95).
// Ground truth comes from tests/trig_ulp_vectors.inc: cos/sin evaluated at
// 320-bit mpmath precision, each rounded once to nearest double
// (scripts/gen_trig_ulp_vectors.py; ported from libhmm's #74 generator,
// same owner, MIT).
//
// Since v2.5.0 both entry points are corvus kernels, which dispatch on their
// own CPUID; libstats no longer carries per-tier trig kernels, so this gate
// holds the dispatched entry points on the host's tier (corvus's own suite
// gates every tier). The tight budget applies everywhere: corvus's bound is
// max 1 ULP over the full double range on every tier.
//
// Do not loosen any budget below without a matching kernel fix; a budget
// miss here is a kernel bug for the orchestrator, not a test-tuning
// problem. Measured on Zen 4 (2026-08-20, all four x86 tiers, cos and sin):
// max 1 ULP, mean 0.022-0.028 ULP on the main set; 0 ULP on specials.
//
// Demonstrated failing against the unfixed x86 cos kernels prior to #95
// landing (repo rule: a new regression guard must be shown to fail against
// the unfixed state on the platform it targets) -- see PLAN.md's #95 entry
// for the ~6.5e-11 absolute (~2.9e5 ULP) measurement this gate reproduces.

// vector_sin landed with #95's kernel change set; the sin half of the gate
// is active. (Was 0 during the staged fail-first demonstration against the
// pre-#95 cos kernels, when vector_sin did not yet exist.)
#define LIBSTATS_TRIG_GATES_HAVE_SIN 1

#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace {

// Correctly-rounded cos()/sin() reference vectors (input_bits, cos_bits,
// sin_bits), evaluated at 320-bit precision with mpmath then each rounded
// once to nearest double. Defines struct TrigUlpVector, kTrigUlpVectors[]
// (main gate budget) and kTrigUlpSpecials[] (domain-edge/NaN/Inf, gated
// separately). See scripts/gen_trig_ulp_vectors.py.
#include "trig_ulp_vectors.inc"

// The vectorized domain the corrected cos/sin kernels are expected to gate
// their per-lane scalar-libm fixup on once #95 lands (matches
// docs/NEON_TRIG_DERIVATION.md's D_max, already the NEON tier's contract).
constexpr double kTrigDMax = 0x1.0000000000000p+23;  // 2^23

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// Sign-aware ULP distance on the integer lattice; ported from libstats'
// own cosUlpError (scripts/gen_cos_ulp_vectors.py /
// test_simd_neon_cos_accuracy.cpp). Both cos and sin span both signs and
// both have zero crossings, so cross-zero distance is charged at full
// weight rather than being treated as "near enough". inf/NaN handled
// explicitly so the metric never has to trust IEEE comparison operators on
// non-finite values.
double trigUlpError(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (std::isinf(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    const auto ordered = [](double v) -> std::int64_t {
        std::int64_t i;
        std::memcpy(&i, &v, sizeof i);
        return i < 0 ? static_cast<std::int64_t>(0x8000000000000000ULL) - i : i;
    };
    const std::int64_t g = ordered(got), r = ordered(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

}  // namespace

// =========================================================================
// Unit self-tests for trigUlpError. A gate that cannot fail is worthless --
// these pin the metric's own behaviour before it is trusted to gate
// kernels.
// =========================================================================

TEST(TrigUlpErrorSelfTest, AdjacentDoublesAreOneUlp) {
    const double a = 1.0;
    const double b = std::nextafter(a, 2.0);
    EXPECT_DOUBLE_EQ(trigUlpError(a, b), 1.0);
    EXPECT_DOUBLE_EQ(trigUlpError(b, a), 1.0);
}

TEST(TrigUlpErrorSelfTest, EqualIsZero) {
    EXPECT_DOUBLE_EQ(trigUlpError(0.5, 0.5), 0.0);
    EXPECT_DOUBLE_EQ(trigUlpError(-3.25, -3.25), 0.0);
    EXPECT_DOUBLE_EQ(trigUlpError(0.0, 0.0), 0.0);
}

TEST(TrigUlpErrorSelfTest, CrossZeroChargedFullWeight) {
    // Smallest positive and smallest negative subnormals are each exactly
    // one representable step from zero; crossing zero must cost 2 ULP, not
    // be treated as "close" by an unsigned bit-pattern distance.
    const double tiny_pos = std::numeric_limits<double>::denorm_min();
    const double tiny_neg = -tiny_pos;
    EXPECT_DOUBLE_EQ(trigUlpError(tiny_pos, tiny_neg), 2.0);
    EXPECT_DOUBLE_EQ(trigUlpError(tiny_neg, tiny_pos), 2.0);
}

TEST(TrigUlpErrorSelfTest, NanVsNumberIsHuge) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_GT(trigUlpError(nan, 1.0), 1e17);  // got=NaN, ref finite
    EXPECT_GT(trigUlpError(1.0, nan), 1e17);  // got finite, ref=NaN
}

TEST(TrigUlpErrorSelfTest, InfVsNumberIsHuge) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_GT(trigUlpError(inf, 1.0), 1e17);  // got=Inf, ref finite
    EXPECT_GT(trigUlpError(1.0, inf), 1e17);  // got finite, ref=Inf
}

TEST(TrigUlpErrorSelfTest, NanVsNanIsZero) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_DOUBLE_EQ(trigUlpError(nan, nan), 0.0);
}

namespace {

// =========================================================================
// Budgets. FMA tiers (avx2/avx512) get the tight budget since the
// port-back's compensated reduction and fused cores are designed for
// sub-ULP accuracy (already proven at NEON, and validated max 1 ULP on Zen
// 4 per libhmm's #74 rerun -- PLAN.md); sse2/avx have no guaranteed FMA in
// the polynomial cores so get a looser floor. NEON is included for
// completeness (mirrors the existing test_simd_neon_cos_accuracy.cpp
// floor: measured 0.50-0.78 ULP) but skips at runtime on this x86 machine.
// PROVISIONAL until the orchestrator's verification pass records measured
// values back into the design doc -- do not loosen without a kernel fix;
// see the file banner.
// =========================================================================
constexpr double kBudgetTight = 1.0;  // avx2 / avx512 / neon
// [[maybe_unused]]: on a NEON-only (no x86 tier) build this constant has no
// reader, which trips -Wunused-const-variable under GCC/Clang -Wall.
constexpr double kBudgetSpecials = 4.0;  // libm fixup path, every tier

struct GateResult {
    double cos_max = 0.0, cos_mean = 0.0;
    double sin_max = 0.0, sin_mean = 0.0;
    double cos_worst_x = 0.0, sin_worst_x = 0.0;
};

using CosSinFn = void (*)(const double*, double*, std::size_t) noexcept;

// Runs cos_fn (and sin_fn, when LIBSTATS_TRIG_GATES_HAVE_SIN) as ONE batch
// call each over the full vector set, computes max/mean ULP vs. the mpmath
// references, and prints a machine-readable-ish one-liner per function
// (consumed by the orchestrator afterward to record measured values in
// docs).
GateResult run_gate(const char* tier, const char* set_name, const TrigUlpVector* vecs,
                    std::size_t n, CosSinFn cos_fn, CosSinFn sin_fn) {
    std::vector<double> in(n), cos_out(n), sin_out(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = bitsToF64(vecs[i].x_bits);

    cos_fn(in.data(), cos_out.data(), n);
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    sin_fn(in.data(), sin_out.data(), n);
#else
    (void)sin_fn;
    std::fill(sin_out.begin(), sin_out.end(), std::numeric_limits<double>::quiet_NaN());
#endif

    GateResult r;
    double cos_sum = 0.0, sin_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ce = trigUlpError(cos_out[i], bitsToF64(vecs[i].cos_bits));
        cos_sum += ce;
        if (ce > r.cos_max) {
            r.cos_max = ce;
            r.cos_worst_x = in[i];
        }
#if LIBSTATS_TRIG_GATES_HAVE_SIN
        const double se = trigUlpError(sin_out[i], bitsToF64(vecs[i].sin_bits));
        sin_sum += se;
        if (se > r.sin_max) {
            r.sin_max = se;
            r.sin_worst_x = in[i];
        }
#endif
    }
    r.cos_mean = cos_sum / static_cast<double>(n);
    r.sin_mean = sin_sum / static_cast<double>(n);

    std::cout << std::setprecision(17) << "cos " << tier << " " << set_name
              << " max_ulp=" << r.cos_max << " mean_ulp=" << r.cos_mean
              << " worst_x=" << r.cos_worst_x << "\n";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    std::cout << "sin " << tier << " " << set_name << " max_ulp=" << r.sin_max
              << " mean_ulp=" << r.sin_mean << " worst_x=" << r.sin_worst_x << "\n";
#else
    std::cout << "sin " << tier << " " << set_name << " SKIPPED (vector_sin not yet landed, #95)\n";
#endif
    return r;
}

constexpr std::size_t kMainN = sizeof(kTrigUlpVectors) / sizeof(kTrigUlpVectors[0]);
constexpr std::size_t kSpecialsN = sizeof(kTrigUlpSpecials) / sizeof(kTrigUlpSpecials[0]);
// Index contract with scripts/gen_trig_ulp_vectors.py: +/-0 at 0-1, +inf/-inf/NaN at 2-4,
// so all five asserted specials sit inside the first 8-lane vector on every tier.
static_assert(kSpecialsN >= 8, "specials must fill one AVX-512 vector");

void run_main_gate(const char* tier, CosSinFn cos_fn, CosSinFn sin_fn, double budget) {
    const GateResult r = run_gate(tier, "main", kTrigUlpVectors, kMainN, cos_fn, sin_fn);
    EXPECT_LE(r.cos_max, budget) << tier << " cos max ULP over budget (worst x=" << r.cos_worst_x
                                 << ")";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    EXPECT_LE(r.sin_max, budget) << tier << " sin max ULP over budget (worst x=" << r.sin_worst_x
                                 << ")";
#endif
}

// Specials gate: beyond-kTrigDMax finite points route through the batch
// wrappers' per-lane scalar-libm fixup once #95 lands, so every tier is
// held to the libm budget here (not the tight FMA budget). +/-Inf and NaN
// are additionally required to produce NaN EXACTLY, independent of the ULP
// metric (which already scores NaN-vs-NaN as 0, but an explicit check makes
// a silent "kernel returns something finite for Inf" regression fail
// loudly). cos(+/-0)=1 and sin(+/-0)=+/-0 (sign preserved) are asserted
// explicitly too -- the reference set encodes both signs of zero exactly
// (scripts/gen_trig_ulp_vectors.py works around mpmath's float->mpf sign-of-
// zero loss for this).
void run_specials_gate(const char* tier, CosSinFn cos_fn, CosSinFn sin_fn, double budget) {
    std::vector<double> in(kSpecialsN), cos_out(kSpecialsN), sin_out(kSpecialsN);
    for (std::size_t i = 0; i < kSpecialsN; ++i)
        in[i] = bitsToF64(kTrigUlpSpecials[i].x_bits);

    cos_fn(in.data(), cos_out.data(), kSpecialsN);
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    sin_fn(in.data(), sin_out.data(), kSpecialsN);
#else
    (void)sin_fn;
    std::fill(sin_out.begin(), sin_out.end(), std::numeric_limits<double>::quiet_NaN());
#endif

    for (std::size_t i = 0; i < kSpecialsN; ++i) {
        if (!std::isfinite(in[i])) {
            EXPECT_TRUE(std::isnan(cos_out[i]))
                << tier << " specials: cos(" << in[i] << ") must be NaN, got " << cos_out[i];
#if LIBSTATS_TRIG_GATES_HAVE_SIN
            EXPECT_TRUE(std::isnan(sin_out[i]))
                << tier << " specials: sin(" << in[i] << ") must be NaN, got " << sin_out[i];
#endif
        }
    }

    // cos(+0)=1, cos(-0)=1; sin(+0)=+0, sin(-0)=-0 with sign preserved.
    // kTrigUlpSpecials[0] is +0.0, kTrigUlpSpecials[1] is -0.0 (generator
    // order).
    ASSERT_GE(kSpecialsN, 2u);
    EXPECT_EQ(cos_out[0], 1.0) << tier << " cos(+0) must be exactly 1.0";
    EXPECT_EQ(cos_out[1], 1.0) << tier << " cos(-0) must be exactly 1.0";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    EXPECT_EQ(sin_out[0], 0.0) << tier << " sin(+0) must be exactly +0.0";
    EXPECT_TRUE(std::signbit(sin_out[0]) == false) << tier << " sin(+0) sign must be positive";
    EXPECT_EQ(sin_out[1], 0.0) << tier << " sin(-0) must be exactly -0.0";
    EXPECT_TRUE(std::signbit(sin_out[1])) << tier << " sin(-0) sign must be negative";
#endif

    double cos_max = 0.0, cos_sum = 0.0, sin_max = 0.0, sin_sum = 0.0;
    for (std::size_t i = 0; i < kSpecialsN; ++i) {
        const double ce = trigUlpError(cos_out[i], bitsToF64(kTrigUlpSpecials[i].cos_bits));
        cos_sum += ce;
        cos_max = std::max(cos_max, ce);
#if LIBSTATS_TRIG_GATES_HAVE_SIN
        const double se = trigUlpError(sin_out[i], bitsToF64(kTrigUlpSpecials[i].sin_bits));
        sin_sum += se;
        sin_max = std::max(sin_max, se);
#endif
    }
    const double cos_mean = cos_sum / static_cast<double>(kSpecialsN);
    const double sin_mean = sin_sum / static_cast<double>(kSpecialsN);
    std::cout << std::setprecision(10) << "cos " << tier << " specials max_ulp=" << cos_max
              << " mean_ulp=" << cos_mean << "\n";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    std::cout << "sin " << tier << " specials max_ulp=" << sin_max << " mean_ulp=" << sin_mean
              << "\n";
#else
    std::cout << "sin " << tier << " specials SKIPPED (vector_sin not yet landed, #95)\n";
#endif
    EXPECT_LE(cos_max, budget) << tier << " specials cos max ULP over budget";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    EXPECT_LE(sin_max, budget) << tier << " specials sin max ULP over budget";
#endif
}

}  // namespace

// =========================================================================
// Self-check: the generator's own domain assertion, re-verified here so a
// stale/hand-edited .inc can't silently violate the vectorized-domain
// contract the per-tier gates below depend on.
// =========================================================================

TEST(TrigUlpGates, MainVectorsRespectDomainBound) {
    for (std::size_t i = 0; i < kMainN; ++i) {
        const double x = bitsToF64(kTrigUlpVectors[i].x_bits);
        ASSERT_LE(std::fabs(x), kTrigDMax) << "main-bucket vector " << i << " outside kTrigDMax";
    }
}

using stats::arch::simd::VectorOps;

// =========================================================================
// Main and specials gates on the dispatched entry points.
// =========================================================================

TEST(TrigUlpGates, Main) {
    run_main_gate("dispatched", VectorOps::vector_cos, VectorOps::vector_sin, kBudgetTight);
}

TEST(TrigUlpSpecialsGates, Dispatched) {
    run_specials_gate("dispatched", VectorOps::vector_cos, VectorOps::vector_sin, kBudgetSpecials);
}

// Non-lane-multiple sub-span: a length (4999) that is not a multiple of any
// lane count (2/4/8) exercises the masked tail on whatever tier corvus picks.
TEST(TrigUlpGates, SubSpanNonLaneMultipleTailPath) {
    constexpr std::size_t n = 4999;
    static_assert(n < kMainN, "sub-span must fit inside the main vector set");
    static_assert(n % 2 != 0, "sub-span length must not be a multiple of any lane count");
    const GateResult r = run_gate("dispatched-subspan", "subspan_4999", kTrigUlpVectors, n,
                                  VectorOps::vector_cos, VectorOps::vector_sin);
    EXPECT_LE(r.cos_max, kBudgetTight)
        << "sub-span cos max ULP over budget (worst x=" << r.cos_worst_x << ")";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    EXPECT_LE(r.sin_max, kBudgetTight)
        << "sub-span sin max ULP over budget (worst x=" << r.sin_worst_x << ")";
#endif
}

// =========================================================================
// Dispatch-entry coverage: the per-tier gates above deliberately bypass the
// runtime-dispatched VectorOps::vector_cos()/vector_sin() entry points, and
// vector_sin has no in-library consumer until #51 lands -- so a wiring
// mistake in makeDispatchTable() (e.g. t.vector_sin = vector_cos_avx2)
// would pass every gate. This holds the dispatched path, whatever tier the
// CPU selects, to the loose budget on the full main set. Uses 4999 elements
// so the dispatched tier's masked/scalar tail runs too.
// =========================================================================

TEST(TrigUlpGates, DispatchedEntryPoints) {
    constexpr std::size_t n = 4999;
    const GateResult r = run_gate("dispatched", "main_4999", kTrigUlpVectors, n,
                                  VectorOps::vector_cos, VectorOps::vector_sin);
    EXPECT_LE(r.cos_max, kBudgetTight)
        << "dispatched cos max ULP over budget (worst x=" << r.cos_worst_x << ")";
#if LIBSTATS_TRIG_GATES_HAVE_SIN
    EXPECT_LE(r.sin_max, kBudgetTight)
        << "dispatched sin max ULP over budget (worst x=" << r.sin_worst_x << ")";
#endif
}

// ---------------------------------------------------------------------------
// In-place aliasing (review 2026-08-21, AV2). Every trig kernel claims its
// out-of-domain/inf fixup is decided from the pre-store register value so that
// output may alias input; until this test the only guard was the NEON-gated
// NeonCosAccuracy.InPlaceAliasingSafe. This one runs through the dispatched
// entry points, so it covers whichever tier the host selects (all five in CI).
// ---------------------------------------------------------------------------
#include <cstring>
#include <limits>
TEST(TrigUlpGates, InPlaceAliasingMatchesOutOfPlace) {
    const auto bits_of = [](double d) {
        std::uint64_t b;
        std::memcpy(&b, &d, sizeof b);
        return b;
    };
    std::vector<double> in;
    for (std::size_t i = 0; i < 4096; ++i)
        in.push_back(-7.0 + 14.0 * static_cast<double>(i) / 4095.0);
    // Lanes that exercise the register-decided fixup and the specials.
    const double extras[] = {0.0,
                             -0.0,
                             1e300,
                             -1e300,
                             16777216.0,
                             8388608.0,
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(),
                             1e9,
                             -1e9};
    in.insert(in.end(), std::begin(extras), std::end(extras));
    const std::size_t n = in.size();  // 4107: not a lane multiple, so the tail runs too

    std::vector<double> cos_sep(n), sin_sep(n);
    VectorOps::vector_cos(in.data(), cos_sep.data(), n);
    VectorOps::vector_sin(in.data(), sin_sep.data(), n);

    std::vector<double> cos_alias(in), sin_alias(in);
    VectorOps::vector_cos(cos_alias.data(), cos_alias.data(), n);
    VectorOps::vector_sin(sin_alias.data(), sin_alias.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(bits_of(cos_sep[i]), bits_of(cos_alias[i]))
            << "cos in-place differs at lane " << i << " (x = " << in[i] << ")";
        EXPECT_EQ(bits_of(sin_sep[i]), bits_of(sin_alias[i]))
            << "sin in-place differs at lane " << i << " (x = " << in[i] << ")";
    }
}

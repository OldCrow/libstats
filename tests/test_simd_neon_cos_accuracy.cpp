/**
 * @file test_simd_neon_cos_accuracy.cpp
 * @brief Regression test: vector_cos_neon holds the <1 ULP accuracy floor.
 *
 * vector_cos_neon was replaced 2026-07-19 with a clean-room quadrant-reduction
 * kernel (4-part exact-product pi/2 split, compensated reduced argument,
 * degree-6 minimax cores; see docs/NEON_TRIG_DERIVATION.md and
 * docs/NEON_TRIG_DIVERGENCE_AUDIT.md). The pre-replacement kernel measured
 * ~6e8 ULP inside [-2pi, 2pi] and produced sign errors near k*pi/2, so this
 * gate is load-bearing: the reference set deliberately includes doubles at and
 * near odd multiples of pi/2 across the whole supported domain (|x| <= 2^23).
 *
 * Architecture-neutral: on non-NEON builds (or if NEON is unavailable at
 * runtime) this test is skipped rather than failing, since vector_cos() will
 * dispatch to a different backend there.
 */

#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

// Correctly-rounded cos() reference vectors (input_bits, cos_bits), evaluated
// at 320-bit precision with mpmath then rounded once to nearest double.
// Defines struct CosUlpVector and kCosUlpVectors[]. See scripts/gen_cos_ulp_vectors.py.
#include "cos_ulp_vectors.inc"

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

std::uint64_t f64ToBits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

// ULP distance on the integer lattice; sign-aware (cos spans both signs, and
// the old kernel's failure mode near k*pi/2 was exactly a cross-zero sign
// error, which this metric charges at full weight). inf/NaN explicit.
double cosUlpError(double got, double ref) {
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

TEST(NeonCosAccuracy, HoldsSubUlpFloorVsCorrectlyRoundedReference) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_cos dispatches elsewhere";
    }

    constexpr std::size_t n = sizeof(kCosUlpVectors) / sizeof(kCosUlpVectors[0]);
    std::vector<double> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = bitsToF64(kCosUlpVectors[i].x_bits);

    stats::arch::simd::VectorOps::vector_cos(in.data(), out.data(), n);

    double max_ulp = 0.0, sum_ulp = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double err = cosUlpError(out[i], bitsToF64(kCosUlpVectors[i].cos_bits));
        sum_ulp += err;
        if (err > max_ulp) {
            max_ulp = err;
            worst = i;
        }
    }

    // Clean-room kernel measured max 0.78 ULP uniform / 0.50 stress in the
    // standalone harness; 1.0 here is the regression floor (the old kernel
    // measured ~6e8 on the same class of points).
    EXPECT_LE(max_ulp, 1.0) << "worst x = " << in[worst] << " (bits 0x" << std::hex
                            << kCosUlpVectors[worst].x_bits << ")";
    EXPECT_LE(sum_ulp / static_cast<double>(n), 0.5);
}

TEST(NeonCosAccuracy, EdgeCasesAndDomainFallback) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_cos dispatches elsewhere";
    }

    const double pinf = std::numeric_limits<double>::infinity();
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double dmax = 0x1p23;  // supported-domain bound

    // even count so every lane goes through the SIMD path; includes both sides
    // of the domain bound (beyond it the kernel must match std::cos bitwise
    // via the scalar fallback) and the tiny/zero region.
    const std::vector<double> in = {0.0,   -0.0,      0x1p-30,  -0x1p-30, 1.5,  -1.5,
                                    dmax,  -dmax,     dmax * 4, 1e300,    pinf, qnan,
                                    -pinf, 0x1p-1074, 3.0,      -3.0};
    std::vector<double> out(in.size());
    stats::arch::simd::VectorOps::vector_cos(in.data(), out.data(), in.size());

    for (std::size_t i = 0; i < in.size(); ++i) {
        const double ref = std::cos(in[i]);
        if (std::isnan(ref)) {
            EXPECT_TRUE(std::isnan(out[i])) << "x = " << in[i];
        } else {
            EXPECT_EQ(f64ToBits(out[i]), f64ToBits(ref)) << "x = " << in[i];
        }
    }
    EXPECT_EQ(out[0], 1.0);  // cos(+0) = 1 exactly
    EXPECT_EQ(out[1], 1.0);  // cos(-0) = 1 exactly
}

TEST(NeonCosAccuracy, EvenSymmetryBitwise) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_cos dispatches elsewhere";
    }

    // cos is even; the kernel's reduction and sign logic must preserve that
    // bitwise (DERIVATION.md sec. 1.2). Deterministic pseudo-random sweep.
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    constexpr std::size_t n = 4096;
    std::vector<double> pos(n), neg(n), outp(n), outn(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double mag = std::ldexp(1.0 + static_cast<double>(next() >> 12) * 0x1p-52,
                                      static_cast<int>(next() % 26) - 2);  // up to ~2^23
        pos[i] = mag;
        neg[i] = -mag;
    }
    stats::arch::simd::VectorOps::vector_cos(pos.data(), outp.data(), n);
    stats::arch::simd::VectorOps::vector_cos(neg.data(), outn.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        ASSERT_EQ(f64ToBits(outp[i]), f64ToBits(outn[i])) << "x = " << pos[i];
}

TEST(NeonCosAccuracy, InPlaceAliasingSafe) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_cos dispatches elsewhere";
    }

    // Must stay correct when output aliases input; include beyond-domain and
    // special lanes so the register-decided fixup path is exercised.
    std::vector<double> buf = {0.5, 2.0, 1e300, 3.5, 0.0, -1.5, 0x1p23 * 2, 100.0};
    std::vector<double> expected(buf.size());
    stats::arch::simd::VectorOps::vector_cos(buf.data(), expected.data(), buf.size());
    stats::arch::simd::VectorOps::vector_cos(buf.data(), buf.data(), buf.size());
    for (std::size_t i = 0; i < buf.size(); ++i)
        EXPECT_EQ(f64ToBits(buf[i]), f64ToBits(expected[i])) << "lane " << i;
}

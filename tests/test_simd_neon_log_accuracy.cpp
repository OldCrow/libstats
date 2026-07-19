/**
 * @file test_simd_neon_log_accuracy.cpp
 * @brief Regression test: vector_log_neon holds the <1 ULP accuracy floor.
 *
 * vector_log_neon was replaced 2026-07-19 with a clean-room table+series
 * kernel (compensated anchors, no division; see docs/NEON_LOG_DERIVATION.md
 * and docs/NEON_LOG_DIVERGENCE_AUDIT.md). This test guards that replacement
 * against future regressions using the correctly-rounded mpmath reference set
 * (Issue #33 Q1 vectors, scripts/gen_log_ulp_vectors.py), plus explicit IEEE
 * edge-case checks the reference sweep does not cover.
 *
 * Architecture-neutral: on non-NEON builds (or if NEON is unavailable at
 * runtime) this test is skipped rather than failing, since vector_log() will
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

// Correctly-rounded log() reference vectors (input_bits, log_bits), evaluated at
// 200-bit precision with mpmath then rounded once to nearest double.
// Defines struct LogUlpVector and kLogUlpVectors[]. See scripts/gen_log_ulp_vectors.py.
#include "log_ulp_vectors.inc"

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

// ULP distance on the integer lattice; sign-aware (log spans both signs),
// inf/NaN handled explicitly.
double logUlpError(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (std::isinf(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    // map to a monotone signed-integer lattice so cross-zero distances are valid
    const auto ordered = [](double v) -> std::int64_t {
        std::int64_t i;
        std::memcpy(&i, &v, sizeof i);
        return i < 0 ? static_cast<std::int64_t>(0x8000000000000000ULL) - i : i;
    };
    const std::int64_t g = ordered(got), r = ordered(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

}  // namespace

TEST(NeonLogAccuracy, HoldsSubUlpFloorVsCorrectlyRoundedReference) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_log dispatches elsewhere";
    }

    constexpr std::size_t n = sizeof(kLogUlpVectors) / sizeof(kLogUlpVectors[0]);
    std::vector<double> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i)
        in[i] = bitsToF64(kLogUlpVectors[i].x_bits);

    stats::arch::simd::VectorOps::vector_log(in.data(), out.data(), n);

    double max_ulp = 0.0, sum_ulp = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double err = logUlpError(out[i], bitsToF64(kLogUlpVectors[i].log_bits));
        sum_ulp += err;
        if (err > max_ulp) {
            max_ulp = err;
            worst = i;
        }
    }

    // Clean-room kernel measured max 0.52 ULP over harsher stress buckets than
    // this sweep; 1.0 here is the regression floor (the old kernel measured 2.0).
    EXPECT_LE(max_ulp, 1.0) << "worst x = " << in[worst] << " (bits 0x" << std::hex
                            << kLogUlpVectors[worst].x_bits << ")";
    EXPECT_LE(sum_ulp / static_cast<double>(n), 0.5);
}

TEST(NeonLogAccuracy, IeeeEdgeCasesBitwiseExact) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_log dispatches elsewhere";
    }

    const double pinf = std::numeric_limits<double>::infinity();
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double min_normal = 0x1p-1022;
    const double min_sub = 0x0.0000000000001p-1022;  // smallest positive subnormal

    // even count so every lane goes through the SIMD path
    const std::vector<double> in = {1.0,       0.0,        -0.0,    -1.0,        pinf, qnan,
                                    -pinf,     min_normal, min_sub, 0x1.8p-1023,  // subnormals
                                    0x1p-1074, 4.0};
    std::vector<double> out(in.size());
    stats::arch::simd::VectorOps::vector_log(in.data(), out.data(), in.size());

    for (std::size_t i = 0; i < in.size(); ++i) {
        const double ref = std::log(in[i]);
        if (std::isnan(ref)) {
            EXPECT_TRUE(std::isnan(out[i])) << "x = " << in[i];
        } else {
            EXPECT_EQ(f64ToBits(out[i]), f64ToBits(ref)) << "x = " << in[i];
        }
    }
    EXPECT_EQ(f64ToBits(out[0]), f64ToBits(+0.0));  // log(1) = +0 exactly
}

TEST(NeonLogAccuracy, InPlaceAliasingSafe) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON unavailable; vector_log dispatches elsewhere";
    }

    // vector_log must stay correct when result aliases the input (the pow
    // composition and log-space pipelines call it in place); include special
    // lanes so the register-decided fixup path is exercised under aliasing.
    std::vector<double> buf = {0.5, 2.0, 0.0, 3.5, 1.0, -1.0, 1e300, 0x1p-1074};
    std::vector<double> expected(buf.size());
    stats::arch::simd::VectorOps::vector_log(buf.data(), expected.data(), buf.size());
    stats::arch::simd::VectorOps::vector_log(buf.data(), buf.data(), buf.size());
    for (std::size_t i = 0; i < buf.size(); ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(buf[i])) << "lane " << i;
        } else {
            EXPECT_EQ(f64ToBits(buf[i]), f64ToBits(expected[i])) << "lane " << i;
        }
    }
}

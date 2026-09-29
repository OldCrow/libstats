#pragma once
/**
 * @file bessel.h
 * @brief Modified Bessel functions of the first kind for VonMisesDistribution.
 *
 * I₀(x), I₁(x), log I₀(x) and the ratio helpers A(κ) = I₁(κ)/I₀(κ) and
 * 1 − A(κ). Since v2.5.0 the kernels are corvus's i0/i1/i0e/i1e (max 1 ULP on
 * every SIMD tier, docs/ACCURACY.md there), reached through the definitions in
 * src/math_utils.cpp. This header declares only: it is installed, and it keeps
 * corvus out of the public include surface.
 *
 * History: v2.0–v2.4 selected between std::cyl_bessel_i (GCC/MSVC) and an
 * A&S §9.8 polynomial fallback (AppleClang, ~1.6e-7) at configure time; the
 * generated libstats_config.h that carried that choice (#97) is gone with it.
 */

namespace stats {
namespace detail {

/// I₀(x). Even; saturates to +inf past |x| ≈ 713.99.
[[nodiscard]] double bessel_i0(double x) noexcept;

/// I₁(x). Odd; saturates to ±inf past |x| ≈ 713.99.
[[nodiscard]] double bessel_i1(double x) noexcept;

/// log I₀(x) = |x| + log i0e(x). Even, finite for every finite x, with no
/// asymptotic seam: i0e never underflows on a finite double.
[[nodiscard]] double log_bessel_i0(double x) noexcept;

// ---------------------------------------------------------------------------
// Ratio helpers: A(κ) = I₁(κ)/I₀(κ) and its complement 1 − A(κ)
//
// WHY THESE EXIST (issue #93)
// ---------------------------
// A(κ) is the von Mises mean resultant length and 1 − A(κ) is its circular
// variance, so both are wanted. Neither can be computed from the other across
// the whole domain, because each is the one that cancels somewhere:
//
//   * A(κ) → 1 − 1/(2κ) as κ grows, so forming `1 − A` in double discards
//     about log₂(2κ) bits NO MATTER HOW ACCURATE A IS. Computing A better
//     cannot fix the variance; the complement needs its own route.
//   * A(κ) → κ/2 as κ → 0, so forming `1 − complement` cancels at the other
//     end. Each therefore has its own direct path in its own regime.
//
// Below kBesselRatioAsymptoticCut, A is i1e(κ)/i0e(κ) — the scaled pair, so
// nothing overflows — and the complement is 1 − A. Above it the complement
// comes from the asymptotic series in 1/κ and A is 1 − complement.
//
// EVERY coefficient of the series is an exact rational, obtained by exact
// rational arithmetic rather than fitted. 1 − A is the ratio of the two Hankel
// expansions (A&S 9.7.1 at ν = 0 and ν = 1); the e^κ/√(2πκ) prefactor cancels,
// so dividing them as power series in t = 1/κ over the rationals yields each
// c_k exactly, with no solve and no conditioning:
//
//   1/2, 1/8, 1/8, 25/128, 13/32, 1073/1024, 103/32,
//   375733/32768, 23797/512, 55384775/262144
//
// The first shipped version (#93) fitted these with a Vandermonde solve and got
// the top three wrong — c10 by 0.199 — corrected in #96. If this series is
// ever extended, derive by series division.
//
// The cut of 50 was measured against mpmath with the pre-corvus kernels
// (std::cyl_bessel_i): the direct branch and the series crossed at ~110–130
// ULP there, the series alone being 1057 ULP at 40 and 17 at 60. corvus's
// 1-ULP i0e/i1e lower the direct branch's error, so the true crossover has
// likely moved up; re-measure before moving the cut. The residual band near
// the cut is intrinsic to double — the complement is ~1/(2κ), so any error in
// A is amplified 2κ-fold — and closing it needs the double-double complement
// filed as corvus #19. Away from the cut the helper is sub-ULP.
//
// The domain is x ≥ 0, matching κ.
// ---------------------------------------------------------------------------

inline constexpr double kBesselRatioAsymptoticCut = 50.0;

/// @brief 1 − I₁(x)/I₀(x) — the von Mises circular variance. Requires x ≥ 0.
[[nodiscard]] double bessel_i1_i0_complement(double x) noexcept;

/// @brief I₁(x)/I₀(x) — the von Mises mean resultant length A(κ). Requires x ≥ 0.
[[nodiscard]] double bessel_i1_over_i0(double x) noexcept;

}  // namespace detail
}  // namespace stats

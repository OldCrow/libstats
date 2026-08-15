#pragma once

/**
 * @file bessel.h
 * @brief Modified Bessel functions of the first kind for VonMisesDistribution.
 *
 * Provides I₀(x), I₁(x), and log I₀(x) via two implementation tiers:
 *
 *   Tier 1 (LIBSTATS_HAS_CXX17_BESSEL defined):
 *     Delegates to std::cyl_bessel_i(ν, x) from <cmath> (C++17 §29.9.3).
 *     Available on GCC 6.1+, MSVC 2017 15.5+.  Not available on AppleClang /
 *     macOS libc++ (unimplemented as of Xcode 16 / macOS 14).
 *
 *   Tier 2 (portable fallback — active on macOS with system AppleClang):
 *     Polynomial approximations from Abramowitz & Stegun §9.8.1–9.8.4.
 *     Accurate to ε < 1.6×10⁻⁷ in the polynomial region.  For log I₀(x) at
 *     large x the asymptotic expansion avoids exp() overflow.
 *
 * CMakeLists.txt detects std::cyl_bessel_i via check_cxx_source_compiles and
 * defines LIBSTATS_HAS_CXX17_BESSEL when available.
 *
 * Ported from libhmm/include/libhmm/math/bessel.h with the following changes:
 *   - Namespace: libhmm::detail → stats::detail
 *   - Macro:     LIBHMM_HAS_CXX17_BESSEL → LIBSTATS_HAS_CXX17_BESSEL
 */

#include <cmath>

namespace stats {
namespace detail {

#if defined(LIBSTATS_HAS_CXX17_BESSEL)

// ---------------------------------------------------------------------------
// Tier 1: delegate to C++17 <cmath> special functions
// ---------------------------------------------------------------------------

[[nodiscard]] inline double bessel_i0(double x) noexcept {
    return std::cyl_bessel_i(0.0, x);
}

[[nodiscard]] inline double bessel_i1(double x) noexcept {
    return std::cyl_bessel_i(1.0, x);
}

// 2π as a double. Identical to 2.0 * M_PI (doubling is exact, and 2×π_double
// is the nearest double to 2π), but M_PI is not standard C++ — it needs
// _USE_MATH_DEFINES before <cmath> on MSVC, which CMakeLists.txt supplies
// globally. Spelling it locally makes this header self-contained for any
// consumer that includes it directly.
inline constexpr double kTwoPi = 6.283185307179586476925286766559;

[[nodiscard]] inline double log_bessel_i0(double x) noexcept {
    // For large x, I₀(x) overflows double (near x ≈ 713.99), so switch to the
    // asymptotic form:
    //     I₀(x) ~ e^x/√(2πx) · Σ_k c_k x^-k,   c_k = ((2k−1)!!)² / (k! 8^k)
    //     log I₀(x) = x − ½·log(2πx) + log1p(Σ_{k≥1} c_k x^-k)
    //
    // FIVE terms, not two (#92). The original carried only c₁ = 1/8 and
    // c₂ = 9/128, truncating at O(x⁻³) — and c₃ = 225/3072 evaluates to
    // 0.0732/700³ = 2.13×10⁻¹⁰, which was precisely the size of the step this
    // branch introduced at the switchover. The branches were not value-matched,
    // so log I₀ was DISCONTINUOUS at x = 700: 0.4 → 1881 ULP between adjacent
    // points, inherited as a visible step by any density built on it.
    //
    // Measured against mpmath at dps 60, error at x = 700 (worst point for the
    // branch, since 1/x is largest there): 1385 ULP at two terms, 0.009 at
    // four, 0.000 at five. Seam discontinuity falls from 2.139e-10 to 4.8e-14,
    // which is the direct path's OWN error — the asymptotic is no longer the
    // limiting side, which is why five and not four.
    //
    // Coefficients are exact dyadic rationals, generated from the c_k formula
    // rather than transcribed: 1/8, 9/128, 225/3072, 11025/98304, 893025/2^22.
    if (x > 700.0) {
        const double t = 1.0 / x;
        const double s =
            t * (0.125 +
                 t * (0.0703125 +
                      t * (0.0732421875 + t * (0.112152099609375 + t * 0.22710800170898438))));
        return x - 0.5 * std::log(kTwoPi * x) + std::log1p(s);
    }
    return std::log(std::cyl_bessel_i(0.0, x));
}

#else

// ---------------------------------------------------------------------------
// Tier 2: A&S polynomial approximations (portable fallback)
//
// I₀(x):  A&S 9.8.1 (|x| ≤ 3.75) and 9.8.2 (|x| > 3.75)
// I₁(x):  A&S 9.8.3 (|x| ≤ 3.75) and 9.8.4 (|x| > 3.75)
//
// Numerical precision: error < 1.6×10⁻⁷ in the polynomial region.
// ---------------------------------------------------------------------------

[[nodiscard]] inline double bessel_i0(double x) noexcept {
    const double ax = std::fabs(x);
    if (ax <= 3.75) {
        const double t = (ax / 3.75) * (ax / 3.75);
        return 1.0 +
               t * (3.5156229 +
                    t * (3.0899424 +
                         t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    } else {
        const double t = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) *
               (0.39894228 +
                t * (0.01328592 +
                     t * (0.00225319 +
                          t * (-0.00157565 +
                               t * (0.00916281 + t * (-0.02057706 +
                                                      t * (0.02635537 + t * (-0.01647633 +
                                                                             t * 0.00392377))))))));
    }
}

[[nodiscard]] inline double bessel_i1(double x) noexcept {
    const double ax = std::fabs(x);
    double result;
    if (ax <= 3.75) {
        const double t = (ax / 3.75) * (ax / 3.75);
        result =
            ax *
            (0.5 +
             t * (0.87890594 +
                  t * (0.51498869 +
                       t * (0.15084934 + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
    } else {
        const double t = 3.75 / ax;
        result =
            (std::exp(ax) / std::sqrt(ax)) *
            (0.39894228 +
             t * (-0.03988024 +
                  t * (-0.00362018 +
                       t * (0.00163801 +
                            t * (-0.01031555 +
                                 t * (0.02282967 + t * (-0.02895312 +
                                                        t * (0.01787654 + t * (-0.00420059)))))))));
    }
    return (x < 0.0) ? -result : result;
}

[[nodiscard]] inline double log_bessel_i0(double x) noexcept {
    // For x > 3.75: use the factored form to avoid exp() overflow.
    //   log I₀(x) = x − 0.5·log(x) + log(P(3.75/x))
    // where P is the A&S 9.8.2 polynomial factor (exp/sqrt already divided out).
    const double ax = std::fabs(x);
    if (ax <= 3.75) {
        return std::log(bessel_i0(ax));
    } else {
        const double t = 3.75 / ax;
        const double poly =
            0.39894228 +
            t * (0.01328592 +
                 t * (0.00225319 +
                      t * (-0.00157565 +
                           t * (0.00916281 +
                                t * (-0.02057706 +
                                     t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
        return ax - 0.5 * std::log(ax) + std::log(poly);
    }
}

#endif  // LIBSTATS_HAS_CXX17_BESSEL

// ---------------------------------------------------------------------------
// Ratio helpers: A(κ) = I₁(κ)/I₀(κ) and its complement 1 − A(κ)
//
// Tier-independent — these build on whichever bessel_i0/bessel_i1 was selected
// above, so they need no per-tier duplication.
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
// Above kBesselRatioAsymptoticCut the complement comes from the asymptotic
// series in 1/κ; below it, directly from the two Bessel values.
//
// The cut also repairs an overflow defect for free: I₀ and I₁ both exceed
// DBL_MAX around κ ≈ 713, and the old `(i0 > 0.0)` guard does not catch inf,
// so `inf/inf` made the circular variance NaN for any κ past that. Every κ in
// the overflow region now takes the series branch, which never evaluates
// either Bessel function.
//
// Coefficients were derived by solving a Vandermonde system in 1/κ against
// mpmath at dps 220, not quoted from a table. The low orders come out exactly
// dyadic — 1/2, 1/8, 1/8, 25/128, 13/32, 1073/1024 — which is what confirms
// the solve; the top three are the numerical continuation.
//
// The cut was chosen by evaluating BOTH branches at the same κ in this
// compiled header and scoring against mpmath — not from a model. An earlier
// estimate that rounded an exact I₁/I₀ once put the naive branch at ~25 ULP
// and suggested a cut of 60; the shipped path calls bessel_i0 and bessel_i1
// separately, so two independent multi-ULP errors are amplified by the same
// 2κ factor and the real figures are an order worse and much noisier:
//
//        κ      series      naive
//       30    19749 ULP      76 ULP
//       45      318 ULP     125 ULP
//       50      110 ULP     127 ULP   <- they cross here
//       60       17 ULP     177 ULP
//       80      0.6 ULP     206 ULP
//      100      0.4 ULP      47 ULP
//
// Worst case over κ ∈ [15, 105] by cut: 1050 ULP at 40, 318 at 45, **129 at
// 50**, 298 at 60. Hence 50.
//
// The residual ~130 ULP near the cut is close to intrinsic for double here,
// not slack left on the table. The complement is ~1/(2κ), so ANY error in A is
// amplified 2κ-fold: even a correctly-rounded A leaves a floor around κ ULP
// (~50 at the cut). Extra series terms make it worse below the cut, since the
// expansion is asymptotic and diverges — ten terms already give 1050 ULP at
// κ = 40. Closing that band would need extended precision, which this library
// has no layer for. Away from the cut the helper is sub-ULP: < 1 ULP for
// κ ≳ 80 and for κ ≲ 20.
//
// The domain is x ≥ 0, matching κ.
// ---------------------------------------------------------------------------

inline constexpr double kBesselRatioAsymptoticCut = 50.0;

/// @brief 1 − I₁(x)/I₀(x) — the von Mises circular variance. Requires x ≥ 0.
[[nodiscard]] inline double bessel_i1_i0_complement(double x) noexcept {
    if (x >= kBesselRatioAsymptoticCut) {
        // Horner in t = 1/x. Finite for every x up to +inf (t → 0 → 0).
        const double t = 1.0 / x;
        return t * (0.5 +
                    t * (0.125 +
                         t * (0.125 +
                              t * (0.1953125 +
                                   t * (0.40625 +
                                        t * (1.0478515625 +
                                             t * (3.21875 +
                                                  t * (11.466461181921275 +
                                                       t * (46.478503876575118 +
                                                            t * 211.47489057159319)))))))));
    }
    const double i0 = bessel_i0(x);
    const double i1 = bessel_i1(x);
    return (i0 > 0.0) ? (1.0 - i1 / i0) : 1.0;
}

/// @brief I₁(x)/I₀(x) — the von Mises mean resultant length A(κ). Requires x ≥ 0.
[[nodiscard]] inline double bessel_i1_over_i0(double x) noexcept {
    // Past the cut the complement is small, so 1 − complement is the stable
    // direction here; below it, the ratio is taken directly for the reason
    // given above (1 − complement would cancel as κ → 0).
    if (x >= kBesselRatioAsymptoticCut) {
        return 1.0 - bessel_i1_i0_complement(x);
    }
    const double i0 = bessel_i0(x);
    const double i1 = bessel_i1(x);
    return (i0 > 0.0) ? (i1 / i0) : 0.0;
}

}  // namespace detail
}  // namespace stats

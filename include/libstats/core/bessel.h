#pragma once

/**
 * @file bessel.h
 * @brief Modified Bessel functions of the first kind for VonMisesDistribution.
 *
 * Provides I₀(x), I₁(x), and log I₀(x) via three implementation tiers:
 *
 *   Tier 0 (LIBSTATS_USE_CORVUS defined — opt-in, OFF by default):
 *     Delegates to corvus, a SIMD special-function library with per-tier
 *     audited ULP bounds (max 1 ULP for i0/i1/i0e/i1e on every validated
 *     SIMD target).  Retires Tier 2's 1.6×10⁻⁷ accuracy ceiling on
 *     macOS/AppleClang, which is what issue #47 is about.
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

#if defined(LIBSTATS_USE_CORVUS)
    #include <corvus/corvus.h>

    #include <span>
#endif

namespace stats {
namespace detail {

#if defined(LIBSTATS_USE_CORVUS)

// ---------------------------------------------------------------------------
// Tier 0: delegate to corvus (opt-in via LIBSTATS_USE_CORVUS)
//
// corvus takes spans, not scalars. Every call site in this repo is scalar and
// sits in parameter-cache or fit-time code (src/von_mises.cpp, 8 sites, none
// in a hot loop), so a span-of-1 wrapper is the right shape: it costs one
// runtime-dispatch indirect call per invocation and buys the accuracy. If a
// batch von Mises path ever wants these, it should call corvus with a real
// span rather than looping over these wrappers.
//
// WHY log I₀ COMPOSES INSTEAD OF HAVING ITS OWN KERNEL
// ----------------------------------------------------
// corvus deliberately exports no log_i0. It documents the composition
//     log I₀(x) = log(i0e(x)) + |x|          [i0e(x) = I₀(x)·e^{−|x|}]
// as relative error < 1 ULP for x ≳ 2, and absolute error ≤ 3.3×10⁻¹⁶ on the
// whole axis. The composition is relatively WEAK as x → 0: log I₀(x) ~ x²/4
// there, so log(i0e) ≈ x²/4 − x cancels against the +x and the small result
// keeps only absolute accuracy.
//
// That weakness is unreachable from this repo, at all three call sites:
//
//   updateCacheUnsafe()  logNormaliser_ = LN_2PI + log I₀(κ)
//   getDifferentialEntropy()  H = LN_2PI − log I₀(κ) + κ·A(κ)
//
// Both embed log I₀ in a sum anchored by LN_2PI ≈ 1.8379, so the governing
// contract is ABSOLUTE, not relative: 3.3×10⁻¹⁶ against a result of magnitude
// ≥ 1.8 is ≤ 1.5×10⁻¹⁶ relative, inside one ULP of the answer (ulp(1.84) =
// 2.22×10⁻¹⁶). The entropy site additionally short-circuits on isUniform_
// (κ < 1e-10) before reaching here. So the small-κ regime where the
// composition is weak either cannot be reached or cannot be observed.
//
// At LARGE κ the composition is strictly BETTER than Tier 1, and that is
// MEASURED rather than argued. Tier 1 cannot call std::cyl_bessel_i above
// x = 700 (I₀ overflows double) and falls back to the two-term A&S
// asymptotic below, whose truncation is O(1/x³). Against mpmath at dps 50:
// at κ = 1000 that fallback is 7.3×10⁻¹¹ absolute, while this composition is
// 2.2×10⁻¹⁴ — better by ~3300×. corvus's i0e is 1-ULP everywhere and |x| is
// exact, so the composition has no such seam.
//
// Every κ ≤ 700 row of the spike sweep is bit-identical between the two
// tiers, with one exception: at κ = 8.1 the composition is 0.7 ULP and
// Tier 1 is 0.3 ULP. Both sit inside the documented bound; neither is a
// defect. (Spike measurement, 2026-08-15.)
//
// VERDICT: adopt the composition as-is; no dedicated log-I₀ kernel is needed
// for this consumer. (Spike adjudication, 2026-08-15.)
// ---------------------------------------------------------------------------

[[nodiscard]] inline double bessel_i0(double x) noexcept {
    const double in = x;  // I₀ is even; corvus handles both signs natively
    double out;
    corvus::i0(std::span<const double>(&in, 1), std::span<double>(&out, 1));
    return out;
}

[[nodiscard]] inline double bessel_i1(double x) noexcept {
    const double in = x;  // I₁ is odd; corvus reapplies the sign internally
    double out;
    corvus::i1(std::span<const double>(&in, 1), std::span<double>(&out, 1));
    return out;
}

[[nodiscard]] inline double log_bessel_i0(double x) noexcept {
    // |x| on both sides of the composition, so this is correct for either
    // scaling convention and matches Tier 2's use of fabs. I₀ is even.
    const double ax = std::fabs(x);
    double scaled;
    corvus::i0e(std::span<const double>(&ax, 1), std::span<double>(&scaled, 1));
    return std::log(scaled) + ax;
}

#elif defined(LIBSTATS_HAS_CXX17_BESSEL)

// ---------------------------------------------------------------------------
// Tier 1: delegate to C++17 <cmath> special functions
// ---------------------------------------------------------------------------

[[nodiscard]] inline double bessel_i0(double x) noexcept {
    return std::cyl_bessel_i(0.0, x);
}

[[nodiscard]] inline double bessel_i1(double x) noexcept {
    return std::cyl_bessel_i(1.0, x);
}

[[nodiscard]] inline double log_bessel_i0(double x) noexcept {
    // For large x, I₀(x) overflows double; use the asymptotic form instead.
    // I₀(x) ≈ exp(x)/√(2πx) · [1 + 1/(8x) + 9/(128x²) + ...]
    // log I₀(x) ≈ x − 0.5·log(2πx) + log(1 + 1/(8x) + ...)
    if (x > 700.0) {  // exp(710) ≈ DBL_MAX
        const double t = 1.0 / x;
        return x - 0.5 * std::log(2.0 * M_PI * x) + std::log1p(0.125 * t + 0.0703125 * t * t);
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

}  // namespace detail
}  // namespace stats

#include "libstats/distributions/von_mises.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/bessel.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stats {

//==============================================================================
// Private helper: angle wrapping
//==============================================================================

// Upper bound of the validated #51 Bessel-series CDF range; above it the
// pre-#51 wrapped-normal fallback is used (two code sites, one constant).
constexpr double kCdfSeriesKappaMax = 1000.0;

double VonMisesDistribution::wrapAngle(double x) noexcept {
    if (!std::isfinite(x))
        return x;
    x = std::fmod(x, detail::TWO_PI);
    if (x <= -detail::PI)
        x += detail::TWO_PI;
    if (x > detail::PI)
        x -= detail::TWO_PI;
    return x;
}

//==============================================================================
// Private helper: κ from mean resultant length R̄
//
// Mardia–Jupp approximation (Mardia & Jupp 2000, Directional Statistics §A.2),
// refined by Newton–Raphson on A(κ) = I₁(κ)/I₀(κ) = R̄.
// Derivative: A'(κ) = 1 − A(κ)² − A(κ)/κ.
// Always converges (function is monotone increasing); 3–5 Newton steps suffice.
//==============================================================================

namespace {

[[nodiscard]] double kappa_from_r_bar(double R_bar) noexcept {
    if (R_bar <= 0.0)
        return 0.0;
    if (R_bar >= 1.0)
        return 1.0e6;  // effectively point mass

    double kappa;
    if (R_bar < 0.53) {
        kappa = detail::TWO * R_bar + R_bar * R_bar * R_bar +
                (5.0 / 6.0) * R_bar * R_bar * R_bar * R_bar * R_bar;
    } else if (R_bar < 0.85) {
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (detail::ONE - R_bar);
    } else {
        const double r = R_bar;
        kappa = detail::ONE / (r * r * r - 4.0 * r * r + 3.0 * r);
    }
    if (kappa < 0.0)
        kappa = 0.0;

    for (int iter = 0; iter < 20 && kappa > 0.0; ++iter) {
        // A(κ) via the ratio helper (#93): the direct i1/i0 form returned NaN
        // above κ ≈ 713, where both values overflow and the old `i0 <= 0.0`
        // guard did not catch inf — the Newton step then propagated NaN and
        // the fit silently produced a NaN concentration.
        const double A = detail::bessel_i1_over_i0(kappa);
        const double Ap = detail::ONE - A * A - A / kappa;
        if (std::fabs(Ap) < 1e-15)
            break;
        const double dk = (A - R_bar) / Ap;
        kappa -= dk;
        if (kappa < 0.0) {
            kappa = 0.0;
            break;
        }
        if (std::fabs(dk) < 1e-12 * (detail::ONE + kappa))
            break;
    }
    return kappa;
}

//==============================================================================
// CDF Bessel-series coefficients (issue #51)
//
// F(x) = (t+pi)/(2pi) + sum_{j=1}^{j_max} b_j * sin(j*t),  t = wrap(x-mu)
// b_j = I_j(kappa) / (j * pi * I0(kappa))
//
// Miller backward recurrence: I_{j-1}(kappa) = I_{j+1}(kappa) + (2j/kappa)*I_j(kappa),
// run downward from an arbitrary seed at N = j_max+15 (the forward recurrence is
// unstable once j exceeds ~kappa and must not be used -- see issue #51). No Bessel
// anchor is needed: every series term uses only the ratio I_j/I0, and backward
// recurrence delivers every f_j proportional to I_j up to one common (arbitrary)
// scale factor, so f_j/f_0 IS I_j/I0 exactly -- no corvus/A&S dependency, no #47
// exposure. Rescaling mid-recurrence by 1e250 whenever magnitudes grow guards
// overflow; the ratios are invariant under a uniform rescale of the array.
//==============================================================================

[[nodiscard]] std::vector<double> vonmises_cdf_series_coeffs(double kappa, int j_max) {
    const int N = j_max + 15;
    std::vector<double> f(static_cast<std::size_t>(N + 2), 0.0);
    f[static_cast<std::size_t>(N + 1)] = 0.0;
    f[static_cast<std::size_t>(N)] = 1e-30;

    for (int j = N; j >= 1; --j) {
        const auto jm1 = static_cast<std::size_t>(j - 1);
        const auto jj = static_cast<std::size_t>(j);
        const auto jp1 = static_cast<std::size_t>(j + 1);
        f[jm1] = f[jp1] + (detail::TWO * static_cast<double>(j) / kappa) * f[jj];
        if (std::fabs(f[jm1]) > 1e250) {
            for (std::size_t m = jm1; m <= static_cast<std::size_t>(N + 1); ++m)
                f[m] /= 1e250;
        }
    }

    std::vector<double> b(static_cast<std::size_t>(j_max));
    const double f0 = f[0];
    for (int j = 1; j <= j_max; ++j) {
        b[static_cast<std::size_t>(j - 1)] =
            (f[static_cast<std::size_t>(j)] / f0) / (static_cast<double>(j) * detail::PI);
    }
    return b;
}

// The series CDF at t = wrap(x - mu), before the tail correction. Terms are
// summed j_max -> 1 (smallest first) so the largest terms accumulate last,
// keeping the round-off floor low.
[[nodiscard]] double vonmises_series_cdf(double t, const std::vector<double>& b) noexcept {
    double sum = detail::ZERO_DOUBLE;
    for (std::size_t j = b.size(); j >= 1; --j)
        sum += b[j - 1] * std::sin(static_cast<double>(j) * t);
    return (t + detail::PI) / detail::TWO_PI + sum;
}

// The same series with sin(j t) by the rotation recurrence instead of one
// sin per term. sin(j t) then carries ~j ulp, so the sum carries ~eps times
// sum_j j b_j = (e^k / I0(k) - 1) / (2 pi), ~13 ulp at kappa = 1000: too much
// for the CDF, ample for a quantile seed, whose error costs iterations only.
[[nodiscard]] double vonmises_series_cdf_fast(double t, const std::vector<double>& b) noexcept {
    const double s1 = std::sin(t);
    const double c1 = std::cos(t);
    double s = s1;
    double c = c1;
    double sum = detail::ZERO_DOUBLE;
    for (std::size_t j = 1; j <= b.size(); ++j) {
        sum += b[j - 1] * s;
        const double s_next = s * c1 + c * s1;
        c = c * c1 - s * s1;
        s = s_next;
    }
    return (t + detail::PI) / detail::TWO_PI + sum;
}

//==============================================================================
// Tail mass by direct quadrature (CDF tails and the quantile)
//
// The Bessel series above sums terms of order 1, so it carries only absolute
// accuracy (~1e-16): it returned 5.55e-17 or 0 where F is 1e-40 or 1e-100. On
// the small side of the probability scale the mass is integrated directly
// instead. For mu = 0, t in [-pi, 0] and d = t + pi (the distance from the
// left edge, with pi exact: d = (t + PI) + kPiLo),
//
//   G(t) = int_{-pi}^{t} f = g(t) * J(t),
//   J(t) = int_0^d exp(-2k sin(d - u/2) sin(u/2)) du,
//
// where g(t) = exp(-2k cos^2(d/2)) / (2 pi I0e(k)) is the density at t. The
// identity cos(t - u) - cos(t) = -2 sin(d - u/2) sin(u/2) keeps the exponent
// free of cancellation; sin(d - u/2) = sin(u/2 - t) is used for t > -pi/2,
// where d is near pi. The integrand is 1 at u = 0 and decreases
// monotonically, so J is a sum of positive terms with a relative error of a
// few ulp. ln G = ln g + ln J does not underflow, which the quantile's Newton
// step uses (d ln G/dt = 1/J). The upper tail is the mirror: 1 - F(t) = G(-t).
//
// J is integrated by adaptive Gauss-Kronrod 7/15 (QUADPACK qk15 nodes) over
// [0, U], where U cuts the integrand at e^-80 by bisection on the monotone
// exponent. The exponent is >= -k u, so whenever the cut applies J >= (1 -
// 1/e)/k; the dropped piece, below pi e^-80, is then under 1e-34 k relative.
//
// I0e(k) = I0(k) e^-k comes from the trapezoid rule on the periodic integrand
// exp(k (cos phi - 1)), exponentially convergent (aliasing ~ e^(-N^2/(2k)),
// N^2/(2k) >= 50 here), and not from log_bessel_i0(k) - k, which cancels
// ~log2(k) bits and is only 1.6e-7 accurate on the Tier 2 Bessel path. Above
// kappa = 1000 it comes from the Hankel series log_bessel_i0 uses there,
// without its leading k.
//==============================================================================

// pi - PI: the double PI is below pi by this much (to ~1e-32).
constexpr double kPiLo = 1.2246467991473532e-16;

// The series CDF is replaced by the tail quadrature where it falls below this
// (or above 1 minus this). Inside the band the series' absolute error, at
// most ~2e-16 (measured at kappa = 1000), is within the accuracy law relative
// to F.
constexpr double kTailSwitch = 0.25;

// Below kTailSwitch the quantile still seeds the tail solve from the series
// (vonmises_series_cdf_fast) down to this m, where the recurrence's absolute
// error, ~1e-14, is 1e-8 relative: one tail evaluation then converges.
constexpr double kSeriesSeedMin = 1e-6;

// ln(2 pi I0(kappa) e^-kappa), the log normaliser of the density exp(kappa (cos t - 1)).
[[nodiscard]] double log_scaled_normaliser(double kappa) noexcept {
    if (kappa > kCdfSeriesKappaMax) {
        const double t = detail::ONE / kappa;
        const double s =
            t * (0.125 + t * (0.0703125 + t * (0.0732421875 +
                                               t * (0.112152099609375 + t * 0.22710800170898438))));
        return detail::LN_2PI - detail::HALF * std::log(detail::TWO_PI * kappa) + std::log1p(s);
    }
    // N = 2*half points over the full period; the two halves are mirror images.
    const int half = static_cast<int>(std::ceil(detail::HALF * std::sqrt(100.0 * kappa))) + 8;
    const double n = 2.0 * static_cast<double>(half);
    double sum = std::exp(-detail::TWO * kappa);  // phi = pi; terms summed smallest first
    for (int j = half - 1; j >= 1; --j) {
        const double s = std::sin(detail::PI * static_cast<double>(j) / n);
        sum += detail::TWO * std::exp(-detail::TWO * kappa * s * s);
    }
    sum += detail::ONE;  // phi = 0
    return detail::LN_2PI + std::log(sum / n);
}

struct LeftTail {
    double log_mass;  ///< ln G(t)
    double j;         ///< J(t) = G(t) / g(t)
};

// G(t) for mu = 0 and t in [-PI, 0], in log form; see the block comment above.
// Since J <= d, ln G <= ln g + ln d; when that bound is below log_floor the
// bound itself is returned (with j = 0) and the quadrature skipped -- the CDF
// passes a floor below which its result rounds to 0 (or 1 - G to 1) anyway.
[[nodiscard]] LeftTail vonmises_left_tail(
    double t, double kappa, double log_scaled_norm,
    double log_floor = -std::numeric_limits<double>::infinity()) noexcept {
    const double d = (t + detail::PI) + kPiLo;
    const bool near_edge = t < -detail::HALF * detail::PI;
    // cos(d/2) = sin(-t/2); each form is taken where its argument is accurate.
    const double c = near_edge ? std::cos(detail::HALF * d) : std::sin(-detail::HALF * t);
    const double log_density = -detail::TWO * kappa * c * c - log_scaled_norm;
    const double log_bound = log_density + std::log(d);
    if (log_bound < log_floor)
        return {log_bound, detail::ZERO_DOUBLE};

    auto exponent = [&](double u) {
        const double s =
            near_edge ? std::sin(d - detail::HALF * u) : std::sin(detail::HALF * u - t);
        return -detail::TWO * kappa * s * std::sin(detail::HALF * u);
    };

    // Cut the integration range where the integrand falls below e^-80.
    constexpr double kCut = -80.0;
    double upper = d;
    if (exponent(d) < kCut) {
        double lo = detail::ZERO_DOUBLE;
        for (int i = 0; i < 60 && upper - lo > 1e-3 * upper; ++i) {
            const double mid = detail::HALF * (lo + upper);
            if (exponent(mid) < kCut)
                upper = mid;
            else
                lo = mid;
        }
    }

    // Adaptive Gauss-Kronrod 7/15.
    static constexpr double xgk[8] = {
        0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
        0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
        0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
        0.207784955007898467600689403773245, 0.000000000000000000000000000000000};
    static constexpr double wgk[8] = {
        0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
        0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
        0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
        0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static constexpr double wg[4] = {
        0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
        0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    auto gk15 = [&](double a, double b, double& kronrod) {
        const double centre = detail::HALF * (a + b);
        const double halfLength = detail::HALF * (b - a);
        const double fc = std::exp(exponent(centre));
        double resk = fc * wgk[7];
        double resg = fc * wg[3];
        for (int i = 0; i < 7; ++i) {
            const double dx = halfLength * xgk[i];
            const double fsum = std::exp(exponent(centre - dx)) + std::exp(exponent(centre + dx));
            resk += wgk[i] * fsum;
            if (i % 2 == 1)
                resg += wg[i / 2] * fsum;
        }
        kronrod = resk * halfLength;
        return std::fabs((resk - resg) * halfLength);
    };

    double whole = detail::ZERO_DOUBLE;
    gk15(detail::ZERO_DOUBLE, upper, whole);
    // Accept an interval once |K15 - G7| is below 1e-10 of the whole, pro rata
    // by length. That difference is the G7 error; K15, exact to twice the
    // polynomial degree, is then accurate to well below an ulp (QUADPACK's
    // estimate, 200|K-G| (200|K-G|/I)^1.5, puts it near 1e-19 relative).
    const double tolerance = 1e-10 * whole / upper;
    constexpr int kMaxDepth = 48;
    struct Interval {
        double a, b;
        int depth;
    };
    Interval stack[kMaxDepth + 2];
    int top = 0;
    stack[top++] = {detail::ZERO_DOUBLE, upper, 0};
    double j = detail::ZERO_DOUBLE;
    int calls_left = 2000;  // backstop against a degenerate tolerance
    while (top > 0) {
        const Interval iv = stack[--top];
        double k15 = detail::ZERO_DOUBLE;
        const double err = gk15(iv.a, iv.b, k15);
        if (err <= tolerance * (iv.b - iv.a) || iv.depth >= kMaxDepth || --calls_left <= 0) {
            j += k15;
        } else {
            const double mid = detail::HALF * (iv.a + iv.b);
            stack[top++] = {mid, iv.b, iv.depth + 1};
            stack[top++] = {iv.a, mid, iv.depth + 1};  // the left (larger) half first
        }
    }
    return {log_density + std::log(j), j};
}

// One Halley step for h(v) = 0 from the value h and its first three
// derivatives h1, h2, h3 at the current point; `predicted` is Halley's
// asymptotic error after the step, (c2^2 - c3) e^3 with c2 = h2/(2 h1) and
// c3 = h3/(6 h1), bounded here by (c2^2 + |c3|) |step|^3 so that the two terms
// cannot cancel. A Newton step is taken where the Halley denominator is far
// from 1 (the cubic model is then not trustworthy); its error is c2 step^2.
struct HalleyStep {
    double step;
    double predicted;  ///< |error| after the step, to leading order
};

[[nodiscard]] HalleyStep halley_step(double h, double h1, double h2, double h3) noexcept {
    const double newton = -h / h1;
    const double c2 = h2 / (detail::TWO * h1);
    const double c3 = h3 / (6.0 * h1);
    const double den = detail::ONE + newton * c2;
    if (den > detail::HALF && den < detail::TWO) {
        const double step = newton / den;
        const double a = std::fabs(step);
        return {step, (c2 * c2 + std::fabs(c3)) * a * a * a};
    }
    return {newton, std::fabs(c2) * newton * newton};
}

// t in [-PI, 0] with G(t) = m, for 0 < m < 1/2 (mu = 0). Halley on ln G, which
// is well scaled from 1e-300 to 1/2: in t away from the edge, in s = ln d near
// it, where G is nearly linear in d and a t-step would overshoot; bracketed,
// with bisection (geometric in d when the bracket spans decades) whenever a
// step leaves the bracket. Returns -PI when the answer is within an ulp of the
// edge.
//
// With L = ln G = ln g + ln J and g'/g = -kappa sin t, J = G/g satisfies
// J' = 1 + kappa sin(t) J, so one quadrature gives every derivative:
//   L' = 1/J,  L'' = -J'/J^2,  L''' = 2 J'^2/J^3 - J''/J^2,
//   J'' = kappa (cos(t) J + sin(t) J').
// In s (dt/ds = d2t/ds2 = d3t/ds3 = d): L_s = d L', L_ss = d^2 L'' + d L',
// L_sss = d^3 L''' + 3 d^2 L'' + d L'.
[[nodiscard]] double vonmises_left_quantile(double m, double kappa, double log_scaled_norm,
                                            double seed) noexcept {
    const double log_m = std::log(m);
    // g is increasing on [-pi, 0], so m / g(0) <= d <= m / g(-pi).
    const double log_d_lo = log_m + log_scaled_norm;
    const double log_d_hi = log_m + detail::TWO * kappa + log_scaled_norm;
    if (log_d_hi < std::log(kPiLo + 2.3e-16))
        return -detail::PI;
    auto t_of = [](double d) { return std::max(-detail::PI, (d - kPiLo) - detail::PI); };
    double t_lo = std::max(-detail::PI, t_of(std::exp(log_d_lo)) - 1e-15);
    double t_hi =
        log_d_hi < std::log(detail::PI) ? t_of(std::exp(log_d_hi)) + 1e-15 : detail::ZERO_DOUBLE;
    t_hi = std::min(t_hi, detail::ZERO_DOUBLE);

    // Seed: the caller's (NaN for none), else the wrapped normal at moderate
    // kappa and the uniform below it.
    double t = seed;
    if (!(t > t_lo && t < t_hi))
        t = kappa >= detail::ONE ? detail::inverse_normal_cdf(m) / std::sqrt(kappa)
                                 : detail::TWO_PI * m - detail::PI;
    if (!(t > t_lo && t < t_hi))
        t = t_of(std::exp(detail::HALF * (log_d_lo + std::min(log_d_hi, std::log(detail::PI)))));
    if (!(t > t_lo && t < t_hi))
        t = detail::HALF * (t_lo + t_hi);

    constexpr double kEps = std::numeric_limits<double>::epsilon();
    for (int iter = 0; iter < 100; ++iter) {
        const LeftTail tail = vonmises_left_tail(t, kappa, log_scaled_norm);
        const double r = tail.log_mass - log_m;
        if (r == detail::ZERO_DOUBLE)
            return t;
        if (r < detail::ZERO_DOUBLE)
            t_lo = t;
        else
            t_hi = t;

        const double d = (t + detail::PI) + kPiLo;
        const bool near_edge = t < -detail::HALF * detail::PI;
        const double sin_t = near_edge ? -std::sin(d) : std::sin(t);
        const double cos_t = near_edge ? -std::cos(d) : std::cos(t);
        const double j = tail.j;
        const double j1 = detail::ONE + kappa * sin_t * j;
        const double j2 = kappa * (cos_t * j + sin_t * j1);
        double l1 = detail::ONE / j;
        double l2 = -j1 / (j * j);
        double l3 = (detail::TWO * j1 * j1 / j - j2) / (j * j);
        const bool log_step = d < detail::HALF;
        if (log_step) {
            l3 = d * (d * (d * l3 + 3.0 * l2) + l1);
            l2 = d * (d * l2 + l1);
            l1 = d * l1;
        }
        const HalleyStep hs = halley_step(r, l1, l2, l3);
        const double d_next = log_step ? d * std::exp(hs.step) : d + hs.step;
        double next = log_step ? t_of(d_next) : t + hs.step;
        // The predicted error, in t; trusted only once G is within 0.1% of m,
        // where the step is well inside the radius of the cubic model.
        double predicted = std::fabs(r) > 1e-3 ? std::numeric_limits<double>::infinity()
                           : log_step          ? hs.predicted * d_next
                                               : hs.predicted;
        // Converged once the step, or the error it leaves, is below the noise
        // that ln G's own error, ~|ln m| ulp absolute, leaves in t -- or below
        // half an ulp of t, which near the edge is the larger. A step that
        // small can round onto t itself, outside the open bracket.
        const double tolerance = std::max(detail::TWO * kEps * (detail::ONE - log_m) * j,
                                          detail::HALF * kEps * std::fabs(t));
        if (std::fabs(next - t) <= tolerance)
            return next;
        if (!(next > t_lo && next < t_hi)) {
            const double d_lo = (t_lo + detail::PI) + kPiLo;
            const double d_hi = (t_hi + detail::PI) + kPiLo;
            next = d_hi > 4.0 * d_lo ? t_of(std::sqrt(d_lo * d_hi)) : detail::HALF * (t_lo + t_hi);
            if (!(next > t_lo && next < t_hi))
                return t;  // the bracket is down to adjacent doubles
            predicted = std::numeric_limits<double>::infinity();
        }
        if (predicted <= tolerance)
            return next;
        t = next;
    }
    return t;
}

// t in [-PI, 0] with S(t) = m (mu = 0), S the Bessel series CDF on the cached
// coefficients b: by the CDF's own summation when `exact`, else by
// vonmises_series_cdf_fast. Halley in t from the seed, with S' = f,
// S'' = -kappa sin(t) f, S''' = kappa (kappa sin^2(t) - cos(t)) f; bracketed by
// [-PI, 0], with bisection whenever a step leaves the bracket. Converged once
// the step, or the error it leaves, is below the evaluation's absolute error
// mapped to t (noise / f): ~1 ulp of m for the exact sum, 64 ulp of 1 for the
// recurrence.
[[nodiscard]] double vonmises_series_quantile(double m, double kappa, double log_scaled_norm,
                                              const std::vector<double>& b, double t,
                                              bool exact) noexcept {
    double t_lo = -detail::PI;
    double t_hi = detail::ZERO_DOUBLE;
    if (!(t > t_lo && t < t_hi))
        t = detail::HALF * (t_lo + t_hi);

    constexpr double kEps = std::numeric_limits<double>::epsilon();
    const double noise = exact ? kEps * m : 64.0 * kEps;
    for (int iter = 0; iter < 100; ++iter) {
        const double r = (exact ? vonmises_series_cdf(t, b) : vonmises_series_cdf_fast(t, b)) - m;
        if (r == detail::ZERO_DOUBLE)
            return t;
        if (r < detail::ZERO_DOUBLE)
            t_lo = t;
        else
            t_hi = t;
        const double sin_t = std::sin(t);
        const double cos_t = std::cos(t);
        const double f = std::exp(kappa * (cos_t - detail::ONE) - log_scaled_norm);
        const HalleyStep hs = halley_step(r, f, -kappa * sin_t * f,
                                          kappa * (kappa * sin_t * sin_t - cos_t) * f);
        double next = t + hs.step;
        double predicted = std::fabs(r) > 1e-3 * m ? std::numeric_limits<double>::infinity()
                                                   : hs.predicted;
        const double tolerance = std::max(noise / f, detail::HALF * kEps * std::fabs(t));
        if (std::fabs(next - t) <= tolerance)
            return next;
        if (!(next > t_lo && next < t_hi)) {
            next = detail::HALF * (t_lo + t_hi);
            if (!(next > t_lo && next < t_hi))
                return t;
            predicted = std::numeric_limits<double>::infinity();
        }
        if (predicted <= tolerance)
            return next;
        t = next;
    }
    return t;
}

// The series CDF value `series` at t = wrap(x - mu), with each tail replaced
// by the directly integrated mass (the series has absolute accuracy only).
// The upper tail is the mirror of the lower: 1 - F(t) = G(-t).
[[nodiscard]] double tail_corrected_cdf(double series, double t, double kappa,
                                        double log_scaled_norm) noexcept {
    // Floors: e^-746 is below half the smallest subnormal, and 1 - e^-40 rounds to 1.
    if (series < kTailSwitch && t <= detail::ZERO_DOUBLE)
        return std::exp(vonmises_left_tail(t, kappa, log_scaled_norm, -746.0).log_mass);
    if (series > detail::ONE - kTailSwitch && t >= detail::ZERO_DOUBLE)
        return detail::ONE -
               std::exp(vonmises_left_tail(-t, kappa, log_scaled_norm, -40.0).log_mass);
    return std::clamp(series, detail::ZERO_DOUBLE, detail::ONE);
}

}  // anonymous namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

VonMisesDistribution::VonMisesDistribution(double mu, double kappa)
    : DistributionBase(), mu_(wrapAngle(mu)), kappa_(kappa) {
    validateParameters(mu_, kappa_);
    updateCacheUnsafe();
}

VonMisesDistribution::VonMisesDistribution(const VonMisesDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    kappa_ = other.kappa_;
    logNormaliser_ = other.logNormaliser_;
    logScaledNormaliser_ = other.logScaledNormaliser_;
    circularVariance_ = other.circularVariance_;
    isUniform_ = other.isUniform_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
}

VonMisesDistribution& VonMisesDistribution::operator=(const VonMisesDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_ = other.mu_;
        kappa_ = other.kappa_;
        logNormaliser_ = other.logNormaliser_;
        logScaledNormaliser_ = other.logScaledNormaliser_;
        circularVariance_ = other.circularVariance_;
        isUniform_ = other.isUniform_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicKappa_.store(kappa_, std::memory_order_release);
    }
    return *this;
}

VonMisesDistribution::VonMisesDistribution(VonMisesDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mu_ = other.mu_;
    kappa_ = other.kappa_;
    logNormaliser_ = other.logNormaliser_;
    logScaledNormaliser_ = other.logScaledNormaliser_;
    circularVariance_ = other.circularVariance_;
    isUniform_ = other.isUniform_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.kappa_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
}

VonMisesDistribution& VonMisesDistribution::operator=(VonMisesDistribution&& other) noexcept {
    if (this != &other) {
        mu_ = other.mu_;
        kappa_ = other.kappa_;
        logNormaliser_ = other.logNormaliser_;
        logScaledNormaliser_ = other.logScaledNormaliser_;
        circularVariance_ = other.circularVariance_;
        isUniform_ = other.isUniform_;
        other.mu_ = detail::ZERO_DOUBLE;
        other.kappa_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicKappa_.store(kappa_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

VonMisesDistribution VonMisesDistribution::createUnchecked(double mu, double kappa) noexcept {
    return VonMisesDistribution(wrapAngle(mu), kappa, true);
}

VonMisesDistribution::VonMisesDistribution(double mu, double kappa,
                                           bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), kappa_(kappa) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void VonMisesDistribution::setMu(double mu) {
    validateParameters(mu, getKappa());
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    // Changing only μ doesn't affect logNormaliser_ or circularVariance_
    // (those depend only on κ), but cache_valid_ is reset for thread safety.
    updateCacheUnsafe();
}

void VonMisesDistribution::setKappa(double kappa) {
    validateParameters(getMu(), kappa);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void VonMisesDistribution::setParameters(double mu, double kappa) {
    validateParameters(mu, kappa);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double VonMisesDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mu_;
}

double VonMisesDistribution::getVariance() const {
    double cv;
    withCacheSnapshot([&] { cv = circularVariance_; });
    return cv;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult VonMisesDistribution::trySetMu(double mu) noexcept {
    auto v = validateVonMisesParameters(mu, getKappa());
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult VonMisesDistribution::trySetKappa(double kappa) noexcept {
    auto v = validateVonMisesParameters(getMu(), kappa);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult VonMisesDistribution::trySetParameters(double mu, double kappa) noexcept {
    auto v = validateVonMisesParameters(mu, kappa);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult VonMisesDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateVonMisesParameters(mu_, kappa_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double VonMisesDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;  // ±inf → PDF is 0

    double k, mu, lnorm;
    withCacheSnapshot([&] {
        k = kappa_;
        mu = mu_;
        lnorm = logNormaliser_;
    });
    return std::exp(k * std::cos(x - mu) - lnorm);
}

double VonMisesDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;  // ±inf → log PDF is -∞

    double k, mu, lnorm;
    withCacheSnapshot([&] {
        k = kappa_;
        mu = mu_;
        lnorm = logNormaliser_;
    });
    return k * std::cos(x - mu) - lnorm;
}

double VonMisesDistribution::getCumulativeProbability(double x) const {
    if (!std::isfinite(x)) {
        if (std::isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        return (x > 0 ? detail::ONE : detail::ZERO_DOUBLE);
    }

    double result = detail::ZERO_DOUBLE;
    withCacheSnapshot([&] {
        const double kappa = kappa_;
        const double mu = mu_;

        // kappa = 0 (uniform circular distribution): exact linear CDF.
        if (isUniform_) {
            const double t = wrapAngle(x - mu);
            result = std::clamp(((t + detail::PI) + kPiLo) / detail::TWO_PI, detail::ZERO_DOUBLE,
                                detail::ONE);
            return;
        }

        // kappa > 1000: unvalidated range for the #51 series -- use the
        // pre-#51 wrapped-normal approximation. VM(mu, kappa) ~ N(mu, 1/kappa)
        // on the circle; approximation error is ~0.043/kappa absolute (measured
        // against a quadrature oracle at kappa = 1e3, 2e3, 1e4 -- O(1/kappa),
        // not O(1/kappa^2)). The standardised argument is the WRAPPED
        // DIFFERENCE, matching the series branch below: wrapping x alone and
        // subtracting mu afterwards leaves a 2*pi offset for every x on the far
        // side of the +-pi cut from mu (#106).
        if (kappa > kCdfSeriesKappaMax) {
            const double z = wrapAngle(x - mu) * std::sqrt(kappa);
            result = std::clamp(detail::HALF * (detail::ONE + std::erf(z * detail::INV_SQRT_2)),
                                detail::ZERO_DOUBLE, detail::ONE);
            return;
        }

        // 0 < kappa <= 1000: Bessel-series CDF (issue #51), PROVISIONAL pending
        // the mpmath-oracle accuracy gate (tests/test_vonmises_cdf_accuracy.cpp).
        //   F(t) = (t+pi)/(2pi) + sum_{j=1}^{j_max} b_j * sin(j*t),  t = wrap(x-mu)
        // b_j = cdfSeriesCoeffs_[j-1], from the Miller recurrence in
        // updateCacheUnsafe() (no Bessel anchor -- f_j/f_0 IS I_j/I0 exactly).
        const double t = wrapAngle(x - mu);
        result = tail_corrected_cdf(vonmises_series_cdf(t, cdfSeriesCoeffs_), t, kappa,
                                    logScaledNormaliser_);
    });

    return result;
}

double VonMisesDistribution::getQuantile(double p) const {
    if (std::isnan(p))
        return std::numeric_limits<double>::quiet_NaN();
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }

    // Solve on the small side of the probability scale: the CDF below the
    // median, the survival above it (1 - p is exact there). In the band where
    // the CDF uses the Bessel series (m >= kTailSwitch), the quantile solves on
    // that series: on its fast form first, then on the CDF's own sum, which
    // then converges in one evaluation. Below the band it solves on the
    // directly integrated tail mass with Halley on its logarithm
    // (vonmises_left_quantile), seeded from the fast series down to
    // kSeriesSeedMin. By symmetry the upper quantile is the mirror of the
    // lower one.
    const bool upper = p > detail::HALF;
    const double m = upper ? detail::ONE - p : p;
    double t = -detail::PI;
    bool solved = false;
    double seed = std::numeric_limits<double>::quiet_NaN();
    double mu, kappa, log_scaled_norm;
    withCacheSnapshot([&] {
        mu = mu_;
        kappa = kappa_;
        log_scaled_norm = logScaledNormaliser_;
        if (m < kSeriesSeedMin || m >= detail::HALF || cdfSeriesCoeffs_.empty())
            return;
        const double t0 = kappa >= detail::ONE ? detail::inverse_normal_cdf(m) / std::sqrt(kappa)
                                               : detail::TWO_PI * m - detail::PI;
        seed = vonmises_series_quantile(m, kappa, log_scaled_norm, cdfSeriesCoeffs_, t0, false);
        if (m >= kTailSwitch) {
            t = vonmises_series_quantile(m, kappa, log_scaled_norm, cdfSeriesCoeffs_, seed, true);
            solved = true;
        }
    });

    if (m >= detail::HALF)
        t = detail::ZERO_DOUBLE;
    else if (!solved && m > detail::ZERO_DOUBLE)
        t = vonmises_left_quantile(m, kappa, log_scaled_norm, seed);

    if (upper) {
        t = -t;
    } else if (t <= -detail::PI) {
        // The left end stays at the left end. The support is (-pi, pi] and
        // wrapAngle reads -PI as +PI, so the smallest double above -PI stands
        // for a quantile within an ulp of -pi (p = 1e-300 used to return +pi).
        t = std::nextafter(-detail::PI, detail::ZERO_DOUBLE);
    }
    return wrapAngle(t + mu);
}

double VonMisesDistribution::sample(std::mt19937& rng) const {
    double kappa, mu;
    withCacheSnapshot([&] {
        kappa = kappa_;
        mu = mu_;
    });

    // Near-uniform case (κ ≈ 0): sample uniformly on the circle.
    if (kappa < 1e-9) {
        std::uniform_real_distribution<double> u(-detail::PI, detail::PI);
        return u(rng);
    }

    // Best (1979) rejection sampler for the Von Mises distribution.
    // Reference: D.J. Best and N.I. Fisher (1979). Efficient simulation of
    //            the von Mises distribution. Applied Statistics 28(2), 152–157.
    const double tau = detail::ONE + std::sqrt(detail::ONE + 4.0 * kappa * kappa);
    const double rho = (tau - std::sqrt(detail::TWO * tau)) / (detail::TWO * kappa);
    const double r = (detail::ONE + rho * rho) / (detail::TWO * rho);

    std::uniform_real_distribution<double> u01(detail::ZERO_DOUBLE, detail::ONE);

    for (;;) {
        const double u1 = u01(rng);
        const double z = std::cos(detail::PI * u1);
        const double f = (detail::ONE + r * z) / (r + z);
        const double c = kappa * (r - f);
        const double u2 = u01(rng);

        bool accept = false;
        if (c * (detail::TWO - c) > u2) {
            accept = true;
        } else if (c > detail::ZERO_DOUBLE) {
            accept = (std::log(c / u2) + detail::ONE - c >= detail::ZERO_DOUBLE);
        }

        if (accept) {
            const double u3 = u01(rng);
            const double angle = (u3 > detail::HALF) ? std::acos(f) : -std::acos(f);
            return wrapAngle(mu + angle);
        }
    }
}

std::vector<double> VonMisesDistribution::sample(std::mt19937& rng, size_t n) const {
    double kappa, mu;
    withCacheSnapshot([&] {
        kappa = kappa_;
        mu = mu_;
    });

    std::vector<double> samples;
    samples.reserve(n);

    if (kappa < 1e-9) {
        std::uniform_real_distribution<double> u(-detail::PI, detail::PI);
        for (size_t i = 0; i < n; ++i)
            samples.push_back(u(rng));
        return samples;
    }

    // Best (1979) rejection sampler — precompute constants outside the loop.
    const double tau = detail::ONE + std::sqrt(detail::ONE + 4.0 * kappa * kappa);
    const double rho = (tau - std::sqrt(detail::TWO * tau)) / (detail::TWO * kappa);
    const double r = (detail::ONE + rho * rho) / (detail::TWO * rho);
    std::uniform_real_distribution<double> u01(detail::ZERO_DOUBLE, detail::ONE);

    for (size_t i = 0; i < n; ++i) {
        for (;;) {
            const double u1 = u01(rng);
            const double z = std::cos(detail::PI * u1);
            const double f = (detail::ONE + r * z) / (r + z);
            const double c = kappa * (r - f);
            const double u2 = u01(rng);
            bool accept = (c * (detail::TWO - c) > u2) ||
                          (c > detail::ZERO_DOUBLE &&
                           std::log(c / u2) + detail::ONE - c >= detail::ZERO_DOUBLE);
            if (accept) {
                const double u3 = u01(rng);
                const double angle = (u3 > detail::HALF) ? std::acos(f) : -std::acos(f);
                samples.push_back(wrapAngle(mu + angle));
                break;
            }
        }
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void VonMisesDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    double S = detail::ZERO_DOUBLE, C = detail::ZERO_DOUBLE;
    for (double x : values) {
        // FIT-4: NaN or Inf corrupts the sin/cos accumulation silently.
        if (!std::isfinite(x))
            throw std::invalid_argument("All values must be finite for VonMises fit");
        S += std::sin(x);
        C += std::cos(x);
    }
    const double n = static_cast<double>(values.size());
    const double mu_hat = wrapAngle(std::atan2(S / n, C / n));
    const double R_bar = std::sqrt(S * S + C * C) / n;
    const double kappa_hat = kappa_from_r_bar(R_bar);

    setParameters(mu_hat, kappa_hat);
}

void VonMisesDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<VonMisesDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void VonMisesDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    kappa_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);  // NEW-TS-4
    updateCacheUnsafe();
}

std::string VonMisesDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "VonMisesDistribution(mu=" << mu_ << ",kappa=" << kappa_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double VonMisesDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicMu_.load(std::memory_order_acquire);
    return getMu();
}

double VonMisesDistribution::getKappaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicKappa_.load(std::memory_order_acquire);
    return getKappa();
}

double VonMisesDistribution::getCircularVariance() const {
    return getVariance();
}

double VonMisesDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mu_;
}

double VonMisesDistribution::getEntropy() const {
    bool is_uniform;
    double k;
    withCacheSnapshot([&] {
        is_uniform = isUniform_;
        k = kappa_;
    });
    // H = log(2π) − log I₀(κ) + κ·I₁(κ)/I₀(κ)
    // At κ=0: H = log(2π) − log(1) + 0 = log(2π) ✓ (uniform on the circle)
    if (is_uniform)
        return detail::LN_2PI;
    const double log_i0 = detail::log_bessel_i0(k);
    // A(κ) via the ratio helper: the direct i1/i0 form returned NaN above
    // κ ≈ 713 where both values overflow (#93).
    const double A1 = detail::bessel_i1_over_i0(k);
    return detail::LN_2PI - log_i0 + k * A1;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void VonMisesDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const VonMisesDistribution& d, double x) { return d.getProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            d.getProbabilityBatchUnsafeImpl(vals, res, count, k, mu, lnorm);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        return;
                    }
                    res[i] = std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm)
                                              : detail::ZERO_DOUBLE;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        continue;
                    }
                    res[i] = std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm)
                                              : detail::ZERO_DOUBLE;
                }
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                    res[i] = x;
                    return;
                }
                res[i] =
                    std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm) : detail::ZERO_DOUBLE;
            });
            pool.waitForAll();
        });
}

void VonMisesDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const VonMisesDistribution& d, double x) { return d.getLogProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, k, mu, lnorm);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        return;
                    }
                    res[i] =
                        std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        continue;
                    }
                    res[i] =
                        std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
                }
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                    res[i] = x;
                    return;
                }
                res[i] =
                    std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
            });
            pool.waitForAll();
        });
}

void VonMisesDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const VonMisesDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            double mu, kappa, log_scaled_norm;
            std::vector<double> coeffs;
            d.withCacheSnapshot([&] {
                mu = d.mu_;
                kappa = d.kappa_;
                log_scaled_norm = d.logScaledNormaliser_;
                coeffs = d.cdfSeriesCoeffs_;  // copy: batch runs outside the lock
            });
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, mu, kappa, log_scaled_norm,
                                                      coeffs);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = d.getCumulativeProbability(vals[i]);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = d.getCumulativeProbability(vals[i]);
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getCumulativeProbability(vals[i]); });
            pool.waitForAll();
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool VonMisesDistribution::operator==(const VonMisesDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(kappa_ - other.kappa_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool VonMisesDistribution::operator!=(const VonMisesDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const VonMisesDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, VonMisesDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("VonMisesDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t mu_pos = token.find("mu=");
    const size_t comma = token.find(",", mu_pos);
    const size_t kappa_pos = token.find("kappa=");
    const size_t close = token.find(")", kappa_pos);
    if (mu_pos == std::string::npos || comma == std::string::npos ||
        kappa_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double mu = std::stod(token.substr(mu_pos + 3, comma - mu_pos - 3));
        const double kappa = std::stod(token.substr(kappa_pos + 6, close - kappa_pos - 6));
        auto result = d.trySetParameters(mu, kappa);
        if (result.isError())
            is.setstate(std::ios::failbit);
    } catch (...) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// LogPDF batch:  z[i] = x[i] − μ  |  c[i] = vector_cos(z)  |  r[i] = κ·c[i] − ln Z
// PDF batch:     same as LogPDF then r[i] = vector_exp(r)
// CDF batch (#51): t[i] = wrap(x[i]−μ); r[i] = (t[i]+π)/(2π); for j = j_max..1:
//                  r[i] += b_j · vector_sin(j·t[i])  (κ=0 or κ>1000: scalar fallback);
//                  lanes below 1/4 or above 3/4 then take the scalar tail quadrature
//
// PDF/LogPDF use VectorOps::vector_cos; CDF uses VectorOps::vector_sin (#95) —
// both AVX/AVX2/SSE2/NEON/AVX-512. Non-finite inputs receive an exact sentinel
// value via a scalar fixup pass after the SIMD kernel.
//
// The primary performance gain over per-element calls remains avoiding the
// cache-validity check and lock acquisition on every element.
//==============================================================================

void VonMisesDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_kappa, double cached_mu,
    double cached_log_normaliser) const noexcept {
    // Step 1: z[i] = values[i] - mu  (scalar_add with -mu)
    arch::simd::VectorOps::scalar_add(values, -cached_mu, results, count);

    // Step 2: results[i] = cos(z[i])  (vectorised)
    arch::simd::VectorOps::vector_cos(results, results, count);

    // Step 3: results[i] = kappa * results[i] - log_normaliser
    arch::simd::VectorOps::scalar_multiply(results, cached_kappa, results, count);
    arch::simd::VectorOps::scalar_add(results, -cached_log_normaliser, results, count);

    // Fixup: NaN propagates; ±inf must produce -∞ regardless of the SIMD result
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i]))
            results[i] = values[i];
        else if (!std::isfinite(values[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void VonMisesDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_kappa, double cached_mu,
    double cached_log_normaliser) const noexcept {
    // Compute log-PDF then exponentiate
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_kappa, cached_mu,
                                     cached_log_normaliser);

    // Step 4: results[i] = exp(results[i])
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: NaN propagates; ±inf must produce 0
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i]))
            results[i] = values[i];
        else if (!std::isfinite(values[i]))
            results[i] = detail::ZERO_DOUBLE;
    }
}

void VonMisesDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu, double cached_kappa,
    double cached_log_scaled_norm, const std::vector<double>& cached_coeffs) const noexcept {
    // kappa = 0 or kappa > 1000: series not applicable (see updateCacheUnsafe).
    // Fall back to the per-element scalar path -- itself now O(j_max) or the
    // wrapped-normal approximation, not the old O(512) trapezoid.
    if (cached_coeffs.empty()) {
        for (std::size_t i = 0; i < count; ++i)
            results[i] = getCumulativeProbability(values[i]);
        return;
    }

    const int j_max = static_cast<int>(cached_coeffs.size());

    std::vector<double, arch::simd::aligned_allocator<double>> t(count);
    std::vector<double, arch::simd::aligned_allocator<double>> jt(count);
    std::vector<double, arch::simd::aligned_allocator<double>> s(count);

    // t[i] = wrap(values[i] - mu). wrapAngle is scalar-only (fmod-based); the
    // subtraction runs through SIMD, the wrap through a plain loop.
    arch::simd::VectorOps::scalar_add(values, -cached_mu, t.data(), count);
    for (std::size_t i = 0; i < count; ++i)
        t[i] = wrapAngle(t[i]);

    // results[i] = (t[i] + pi) / (2*pi)  -- the linear part of the series.
    arch::simd::VectorOps::scalar_add(t.data(), detail::PI, results, count);
    arch::simd::VectorOps::scalar_multiply(results, detail::ONE / detail::TWO_PI, results, count);

    // Accumulate j_max -> 1 (smallest terms first, matching the scalar path):
    //   jt[i] = j * t[i]; s[i] = sin(jt[i]) (#95, vector_sin, max 1 ULP);
    //   results[i] += b_j * s[i]
    for (int j = j_max; j >= 1; --j) {
        const double bj = cached_coeffs[static_cast<std::size_t>(j - 1)];
        arch::simd::VectorOps::scalar_multiply(t.data(), static_cast<double>(j), jt.data(), count);
        arch::simd::VectorOps::vector_sin(jt.data(), s.data(), count);
        arch::simd::VectorOps::scalar_multiply(s.data(), bj, s.data(), count);
        arch::simd::VectorOps::vector_add(results, s.data(), results, count);
    }

    // Fixup: wrapAngle passes NaN/+-inf through unchanged, so those lanes are
    // poisoned (NaN) by this point -- restore the scalar contract exactly:
    // NaN -> NaN, +inf -> 1, -inf -> 0. Finite lanes take the scalar path's
    // tail correction (the integrated tail mass where the series is within
    // kTailSwitch of 0 or 1) and are clamped to [0,1].
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            results[i] = std::numeric_limits<double>::quiet_NaN();
        } else if (std::isinf(values[i])) {
            results[i] = (values[i] > 0.0) ? detail::ONE : detail::ZERO_DOUBLE;
        } else {
            results[i] = tail_corrected_cdf(results[i], t[i], cached_kappa, cached_log_scaled_norm);
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void VonMisesDistribution::updateCacheUnsafe() const noexcept {
    // logNormaliser = log(2π) + log I₀(κ)
    // When κ = 0: I₀(0) = 1, log I₀ = 0, logNormaliser = log(2π). ✓
    logNormaliser_ = detail::LN_2PI + detail::log_bessel_i0(kappa_);
    // The same constant less kappa, formed without the cancellation, for the
    // tail quadrature behind the CDF tails and the quantile.
    logScaledNormaliser_ = log_scaled_normaliser(kappa_);

    isUniform_ = (kappa_ < 1e-10);

    // Circular variance = 1 − I₁(κ)/I₀(κ), via the dedicated complement helper
    // (#93). Forming it as `ONE - i1 / i0` discarded ~log₂(2κ) bits — about 9
    // at κ = 200 — and returned NaN above κ ≈ 713, where I₀ and I₁ both
    // overflow and the `i0 > 0.0` guard does not catch inf.
    if (isUniform_) {
        circularVariance_ = detail::ONE;
    } else {
        circularVariance_ = detail::bessel_i1_i0_complement(kappa_);
    }

    // CDF Bessel-series coefficients (#51). Not applicable at kappa=0 (isUniform_
    // uses the exact linear CDF) or kappa>1000 (unvalidated range for the series;
    // the wrapped-normal fallback is used instead) -- see vonmises_cdf_series_coeffs
    // above and the class-level CDF doc for the derivation.
    if (isUniform_ || kappa_ > kCdfSeriesKappaMax) {
        cdfSeriesCoeffs_.clear();
    } else {
        const int j_max = static_cast<int>(std::ceil(10.0 + 8.5 * std::sqrt(kappa_)));
        cdfSeriesCoeffs_ = vonmises_cdf_series_coeffs(kappa_, j_max);
    }

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

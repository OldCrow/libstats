#include "libstats/core/math_utils.h"

#include "libstats/common/cpu_detection_fwd.h"         // CPU feature queries (lightweight)
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/common/simd_policy_fwd.h"           // SIMD policy decisions (lightweight)
#include "libstats/core/bessel.h"
#include "libstats/core/distribution_base.h"
#include "libstats/core/math_constants.h"
#include "libstats/core/safety.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/stats/analysis/statistical_utilities.h"
#include "portable_fma.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <corvus/corvus.h>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

#if defined(__APPLE__) && !defined(_REENTRANT)
// POSIX lgamma_r: in libSystem, declared by Apple's <math.h> only under _REENTRANT.
extern "C" double lgamma_r(double, int*);
#endif

namespace stats {
namespace detail {

// Forward declarations
static double beta_continued_fraction(double x, double y, double a, double b) noexcept;
static double beta_continued_fraction_unscaled(double x, double y, double a, double b) noexcept;

// Iteration cap for the incomplete beta continued fraction (#166). Near the median it needs O(√a)
// terms — the terms fall like exp(−n²/2a), so reaching 3ε takes n ≈ 9√a — and the fixed cap of 100
// stopped it early at large shape. The cap bounds non-convergence only; a converging expansion
// breaks out long before it. The std::min also sends a NaN scale to the 1e6 bound, which keeps the
// cast defined.
static int expansion_iteration_cap(double shape) noexcept {
    return static_cast<int>(std::min(1e6, 1000.0 + 20.0 * std::sqrt(std::fabs(shape))));
}

// The prefactors below switch to Stirling's form at this shape; under it the direct form keeps its
// pre-v2.5.0 bits. It is not the more accurate of the two there: it loses about
// min(a, b)·ln(max(a, b))·ε, 10-50 ulp in Gamma at shapes 8-19 and up to ~1e3 ulp in Beta with one
// huge shape, against 2-8 ulp just above 20, so values jump by up to ~200 ulp across the switch
// (M1 seam hunt, 2026-10-12). Within the accuracy contract; moving the switch is deferred to the
// corvus kernels (#266).
constexpr double kStirlingPrefactorShape = detail::STIRLING_PREFACTOR_SHAPE;

// Stirling remainder c(z) = lgamma(z) − [(z − ½)·log(z) − z + ½·log(2π)], truncated after its
// 1/z⁹ term (the next is < 2e-17 at z = 20).
static double stirling_remainder(double z) noexcept {
    const double r = detail::ONE / z;
    const double r2 = r * r;
    return r * (1.0 / 12.0 -
                r2 * (1.0 / 360.0 - r2 * (1.0 / 1260.0 - r2 * (1.0 / 1680.0 - r2 / 1188.0))));
}

// log1p(t) − t without the cancellation of the direct difference near t = 0, where both terms
// are of size t and the result t²/2. Below |t| = ½ the atanh-form series log1pmx_series
// (math_utils.h), which replaced a t-series summing 10–20 dependent divisions below ¼ and the
// direct difference on [¼, ½) (1.7 ulp worst case, against 5.4 and 7). From ½ the direct
// difference, which cancels at most threefold there.
static double log1pmx(double t) noexcept {
    if (std::fabs(t) >= LOG1PMX_SERIES_LIMIT)
        return std::log1p(t) - t;
    return log1pmx_series(t);
}

// log of the incomplete-gamma prefactor x^a·e^{−x}/Γ(a) (#166). Formed directly,
// −x + a·log(x) − lgamma(a) cancels terms of size a·log(x), so its absolute error — the
// prefactor's relative error — is ~a·log(x)·ε: 6e-11 at a = 5e4. Stirling's series rewrites it as
// a·(log1p(t) − t) + ½·log(a/2π) − c(a) with t = (x − a)/a, whose error is ~a·|t|·ε (≈ √a·ε near
// the median). The shape-only part ½·log(a/2π) − c(a) is log_gamma_prefactor_constant(a), which
// batch callers hoist.
double log_gamma_prefactor_constant(double a) noexcept {
    return detail::HALF * (std::log(a) - detail::LN_2PI) - stirling_remainder(a);
}

// log1p(t) − t for t = (x − m)/m, m > 0: the series below |t| = ½; beyond, log1p(t) is formed as
// log(x/m), one rounding, not from the rounded t. Where x ≪ m, t ≈ −1 has lost most of x
// (t = −1 exactly at x = 1e-15, m = 1000, so log1p(t) = −inf), and the shape multiplies the
// loss.
static double log1pmx_ratio(double t, double x, double m) noexcept {
    return std::fabs(t) < LOG1PMX_SERIES_LIMIT ? log1pmx_series(t) : std::log(x / m) - t;
}

double log_gamma_prefactor(double a, double x, double shape_constant) noexcept {
    return a * log1pmx_ratio((x - a) / a, x, a) + shape_constant;
}

double log_gamma_prefactor(double a, double x) noexcept {
    if (a < kStirlingPrefactorShape)
        return -x + a * std::log(x) - lgamma(a);
    return log_gamma_prefactor(a, x, log_gamma_prefactor_constant(a));
}

// The constant beta_i's four-argument overload takes: −log B(a, b) in the direct form, and in
// Stirling's form (both shapes from 20) its shape-only part ½·log(ab / (2π(a + b))) + c(a + b) −
// c(a) − c(b). Both are symmetric in a and b, as the I_{1−x}(b, a) reflection needs.
double beta_prefactor_constant(double a, double b) noexcept {
    if (a < kStirlingPrefactorShape || b < kStirlingPrefactorShape)
        return -lbeta(a, b);
    const double sum = a + b;
    return detail::HALF * (std::log(a) + std::log(b) - std::log(sum) - detail::LN_2PI) +
           stirling_remainder(sum) - stirling_remainder(a) - stirling_remainder(b);
}

// log x and log(1 − x) from x and omx = 1 − x, each held to relative ε. The smaller of the two is
// the one the caller holds exactly. While that is x, log(1 − x) is log1p(−x). Re-formed from the
// rounded 1 − x, the log carried that rounding as an absolute ε/2, which the shape multiplied:
// relative b·ε/4 in I_x(a, b) (Beta(1, 1e6).cdf(1.5e-6) erred 1.2e-11). Once b·x is O(1) with
// x < ε, 1 − x rounds to 1, b·log(1 − x) = 0, and the prefactor is e^{b·x} too large
// (Beta(1, 4.19e46).cdf(3.65e-47) was 3.6). Above ½, 1 − x from x is exact, and each log is taken
// directly while omx is that complement; in a pair that is not exactly complementary (beta_i_pair:
// F's y next to 1 beside its ybar) the small omx is the exact one, and log x is log1p(−omx).
static void pair_logs(double x, double omx, double& log_x, double& log_omx) noexcept {
    if (x <= omx) {
        log_x = std::log(x);
        log_omx = std::log1p(-x);
        return;
    }
    log_x = omx == detail::ONE - x ? std::log(x) : std::log1p(-omx);
    log_omx = std::log(omx);
}

// log of the incomplete-beta prefactor x^a·(1 − x)^b / B(a, b) (#166), with the same
// cancellation as the gamma one: lgamma(a + b) − lgamma(a) − lgamma(b) + a·log(x) + b·log(1 − x)
// carries terms of size (a + b)·log 2, 3e-11 relative at a = b = 1e4. For a, b ≥ 20, Stirling
// gives a·(log1p(u) − u) + b·(log1p(v) − v) + ½·log(ab / (2π(a + b))) + c(a + b) − c(a) − c(b),
// with x₀ = a/(a + b), u = (x − x₀)/x₀ and v = (x₀ − x)/(1 − x₀); the linear terms a·u + b·v
// cancel exactly, and x₀ is the stationary point, so its rounding enters only at second order.
// shape_constant is beta_prefactor_constant(a, b), in whichever form the shapes select.
//
// omx is 1 − x, and log_x and log_omx are log x and log(1 − x): the public overload passes
// 1 − x and pair_logs(x, 1 − x), beta_i_pair the caller's complement, the quantile solver the
// exact logs of its logit, which it holds where x itself has rounded to 1 (Beta(0.001, 1).Q(0.7):
// the reflected root is 1 − 0.7^1000, and 1 − x re-formed from x was 0, so b·log(1 − x) = −∞
// zeroed the residual and the Newton slope, and the solver returned 6.6e-37 for 1.25e-155 — B1)
// and where a·log x magnifies the rounding of x (Beta(0.001, 1e6).Q(0.99): a = 1e6 against
// x = 1 − 2e-11 put 1e-10 into the prefactor and 4e-8 into the root). The Stirling branch reads
// x − x₀ from the smaller of x and 1 − x.
double log_beta_prefactor_logs(double x, double omx, double a, double b, double log_x,
                               double log_omx, double shape_constant) noexcept {
    if (a < kStirlingPrefactorShape || b < kStirlingPrefactorShape)
        return shape_constant + a * log_x + b * log_omx;
    const double sum = a + b;
    const double min_shape = std::min(a, b);
    const double bv = sum - a;
    const double sum_err = (a - (sum - bv)) + (b - bv);  // a + b = sum + sum_err exactly
    if (sum * sum * std::numeric_limits<double>::epsilon() <= min_shape) {
        // x₀ = a/(a + b) and 1 − x₀ as an exact complementary pair: 1 − (the smaller quotient)
        // rounded, the other its complement, exact by Sterbenz. The sum
        //   a·log(x/x₀) + b·log((1 − x)/(1 − x₀))
        // then differs from its value at the exact a/(a + b) only at second order, but the
        // dropped linear term a·u + b·v is first order in that rounding:
        //   a·u + b·v = (x − x₀)·(a − (a + b)·x₀)/(x₀(1 − x₀)),
        // of size (x − x₀)·(a + b)²·ε/b, 1e-11 of the prefactor at (1e6, 30) near the branch
        // point. So it is added back, its numerator formed from a + b split exactly by TwoSum (no
        // products: contraction-safe) and one explicit fma. The second-order cost of the pair is
        // (a + b)²ε²/(4·min(a, b)), below ε/4 while (a + b)²·ε ≤ min(a, b).
        double x0;
        double one_minus_x0;
        if (a <= b) {
            one_minus_x0 = detail::ONE - a / sum;
            x0 = detail::ONE - one_minus_x0;
        } else {
            x0 = detail::ONE - b / sum;
            one_minus_x0 = detail::ONE - x0;
        }
        // x − x₀ = (1 − x₀) − (1 − x) exactly when 1 − x is x's exact complement, so both forms
        // round the same real number; from 1 − x where that is the smaller, which the caller may
        // hold more exactly than x.
        const double d = x <= omx ? x - x0 : one_minus_x0 - omx;
        const double lin_num = detail::fma_portable(-sum, x0, a) - sum_err * x0;
        const double linear = d * (lin_num / (x0 * one_minus_x0));
        const double u = d / x0;
        const double v = -d / one_minus_x0;
        // log(1 + v) = log((1 − x)/(1 − x₀)) beyond the series: from log1p(−x) while x is the
        // smaller, 1 − x then near 1; else as the log of the ratio, with 1 − x as held. Near
        // x = 1 the rounded v has lost 1 − x as u loses x near 0, log1p(−x) of a rounded x next to
        // 1 loses it too (beta_i_pair: 2e7·ε at (2.6e8, 50) from F's y), and the difference of
        // two logs of small quantities cancels terms of size log(a + b) (520·ε at (3e8, 20)).
        const double lv = std::fabs(v) < LOG1PMX_SERIES_LIMIT
                              ? log1pmx_series(v)
                              : (x <= omx ? std::log1p(-x) - std::log(one_minus_x0)
                                          : std::log(omx / one_minus_x0)) -
                                    v;
        return a * log1pmx_ratio(u, x, x0) + b * lv + linear + shape_constant;
    }
    // Shape ratios beyond (a + b)²·ε > min(a, b) (~1e8 at min 20): the complement of the small
    // quotient rounds far from the stationary point, and the pair above would cost more than ε
    // at second order. Separately rounded quotients lost x − x₀ instead: x₀ next to 1 holds it
    // to an absolute ε/2, which the large shape multiplies (1.5e11·ε at (1e11, 20); F(50,
    // 1e20).Q(0.5) was 163.84 for 0.9867). Here only the small quotient m = min(a, b)/(a + b)
    // is formed, with one rounding; x₀ is m or 1 − m held implicitly, x − x₀ is read off x
    // where x₀ = m and off 1 − x where x₀ = 1 − m, the complement of m enters only as a
    // denominator and through log1p(−m), and the linear term is added back with the exact
    // numerator a − (a + b)·x₀ as above. m's rounding is relative, so the second-order cost is
    // min(a, b)·ε², and nothing in the form grows with the ratio.
    double d;
    double num;  // a − (a + b)·x₀
    double x0;
    double one_minus_x0;
    double lu_far;  // log(x/x₀), used beyond the series
    double lv_far;  // log((1 − x)/(1 − x₀)), likewise
    if (a <= b) {
        x0 = a / sum;
        one_minus_x0 = detail::ONE - x0;
        d = x - x0;
        num = detail::fma_portable(-sum, x0, a) - sum_err * x0;
        lu_far = std::log(x / x0);
        lv_far = log_omx - std::log1p(-x0);
    } else {
        one_minus_x0 = b / sum;
        x0 = detail::ONE - one_minus_x0;
        d = one_minus_x0 - omx;
        num = sum_err * one_minus_x0 - detail::fma_portable(-sum, one_minus_x0, b);
        lu_far = log_x - std::log1p(-one_minus_x0);
        lv_far = std::log(omx / one_minus_x0);
    }
    // The ratio of the two small quantities is logged as one, not as a difference of two logs of
    // size log(a + b): that cancellation cost 7e-13 at (8e85, 50).
    const double linear = d * (num / (x0 * one_minus_x0));
    const double u = d / x0;
    const double v = -d / one_minus_x0;
    const double lu = std::fabs(u) < LOG1PMX_SERIES_LIMIT ? log1pmx_series(u) : lu_far - u;
    const double lv = std::fabs(v) < LOG1PMX_SERIES_LIMIT ? log1pmx_series(v) : lv_far - v;
    return a * lu + b * lv + linear + shape_constant;
}

double log_beta_prefactor(double x, double a, double b, double shape_constant) noexcept {
    const double omx = detail::ONE - x;
    double log_x, log_omx;
    pair_logs(x, omx, log_x, log_omx);
    return log_beta_prefactor_logs(x, omx, a, b, log_x, log_omx, shape_constant);
}

// Discrete log-pmfs at large counts (#172). Formed directly, k·log λ − λ − lgamma(k + 1) and the
// binomial and negative-binomial analogues add terms of size n·log n to a result of order one:
// 2e-9 relative in the pmf at counts ~1e5. Splitting each lgamma(m + 1) by Stirling into
// (m + ½)·log m − m + ½·log 2π + c(m) regroups the large terms into deviances
// x·log(x/M) + M − x around the means M, each of the size of the result.
namespace {
// Stirling error c(m) = lgamma(m + 1) − [(m + ½)·log m − m + ½·log 2π] for real m > 0, which is
// stirling_remainder(m): the series from m = 20, the direct difference below, where its terms
// are small.
// lgamma_m1() is lgamma(m + 1), asked for only below the series: a caller that holds the value
// (a batch's fixed count) supplies it, everyone else passes the call.
template <class LgammaM1>
double stirling_error_with(double m, LgammaM1&& lgamma_m1) noexcept {
    if (m >= kStirlingPrefactorShape)
        return stirling_remainder(m);
    return lgamma_m1() - ((m + detail::HALF) * std::log(m) - m + detail::HALF * detail::LN_2PI);
}

double stirling_error(double m) noexcept {
    return stirling_error_with(m, [m]() noexcept { return detail::lgamma(m + detail::ONE); });
}

// The deviance x·log(x/M) + M − x ≥ 0. Within M/2 of M it is M·[(1 + t)·log1pmx(t) + t²] with
// t = (x − M)/M: no cancellation below |t| = ¼, where log1pmx sums its series, and about 3 bits
// in log1pmx's direct difference from there to ½; farther out the direct form loses a few bits
// at most.
//
// log(x/m) of two positive doubles: the quotient logged as one where it is a normal double, the
// difference of the two logs where it is not. A quotient that underflows below DBL_MIN or overflows
// carries no precision (NegativeBinomial(1e-300, ½) at k = 1e300 logged 0, and its −∞ met a +∞
// downstream as NaN); the difference of two finite logs is exact to a few ulp of their size.
double log_ratio(double x, double m) noexcept {
    const double q = x / m;
    if (q >= std::numeric_limits<double>::min() && q <= std::numeric_limits<double>::max())
        return std::log(q);
    return std::log(x) - std::log(m);
}

// log(n/(a·b)), as one quotient where the product and the quotient are normal doubles (the
// original form, bit-identical there), from log_ratio(n, a) − log b where either is not: a count
// of 1e-300 beside one of 1e300 overflows n/(a·b).
double log_ratio3(double n, double a, double b) noexcept {
    const double prod = a * b;
    const double q = n / prod;
    constexpr double kMin = std::numeric_limits<double>::min();
    constexpr double kMax = std::numeric_limits<double>::max();
    if (prod >= kMin && prod <= kMax && q >= kMin && q <= kMax)
        return std::log(q);
    return log_ratio(n, a) - std::log(b);
}

double deviance(double x, double m) noexcept {
    if (x == detail::ZERO_DOUBLE)
        return m;
    const double d = x - m;
    if (std::fabs(d) < detail::HALF * m) {
        const double t = d / m;
        return m * ((detail::ONE + t) * log1pmx(t) + t * t);
    }
    return x * log_ratio(x, m) + m - x;
}

// Error-free transforms for the binomial means, every fusion spelled out (AGENTS.md
// FP-contraction rule): a + b = hi + lo and a·b = hi + lo exactly.
struct DoubleDouble {
    double hi;
    double lo;
};

DoubleDouble two_sum(double a, double b) noexcept {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

DoubleDouble two_prod(double a, double b) noexcept {
    const double p = a * b;
    return {p, detail::fma_portable(a, b, -p)};
}
}  // namespace

double poisson_log_pmf(double k, double lambda) noexcept {
    if (k == detail::ZERO_DOUBLE)
        return -lambda;
    if (k < kStirlingPrefactorShape && lambda < kStirlingPrefactorShape)
        return k * std::log(lambda) - lambda - detail::lgamma(k + detail::ONE);
    // log pmf = −D(k, λ) − ½·log(2πk) − c(k).
    return -deviance(k, lambda) - detail::HALF * (std::log(k) + detail::LN_2PI) - stirling_error(k);
}

namespace {
// binomial_log_pmf with its two parameter-dependent lgamma values behind callables: lgamma_xa1()
// is lgamma(xa + 1) and lgamma_n1() is lgamma(fl(xa + xb) + 1). Each is called at most once and
// only where the three-argument form calls detail::lgamma on that argument, so supplying the
// same values gives the same bits.
template <class LgammaXa1, class LgammaN1>
double binomial_log_pmf_with(double xa, double xb, double pa, LgammaXa1&& lgamma_xa1,
                             LgammaN1&& lgamma_n1) noexcept {
    // pa = 0 or 1 zeroes a mean below, and its low-part term would be 0·∞; the limits instead.
    if (pa <= detail::ZERO_DOUBLE)
        return xa == detail::ZERO_DOUBLE ? detail::ZERO_DOUBLE : detail::NEGATIVE_INFINITY;
    if (pa >= detail::ONE)
        return xb == detail::ZERO_DOUBLE ? detail::ZERO_DOUBLE : detail::NEGATIVE_INFINITY;
    if (xa == detail::ZERO_DOUBLE)
        return xb * std::log1p(-pa);
    if (xb == detail::ZERO_DOUBLE)
        return xa * std::log(pa);
    const DoubleDouble n = two_sum(xa, xb);
    if (n.hi < kStirlingPrefactorShape)
        return lgamma_n1() - lgamma_xa1() - detail::lgamma(xb + detail::ONE) + xa * std::log(pa) +
               xb * std::log1p(-pa);

    // The means Ma = n·pa and Mb = n − Ma as double-doubles, so Ma + Mb = xa + xb to ~ε²·n. Then
    //   log pmf = c(n) − c(xa) − c(xb) + ½·log(n/(2π·xa·xb)) − D(xa, Ma) − D(xb, Mb)
    //             + Σ lo·(x/hi − 1),
    // where the last sum carries the low parts of the means into the deviances' linear terms,
    // which cancel exactly when the means sum to n. Rounding the means instead costs n·ε.
    const DoubleDouble pa_n = two_prod(n.hi, pa);
    const DoubleDouble ma = two_sum(pa_n.hi, pa_n.lo + n.lo * pa);
    const DoubleDouble rest = two_sum(n.hi, -ma.hi);
    const DoubleDouble mb = two_sum(rest.hi, rest.lo + (n.lo - ma.lo));
    return stirling_error(n.hi) - stirling_error_with(xa, lgamma_xa1) - stirling_error(xb) +
           detail::HALF * (log_ratio3(n.hi, xa, xb) - detail::LN_2PI) - deviance(xa, ma.hi) -
           deviance(xb, mb.hi) + ma.lo * (xa / ma.hi - detail::ONE) +
           mb.lo * (xb / mb.hi - detail::ONE);
}
}  // namespace

double binomial_log_pmf(double xa, double xb, double pa) noexcept {
    return binomial_log_pmf_with(
        xa, xb, pa, [xa]() noexcept { return detail::lgamma(xa + detail::ONE); },
        [xa, xb]() noexcept { return detail::lgamma((xa + xb) + detail::ONE); });
}

double binomial_log_pmf(double xa, double xb, double pa, double lgamma_xa_plus_1,
                        double lgamma_n_plus_1) noexcept {
    return binomial_log_pmf_with(
        xa, xb, pa, [lgamma_xa_plus_1]() noexcept { return lgamma_xa_plus_1; },
        [lgamma_n_plus_1]() noexcept { return lgamma_n_plus_1; });
}

// =============================================================================
// SPECIAL MATHEMATICAL FUNCTIONS
// =============================================================================
namespace {

// One value through a corvus span function. Only for the functions corvus exports as spans alone
// (the erf family, lgamma, lbeta, digamma, trigamma, the Bessel functions). gamma_p, gamma_q,
// beta_p, beta_q and their four inverses have scalar entry points since corvus v1.1.0, called
// directly below: each name is an overload set, which Fn cannot deduce, and the scalar entry
// returns the same bits as a length-1 span without this hop.
template <class Fn>
[[nodiscard]] inline double corvus_scalar(Fn fn, double x) noexcept {
    double out;
    fn(std::span<const double>{&x, 1}, std::span<double>{&out, 1});
    return out;
}

template <class Fn>
[[nodiscard]] inline double corvus_scalar(Fn fn, double a, double b) noexcept {
    double out;
    fn(std::span<const double>{&a, 1}, std::span<const double>{&b, 1}, std::span<double>{&out, 1});
    return out;
}

}  // namespace

double erf(double x) noexcept {
    return corvus_scalar(corvus::erf, x);
}

double erfc(double x) noexcept {
    return corvus_scalar(corvus::erfc, x);
}

double erf_inv(double x) noexcept {
    return corvus_scalar(corvus::erfinv, x);
}

// log Γ(x) from corvus (v3.0.0). #173's signgam race is moot here: corvus::lgamma writes no
// global, so the lgamma_r detour of v2.5.0 is gone with it.
//
// PROTOTYPE: integer arguments 1..kLgammaTableSize come from a table. The discrete log-pmfs below
// ask for lgamma(k + 1) at counts under STIRLING_PREFACTOR_SHAPE, one to four times per element of
// a Poisson, Binomial or NegativeBinomial batch and on every scalar call, and corvus v1.2.0 has
// no scalar lgamma entry (corvus #110): each call is a length-1 span, about 140 ns on a 2-lane
// tier against 2 ns for the lookup. The table is filled once, on first use, by the same length-1
// span call, so every entry is the value that call returns in this process (corvus's dispatch
// target is fixed per process): bit-identical by construction, on every tier.
namespace {
constexpr int kLgammaTableSize = 32;

const std::array<double, kLgammaTableSize + 1>& lgamma_integer_table() noexcept {
    static const std::array<double, kLgammaTableSize + 1> table = [] {
        std::array<double, kLgammaTableSize + 1> t{};
        for (int i = 1; i <= kLgammaTableSize; ++i)
            t[static_cast<std::size_t>(i)] = corvus_scalar(corvus::lgamma, static_cast<double>(i));
        return t;
    }();
    return table;
}
}  // namespace

double lgamma(double x) noexcept {
    // NaN fails both comparisons; a non-integer fails the round trip through int.
    if (x >= detail::ONE && x <= static_cast<double>(kLgammaTableSize)) {
        const int i = static_cast<int>(x);
        if (static_cast<double>(i) == x)
            return lgamma_integer_table()[static_cast<std::size_t>(i)];
    }
    return corvus_scalar(corvus::lgamma, x);
}

// log Γ(1 + a) for |a| ≤ ½. std::lgamma(1 + a) first rounds a into 1 + a, a relative error of
// ε/a in a. The Taylor series about 1, −γ·a + Σ_{k≥2} (−a)^k·ζ(k)/k, regrouped as
//   −(log1p(a) − a) − γ·a + Σ_{k≥2} (−a)^k·(ζ(k) − 1)/k,
// has terms that fall at least fourfold, since ζ(k) − 1 ~ 2^−k.
static double lgamma1p_small(double a) noexcept {
    static constexpr double kZetaMinusOne[] = {
        6.4493406684822644e-1, 2.0205690315959429e-1, 8.2323233711138192e-2, 3.6927755143369926e-2,
        1.734306198444914e-2,  8.3492773819228268e-3, 4.0773561979443394e-3, 2.0083928260822144e-3,
        9.9457512781808534e-4, 4.9418860411946456e-4, 2.460865533080483e-4,  1.2271334757848915e-4,
        6.1248135058704829e-5, 3.0588236307020494e-5, 1.5282259408651872e-5, 7.6371976378997623e-6,
        3.8172932649998399e-6, 1.9082127165539389e-6, 9.5396203387279611e-7, 4.7693298678780646e-7,
        2.3845050272773299e-7, 1.1921992596531107e-7, 5.960818905125948e-8,  2.980350351465228e-8,
        1.4901554828365041e-8, 7.4507117898354295e-9, 3.7253340247884571e-9, 1.862659723513049e-9,
        9.3132743241966818e-10};  // ζ(k) − 1, k = 2…30 (mpmath)
    double sum = detail::ZERO_DOUBLE;
    double power = a;  // a^(k−1)
    for (int k = 2; k < 31; ++k) {
        power *= a;
        const double term = (k % 2 == 0 ? power : -power) * kZetaMinusOne[k - 2] / k;
        sum += term;
        if (std::fabs(term) <= std::numeric_limits<double>::epsilon() * std::fabs(sum))
            break;
    }
    return -log1pmx(a) - detail::EULER_MASCHERONI * a + sum;
}

double gamma_p(double a, double x) noexcept {
    // P(a, x) from corvus (v3.0.0). The guards keep the pre-adoption edge contract: NaN in, NaN
    // out; x < 0 or a <= 0 reads 0; x = +inf is a caller's x·rate that overflowed (#224).
    if (std::isnan(a) || std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE || a <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (x == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (x == std::numeric_limits<double>::infinity())
        return detail::ONE;
    return corvus::gamma_p(a, x);
}

double gamma_p_from_log_x(double a, double log_x) noexcept {
    // x < DBL_MIN: the series P = x^a e^{-x} Σ x^n / Γ(a+n+1) is its n = 0 term to double.
    if (std::isnan(a) || std::isnan(log_x))
        return std::numeric_limits<double>::quiet_NaN();
    if (a <= detail::ZERO_DOUBLE || log_x == -std::numeric_limits<double>::infinity())
        return detail::ZERO_DOUBLE;
    return std::exp(a * log_x - detail::lgamma(a + detail::ONE));
}

double gamma_q_from_log_x(double a, double log_x) noexcept {
    if (std::isnan(a) || std::isnan(log_x))
        return std::numeric_limits<double>::quiet_NaN();
    if (a <= detail::ZERO_DOUBLE || log_x == -std::numeric_limits<double>::infinity())
        return detail::ONE;
    return -std::expm1(a * log_x - detail::lgamma(a + detail::ONE));
}

double gamma_q(double a, double x) noexcept {
    // Q(a, x) from corvus (v3.0.0); guards as in gamma_p, complemented.
    if (std::isnan(a) || std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE || a <= detail::ZERO_DOUBLE)
        return detail::ONE;
    if (x == detail::ZERO_DOUBLE)
        return detail::ONE;
    if (x == std::numeric_limits<double>::infinity())
        return detail::ZERO_DOUBLE;  // #224
    return corvus::gamma_q(a, x);
}

// Newton on a residual f(t) that is monotone and concave in t, from any start in (lo, hi) (#160,
// #159). Concavity makes the iteration globally convergent: at most one step overshoots onto the
// f < 0 side, and from there the iterates approach the root monotonically, so a positive residual
// after a Newton step off the f < 0 side is rounding noise. A bisection step carries no such
// guarantee, and a non-finite f (an underflowed tail) is not a sighting of that side: counting
// either as one ended the #223 solve after two evaluations. `residual(t, slope)` returns f(t)
// and sets slope to |f'(t)|; `rising` gives the sign of f'. A non-finite f or a step out of the
// bracket bisects. Returns the root in t; with `last_step`, returns instead the last evaluated t
// and the final Newton correction separately, so a caller can apply the correction to exp(t)
// rather than to t, where it is below the resolution of t once |t| is large (#226).
// `exp_cliff` says the residual falls like −eᵗ far to the right (a log tail in t = log x, whose
// underflow once bisected the solver out of there): a Newton step in t from that side moves one
// unit per step, so a long falling step is taken in eᵗ instead, next = t + log1p(step), which
// lands by the root in one step or bisects where eᵗ·(1 + step) would not be positive.
template <typename Residual>
static double solve_concave(double t, double lo, double hi, bool rising, Residual residual,
                            double* last_step = nullptr, bool exp_cliff = false,
                            double* last_slope = nullptr) noexcept {
    bool stepped_off_negative = false;  // a Newton step has been taken off the f < 0 side
    int noise_flips = 0;
    double best_t = t;
    double best_f = std::numeric_limits<double>::infinity();
    if (last_step)
        *last_step = detail::ZERO_DOUBLE;
    for (int i = 0; i < 100; ++i) {
        double slope = detail::ZERO_DOUBLE;
        const double f = residual(t, slope);
        if (f == detail::ZERO_DOUBLE)
            return t;
        if (std::abs(f) < best_f) {
            best_f = std::abs(f);
            best_t = t;
        }
        // A negative residual bounds the root from below for a rising f, from above for a falling
        // one.
        if ((f < detail::ZERO_DOUBLE) == rising)
            lo = t;
        else
            hi = t;
        // Once Newton has stepped off the f < 0 side, exact iterates stay there; a return to
        // f > 0 is rounding noise. The second one ends a ping-pong the step test below may not
        // catch at small slope.
        if (f > detail::ZERO_DOUBLE && stepped_off_negative && ++noise_flips == 2)
            return best_t;

        const double step = rising ? -f / slope : f / slope;
        double next = t + step;
        // Converged before the bracket test: a step below one ulp leaves next == t, which is now
        // a bracket end.
        if (std::abs(step) <= detail::TWO * std::numeric_limits<double>::epsilon() *
                                  std::max(detail::ONE, std::abs(t))) {
            if (!last_step)
                return next;
            *last_step = step;
            if (last_slope)
                *last_slope = slope;
            return t;
        }
        // A long falling step on an exp_cliff residual is Newton in eᵗ: eᵗ·(1 + step), or a
        // bisection where that is not positive.
        const bool cliff = exp_cliff && !rising && step < -0.9;
        if (cliff)
            next = t + std::log1p(step);
        const bool newton = std::isfinite(next) && next > lo && next < hi;
        if (!newton)
            next = detail::HALF * (lo + hi);  // underflowed tail or a step out of the bracket
        if (next == t)
            return best_t;  // the bracket has closed on t
        if (newton && !cliff && f < detail::ZERO_DOUBLE)
            stepped_off_negative = true;
        t = next;
    }
    return best_t;
}

// corvus's inverse since the v1.2.0 pin. v2.5.0's log-x Newton solver (#160) and its closure
// (log_gamma_tail, the Temme zone, the series and continued fraction, gamma_q_small_shape) stayed
// through the v1.0.1 port because that corvus failed the small-shape upper tail (corvus #48) and
// deep-subnormal targets (corvus #72); both are fixed in v1.1.0 and test_gamma_quantile_accuracy
// and the Gamma/Erlang/ChiSquared identities pass on corvus alone. The guards keep libstats'
// contract at the ends of p, where corvus returns NaN outside [0, 1]. The Wilson-Hilferty
// shortcut for shape >= 1e16 (#226) is gone with the solver: at 1e32 it returned the shape
// itself, 0.711 ulp from the p = 0.1 and 0.9 quantiles, where corvus rounds correctly.
double gamma_p_inv(double a, double p) noexcept {
    if (std::isnan(a) || std::isnan(p) || a <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    if (p <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p >= detail::ONE)
        return std::numeric_limits<double>::infinity();
    return corvus::gamma_p_inv(a, p);
}

double log_small_shape_beta(double s, double t) noexcept {
    // log(s·B(s, t)) = lgamma(1 + s) − (lgamma(t + s) − lgamma(t)), each part of order s; the
    // second as lgamma_ratio_excess(t, s) + s·log(t + s), but to relative rather than absolute
    // ε: there c(z) alone absorbed the O(s) terms (0 at s = 1e-228), and c(z + s) − c(z), of
    // order s/(12z²), is formed term by term from 1/(z + s) − 1/z = −s/(z(z + s)).
    double z = t;
    double acc = detail::ZERO_DOUBLE;
    while (z < kStirlingPrefactorShape) {
        acc -= std::log1p(s / z);
        z += detail::ONE;
    }
    if (z != t)
        acc += s * std::log((z + s) / (t + s));
    // c(r) = r/12 − r³/360 + r⁵/1260 − r⁷/1680 + r⁹/1188 at r = 1/(z + s) less at r = 1/z:
    // qⁿ − pⁿ = (q − p)·(qⁿ⁻¹ + qⁿ⁻²p + … + pⁿ⁻¹).
    const double p = detail::ONE / z;
    const double q = detail::ONE / (z + s);
    const double dq = -s / (z * (z + s));
    static constexpr double kStirlingCoeff[] = {1.0 / 12.0, -1.0 / 360.0, 1.0 / 1260.0,
                                                -1.0 / 1680.0, 1.0 / 1188.0};
    double dc = detail::ZERO_DOUBLE;
    double geometric = detail::ONE;  // q^(n−1) + … + p^(n−1) for n = 1, 3, 5, …
    double q_pow = detail::ONE;      // q^(n−1)
    for (int k = 0; k < 5; ++k) {
        dc += kStirlingCoeff[k] * geometric;
        // n → n + 2 by S_{n+1} = qⁿ + p·S_n twice.
        const double s_next = q_pow * q + p * geometric;
        q_pow *= q * q;
        geometric = q_pow + p * s_next;
    }
    const double excess = acc + (z - detail::HALF) * std::log1p(s / z) - s + dq * dc;
    return lgamma1p_small(s) - excess - s * std::log(t + s);
}

double beta_i(double x, double a, double b) noexcept {
    // I_x(a, b) from corvus (v3.0.0); the guards keep the pre-adoption edge contract.
    if (std::isnan(x) || std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE || x > detail::ONE || a <= detail::ZERO_DOUBLE ||
        b <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (x == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (x == detail::ONE)
        return detail::ONE;
    // I_½(a, a) = ½ by symmetry; a kernel an ulp off it moved a discrete quantile at that exact
    // tie (NegativeBinomial(5, ½) at p = ½ gave 5, not 4).
    if (a == b && x == detail::HALF)
        return detail::HALF;
    return corvus::beta_p(a, b, x);
}

double beta_i(double x, double a, double b, double /*log_beta_prefix*/) noexcept {
    // The prefix was the caller's hoisted lgamma triple for the retired local core; corvus forms
    // its own prefactor. Kept for source compatibility; batch callers belong on vector_beta_i.
    return beta_i(x, a, b);
}

double beta_i_pair(double x, double y, double a, double b, double /*log_beta_prefix*/) noexcept {
    // I_x(a, b) with y = 1 − x supplied exactly by the caller (#159). corvus takes whichever side
    // is the small one: I_x(a, b) directly, or 1 − I_y(b, a) = beta_q(b, a, y) by the swap
    // identity, so a tiny y is never formed as 1 − x.
    if (std::isnan(x) || std::isnan(y) || std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    if (a <= detail::ZERO_DOUBLE || b <= detail::ZERO_DOUBLE || x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (y <= detail::ZERO_DOUBLE)
        return detail::ONE;
    if (a == b && (x == y || x == detail::HALF))
        return detail::HALF;  // by symmetry; see beta_i
    return x <= y ? corvus::beta_p(a, b, x) : corvus::beta_q(b, a, y);
}

// Helper function for beta incomplete function continued fraction
// Based on Numerical Recipes algorithm
static double beta_continued_fraction(double x, double y, double a, double b) noexcept {
    // Return the continued fraction value multiplied by 1/a
    // This is part of the standard algorithm for regularized incomplete beta
    return beta_continued_fraction_unscaled(x, y, a, b) / a;
}

// The continued fraction h itself, without the leading 1/a (folded into the log prefactor by
// the tiny-shape path, where exp(·)/a cancels).
//
// h = 1/(1 + d₁/(1 + d₂/(1 + d₃/…))), the Numerical Recipes fraction with
//   d₂ₘ = m(b − m)x / ((a + 2m − 1)(a + 2m)),
//   d₂ₘ₊₁ = −(a + m)(a + b + m)x / ((a + 2m)(a + 2m + 1)),
// evaluated as its odd contraction (Lentz): 1/h = B₀ + A₁/(B₁ + A₂/(B₂ + …)) with
//   B₀ = 1 + d₁,   Bₘ = 1 + d₂ₘ + d₂ₘ₊₁,   Aₘ = −d₂ₘ₋₁·d₂ₘ.
// One step of it is two of the uncontracted fraction. The point of contracting is 1 + d₂ₘ₊₁:
// near x = 1 with a large it is 1 − (1 − O(1/a)), and the uncontracted Lentz steps form that
// difference implicitly from a rounded x·(a + m)(a + b + m)/(…), so h came out as if x were
// perturbed by an ulp — a relative error of cond_x(h)·ε, cond_x(h) ≈ 0.68·a at the branch point
// (4e-11 at a = 1e6, b = 1e-3; 1e-10 just below it). No stopping rule fixes that: the
// uncontracted error random-walks at that level for hundreds of iterations past the stop.
// Here the difference is formed in closed form:
//   (a + 2m)(a + 2m + 1)(1 + d₂ₘ₊₁) = (a + m)·λ + a + 2m + am(3 − x) + m²(4 − x),
//   λ = a − (a + b)x = (a + b)(1 − x) − b,
// every term but (a + m)·λ positive, and λ itself formed with one rounding: a + b split exactly
// (TwoSum, no products, so FP contraction cannot touch it) and the product fused explicitly,
// from the smaller of x and y. y is the caller's 1 − x: exact from x ≥ ½ (Sterbenz), and where
// the caller holds it more exactly than 1 − x — the tail integral's 1 − x_b = s₀, the quantile
// solver's logit — re-forming it from x would put the ulp of x back in. Below the branch point
// x_b = (a + 1)/(a + b + 2), where every caller evaluates it, 1 + λ = (a + 1)(1 + d₁) ≥
// 2(a + 1)/(a + b + 2) > 0, so B₀ never cancels. The contracted error is a few ε per step
// (≤ 6e-15 over 325 mpmath rows, shapes 1e-3…1e6, at and below x_b), and the single-step
// stopping rule is sound once the computed steps track the true ones.
static double beta_continued_fraction_unscaled(double x, double y, double a, double b) noexcept {
    const int max_iterations = expansion_iteration_cap(std::max(a, b));
    const double tolerance = detail::SPECIAL_FUNCTION_TOLERANCE;

    // a + b = s + e exactly (TwoSum).
    const double s = a + b;
    const double bv = s - a;
    const double e = (a - (s - bv)) + (b - bv);
    double lambda;
    if (x <= y)
        lambda = detail::fma_portable(-s, x, a) - e * x;
    else
        lambda = detail::fma_portable(s, y, -b) + e * y;

    // Past a shape of 1e100 the products of two shape-sized factors below — (a + m)·(a + b + m),
    // (a + m)·λ, (a + 2m)(a + 2m + 1) — overflow, to inf/inf = NaN (F(2, 1e308).cdf(1.5) =
    // I_y(1, 5e307) at y = 3e-308). There each is divided down before it is multiplied up, and
    // the fraction is carried in the equivalent form with every Bₘ multiplied by w = max(a, 1)
    // and every Aₘ by w² (f then holds w/h): at large a the Bₘ are of order (m + λ)/a and the Aₘ
    // of order 1/a², which left the normal range from a ~ 1e154 (I_{1−9e-170}(1e168, 21) was
    // 1e16·ε off). Below it the unscaled forms keep their bits.
    const bool scaled = std::max(a, b) > 1e100;
    const double w = std::max(a, detail::ONE);

    // Lentz's guard against a zero denominator: only has to stay clear of 0. The usual 1e-30
    // replaced B₀ = (1 + λ)/(a + 1), legitimately of order 1/a near x = 1, from a ~ 1e30
    // (I_{1−3.6e-40}(1e40, 21) was 5e-12 for 3e-3).
    constexpr double kLentzTiny = std::numeric_limits<double>::min();
    const double ap1 = a + detail::ONE;
    double f = scaled ? (detail::ONE + lambda) * (w / ap1)
                      : (detail::ONE + lambda) / ap1;  // B₀ = 1 + d₁ (times w when scaled)
    if (std::abs(f) < kLentzTiny)
        f = kLentzTiny;
    double c = f;
    double d = detail::ZERO_DOUBLE;
    // d₂ₘ₋₁, starting at d₁ (times w when scaled)
    double d_odd = scaled ? -(s * x) * (w / ap1) : -(s * x) / ap1;

    for (int m = 1; m <= max_iterations; ++m) {
        const double md = static_cast<double>(m);
        const double apm = a + md;
        const double a2m = a + detail::TWO * md;
        const double a2m1 = a2m + detail::ONE;
        double d_even;  // d₂ₘ (times w when scaled)
        double bm;
        if (!scaled) {
            d_even = md * (b - md) * x / ((a2m - detail::ONE) * a2m);
            const double num =
                apm * lambda + (a + detail::TWO * md + a * md * (3.0 - x) + md * md * (4.0 - x));
            bm = num / (a2m * a2m1) + d_even;
        } else {
            const double w_over = w / a2m1;
            d_even = md * x * ((b - md) / (a2m - detail::ONE)) * (w / a2m);
            bm = (apm / a2m) * (lambda * w_over) +
                 ((a + detail::TWO * md) / a2m + (a / a2m) * (md * (3.0 - x)) +
                  (md / a2m) * (md * (4.0 - x))) *
                     w_over +
                 d_even;
        }
        const double am = -d_odd * d_even;

        d = bm + am * d;
        if (std::abs(d) < kLentzTiny)
            d = kLentzTiny;
        c = bm + am / c;
        if (std::abs(c) < kLentzTiny)
            c = kLentzTiny;
        d = detail::ONE / d;
        const double delta = c * d;
        f *= delta;

        // d₂ₘ₊₁, the next step's d₂ₘ₋₁
        d_odd = scaled ? -(apm / a2m) * ((s + md) * (w / a2m1)) * x
                       : -apm * (s + md) * x / (a2m * a2m1);
        if (std::abs(delta - detail::ONE) < tolerance)
            break;
    }

    if (scaled)
        return w / f;
    return detail::ONE / f;
}

// log B(a, b). Formed directly, lgamma(a) + lgamma(b) − lgamma(a + b) cancels terms of size
// max(a, b)·log(a + b): 2e-10 relative in the Student-t normaliser at ν = 1e6. Once the larger
// argument x reaches kStirlingPrefactorShape, Stirling's series gives the difference for it
// without forming either lgamma; with y the smaller argument,
//   lgamma(x) − lgamma(x + y) = −(x − ½)·log1p(y/x) − y·log(x + y) + y + c(x) − c(x + y),
// whose terms are of the size of the result.
double lbeta(double a, double b) noexcept {
    if (std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    return corvus_scalar(corvus::lbeta, a, b);
}

double digamma(double x) noexcept {
    // ψ(x) from corvus (v3.0.0); x ≤ 0 keeps the pre-adoption NaN contract.
    if (x <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    return corvus_scalar(corvus::digamma, x);
}

double trigamma(double x) noexcept {
    if (x <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    return corvus_scalar(corvus::trigamma, x);
}

// x with I_x(a, b) = p (#137), by Newton in the logit u = log(x/(1 − x)) on the small side of
// the probability scale, the gamma_p_inv pattern. For p > ½ the problem is reflected,
// I_y(b, a) = 1 − p with y = 1 − x, so the target is always exact (1 − p is exact for p ≥ ½)
// and beta_i evaluates its direct continued fraction, never 1 − (something near 1). The
// density of U = logit(X), e^{au}/(1 + e^u)^{a+b}, is log-concave for every a, b > 0 (its log
// has second derivative −(a + b)·e^u/(1 + e^u)²), so f(u) = log I − log p is concave and
// solve_concave converges from any start. |f'(u)| = x(1 − x)·pdf(x)/I, the logit prefactor
// over I. Before this the linear-CDF Newton stopped at |I − p| < 1e-8 and clamped its start to
// [1e-8, 1 − 1e-8]: Beta(1, 1).Q(1e-100) = 1e-8 and Beta(2, 3).Q(1 − 1.7e-6) was 1.4e-5 off.
double inverse_beta_i(double p, double a, double b) noexcept {
    // x with I_x(a, b) = p from corvus beta_p_inv (v3.0.0), which solves against the smaller of p
    // and 1 − p itself. Guards keep the pre-adoption edge contract (#137): an infinite shape is
    // NaN.
    if (std::isnan(p) || !(a > detail::ZERO_DOUBLE && b > detail::ZERO_DOUBLE) || std::isinf(a) ||
        std::isinf(b))
        return std::numeric_limits<double>::quiet_NaN();
    if (p <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p >= detail::ONE)
        return detail::ONE;
    if (a == b && p == detail::HALF)
        return detail::HALF;  // by symmetry; beta_i returns ½ at that point too
    return corvus::beta_p_inv(a, b, p);
}

// The corvus-order inverses added with the adoption (v3.0.0): parameters first, probability last.
// Each solves against whichever of p and 1 − p is the smaller, so one call is accurate on both
// sides of the median (corvus.h).
double erfc_inv(double y) noexcept {
    return corvus_scalar(corvus::erfcinv, y);
}

double gamma_q_inv(double a, double q) noexcept {
    return corvus::gamma_q_inv(a, q);
}

double beta_q_inv(double a, double b, double q) noexcept {
    return corvus::beta_q_inv(a, b, q);
}

// =============================================================================
// MODIFIED BESSEL FUNCTIONS (bessel.h) — corvus i0 / i1 / i0e / i1e
// =============================================================================

double bessel_i0(double x) noexcept {
    return corvus_scalar(corvus::i0, x);
}

double bessel_i1(double x) noexcept {
    return corvus_scalar(corvus::i1, x);
}

double log_bessel_i0(double x) noexcept {
    // |x| + log i0e(|x|): i0e = e^-|x| I0 never underflows on a finite double, so one form serves
    // the whole axis and there is no seam to value-match (#92). Absolute error ~ulp(|x|), which
    // is what the von Mises normaliser LN_2PI + log I0 needs. The v2.x edges (I0 = 1 below 1e-8,
    // the log(2πx) split past DBL_MAX/2π, Fix-9) have no counterpart here: nothing is multiplied
    // by x and corvus i0e carries the small-x limit itself.
    const double ax = std::fabs(x);
    if (std::isinf(ax)) {
        return ax;
    }
    if (ax < 1e-8) {
        // log I0(x) = x²/4 + O(x⁴) here; |x| + log i0e(|x|) would cancel to |x| once e^-|x| rounds
        // to 1 (fix-9's gate: log_bessel_i0(5e-324) = 0, as I0 is 1 to double).
        return 0.25 * ax * ax;
    }
    return ax + std::log(corvus_scalar(corvus::i0e, ax));
}

double log_bessel_i0_scaled(double x) noexcept {
    // log I0(x) − x = log i0e(x) directly; the series past kBesselRatioAsymptoticCut that v2.5.0
    // needed to avoid e^x is unnecessary when the scaled function is the primitive.
    return std::log(corvus_scalar(corvus::i0e, x));
}

double bessel_i1_i0_complement(double x) noexcept {
    if (x >= kBesselRatioAsymptoticCut) {
        // Horner in t = 1/x. Finite for every x up to +inf (t → 0 → 0).
        const double t = 1.0 / x;
        return t *
               (0.5 +
                t * (0.125 +
                     t * (0.125 +
                          t * (0.1953125 +
                               t * (0.40625 +
                                    t * (1.0478515625 +
                                         t * (3.21875 + t * (11.466461181640625 +
                                                             t * (46.478515625 +
                                                                  t * 211.27614974975586)))))))));
    }
    return 1.0 - bessel_i1_over_i0(x);
}

double bessel_i1_over_i0(double x) noexcept {
    // Past the cut the complement is small, so 1 − complement is the stable direction; below it
    // the scaled ratio is direct (1 − complement would cancel as κ → 0). i1e/i0e never forms the
    // overflowing I0/I1 (#93).
    if (x >= kBesselRatioAsymptoticCut) {
        return 1.0 - bessel_i1_i0_complement(x);
    }
    return corvus_scalar(corvus::i1e, x) / corvus_scalar(corvus::i0e, x);
}

// =============================================================================
// NUMERICAL INTEGRATION AND ROOT FINDING
// =============================================================================
//
// Note: Template function implementations are in the header file
// These are placeholder implementations for debugging and non-template fallbacks

// =============================================================================
// STATISTICAL UTILITIES
// =============================================================================

std::vector<double> empirical_cdf(std::span<const double> data) {
    // Placeholder: Calculate empirical CDF
    // Sort data and compute CDF values
    std::vector<double> sorted_data(data.begin(), data.end());
    std::sort(sorted_data.begin(), sorted_data.end());
    std::vector<double> cdf(sorted_data.size());
    for (std::size_t i = 0; i < sorted_data.size(); ++i) {
        cdf[i] = static_cast<double>(i + 1) / static_cast<double>(sorted_data.size());
    }
    return cdf;
}

std::vector<double> calculate_quantiles(std::span<const double> data,
                                        std::span<const double> quantiles) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot calculate quantiles from empty data");
    }

    // Sort data for quantile calculation
    std::vector<double> sorted_data(data.begin(), data.end());
    std::sort(sorted_data.begin(), sorted_data.end());

    std::vector<double> result;
    result.reserve(quantiles.size());

    for (double q : quantiles) {
        if (q < detail::ZERO_DOUBLE || q > detail::ONE) {
            throw std::invalid_argument("Quantile values must be in [0, 1]");
        }

        if (q == detail::ZERO_DOUBLE) {
            result.push_back(sorted_data.front());
        } else if (q == detail::ONE) {
            result.push_back(sorted_data.back());
        } else {
            // Use linear interpolation between data points
            double pos = q * static_cast<double>(sorted_data.size() - 1);
            std::size_t lower_idx = static_cast<std::size_t>(std::floor(pos));
            std::size_t upper_idx = static_cast<std::size_t>(std::ceil(pos));

            if (lower_idx == upper_idx) {
                result.push_back(sorted_data[lower_idx]);
            } else {
                double weight = pos - static_cast<double>(lower_idx);
                double interpolated = sorted_data[lower_idx] * (detail::ONE - weight) +
                                      sorted_data[upper_idx] * weight;
                result.push_back(interpolated);
            }
        }
    }

    return result;
}

std::array<double, 4> sample_moments(std::span<const double> data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot calculate moments from empty data");
    }

    const std::size_t n = data.size();

    // Calculate mean
    double sum = detail::ZERO_DOUBLE;
    for (double x : data) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("Data contains non-finite values");
        }
        sum += x;
    }
    double mean = sum / static_cast<double>(n);

    // Calculate central moments
    double m2 = detail::ZERO_DOUBLE, m3 = detail::ZERO_DOUBLE, m4 = detail::ZERO_DOUBLE;
    for (double x : data) {
        double diff = x - mean;
        double diff2 = diff * diff;
        double diff3 = diff2 * diff;
        double diff4 = diff3 * diff;

        m2 += diff2;
        m3 += diff3;
        m4 += diff4;
    }

    m2 /= static_cast<double>(n);
    m3 /= static_cast<double>(n);
    m4 /= static_cast<double>(n);

    // Calculate variance (sample variance with Bessel's correction)
    double variance = (n > 1) ? (m2 * static_cast<double>(n)) / static_cast<double>(n - 1) : m2;

    // Calculate skewness and kurtosis
    double skewness = std::numeric_limits<double>::quiet_NaN();
    double kurtosis = std::numeric_limits<double>::quiet_NaN();

    if (m2 > detail::ZERO) {
        double sigma = std::sqrt(m2);
        double sigma3 = sigma * sigma * sigma;
        double sigma4 = sigma3 * sigma;

        skewness = m3 / sigma3;
        kurtosis = (m4 / sigma4) - detail::EXCESS_KURTOSIS_OFFSET;  // Excess kurtosis
    }

    return {mean, variance, skewness, kurtosis};
}

bool validate_fitting_data(std::span<const double> data) noexcept {
    return std::all_of(data.begin(), data.end(), [](double x) { return std::isfinite(x); });
}

// =============================================================================
// GOODNESS-OF-FIT TESTING
// =============================================================================

double calculate_ks_statistic(const std::vector<double>& data,
                              const DistributionBase& dist) noexcept {
    if (data.empty()) {
        return detail::ZERO_DOUBLE;
    }

    // Create a copy of the data for sorting
    std::vector<double> sorted_data(data);
    std::sort(sorted_data.begin(), sorted_data.end());

    const auto n = static_cast<double>(sorted_data.size());
    double max_diff = detail::ZERO_DOUBLE;

    // Calculate KS statistic: max |F_n(x) - F(x)|
    for (std::size_t i = 0; i < sorted_data.size(); ++i) {
        double ecdf_i = static_cast<double>(i + 1) / n;  // empirical CDF at step i
        double theoretical_cdf = dist.getCumulativeProbability(sorted_data[i]);

        // Check both F_n(x) - F(x) and F(x) - F_{n-1}(x)
        double diff1 = std::abs(ecdf_i - theoretical_cdf);
        double diff2 = std::abs(theoretical_cdf - static_cast<double>(i) / n);

        max_diff = std::max(max_diff, std::max(diff1, diff2));
    }

    return max_diff;
}

double calculate_ad_statistic(const std::vector<double>& data,
                              const DistributionBase& dist) noexcept {
    if (data.empty()) {
        return detail::ZERO_DOUBLE;
    }

    // Create a copy of the data for sorting
    std::vector<double> sorted_data(data);
    std::sort(sorted_data.begin(), sorted_data.end());

    const auto n = static_cast<double>(sorted_data.size());
    double ad_sum = detail::ZERO_DOUBLE;

    // Calculate Anderson-Darling statistic using numerically stable approach
    for (std::size_t i = 0; i < sorted_data.size(); ++i) {
        double F_xi = dist.getCumulativeProbability(sorted_data[i]);

        // Clamp F values to avoid numerical issues with log(0) and log(negative)
        // Use
        F_xi = std::max(detail::ANDERSON_DARLING_MIN_PROB,
                        std::min(detail::ONE - detail::ANDERSON_DARLING_MIN_PROB, F_xi));

        // Calculate log terms with safe bounds
        double log_F_xi = std::log(F_xi);
        double log_one_minus_F_xi = std::log(detail::ONE - F_xi);

        // Correct Anderson-Darling formula with proper indexing (i+1 for 1-based indexing)
        ad_sum += (detail::TWO * static_cast<double>(i + 1) - detail::ONE) * log_F_xi +
                  (detail::TWO * n - detail::TWO * static_cast<double>(i + 1) + detail::ONE) *
                      log_one_minus_F_xi;
    }

    return -n - ad_sum / n;
}

// =============================================================================
// SIMD VECTORIZED SPECIAL FUNCTIONS
// =============================================================================

// Block length for the constant-argument fills below (PLAN.md Decided 2026-09-29):
// a multiple of the widest lane count (8 doubles at AVX-512), so only the last
// block is ragged, and corvus handles that itself. 256 doubles is 2 KB of stack
// per filled argument.
namespace {
constexpr std::size_t kCorvusBlock = 256;
}  // namespace

void vector_erf(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;
    }
    corvus::erf(input, output);
}

void vector_gamma_p(double a, std::span<const double> x_values, std::span<double> output) noexcept {
    if (x_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> a_fill;
    a_fill.fill(a);
    const std::size_t n = x_values.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::gamma_p(std::span<const double>{a_fill.data(), len}, x_values.subspan(i, len),
                        output.subspan(i, len));
    }
}

void vector_gamma_q(double a, std::span<const double> x_values, std::span<double> output) noexcept {
    if (x_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> a_fill;
    a_fill.fill(a);
    const std::size_t n = x_values.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::gamma_q(std::span<const double>{a_fill.data(), len}, x_values.subspan(i, len),
                        output.subspan(i, len));
    }
}

void vector_gamma_q(std::span<const double> a_values, double x, std::span<double> output) noexcept {
    // Poisson's shape: Q(k+1, λ) varies the FIRST argument.
    if (a_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> x_fill;
    x_fill.fill(x);
    const std::size_t n = a_values.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::gamma_q(a_values.subspan(i, len), std::span<const double>{x_fill.data(), len},
                        output.subspan(i, len));
    }
}

void vector_beta_i(std::span<const double> x_values, double a, double b,
                   std::span<double> output) noexcept {
    if (x_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> a_fill;
    std::array<double, kCorvusBlock> b_fill;
    a_fill.fill(a);
    b_fill.fill(b);
    const std::size_t n = x_values.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::beta_p(std::span<const double>{a_fill.data(), len},
                       std::span<const double>{b_fill.data(), len}, x_values.subspan(i, len),
                       output.subspan(i, len));
    }
}

void vector_beta_q_x(std::span<const double> x_values, double a, double b,
                     std::span<double> output) noexcept {
    if (x_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> a_fill;
    std::array<double, kCorvusBlock> b_fill;
    a_fill.fill(a);
    b_fill.fill(b);
    const std::size_t n = x_values.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::beta_q(std::span<const double>{a_fill.data(), len},
                       std::span<const double>{b_fill.data(), len}, x_values.subspan(i, len),
                       output.subspan(i, len));
    }
}

void vector_beta_p_shapes(std::span<const double> a_values, std::span<const double> b_values,
                          double x, std::span<double> output) noexcept {
    if (a_values.size() != output.size() || b_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> x_fill;
    x_fill.fill(x);
    const std::size_t n = output.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::beta_p(a_values.subspan(i, len), b_values.subspan(i, len),
                       std::span<const double>{x_fill.data(), len}, output.subspan(i, len));
    }
}

void vector_beta_q_shapes(std::span<const double> a_values, std::span<const double> b_values,
                          double x, std::span<double> output) noexcept {
    if (a_values.size() != output.size() || b_values.size() != output.size()) {
        return;
    }
    std::array<double, kCorvusBlock> x_fill;
    x_fill.fill(x);
    const std::size_t n = output.size();
    for (std::size_t i = 0; i < n; i += kCorvusBlock) {
        const std::size_t len = std::min(kCorvusBlock, n - i);
        corvus::beta_q(a_values.subspan(i, len), b_values.subspan(i, len),
                       std::span<const double>{x_fill.data(), len}, output.subspan(i, len));
    }
}

void vector_lgamma(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size()) {
        return;
    }
    corvus::lgamma(input, output);
}

void vector_lbeta(std::span<const double> a_values, std::span<const double> b_values,
                  std::span<double> output) noexcept {
    if (a_values.size() != b_values.size() || a_values.size() != output.size()) {
        return;
    }
    corvus::lbeta(a_values, b_values, output);
}

bool should_use_vectorized_math(std::size_t size) noexcept {
    // Use the same threshold as SIMD operations
    return arch::simd::SIMDPolicy::shouldUseSIMD(size);
}

std::size_t vectorized_math_threshold() noexcept {
    // Return the minimum size for SIMD operations
    return arch::simd::VectorOps::min_simd_size();
}

// =============================================================================
// STATISTICAL DISTRIBUTION FUNCTIONS
// =============================================================================

double normal_cdf(double z) noexcept {
    // Standard normal CDF using error function, tail-branched (#49).
    //
    // The naive cancellation form 0.5*(1+erf(z/sqrt(2))) has an absolute
    // error floor of ~ulp(1)/2 ~= 1.1e-16 for z < 0 regardless of erf's own
    // quality (1+erf(negative-near-1) cancels most of erf's precision), and
    // once z <~ -8.3, std::erf(z/sqrt(2)) returns exactly -1 so the whole
    // expression collapses to exactly 0 -- true relative error there is
    // 1.0, not the tiny value a naive metric reports. Using erfc for the
    // left tail avoids the cancellation entirely: 0.5*erfc(-z/sqrt(2))
    // stays accurate (full relative precision) all the way down to erfc's
    // own underflow floor (~1e-308), instead of pinning to a fixed
    // absolute floor.  Right tail (z >= 0) keeps the original form, which
    // is already well-conditioned there.
    return z < detail::ZERO_DOUBLE ? detail::HALF * erfc(-z * detail::INV_SQRT_2)
                                   : detail::HALF * (detail::ONE + erf(z * detail::INV_SQRT_2));
}

double inverse_normal_cdf(double p) noexcept {
    if (p <= detail::ZERO_DOUBLE || p >= detail::ONE) {
        if (p == detail::ZERO_DOUBLE)
            return -std::numeric_limits<double>::infinity();
        if (p == detail::ONE)
            return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }
    // Φ⁻¹(p) = −√2·erfc⁻¹(2p) on the small side (#158). 2p − 1 rounds to −1 for p < 2^-54; both 2p
    // and 1 − p (for p ≥ ½, Sterbenz) are exact, so each side is solved from an exact argument.
    return p <= detail::HALF ? -detail::SQRT_2 * erfc_inv(detail::TWO * p)
                             : detail::SQRT_2 * erfc_inv(detail::TWO * (detail::ONE - p));
}

// Seed: Abramowitz & Stegun 26.2.23 rational approximation (|error| < 4.5e-4), then Newton on
// the survival residual with analytic derivative:
//   u ← u + (Q(u) − s)/φ(u),  φ(u) = exp(−u²/2)/√(2π)
// Each step evaluates erfc and exp directly — both full relative precision in the tail — so the
// iteration converges to the |ln s|·2⁻⁵² conditioning limit of any double formulation (the #49
// law). Below DBL_MIN, where φ and erfc are themselves subnormal near the root, Newton runs on
// log Q(u) = −u²/2 − log(u·√(2π)) + log g(u) instead, with g's asymptotic series
// Σ (−1)^k·(2k − 1)!!/u^{2k}: u > 37.5 there, so eight terms reach 1e-19. A clamp to DBL_MIN
// returned Φ⁻¹(DBL_MIN) for every subnormal s, 2.5% off at 5e-324. That log branch is
// inv_survival_normal_log, which callers also use directly when the target s is a product
// (p·Z, for TruncatedNormal) that is not representable at all: s = 0 handed to the log branch
// made u = ∞ and the Newton step ∞ − ∞ = NaN (#225); log_s = −∞ now returns u = ∞, the limit.
//
// Not delegated to erf_inv: its extreme-tail branch (|x| ≥ ERF_INV_TAIL_CUTOFF) seeds with a
// Φ⁻¹-domain formula that is off by ~√2 in the erf domain, and its Halley refinement cannot
// recover once std::erf saturates to 1 — measured during #57 bring-up: erf_inv(1−1e-14) ≈ 7.59
// vs the true 5.46. Until v2.5.0 this helper was duplicated in half_normal.cpp and
// truncated_normal.cpp (#158 promoted it here).
double inv_survival_normal_log(double log_s) noexcept {
    if (std::isnan(log_s))
        return log_s;
    if (log_s >= std::log(std::numeric_limits<double>::min()))
        return inv_survival_normal(std::exp(log_s));
    if (log_s == -std::numeric_limits<double>::infinity())
        return std::numeric_limits<double>::infinity();  // s = 0: the tail's limit, not a NaN
    double u = std::sqrt(-detail::TWO * log_s);
    for (int i = 0; i < 8; ++i) {
        const double z = detail::ONE / (u * u);
        double g = detail::ONE;
        double term = detail::ONE;
        for (int k = 1; k <= 8; ++k) {
            term *= -(2.0 * k - detail::ONE) * z;
            g += term;
        }
        const double log_q =
            -detail::HALF * u * u - std::log(u) - detail::HALF * detail::LN_2PI + std::log(g);
        const double step = (log_q - log_s) * g / u;  // f/|f'|, f'(u) = −u/g
        u += step;
        if (std::fabs(step) <= 1e-16 * u)
            break;
    }
    return u;
}

double inv_survival_normal(double s) noexcept {
    // u with Q(u) = ½·erfc(u/√2) = s, s ∈ (0, ½]: u = √2·erfc⁻¹(2s) from corvus (v3.0.0). Below
    // DBL_MIN the log-domain Newton above keeps the subnormal targets (#225).
    if (s >= detail::HALF)
        return detail::ZERO_DOUBLE;
    if (s < std::numeric_limits<double>::min())
        return inv_survival_normal_log(std::log(s));
    return detail::SQRT_2 * erfc_inv(detail::TWO * s);
}

// I_x(a, ½) for large a and x near 1 (#159): DiDonato & Morris's BGRAT expansion (ACM TOMS 708,
// eq. 9 to 9.6), I_x(a, b) = Γ(a + b)/(Γ(a)·T^b) · Σ p_n·J_n with T = a + (b − 1)/2,
// u = −T·log x, J_0 = Q(b, u) and J_n scaled by the gamma prefactor h = u^b·e^{−u}/Γ(b) so that an
// underflowed h cannot overflow them. There the continued fraction runs thousands of terms and
// accumulates ~1e-12; this needs a handful. Written for b = ½, where Q(½, u) = erfc(√u).
static double beta_i_large_a_half(double a, double log_x) noexcept {
    constexpr double b = 0.5;
    constexpr int kTerms = 30;
    const double T = a + (b - detail::ONE) * detail::HALF;
    const double u = -T * log_x;
    // lead = Γ(a + ½)/(Γ(a)·√T), with lbeta(a, ½) + ½·log T written out in Stirling's form so
    // that its two ½·log(a)-sized halves never meet (a ≥ 20 here): the remainder is O(1/a),
    // and lead is 1 − 3/(8a) + … to a rounding of its own size at every a up to 1e308 (B2).
    const double log_lead = (a - detail::HALF) * std::log1p(detail::HALF / a) - detail::HALF -
                            stirling_remainder(a) + stirling_remainder(a + detail::HALF) -
                            detail::HALF * std::log1p(-0.75 / (a + detail::HALF));
    const double lead = std::exp(log_lead);
    const double h = std::sqrt(u / detail::PI) * std::exp(-u);
    const double lx2 = detail::HALF * log_x * detail::HALF * log_x;
    const double t4 = 4.0 * T * T;

    double odd_factorial[kTerms + 1];  // (2k + 1)!
    odd_factorial[0] = detail::ONE;
    for (int k = 1; k <= kTerms; ++k)
        odd_factorial[k] = odd_factorial[k - 1] * (2.0 * k) * (2.0 * k + 1.0);

    double p[kTerms];
    p[0] = detail::ONE;
    double J = std::erfc(std::sqrt(u));
    double sum = J;
    double lxp = detail::ONE;
    double b2n = b;
    for (int n = 1; n < kTerms; ++n) {
        p[n] = detail::ZERO_DOUBLE;
        for (int m = 1; m < n; ++m)
            p[n] += (m * b - n) * p[n - m] / odd_factorial[m];
        p[n] = p[n] / n + (b - detail::ONE) / odd_factorial[n];
        J = (b2n * (b2n + detail::ONE) * J + (u + b2n + detail::ONE) * lxp * h) / t4;
        lxp *= lx2;
        b2n += detail::TWO;
        const double r = p[n] * J;
        sum += r;
        if (std::abs(r) <= std::numeric_limits<double>::epsilon() * std::abs(sum))
            break;
    }
    return lead * sum;
}

namespace {
// Both tails of Student's t at |t|: tail = P(T < −|t|) = ½·I_x(ν/2, ½) and central =
// P(|T| < |t|) = I_y(½, ν/2), with x = ν/(ν + t²) and y = 1 − x (#159). x and y come from u = t²/ν
// or 1/u, whichever is below 1, and their logs from log1p, so neither is formed by subtraction:
// relative-accurate at large ν, where x is within 1/ν of 1, and past |t| ~ 1e154, where t²
// overflows and x underflows. The continued fraction runs on whichever side converges; the other
// value is formed from it. log_t_pdf = log(|t|·pdf(t)), which is the prefactor x^a·y^½/B(a, ½);
// lbeta_a_half = lbeta(ν/2, ½), which callers hoist (it is three lgamma calls).
struct TTails {
    double tail;
    double central;
    double log_t_pdf;
};

TTails t_tails(double abs_t, double df, double lbeta_a_half) noexcept {
    const double a = detail::HALF * df;
    const double u = abs_t * abs_t / df;  // t²/ν; +inf past |t| ~ 1e154, which x takes as 0
    double x, y, log_x, log_y;
    if (u < detail::ONE) {
        x = detail::ONE / (detail::ONE + u);
        y = u / (detail::ONE + u);
        log_x = -std::log1p(u);
        log_y = detail::TWO * std::log(abs_t) - std::log(df) + log_x;
    } else {
        const double inv_u = df / abs_t / abs_t;  // underflows gracefully past |t| ~ 1e154
        x = inv_u / (detail::ONE + inv_u);
        y = detail::ONE / (detail::ONE + inv_u);
        log_y = -std::log1p(inv_u);
        // log x = log(1/u) + log y. inv_u carries about 1 ulp, so its log is good to ~2e-16
        // absolute; log nu - 2 log|t| cancelled to an error of order |log nu|·eps, which a = nu/2
        // then multiplied (2-7x the conditioning law beyond |t| = sqrt(nu), and non-monotone by
        // 113 ulp at nu = 100: M1 seam hunt). The difference of logs stays only where inv_u has
        // lost its relative precision (subnormal or 0).
        log_x = inv_u >= std::numeric_limits<double>::min()
                    ? std::log(inv_u) + log_y
                    : std::log(df) - detail::TWO * std::log(abs_t) + log_y;
    }
    TTails r{};
    if (a < std::numeric_limits<double>::min()) {
        // ν/2 subnormal or 0 (ν = 4.9e-324 halves to 0): lbeta(a, ½) and the continued
        // fraction's 1/a are infinite there, and the tail came out NaN. I_x(a, ½) = x^a + O(a)
        // as a → 0 (a·B(a, ½) = 1 + O(a), and ∫₀ˣ t^(a−1)((1 − t)^(−½) − 1) dt stays finite), so
        // the tail is ½·x^a to an absolute 1e-308, from log x, which holds where x underflows.
        const double a_log_x = a == detail::ZERO_DOUBLE ? detail::ZERO_DOUBLE : a * log_x;
        r.log_t_pdf = a_log_x + detail::HALF * log_y - lbeta_a_half;
        r.tail = detail::HALF * std::exp(a_log_x);
        r.central = -std::expm1(a_log_x);
        return r;
    }
    if (a >= kStirlingPrefactorShape && u < detail::ONE) {
        // ½·log y − lbeta(a, ½) holds −½·log ν against +½·log a inside lbeta's Stirling form;
        // their difference is K(a) = −½·log 2π + a·log1pmx(1/2a) − c(a) + c(a + ½), free of
        // log ν, so the prefactor keeps its relative precision at every ν (B2: the cancellation
        // cost 1e-13 at ν = 1e300). The hoisted lbeta_a_half is not needed on this side.
        const double k = -detail::HALF * std::log(detail::TWO * detail::PI) +
                         a * log1pmx(detail::HALF / a) - stirling_remainder(a) +
                         stirling_remainder(a + detail::HALF);
        r.log_t_pdf = a * log_x + std::log(abs_t) + detail::HALF * log_x + k;
    } else {
        r.log_t_pdf = a * log_x + detail::HALF * log_y - lbeta_a_half;
    }
    const double prefactor = std::exp(r.log_t_pdf);
    // The tail side is x < (a + 1)/(a + 2.5), i.e. (a + 1)·u > 1.5, decided on u: from
    // ν ≈ 4e17 both x and the bound round to 1 and the comparison on x sent every |t| to the
    // central continued fraction, which is wrong and slow at y ≈ 0 (B2).
    if ((a + detail::ONE) * u > 1.5) {
        // BGRAT where the continued fraction is slow (large a, x near 1); its domain follows
        // Boost's use of it (a ≥ 15, x ≥ 0.3). Its argument −T·log x = T·log1p(t²/ν) carries
        // the O(1/ν) departure from the normal tail exactly, and the J_n corrections fall as
        // (t²/2ν)²ⁿ, so from ν ~ 1e11 the sum is its first term, ½·erfc(√u)·lead, to ε.
        if (a >= kStirlingPrefactorShape && x >= detail::HALF)
            r.tail = detail::HALF * beta_i_large_a_half(a, log_x);
        else if (a >= kStirlingPrefactorShape && prefactor == detail::ZERO_DOUBLE)
            r.tail = detail::ZERO_DOUBLE;  // tail ≤ prefactor/a with x < ½: underflows too
        else
            r.tail = detail::HALF * prefactor * beta_continued_fraction(x, y, a, detail::HALF);
        r.central = detail::ONE - detail::TWO * r.tail;
    } else {
        r.central = prefactor * beta_continued_fraction(y, x, detail::HALF, a);
        r.tail = detail::HALF - detail::HALF * r.central;
    }
    return r;
}
}  // namespace

double t_cdf(double t, double df, double lbeta_a_half) noexcept {
    // Student's t CDF on the regularized incomplete beta, at every df: the df ≥ 1000 normal
    // shortcut this replaced was 1e-3 relative off in the tail (#159).
    if (std::isnan(t) || std::isnan(df) || df <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(t))
        return (t > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
    if (t == detail::ZERO_DOUBLE)
        return detail::HALF;
    const double tail = t_tails(std::fabs(t), df, lbeta_a_half).tail;
    return t < detail::ZERO_DOUBLE ? tail : detail::ONE - tail;
}

double t_cdf(double t, double df) noexcept {
    return t_cdf(t, df, lbeta(detail::HALF * df, detail::HALF));
}

// t with t_cdf(t, df) = p (#159), by Newton in s = log|t| on the smaller probability: log of the
// central mass P(|T| < |t|) against 1 − 2q for q = min(p, 1 − p) above ¼ (1 − 2q is exact there),
// log of the tail against q below. log|T| has the log-concave density
// ∝ e^s·(1 + e^{2s}/ν)^{−(ν+1)/2}, so both are concave in s and solve_concave applies;
// |d/ds| = |t|·pdf over the mass (twice that for the central one). The normal-seeded linear-CDF
// Newton this replaced returned −inf below p ~ 1e-17 and the normal quantile itself for df > 1000.
double inverse_t_cdf(double p, double df) noexcept {
    if (std::isnan(p) || std::isnan(df) || p < detail::ZERO_DOUBLE || p > detail::ONE ||
        df <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    if (p == detail::ZERO_DOUBLE)
        return -std::numeric_limits<double>::infinity();
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();
    if (p == detail::HALF)
        return detail::ZERO_DOUBLE;

    const double sign = p < detail::HALF ? -detail::ONE : detail::ONE;
    const double q = p < detail::HALF ? p : detail::ONE - p;  // exact for p ≥ ½
    // The answer exceeds the double range when even |t| = DBL_MAX leaves more than q in the tail.
    const double lbeta_a_half = lbeta(detail::HALF * df, detail::HALF);
    if (t_tails(std::numeric_limits<double>::max(), df, lbeta_a_half).tail >= q)
        return sign * std::numeric_limits<double>::infinity();

    // |t_q| > |z_q| at every ν: T is a normal scale mixture whose scale has mean below 1, and
    // Φ(x·w) is convex in w for x < 0. So log|z_q| − 1 is a lower bracket and log|z_q| a seed;
    // in the tail the small-x asymptote G ≈ ½·(ν/t²)^a/(a·B(a, ½)) is the better seed.
    const double a = detail::HALF * df;
    const double s_normal = std::log(-inverse_normal_cdf(q));
    const double lo = s_normal - detail::ONE;
    const double hi = 709.78;  // log(DBL_MAX)
    double s = s_normal;
    const double s_asymptote =
        detail::HALF *
        (std::log(df) - (std::log(detail::TWO * q) + std::log(a) + lbeta(a, detail::HALF)) / a);
    if (std::isfinite(s_asymptote) && s_asymptote > s && s_asymptote < hi &&
        s_asymptote > detail::HALF * std::log(df) + detail::ONE)
        s = s_asymptote;

    const bool central = q > 0.25;
    const double log_target = central ? std::log(detail::ONE - detail::TWO * q) : std::log(q);
    return sign * std::exp(solve_concave(s, lo, hi, central, [&](double ss, double& slope) {
               const TTails r = t_tails(std::exp(ss), df, lbeta_a_half);
               const double log_mass = central ? std::log(r.central) : std::log(r.tail);
               slope = std::exp(r.log_t_pdf - log_mass) * (central ? detail::TWO : detail::ONE);
               return log_mass - log_target;
           }));
}

double chi_squared_cdf(double x, double df) noexcept {
    // Chi-squared CDF using regularized incomplete gamma function
    if (x < detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (df <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // Chi-squared with df degrees of freedom is Gamma(df/2, 2)
    // CDF = P(df/2, x/2) = regularized incomplete gamma function
    return gamma_p(df * detail::HALF, x * detail::HALF);
}

double inverse_chi_squared_cdf(double p, double df) noexcept {
    // Chi-squared(df) is Gamma(df/2, scale 2).
    if (p < detail::ZERO_DOUBLE || p > detail::ONE || df <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return detail::TWO * gamma_p_inv(df * detail::HALF, p);
}

// f_cdf / inverse_f_cdf removed in v2.4.0 — see the note in math_utils.h.

double gamma_cdf(double x, double shape, double scale) noexcept {
    // Gamma distribution CDF using regularized incomplete gamma function
    if (x < detail::ZERO_DOUBLE || shape <= detail::ZERO_DOUBLE || scale <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // Gamma CDF: F(x; α, β) = P(α, x/β) where α=shape, β=scale
    // P(a,x) is the regularized incomplete gamma function
    return gamma_p(shape, x / scale);
}

double gamma_inverse_cdf(double p, double shape, double scale) noexcept {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE || shape <= detail::ZERO_DOUBLE ||
        scale <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return scale * gamma_p_inv(shape, p);
}

}  // namespace detail
}  // namespace stats

// =============================================================================
// stats::analysis:: public wrappers
// Delegate to the detail:: implementations above; no logic is duplicated.
// =============================================================================
namespace stats {
namespace analysis {

std::vector<double> empirical_cdf(std::span<const double> data) {
    return detail::empirical_cdf(data);
}

std::vector<double> calculate_quantiles(std::span<const double> data,
                                        std::span<const double> quantiles) {
    return detail::calculate_quantiles(data, quantiles);
}

std::array<double, 4> sample_moments(std::span<const double> data) {
    return detail::sample_moments(data);
}

bool validate_fitting_data(std::span<const double> data) noexcept {
    return detail::validate_fitting_data(data);
}

}  // namespace analysis
}  // namespace stats

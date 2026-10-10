#include "libstats/core/math_utils.h"

#include "libstats/common/cpu_detection_fwd.h"         // CPU feature queries (lightweight)
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/common/simd_policy_fwd.h"           // SIMD policy decisions (lightweight)
#include "libstats/core/distribution_base.h"
#include "libstats/core/math_constants.h"
#include "libstats/core/safety.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/stats/analysis/statistical_utilities.h"

#include <algorithm>
#include <array>
#include <cmath>
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
static double log_beta_prefactor_over_a(double x, double omx, double a, double b) noexcept;
static double log_beta_prefactor_over_a_logs(double log_x, double log_omx, double a,
                                             double b) noexcept;
static double gamma_p_series(double a, double x) noexcept;
double gamma_q(double a, double x) noexcept;

// Iteration cap for the incomplete gamma and beta expansions (#166). Near the median they need
// O(√a) terms — the series' terms fall like exp(−n²/2a), so reaching 3ε takes n ≈ 9√a — and the
// fixed caps of 100 (beta) and 1000 (gamma) stopped them early at large shape. The cap bounds
// non-convergence only; a converging expansion breaks out long before it. The std::min also
// sends a NaN scale to the 1e6 bound, which keeps the cast defined.
static int expansion_iteration_cap(double shape) noexcept {
    return static_cast<int>(std::min(1e6, 1000.0 + 20.0 * std::sqrt(std::fabs(shape))));
}

// The prefactors below switch to Stirling's form at this shape; under it the direct form is the
// more accurate of the two and keeps its pre-v2.4.2 bits.
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

// log of the incomplete-beta prefactor x^a·(1 − x)^b / B(a, b) (#166), with the same
// cancellation as the gamma one: lgamma(a + b) − lgamma(a) − lgamma(b) + a·log(x) + b·log(1 − x)
// carries terms of size (a + b)·log 2, 3e-11 relative at a = b = 1e4. For a, b ≥ 20, Stirling
// gives a·(log1p(u) − u) + b·(log1p(v) − v) + ½·log(ab / (2π(a + b))) + c(a + b) − c(a) − c(b),
// with x₀ = a/(a + b), u = (x − x₀)/x₀ and v = (x₀ − x)/(1 − x₀); the linear terms a·u + b·v
// cancel exactly, and x₀ is the stationary point, so its rounding enters only at second order.
// shape_constant is beta_prefactor_constant(a, b), in whichever form the shapes select.
//
// log_x and log_omx are log x and log(1 − x) for the factors below the Stirling shapes: the
// public overload passes log(x) and log(1 − x), the quantile solver the exact logs of its
// logit, which it holds where x itself has rounded to 1 (Beta(0.001, 1).Q(0.7): the reflected
// root is 1 − 0.7^1000, and 1 − x re-formed from x was 0, so b·log(1 − x) = −∞ zeroed the
// residual and the Newton slope, and the solver returned 6.6e-37 for 1.25e-155 — B1) and where
// a·log x magnifies the rounding of x (Beta(0.001, 1e6).Q(0.99): a = 1e6 against x = 1 − 2e-11
// put 1e-10 into the prefactor and 4e-8 into the root). The Stirling branch keeps reading x:
// both shapes ≥ 20 put the root of p ≤ ½ at 1 − x ≳ 20/(a + b), representable unless the
// larger shape exceeds ~2e17.
static double log_beta_prefactor_logs(double x, double a, double b, double log_x, double log_omx,
                                      double shape_constant) noexcept {
    if (a < kStirlingPrefactorShape || b < kStirlingPrefactorShape)
        return shape_constant + a * log_x + b * log_omx;
    // x₀ = a/(a + b) and 1 − x₀ as an exact complementary pair: 1 − (the smaller quotient)
    // rounded, the other its complement, exact by Sterbenz. a·log(x/x₀) + b·log((1 − x)/(1 − x₀))
    // then differs from its value at the exact a/(a + b) only at second order, but the dropped
    // linear term a·u + b·v is first order in that rounding:
    //   a·u + b·v = (x − x₀)·(a − (a + b)·x₀)/(x₀(1 − x₀)),
    // of size (x − x₀)·(a + b)²·ε/b, 1e-11 of the prefactor at (1e6, 30) near the branch point.
    // So it is added back, its numerator formed from a + b split exactly by TwoSum (no
    // products: contraction-safe) and one explicit fma. The second-order cost of the pair is
    // (a + b)²ε²/(4·min(a, b)), below ε/4 while (a + b)²·ε ≤ min(a, b); past that (shape ratios
    // beyond ~1e10, where the complement of the small quotient rounds to 1) the separately
    // rounded quotients below are kept, with their pre-existing first-order error.
    const double sum = a + b;
    const double small = std::min(a, b);
    double x0;
    double one_minus_x0;
    double linear = detail::ZERO_DOUBLE;  // a·u + b·v
    if (sum * sum * std::numeric_limits<double>::epsilon() <= small) {
        const double bv = sum - a;
        const double sum_err = (a - (sum - bv)) + (b - bv);  // a + b = sum + sum_err exactly
        if (a <= b) {
            one_minus_x0 = detail::ONE - a / sum;
            x0 = detail::ONE - one_minus_x0;
        } else {
            x0 = detail::ONE - b / sum;
            one_minus_x0 = detail::ONE - x0;
        }
        const double lin_num = std::fma(-sum, x0, a) - sum_err * x0;
        linear = (x - x0) * (lin_num / (x0 * one_minus_x0));
    } else {
        x0 = a / sum;
        one_minus_x0 = b / sum;
    }
    const double u = (x - x0) / x0;
    const double v = (x0 - x) / one_minus_x0;
    // log(1 + v) = log((1 − x)/(1 − x₀)) beyond the series, from log1p(−x): 1 − x near 1 as x
    // near 0 is exact, and near x = 1 the rounded v has lost 1 − x as u loses x near 0.
    const double lv = std::fabs(v) < LOG1PMX_SERIES_LIMIT
                          ? log1pmx_series(v)
                          : (std::log1p(-x) - std::log(one_minus_x0)) - v;
    return a * log1pmx_ratio(u, x, x0) + b * lv + linear + shape_constant;
}

double log_beta_prefactor(double x, double a, double b, double shape_constant) noexcept {
    return log_beta_prefactor_logs(x, a, b, std::log(x), std::log(detail::ONE - x), shape_constant);
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
double stirling_error(double m) noexcept {
    if (m >= kStirlingPrefactorShape)
        return stirling_remainder(m);
    return detail::lgamma(m + detail::ONE) -
           ((m + detail::HALF) * std::log(m) - m + detail::HALF * detail::LN_2PI);
}

// The deviance x·log(x/M) + M − x ≥ 0. Within M/2 of M it is M·[(1 + t)·log1pmx(t) + t²] with
// t = (x − M)/M: no cancellation below |t| = ¼, where log1pmx sums its series, and about 3 bits
// in log1pmx's direct difference from there to ½; farther out the direct form loses a few bits
// at most.
double deviance(double x, double m) noexcept {
    if (x == detail::ZERO_DOUBLE)
        return m;
    const double d = x - m;
    if (std::fabs(d) < detail::HALF * m) {
        const double t = d / m;
        return m * ((detail::ONE + t) * log1pmx(t) + t * t);
    }
    return x * std::log(x / m) + m - x;
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
    return {p, std::fma(a, b, -p)};
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

double binomial_log_pmf(double xa, double xb, double pa) noexcept {
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
        return detail::lgamma(n.hi + detail::ONE) - detail::lgamma(xa + detail::ONE) -
               detail::lgamma(xb + detail::ONE) + xa * std::log(pa) + xb * std::log1p(-pa);

    // The means Ma = n·pa and Mb = n − Ma as double-doubles, so Ma + Mb = xa + xb to ~ε²·n. Then
    //   log pmf = c(n) − c(xa) − c(xb) + ½·log(n/(2π·xa·xb)) − D(xa, Ma) − D(xb, Mb)
    //             + Σ lo·(x/hi − 1),
    // where the last sum carries the low parts of the means into the deviances' linear terms,
    // which cancel exactly when the means sum to n. Rounding the means instead costs n·ε.
    const DoubleDouble pa_n = two_prod(n.hi, pa);
    const DoubleDouble ma = two_sum(pa_n.hi, pa_n.lo + n.lo * pa);
    const DoubleDouble rest = two_sum(n.hi, -ma.hi);
    const DoubleDouble mb = two_sum(rest.hi, rest.lo + (n.lo - ma.lo));
    return stirling_error(n.hi) - stirling_error(xa) - stirling_error(xb) +
           detail::HALF * (std::log(n.hi / (xa * xb)) - detail::LN_2PI) - deviance(xa, ma.hi) -
           deviance(xb, mb.hi) + ma.lo * (xa / ma.hi - detail::ONE) +
           mb.lo * (xb / mb.hi - detail::ONE);
}

// =============================================================================
// SPECIAL MATHEMATICAL FUNCTIONS
// =============================================================================

double erf(double x) noexcept {
    // Use std::erf for now, replace with a custom implementation if needed
    return std::erf(x);
}

double erfc(double x) noexcept {
    return std::erfc(x);
}

double erf_inv(double x) noexcept {
    // Standard inverse error function using rational approximation
    // Based on Numerical Recipes and NIST algorithms

    if (x < detail::NEG_ONE || x > detail::ONE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (x == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (x >= detail::ONE)
        return std::numeric_limits<double>::infinity();
    if (x <= detail::NEG_ONE)
        return -std::numeric_limits<double>::infinity();

    // Use symmetry: erf_inv(-x) = -erf_inv(x)
    double sign = (x < detail::ZERO_DOUBLE) ? detail::NEG_ONE : detail::ONE;
    double a = std::abs(x);

    // Rational approximation constants (Moro's method) — central region
    static constexpr double a0 = 2.50662823884;
    static constexpr double a1 = -18.61500062529;
    static constexpr double a2 = 41.39119773534;
    static constexpr double a3 = -25.44106049637;

    static constexpr double b0 = -8.47351093090;
    static constexpr double b1 = 23.08336743743;
    static constexpr double b2 = -21.06224101826;
    static constexpr double b3 = 3.13082909833;

    // Acklam rational approximation coefficients shared by the moderate- and
    // extreme-tail branches.  Hoisted to eliminate copy-paste and ensure both
    // branches use identical values.
    static constexpr double ACKLAM_D0 = 2.515517;
    static constexpr double ACKLAM_D1 = 0.802853;
    static constexpr double ACKLAM_D2 = 0.010328;
    static constexpr double ACKLAM_E0 = 1.432788;
    static constexpr double ACKLAM_E1 = 0.189269;
    static constexpr double ACKLAM_E2 = 0.001308;

    double result;

    if (a <= detail::ERF_INV_CENTRAL_CUTOFF) {
        // Moro's rational approximation for Phi^{-1}(p) converted to erf_inv.
        //
        // Identity: erf_inv(a) = Phi^{-1}((a+1)/2) / sqrt(2).
        // Moro's formula is parameterised by y = p - 0.5 = a/2 (not by a).
        //
        // Bug that was here: used z = a*a instead of z = (a/2)*(a/2),
        // evaluating the polynomial at 4x the correct argument. For a~0.5
        // this produced ~2.8 instead of the true ~0.48, causing Halley's
        // method to diverge over ~48 consecutive grid points.
        double y = a * detail::HALF;  // y = a/2
        double z = y * y;             // z = y^2 as required by Moro
        result = y * (((a3 * z + a2) * z + a1) * z + a0) /
                 ((((b3 * z + b2) * z + b1) * z + b0) * z + detail::ONE) *
                 detail::INV_SQRT_2;  // Phi^{-1} / sqrt(2) = erf_inv
    } else if (a < detail::ERF_INV_TAIL_CUTOFF) {
        // Moderate tail region: use improved asymptotic expansion with better coefficients
        double z = std::sqrt(-std::log((detail::ONE - a) * detail::HALF));

        result = z - (ACKLAM_D0 + ACKLAM_D1 * z + ACKLAM_D2 * z * z) /
                         (detail::ONE + ACKLAM_E0 * z + ACKLAM_E1 * z * z + ACKLAM_E2 * z * z * z);
    } else {
        // Extreme tail region: use specialized asymptotic series
        // For erf(x) very close to 1, use high-precision asymptotic expansion
        double eps = detail::ONE - a;  // Small positive number

        if (eps < detail::ULTRA_HIGH_PRECISION_TOLERANCE) {
            // Ultra-extreme tail: use logarithmic asymptotic expansion
            double log_eps = std::log(eps);
            double sqrt_log_eps = std::sqrt(-log_eps);

            // Leading term from asymptotic series
            result = sqrt_log_eps;

            // Higher order corrections for better accuracy
            double correction = std::log(sqrt_log_eps * detail::SQRT_PI * detail::HALF) /
                                (detail::TWO * sqrt_log_eps);
            result -= correction;

            // Even higher order terms for extreme precision
            if (eps > 1e-15) {
                double log_correction = std::log(sqrt_log_eps * detail::SQRT_PI * detail::HALF);
                double second_order = (log_correction * log_correction - detail::TWO) /
                                      (8.0 * sqrt_log_eps * sqrt_log_eps * sqrt_log_eps);
                result += second_order;
            }
        } else {
            // Standard extreme tail: use refined asymptotic expansion
            double t = std::sqrt(-detail::TWO * std::log(eps));

            result =
                t - (ACKLAM_D0 + ACKLAM_D1 * t + ACKLAM_D2 * t * t) /
                        (detail::ONE + ACKLAM_E0 * t + ACKLAM_E1 * t * t + ACKLAM_E2 * t * t * t);

            // Additional correction term for better accuracy
            double correction = std::log(t * detail::SQRT_PI * detail::HALF) / (detail::TWO * t);
            result -=
                correction * detail::ERF_INV_HALLEY_DAMPING;  // Damped to avoid overcorrection
        }
    }

    // Eight iterations of Halley's method for refinement
    for (int i = 0; i < 8; ++i) {
        double erf_result = erf(result);
        double err = erf_result - a;

        if (std::abs(err) < detail::HIGH_PRECISION_TOLERANCE) {
            break;
        }

        // Halley's method: more stable than Newton-Raphson
        double exp_term = std::exp(-result * result);
        double f_prime = (detail::TWO / detail::SQRT_PI) * exp_term;
        double f_double_prime = -detail::TWO * result * f_prime;

        double denominator = f_prime - detail::HALF * err * f_double_prime / f_prime;
        if (std::abs(denominator) > detail::ZERO) {
            result -= err / denominator;
        }
    }

    return sign * result;
}

// log|Γ(x)| without the global signgam write of std::lgamma, which POSIX marks MT-unsafe and glibc
// documents as race:signgam (#173). lgamma_r is the same libm routine with the sign returned
// through its argument, so the result bits are std::lgamma's. MSVC's lgamma writes no signgam
// and has no lgamma_r.
double lgamma(double x) noexcept {
#if defined(_WIN32)
    return std::lgamma(x);
#else
    int sign = 0;
    return ::lgamma_r(x, &sign);
#endif
}

// Below this shape gamma_q forms Q directly where x ≤ a + 1 (gamma_q_small_shape); above it
// Q ≥ Q(a, a + 1) > 0.08 there, and 1 − P costs at most 4 bits.
constexpr double kSmallShapeQ = 0.5;

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

// Q(a, x) for 0 < a < kSmallShapeQ and 0 < x ≤ a + 1. There P = 1 − O(a) away from x = 0, so
// 1 − P lost about log₁₀(1/a) digits: 4e4·ε at a = 1e-4. From γ(a, x) = x^a·Σ_{n≥0}
// (−x)ⁿ/(n!·(a + n)) (A&S 6.5.29) and Γ(a) = Γ(1 + a)/a,
//   Q = [(Γ(1 + a) − 1) − (x^a − 1) − a·x^a·S] / Γ(1 + a),  S = Σ_{n≥1} (−x)ⁿ/(n!·(a + n)),
// with Γ(1 + a) − 1 and x^a − 1 by expm1, each O(a) like Q itself. x ≤ 1.5 keeps the alternating
// S to a few bits of cancellation.
static double gamma_q_small_shape(double a, double x) noexcept {
    const double lg = lgamma1p_small(a);
    const double xa_m1 = std::expm1(a * std::log(x));
    double s = detail::ZERO_DOUBLE;
    double term = detail::ONE;  // (−x)ⁿ/n!
    for (int n = 1; n < 100; ++n) {
        term *= -x / n;
        const double t = term / (a + n);
        s += t;
        if (std::fabs(t) <= std::numeric_limits<double>::epsilon() * std::fabs(s))
            break;
    }
    return (std::expm1(lg) - xa_m1 - a * (detail::ONE + xa_m1) * s) / std::exp(lg);
}

// ---------------------------------------------------------------------------------------------
// Large shape: Temme's uniform asymptotic expansion (#226). With λ = x/a and
// η = sign(λ − 1)·√(2(λ − 1 − log λ)) (DLMF 8.12.4, 8.12.8),
//   Q(a, x) = ½·erfc(η√(a/2)) + S,   P(a, x) = ½·erfc(−η√(a/2)) − S,
//   S = e^{−½aη²}/√(2πa) · Σ_{k≥0} c_k(η)·a^{−k},
// c_0(η) = 1/(λ − 1) − 1/η and c_k(η) = (1/η)·c_{k−1}'(η) + (−1)^k·g_k/(λ − 1) with g_k the
// Stirling coefficients (DLMF 8.12.12). Near η = 0 each c_k is a removable-singularity
// quotient, so they are evaluated from their Taylor series in η, derived exactly from that
// recursion (scratch script, mpmath, 60 digits; c_k(0) agree with DLMF 8.12.13: −1/3, −1/540,
// 25/6048, 101/155520). The series and continued fraction need O(√a) terms near the median —
// 9·√a to reach 3ε — which expansion_iteration_cap stops short of from a ≈ 1e10, and which costs
// 1e6 terms per call before that; the expansion costs 240 multiply-adds and its truncation error
// is below c_8(η)·a^{−8}. It serves |η| ≤ kTemmeEtaMax, where the Taylor polynomials hold to
// double; outside it λ is far enough from 1 that the series and continued fraction converge in
// under ~60 terms at any shape.
constexpr double kTemmeShape = 1e4;
constexpr double kTemmeEtaMax = 0.6;
constexpr int kTemmeTerms = 8;
constexpr int kTemmeDegree = 30;
static constexpr double kTemmeCoefficients[kTemmeTerms][kTemmeDegree] = {
    {// c_0
     -3.33333333333333315e-01, 8.33333333333333287e-02,  -1.48148148148148154e-02,
     1.15740740740740734e-03,  3.52733686067019424e-04,  -1.78755144032921798e-04,
     3.91926317852243767e-05,  -2.18544851067999198e-06, -1.85406221071515997e-06,
     8.29671134095308652e-07,  -1.76659527368260782e-07, 6.70785354340149841e-09,
     1.02618097842403086e-08,  -4.38203601845335294e-09, 9.14769958223679021e-10,
     -2.55141939949462482e-11, -5.83077213255042561e-11, 2.43619480206674150e-11,
     -5.02766928011417551e-12, 1.10043920319561348e-13,  3.37176326240098514e-13,
     -1.39238872241816207e-13, 2.85348938070474453e-14,  -5.13911183424257231e-16,
     -1.97522882943494422e-15, 8.09952115670456128e-16,  -1.65225312163981622e-16,
     2.53054300974788828e-18,  1.16869397385595764e-17,  -4.77003704982048474e-18},
    {// c_1
     -1.85185185185185192e-03, -3.47222222222222203e-03, 2.64550264550264536e-03,
     -9.90226337448559630e-04, 2.05761316872427979e-04,  -4.01877572016460897e-07,
     -1.80985503344899767e-05, 7.64916091608110982e-06,  -1.61209008945634465e-06,
     4.64712780280743402e-09,  1.37863344691572092e-07,  -5.75254560351770471e-08,
     1.19516285997781477e-08,  -1.75432417197476467e-11, -1.00915437106004126e-09,
     4.16279299184258280e-10,  -8.56390702649298013e-11, 6.06721510160475823e-14,
     7.16249896481148557e-12,  -2.93318664377143705e-12, 5.99669636568368853e-13,
     -2.16717865273233131e-16, -4.97833997236926173e-14, 2.02916288237134252e-14,
     -4.13125571381060994e-15, 8.28651623988309668e-19,  3.41003088693333267e-16,
     -1.38541953028939715e-16, 2.81234665322887471e-17,  -3.40644419414302878e-21},
    {// c_2
     4.13359788359788337e-03,  -2.68132716049382727e-03, 7.71604938271604895e-04,
     2.00938786008230470e-06,  -1.07366532263651599e-04, 5.29234488291201250e-05,
     -1.27606351886187284e-05, 3.42357873409613781e-08,  1.37219573090629342e-06,
     -6.29899213838005482e-07, 1.42806142060642425e-07,  -2.04770984219908661e-10,
     -1.40925299108675203e-08, 6.22897408492202184e-09,  -1.36704883966171141e-09,
     9.42835615901467795e-13,  1.28722524000893180e-10,  -5.56459561343633233e-11,
     1.19759355463669806e-11,  -4.16897822518386344e-15, -1.09406404278845948e-12,
     4.66223994639013565e-13,  -9.90510576390690656e-14, 1.89318767683735153e-17,
     8.85922187259112653e-15,  -3.73782039804640529e-15, 7.86883363903515551e-16,
     -9.00002739574121085e-20, -6.92888122934767126e-17, 2.90203842701647858e-17},
    {// c_3
     6.49434156378600773e-04,  2.29472093621399168e-04,  -4.69189494395255702e-04,
     2.67720632062838854e-04,  -7.56180167188397662e-05, -2.39650511386729680e-07,
     1.10826541153473025e-05,  -5.67495282699159655e-06, 1.42309007324358833e-06,
     -2.78610802915281434e-11, -1.69584040919302782e-07, 8.09946490538808268e-08,
     -1.91111684859736545e-08, 2.39286204398081180e-12,  2.06201318154887967e-09,
     -9.46049666185513302e-10, 2.15410497757749067e-10,  -1.38882333681390304e-14,
     -2.18947616819639379e-11, 9.79099895117168436e-12,  -2.17821918801809609e-12,
     6.20881957340790081e-17,  2.12697836327973708e-13,  -9.34468879151743301e-14,
     2.04536712267828492e-14,  -2.58260790403495020e-19, -1.94052976733445443e-15,
     8.41597929048481583e-16,  -1.82004304395382256e-16, 1.07354436412473090e-21},
    {// c_4
     -8.61888290916711726e-04, 7.84039221720066615e-04,  -2.99072480303190177e-04,
     -1.46384525788434181e-06, 6.64149821546512189e-05,  -3.96836504717943471e-05,
     1.13757269706784187e-05,  2.50749722623753294e-10,  -1.69541495365583054e-06,
     8.90750753220530941e-07,  -2.29293483400080494e-07, 2.95679413754404924e-11,
     2.88658297427087831e-08,  -1.41897394378032191e-08, 3.44635804994648956e-09,
     -2.30245171745280665e-13, -3.94092330280464033e-10, 1.86023389685045010e-10,
     -4.35632300505661772e-11, 1.27860010162962303e-15,  4.67927502665791974e-12,
     -2.14924647061348296e-12, 4.90881561480965202e-13,  -6.33859148489156013e-18,
     -5.04533206908009422e-14, 2.27229582229012859e-14,  -5.09608260847240171e-15,
     3.05520975571713547e-20,  5.06902167631055156e-16,  -2.24938369564818088e-16},
    {// c_5
     -3.36798553366358131e-04, -6.97281375836585711e-05, 2.77275324495939183e-04,
     -1.99325705161888469e-04, 6.79778047793720800e-05,  1.41906292064396713e-07,
     -1.35940481897686926e-05, 8.01847025633420200e-06,  -2.29148117650809516e-06,
     -3.25247355129845377e-10, 3.46528464910852651e-07,  -1.84471871911713436e-07,
     4.82409670378941838e-08,  -1.79894667217435142e-14, -6.30619450001352306e-09,
     3.16241762877456782e-09,  -7.84092425369742885e-10, 5.19267916525404078e-15,
     9.35894424230678423e-11,  -4.51342621616327799e-11, 1.07991299931168276e-11,
     -3.66188671268525198e-17, -1.21090206905515493e-12, 5.68074358499056438e-13,
     -1.32496599163408287e-13, 1.89872407642840755e-19,  1.41933902367947012e-14,
     -6.52321470142469669e-15, 1.49252426362028845e-15,  -8.80038945873236903e-22},
    {// c_6
     5.31307936463992249e-04,  -5.92166437353693932e-04, 2.70878209671804500e-04,
     7.90235323266032815e-07,  -8.15396936756196915e-05, 5.61168275310624970e-05,
     -1.83291165828433752e-05, -3.07961345060330474e-09, 3.46515536880360913e-06,
     -2.02913273960586027e-06, 5.78879286314900390e-07,  2.33863067382665681e-13,
     -8.82860074633048400e-08, 4.74359588804081251e-08,  -1.25454150207103832e-08,
     8.64964885801029260e-14,  1.68460589792640624e-09,  -8.57549282357759428e-10,
     2.15982249292321247e-10,  -7.61323052047615345e-16, -2.66398220085361439e-11,
     1.30657005366110570e-11,  -3.17991639023679772e-12, 4.71097612136743122e-18,
     3.69028008427634655e-13,  -1.76126740462014258e-13, 4.17906678605147798e-14,
     -2.53446793791788044e-20, -4.63206594200160467e-15, 2.16514548596464289e-15},
    {// c_7
     3.44367606892377652e-04,  5.17179090826059187e-05,  -3.34931610811422338e-04,
     2.81269515476323688e-04,  -1.09765822446847311e-04, -1.27410090954844846e-07,
     2.77444515115636454e-05,  -1.82634888057113320e-05, 5.78769494973505252e-06,
     4.93875893393627006e-10,  -1.05953670140260431e-06, 6.16671437611040781e-07,
     -1.75629733590604631e-07, -1.29744732870154394e-12, 2.69542360628896587e-08,
     -1.45783529087312718e-08, 3.88764595938617502e-09,  -3.88100225101941210e-17,
     -5.32799417387728638e-10, 2.74379776433148436e-10,  -6.99579609207056804e-11,
     2.58998638748684806e-17,  8.85668909966963887e-12,  -4.40316881587131093e-12,
     1.08655619470916539e-12,  -2.04679884474166783e-19, -1.29697944216929387e-13,
     6.27892205914772839e-14,  -1.51129483716793956e-14, 4.02469681070584754e-18}};

// Σ_{k<kTemmeTerms} c_k(η)·a^{−k}, each c_k by Horner on its Taylor polynomial.
static double temme_sum(double a, double eta) noexcept {
    const double inv_a = detail::ONE / a;
    double sum = detail::ZERO_DOUBLE;
    double power = detail::ONE;
    for (int k = 0; k < kTemmeTerms; ++k) {
        const double* c = kTemmeCoefficients[k];
        double ck = c[kTemmeDegree - 1];
        for (int n = kTemmeDegree - 2; n >= 0; --n)
            ck = ck * eta + c[n];
        sum += ck * power;
        power *= inv_a;
        if (power < 1e-20)
            break;
    }
    return sum;
}

// The expansion's variables for (a, x), or false outside its zone. z2 = ½aη² is the Gaussian
// exponent, −a·(log1p(μ) − μ) with μ = (x − a)/a, formed as log_gamma_prefactor forms it.
static bool temme_zone(double a, double x, double& eta, double& z2) noexcept {
    if (a < kTemmeShape)
        return false;
    const double mu = (x - a) / a;
    if (std::fabs(mu) > detail::ONE)
        return false;
    z2 = -a * log1pmx_ratio(mu, x, a);
    eta = std::copysign(std::sqrt(detail::TWO * z2 / a), mu);
    return std::fabs(eta) <= kTemmeEtaMax;
}

// erfcx(z) = e^{z²}·erfc(z) for z ≥ 26, from the asymptotic series
// (1/(z√π))·Σ_k (−1)^k·(2k − 1)!!/(2z²)^k: the eighth term is below 2e-18 there.
static double erfcx_asymptotic(double z) noexcept {
    const double r = detail::HALF / (z * z);
    double term = detail::ONE;
    double sum = detail::ONE;
    for (int k = 1; k < 10; ++k) {
        term *= -r * static_cast<double>(2 * k - 1);
        sum += term;
        if (std::fabs(term) < std::numeric_limits<double>::epsilon() * sum)
            break;
    }
    return sum / (z * detail::SQRT_PI);
}

// Q(a, x) (upper) or P(a, x) by the expansion.
static double gamma_tail_temme(double a, double eta, double z2, bool upper) noexcept {
    const double z = std::copysign(std::sqrt(z2), eta);
    const double zs = upper ? z : -z;  // the erfc argument: positive on the small side
    const double s = temme_sum(a, eta) * detail::INV_SQRT_2PI / std::sqrt(a);
    const double correction = std::exp(-z2) * (upper ? s : -s);
    const double tail = detail::HALF * std::erfc(zs) + correction;
    return std::min(detail::ONE, std::max(detail::ZERO_DOUBLE, tail));
}

// log of the same, finite where the tail underflows: e^{−z²}·(½·erfcx(zs) ± Σ/√(2πa)) once
// erfc(zs) is within a few bits of underflow.
static double log_gamma_tail_temme(double a, double eta, double z2, bool upper) noexcept {
    const double z = std::copysign(std::sqrt(z2), eta);
    const double zs = upper ? z : -z;
    const double s = temme_sum(a, eta) * detail::INV_SQRT_2PI / std::sqrt(a);
    const double signed_s = upper ? s : -s;
    if (zs < 26.0)
        return std::log(detail::HALF * std::erfc(zs) + std::exp(-z2) * signed_s);
    return -z2 + std::log(detail::HALF * erfcx_asymptotic(zs) + signed_s);
}

// The series Σ_{n≥0} xⁿ/(a(a + 1)…(a + n)) of P(a, x) = x^a·e^{−x}/Γ(a) · Σ (A&S 6.5.29),
// for x ≤ a + 1. At least 1/a, so its log never underflows.
static double gamma_p_series_sum(double a, double x) noexcept {
    double ap = a;
    double sum = detail::ONE / ap;
    double term = sum;
    const double tolerance = detail::SPECIAL_FUNCTION_TOLERANCE;
    const int max_iterations = expansion_iteration_cap(a);
    for (int n = 1; n < max_iterations; ++n) {
        ap += detail::ONE;
        term *= x / ap;
        sum += term;
        if (std::abs(term) < tolerance * std::abs(sum))
            break;
    }
    return sum;
}

// The continued fraction h of Q(a, x) = x^a·e^{−x}/Γ(a) · h (modified Lentz), for x > a + 1.
// Guard b before dividing: when x = a-1 exactly, b = 0 and d would be ±inf before the
// abs(d)<ZERO clamp inside the loop executes. Mirror the pattern used in
// beta_continued_fraction.
static double gamma_q_continued_fraction(double a, double x) noexcept {
    double b = x + detail::ONE - a;
    if (std::abs(b) < detail::ZERO)
        b = detail::ZERO;
    double c = detail::LARGE_CONTINUED_FRACTION_VALUE;
    double d = detail::ONE / b;
    double h = d;

    const int max_iterations = expansion_iteration_cap(a);
    const double tolerance = detail::SPECIAL_FUNCTION_TOLERANCE;

    for (int i = 1; i <= max_iterations; ++i) {
        double an = -i * (i - a);
        b += detail::TWO;
        d = an * d + b;
        if (std::abs(d) < detail::ZERO) {
            d = detail::ZERO;
        }
        c = b + an / c;
        if (std::abs(c) < detail::ZERO) {
            c = detail::ZERO;
        }
        d = detail::ONE / d;
        double del = d * c;
        h *= del;
        if (std::abs(del - detail::ONE) < tolerance) {
            break;
        }
    }
    return h;
}

double gamma_p(double a, double x) noexcept {
    // Regularized incomplete gamma function P(a,x) = γ(a,x) / Γ(a)
    // where γ(a,x) is the lower incomplete gamma function
    if (std::isnan(a) || std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE || a <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (x == std::numeric_limits<double>::infinity())
        return detail::ONE;  // a caller's x·rate that overflowed (#224)

    double eta, z2;
    if (temme_zone(a, x, eta, z2))
        return gamma_tail_temme(a, eta, z2, false);

    if (x > a + detail::ONE) {
        // For large x, use the complementary function for better convergence
        return detail::ONE - gamma_q(a, x);
    }

    // Use the dedicated series function that has the correct formula
    return gamma_p_series(a, x);
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
    // Regularized complementary incomplete gamma function using continued fraction
    // Q(a,x) = 1 - P(a,x) but for large x, use continued fraction for better convergence
    if (std::isnan(a) || std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE || a <= detail::ZERO_DOUBLE) {
        return detail::ONE;
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ONE;
    }
    if (x == std::numeric_limits<double>::infinity())
        return detail::ZERO_DOUBLE;  // #224

    double eta, z2;
    if (temme_zone(a, x, eta, z2))
        return gamma_tail_temme(a, eta, z2, true);

    if (x <= a + detail::ONE) {
        // For small x, use the series expansion of P(a,x) and compute 1-P; for small a, Q
        // directly, which 1 − P = 1 − (1 − O(a)) cancels.
        if (a < kSmallShapeQ)
            return gamma_q_small_shape(a, x);
        return detail::ONE - gamma_p_series(a, x);
    }

    // For large x, use continued fraction expansion for Q(a,x)
    return std::exp(log_gamma_prefactor(a, x)) * gamma_q_continued_fraction(a, x);
}

// log P(a, x) (or log Q(a, x) with `upper`) for x > 0, finite wherever the exact value is
// nonzero: the prefactor stays in the log and only the series sum or continued fraction is
// formed (#214, #223). `slope` receives x·pdf(x)/tail = prefactor/tail, the quantile residual's
// |d/dt|, from the sum or fraction directly: as exp(log prefactor − log tail) it is garbage where
// both logs are ~−1e17 and the difference is a few units (an iterate at x = 4e17 saw a slope of
// 6e27 for 4e17).
static double log_gamma_tail(double a, double x, bool upper, double& slope) noexcept {
    double eta, z2;
    if (temme_zone(a, x, eta, z2)) {
        const double log_tail = log_gamma_tail_temme(a, eta, z2, upper);
        slope = std::exp(log_gamma_prefactor(a, x) - log_tail);
        return log_tail;
    }
    if (x <= a + detail::ONE) {
        if (!upper) {
            const double sum = gamma_p_series_sum(a, x);
            slope = detail::ONE / sum;
            return log_gamma_prefactor(a, x) + std::log(sum);
        }
        if (a < kSmallShapeQ) {
            const double q = gamma_q_small_shape(a, x);
            slope = std::exp(log_gamma_prefactor(a, x)) / q;
            return std::log(q);
        }
        // Q = 1 − P with P = prefactor·sum: slope = prefactor/Q = P/(sum·Q).
        const double sum = gamma_p_series_sum(a, x);
        const double p = std::exp(log_gamma_prefactor(a, x)) * sum;
        slope = p / (sum * (detail::ONE - p));
        return std::log1p(-p);
    }
    const double h = gamma_q_continued_fraction(a, x);
    if (upper) {
        slope = detail::ONE / h;
        return log_gamma_prefactor(a, x) + std::log(h);
    }
    // P = 1 − Q with Q = prefactor·h: slope = prefactor/P = Q/(h·P).
    const double q = std::exp(log_gamma_prefactor(a, x)) * h;
    slope = q / (h * (detail::ONE - q));
    return std::log1p(-q);
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

// Below this shape the Wilson-Hilferty approximation x ≈ a·(1 − h + z·√h)³, h = 1/(9a), is the
// solver's seed; from it, the answer (#226). Its relative error scales as |z|³·a^{−3/2}: 2e-24
// at a = 1e16 for |z| ≤ 6.4 (mpmath), within the accuracy law from p = 5e-324 to 1 − 1e-15
// (test_gamma_quantile_accuracy, LargeShape). From ~1e32 the central quantiles round to a
// itself and F resolves nothing finer: adjacent doubles differ in F by ~ulp(a)/√(2πa).
constexpr double kWilsonHilfertyExactShape = 1e16;

// x with P(a, x) = p (#160), by Newton in t = log x on the small side of the probability scale:
// f(t) = log P(a, eᵗ) − log p below the median, log Q(a, eᵗ) − log q above it. The density of
// log X, exp(a·t − eᵗ)/Γ(a), is log-concave, so both residuals are concave in t and
// solve_concave applies. |f'(t)| = x·pdf(x)/P is the prefactor over P (or Q). The residual is
// log_gamma_tail, finite wherever the tail is nonzero, so Newton sees the true f on both sides
// of the root instead of −inf where the tail underflows (#214, #223).
double gamma_p_inv(double a, double p) noexcept {
    if (std::isnan(a) || std::isnan(p) || a <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    if (p <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p >= detail::ONE)
        return std::numeric_limits<double>::infinity();

    // Wilson-Hilferty, x ≈ a·(1 − h + z·√h)³ with h = 1/(9a).
    const double h = detail::ONE / (detail::NINE * a);
    const double c = detail::ONE - h + inverse_normal_cdf(p) * std::sqrt(h);
    if (a >= kWilsonHilfertyExactShape)
        return a * c * c * c;  // c > 0: |z|·√h < 2e-7 there

    const bool upper = p > detail::HALF;
    const double log_target = std::log(upper ? detail::ONE - p : p);  // 1 − p exact for p ≥ ½

    // Bracket in t: e^-745 is below the smallest subnormal, e^709.78 near the largest double.
    double lo = -745.0;
    double hi = 709.78;
    if (!upper) {
        // P(a, x) ≤ x^a/Γ(a + 1), so the root is at or above that bound's root. Below t = −700
        // the bound equals P to a relative a·x/(a + 1) < 1e-304, so it is the answer — and the
        // only route to it when x underflows.
        const double t_bound = (log_target + detail::lgamma(a + detail::ONE)) / a;
        if (t_bound < -700.0)
            return std::exp(t_bound);
        lo = t_bound;
        hi = std::log(a);  // P(a, a) > ½: the median is below the mean
    }

    // The seed; the lower bound where Wilson-Hilferty fails (c ≤ 0 at small shape).
    double t = c > detail::ZERO_DOUBLE ? std::log(a) + detail::THREE * std::log(c) : lo;
    if (!(t > lo && t < hi))
        t = upper ? detail::HALF * (lo + hi) : lo;

    double step = detail::ZERO_DOUBLE;
    double slope = detail::ZERO_DOUBLE;
    t = solve_concave(
        t, lo, hi, !upper,
        [&](double tt, double& s) {
            return log_gamma_tail(a, std::exp(tt), upper, s) - log_target;
        },
        &step, true, &slope);
    // The final correction, applied to x: step is below 2ε·|t|, so exp(t)·(1 + step) carries it
    // at the resolution of x, where t + step would lose it to the rounding of t (500 ulps of x at
    // t = 690). Its own error is the residual's, ~16ε, over the slope; it is applied where that
    // is below the ε·|t| the rounding of t costs — at small shape in the deep lower tail the
    // slope is a, both are law-sized, and exp(t + step) keeps its pre-#226 bits.
    if (std::abs(t) * slope < 16.0)
        return std::exp(t + step);
    return std::exp(t) * (detail::ONE + step);
}

// Below this shape I_x is formed as x^a (1 − x)^b / (a·B(a, b)) · h with the 1/a folded into the
// log prefactor. The direct form exp(−lbeta(a, b)) · h/a cancels lgamma(a) ≈ −log a against the
// 1/a: |log a|·ε relative, 1.5e-13 at a = 1e-300, which put I_0.01(1e-300, 1) at 1 + 4.7e-15 and
// I_x(1e-300, 1e-300) on the wrong side of ½ at both ends (#229). At 1e-3 the cancellation is
// 7ε, within the continued fraction's own few ε, so the direct form stays above it (and the
// sweep with it); below, the error grows without bound.
constexpr double kTinyBetaShape = 1e-3;

// log[x^a (1 − x)^b / (a·B(a, b))] with every lgamma at an argument ≥ 1:
//   a·B(a, b) = Γ(a + 1)Γ(b)/Γ(a + b) = ((a + b)/b) · (a + b + 1) · B(a + 1, b + 1),
// and lbeta carries the Stirling form for a large shape — the other shape may be anything up
// to DBL_MAX (a NegativeBinomial CDF at k = DBL_MAX), where lgamma(b + 1) alone is ∞.
// omx is 1 − x, used for the (1 − x)^b factor from x ≥ ½, where the caller may hold it more
// exactly than 1 − x (the quantile solver's logit: x rounds to 1 while 1 − x is e^-661, and
// log1p(−x) = −∞ there zeroed the Newton slope).
static double log_beta_prefactor_over_a(double x, double omx, double a, double b) noexcept {
    return log_beta_prefactor_over_a_logs(std::log(x),
                                          x < detail::HALF ? std::log1p(-x) : std::log(omx), a, b);
}

// The same from log x and log(1 − x) held by the caller: the quantile solver's exact logit
// logs, and the tail integral's branch point x_b = 1 − s₀, whose log x is log1p(−s₀) — the
// integral runs from s₀ exactly, and the rounded x_b sits up to ε from 1 − s₀, which a shape
// of 1e6 turned into 1e-10 of the prefactor. The continued fraction at x_b takes y = s₀ for the
// same reason (see beta_continued_fraction_unscaled).
static double log_beta_prefactor_over_a_logs(double log_x, double log_omx, double a,
                                             double b) noexcept {
    return a * log_x + b * log_omx + std::log(b / (a + b)) - std::log(a + b + detail::ONE) -
           lbeta(a + detail::ONE, b + detail::ONE);
}

// I_x(a, b) below the branch point x_b = (a + 1)/(a + b + 2): the direct orientation, with the
// same underflow rule as the direct form (a prefactor of 0 decides the result without the
// continued fraction, whose products overflow at a shape past ~1e154), and the product clamped
// to 1, where a few ε of rounding could otherwise land above it.
static double beta_i_tiny_direct(double x, double omx, double a, double b) noexcept {
    const double pf = std::exp(log_beta_prefactor_over_a(x, omx, a, b));
    if (pf == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    return std::min(pf * beta_continued_fraction_unscaled(x, omx, a, b), detail::ONE);
}

// I_x(a, b) for min(a, b) < kTinyBetaShape, x in (0, 1).
//
// Below x_b the direct orientation. Above it, with a the tiny shape, I_x is 1 − (something
// small) and the swapped orientation 1 − I_{1−x}(b, a) carries it. With b the tiny shape I_x is
// itself small — b·∫₀ˣ t^(a−1)/(1 − t) dt, 1e-302 at Beta(1, 1e-300) and x = 0.01 — and the
// swapped orientation returns 1 − (1 − small) = 0 above x_b (a decrease from the direct value
// below it), while the direct continued fraction converges like ((1 − √s)/(1 + √s))ᵐ, s = 1 − x,
// too slowly near 1. So the tail is integrated from the branch point instead:
//   I_x = I_{x_b} + (1/B(a, b)) ∫_s^{s₀} (1 − u)^(a−1) u^(b−1) du,   s₀ = 1 − x_b,
//       = I_{x_b} + (1/B(a, b)) Σ_{n≥0} C(a−1, n)(−1)ⁿ (s₀^(n+b) − s^(n+b))/(n + b),
// a·s₀ = a(b + 1)/(a + b + 2) < 1, so the terms fall at least like (a·s₀)ⁿ/n! for a > 1 and
// like s₀ⁿ ≤ 2⁻ⁿ for a ≤ 1 (every coefficient then positive: no cancellation). The n = 0 term's
// (s₀^b − s^b)/b is expm1(b·log(s/s₀))·s₀^b/(−b), finite as b → 0, and 1/B(a, b) is
// b·(a/(a + b))/((a + b + 1)·B(a + 1, b + 1)) as in log_beta_prefactor_over_a. The sum is
// non-negative and increasing in x, so I_x stays ≥ I_{x_b} and non-decreasing.
//
// omx is 1 − x: exact from x for x ≥ x_b ≥ ⅓ (Sterbenz), but the quantile solver holds x and
// 1 − x separately from the logit and passes its own, since 1 − x from an x next to 1 has lost
// the digits the tail integral reads (Beta(1e-16, 1).Q(1 − 6.6e-14) needs 1 − y = e^-661).
static double beta_i_tiny_shape(double x, double omx, double a, double b) noexcept {
    const double x_b = (a + detail::ONE) / (a + b + detail::TWO);
    if (x < x_b)
        return beta_i_tiny_direct(x, omx, a, b);
    if (b >= kTinyBetaShape) {
        const double pf = std::exp(log_beta_prefactor_over_a(omx, x, b, a));
        if (pf == detail::ZERO_DOUBLE)
            return detail::ONE;
        return detail::ONE -
               std::min(pf * beta_continued_fraction_unscaled(omx, x, b, a), detail::ONE);
    }
    const double s0 = (b + detail::ONE) / (a + b + detail::TWO);  // 1 − x_b, formed directly
    const double s = omx;
    if (s >= s0)
        return beta_i_tiny_direct(x, omx, a, b);  // x_b rounded past x
    const double log_rho = std::log(s / s0);
    double q = detail::ONE;  // C(a − 1, n)·(−s₀)ⁿ
    double sum = -std::expm1(b * log_rho) / b;
    for (int n = 1; n < 4000; ++n) {
        q *= (static_cast<double>(n) - a) * s0 / static_cast<double>(n);
        const double t = q * (-std::expm1((static_cast<double>(n) + b) * log_rho)) /
                         (static_cast<double>(n) + b);
        sum += t;
        if (std::fabs(t) <= 1e-17 * std::fabs(sum))
            break;
    }
    const double log_s0 = std::log(s0);
    const double scale = b * (a / (a + b)) *
                         std::exp(b * log_s0 - lbeta(a + detail::ONE, b + detail::ONE) -
                                  std::log(a + b + detail::ONE));
    // I_{x_b} as beta_i_tiny_direct forms it, with log x_b = log1p(−s₀) rather than the log of
    // the rounded x_b (see log_beta_prefactor_over_a_logs).
    const double pf_b = std::exp(log_beta_prefactor_over_a_logs(std::log1p(-s0), log_s0, a, b));
    const double i_b =
        pf_b == detail::ZERO_DOUBLE
            ? detail::ZERO_DOUBLE
            : std::min(pf_b * beta_continued_fraction_unscaled(x_b, s0, a, b), detail::ONE);
    return std::min(i_b + scale * sum, detail::ONE);
}

double beta_i(double x, double a, double b) noexcept {
    // Regularized incomplete beta function I_x(a,b)
    if (std::isnan(x) || std::isnan(a) || std::isnan(b))
        return std::numeric_limits<double>::quiet_NaN();
    if (x < detail::ZERO_DOUBLE || x > detail::ONE || a <= detail::ZERO_DOUBLE ||
        b <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (x == detail::ONE) {
        return detail::ONE;
    }

    // I_½(a, a) = ½ by symmetry; the continued fraction lands an ulp off it, which moved a
    // discrete quantile at that exact tie (NegativeBinomial(5, ½) at p = ½ gave 5, not 4).
    if (a == b && x == detail::HALF)
        return detail::HALF;

    if (std::min(a, b) < kTinyBetaShape)
        return beta_i_tiny_shape(x, detail::ONE - x, a, b);

    // Use continued fraction approximation
    double bt = std::exp(log_beta_prefactor(x, a, b, beta_prefactor_constant(a, b)));
    // An underflowed prefactor decides the result without the continued fraction, which at a
    // shape past ~1e154 overflows its products to inf/inf = NaN (NegativeBinomial cdf(1e300)).
    // bt·cf is then below 2^-1074·cf, so the small side is 0 and its complement 1.
    if (bt == detail::ZERO_DOUBLE)
        return x < (a + detail::ONE) / (a + b + detail::TWO) ? detail::ZERO_DOUBLE : detail::ONE;

    if (x < (a + detail::ONE) / (a + b + detail::TWO)) {
        return bt * beta_continued_fraction(x, detail::ONE - x, a, b);
    } else {
        return detail::ONE - bt * beta_continued_fraction(detail::ONE - x, x, b, a);
    }
}

double beta_i(double x, double a, double b, double log_beta_prefix) noexcept {
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
    if (a == b && x == detail::HALF)
        return detail::HALF;  // by symmetry; see the overload above

    if (std::min(a, b) < kTinyBetaShape)
        return beta_i_tiny_shape(x, detail::ONE - x, a, b);  // the hoisted prefix cancels

    // log_beta_prefix is the caller's hoisted beta_prefactor_constant(a, b), in the direct or the
    // Stirling form as the shapes select (#166).
    double bt = std::exp(log_beta_prefactor(x, a, b, log_beta_prefix));
    if (bt == detail::ZERO_DOUBLE)  // as in the overload above
        return x < (a + detail::ONE) / (a + b + detail::TWO) ? detail::ZERO_DOUBLE : detail::ONE;

    if (x < (a + detail::ONE) / (a + b + detail::TWO)) {
        return bt * beta_continued_fraction(x, detail::ONE - x, a, b);
    } else {
        return detail::ONE - bt * beta_continued_fraction(detail::ONE - x, x, b, a);
    }
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
        lambda = std::fma(-s, x, a) - e * x;
    else
        lambda = std::fma(s, y, -b) + e * y;

    double f = (detail::ONE + lambda) / (a + detail::ONE);  // B₀ = 1 + d₁
    if (std::abs(f) < detail::ZERO)
        f = detail::ZERO;
    double c = f;
    double d = detail::ZERO_DOUBLE;
    double d_odd = -(s * x) / (a + detail::ONE);  // d₂ₘ₋₁, starting at d₁

    for (int m = 1; m <= max_iterations; ++m) {
        const double md = static_cast<double>(m);
        const double apm = a + md;
        const double a2m = a + detail::TWO * md;
        const double a2m1 = a2m + detail::ONE;
        const double d_even = md * (b - md) * x / ((a2m - detail::ONE) * a2m);  // d₂ₘ
        const double am = -d_odd * d_even;
        const double num =
            apm * lambda + (a + detail::TWO * md + a * md * (3.0 - x) + md * md * (4.0 - x));
        const double bm = num / (a2m * a2m1) + d_even;

        d = bm + am * d;
        if (std::abs(d) < detail::ZERO)
            d = detail::ZERO;
        c = bm + am / c;
        if (std::abs(c) < detail::ZERO)
            c = detail::ZERO;
        d = detail::ONE / d;
        const double delta = c * d;
        f *= delta;

        d_odd = -apm * (s + md) * x / (a2m * a2m1);  // d₂ₘ₊₁, the next step's d₂ₘ₋₁
        if (std::abs(delta - detail::ONE) < tolerance)
            break;
    }

    return detail::ONE / f;
}

static double gamma_p_series(double a, double x) noexcept {
    // P(a, x) = exp(-x + a*ln(x) - lgamma(a)) * sum, sum from gamma_p_series_sum
    if (x == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    const double result = std::exp(log_gamma_prefactor(a, x)) * gamma_p_series_sum(a, x);
    return std::min(detail::ONE, std::max(detail::ZERO_DOUBLE, result));  // Clamp to [0,1]
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
    const double x = std::max(a, b);
    const double y = std::min(a, b);
    if (x < kStirlingPrefactorShape || y <= detail::ZERO_DOUBLE)
        return detail::lgamma(a) + detail::lgamma(b) - detail::lgamma(a + b);
    const double s = x + y;
    return detail::lgamma(y) - (x - detail::HALF) * std::log1p(y / x) - y * std::log(s) + y +
           stirling_remainder(x) - stirling_remainder(s);
}

double digamma(double x) noexcept {
    // Digamma ψ(x) = d/dx lnΓ(x)
    // Recurrence ψ(x+1) = ψ(x) + 1/x shifts x > 6 for the asymptotic series.
    if (x <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double result = detail::ZERO_DOUBLE;
    while (x < 6.0) {
        result -= detail::ONE / x;
        x += detail::ONE;
    }

    // Asymptotic expansion: ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)
    const double inv_x = detail::ONE / x;
    const double inv_x2 = inv_x * inv_x;
    result += std::log(x) - detail::HALF * inv_x -
              inv_x2 * (detail::ONE / 12.0 - inv_x2 * (detail::ONE / 120.0 - inv_x2 / 252.0));
    return result;
}

double trigamma(double x) noexcept {
    // Trigamma ψ'(x) = d²/dx² lnΓ(x)  (A&S §6.4.12)
    // Recurrence ψ'(x) = ψ'(x+1) + 1/x² shifts x ≥ 6 for the asymptotic series.
    if (x <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double result = detail::ZERO_DOUBLE;
    while (x < 6.0) {
        result += detail::ONE / (x * x);
        x += detail::ONE;
    }
    const double r = detail::ONE / x;
    const double r2 = r * r;
    // Asymptotic series: 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷) - 1/(30x⁹)
    result += r * (detail::ONE + detail::HALF * r +
                   r2 * (detail::ONE / 6.0 -
                         r2 * (detail::ONE / 30.0 - r2 * (detail::ONE / 42.0 - r2 / 30.0))));
    return result;
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
    if (std::isnan(p) || std::isnan(a) || std::isnan(b) || a <= detail::ZERO_DOUBLE ||
        b <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    if (p <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p >= detail::ONE)
        return detail::ONE;
    if (a == b && p == detail::HALF)
        return detail::HALF;  // by symmetry; beta_i returns ½ at that point too

    const bool upper = p > detail::HALF;
    if (upper) {
        std::swap(a, b);
        p = detail::ONE - p;
    }
    const double log_target = std::log(p);
    const double lb = lbeta(a, b);
    const double pc = beta_prefactor_constant(a, b);
    // Below kTinyBetaShape the residual and the leading-term constant are formed as
    // beta_i_tiny_shape forms I (#229, #230): log(a·B(a, b)) = lbeta(a + 1, b) + log(a + b) with
    // no lgamma(a) against log a, and above the branch point log I from the tail integral
    // rather than log1p(−c) with c = 1 − I, which rounds to 1 when the second shape is tiny —
    // Beta(1, 1e-300).Q(3.7e-299) was 0.53 (true 1 − x = e^-36.8), and a seed landing in that
    // branch left Beta(1, 1e-30).Q(2.3e-34) at 1 − 5.5e-13 (true x = 2.3e-4).
    const bool tiny = std::min(a, b) < kTinyBetaShape;

    // I_x(a, b) = x^a/(a·B)·(1 + a(1 − b)x/(a + 1) + …): the leading term's root t_lb in
    // t = log x bounds the answer from below for b ≥ 1 (then (1 − s)^{b−1} ≤ 1 under the
    // integral) and from above for b < 1. Where x·max(1, b) < 1e-20 the correction is below
    // 1e-20 relative and the leading term is the answer — the only route to it once x cannot
    // hold the logit (below t = −700) or at all (the root underflows to 0 or a subnormal).
    const double t_lb = tiny ? (log_target + lbeta(a + detail::ONE, b) + std::log(a + b)) / a
                             : (log_target + std::log(a) + lb) / a;
    if (t_lb < -700.0 && t_lb + std::log(std::max(detail::ONE, b)) < -46.0)
        return upper ? detail::ONE - std::exp(t_lb) : std::exp(t_lb);

    // Bracket in u. Lower: the leading-term bound (for b < 1 loosened by the (1 − x)^{b−1}
    // factor at x_lb, since the root is below x_lb there), else the smallest logit a double
    // holds. Upper: Markov, P(X ≥ 2·mean) ≤ ½, so the root at p ≤ ½ is at most 2a/(a + b);
    // u = 745 stands for x = 1 when that bound is vacuous.
    double lo = -745.0;
    if (t_lb < detail::ZERO_DOUBLE) {
        lo = t_lb;
        if (b < detail::ONE)
            lo += (detail::ONE - b) / a * std::log1p(-std::exp(t_lb));
        lo = std::max(lo, -745.0);
    }
    double hi = 745.0;
    {
        const double two_mean = detail::TWO * a / (a + b);
        if (two_mean < detail::ONE)
            hi = std::log(two_mean / (detail::ONE - two_mean));
    }
    if (!(lo < hi))
        hi = lo + detail::ONE;

    // Seed: the normal approximation in the centre, the leading term in the lower tail (where the
    // normal approximation lands at or below 0), in the logit; a seed outside the bracket starts
    // at the lower bound, where the leading term already is the root to O(x).
    double u;
    {
        const double mean = a / (a + b);
        const double sd = std::sqrt(a * b / ((a + b) * (a + b) * (a + b + detail::ONE)));
        double x0 = mean + sd * inverse_normal_cdf(p);
        if (p < 0.1)
            x0 = std::max(x0, std::exp(t_lb));
        u = (x0 > detail::ZERO_DOUBLE && x0 < detail::ONE) ? std::log(x0 / (detail::ONE - x0)) : lo;
        if (!(u > lo && u < hi))
            u = lo;
    }

    const double u_root = solve_concave(u, lo, hi, true, [&](double uu, double& slope) {
        // x and 1 − x each to relative ε from the logit, whichever side is the small one, and
        // their logs to absolute ε: log x = u − log1p(e^u), log(1 − x) = −log1p(e^u).
        double x, omx, log_x, log_omx;
        if (uu < detail::ZERO_DOUBLE) {
            const double e = std::exp(uu);
            const double l1pe = std::log1p(e);
            x = e / (detail::ONE + e);
            omx = detail::ONE / (detail::ONE + e);
            log_x = uu - l1pe;
            log_omx = -l1pe;
        } else {
            const double e = std::exp(-uu);
            const double l1pe = std::log1p(e);
            x = detail::ONE / (detail::ONE + e);
            omx = e / (detail::ONE + e);
            log_x = -l1pe;
            log_omx = -uu - l1pe;
        }
        // log I formed as beta_i forms I, but without ever taking the prefactor out of the log:
        // at large shapes it underflows a double long before the tail does (Beta(1e4, 1e4) at
        // x = 0.36 has I = 1e-323 with a prefactor of e^-745), and log of the underflowed I put
        // the root of a subnormal p at x = 1, or in the bulk.
        if (tiny) {
            const double lpo = log_beta_prefactor_over_a_logs(log_x, log_omx, a, b);
            const double log_i =
                x < (a + detail::ONE) / (a + b + detail::TWO)
                    ? lpo + std::log(beta_continued_fraction_unscaled(x, omx, a, b))
                    : std::log(beta_i_tiny_shape(x, omx, a, b));
            slope = std::exp(lpo + std::log(a) - log_i);  // the full prefactor is a·exp(lpo)
            return log_i - log_target;
        }
        const double bt_log = log_beta_prefactor_logs(x, a, b, log_x, log_omx, pc);
        double log_i;
        if (x < (a + detail::ONE) / (a + b + detail::TWO)) {
            log_i = bt_log + std::log(beta_continued_fraction(x, omx, a, b));
        } else {
            const double c = std::exp(bt_log) * beta_continued_fraction(omx, x, b, a);
            log_i = c < detail::ONE ? std::log1p(-c) : -std::numeric_limits<double>::infinity();
        }
        slope = std::exp(bt_log - log_i);  // x(1 − x)·pdf(x)/I = x^a(1 − x)^b/(B·I)
        return log_i - log_target;
    });

    // Back to x. The logit gives y and 1 − y each to relative ε, so the reflected answer is
    // read off as 1 − y directly rather than subtracted (Beta(½, 1e6).Q(1 − 1e-15) = 3.2e-5 is
    // 1 − y with y near 1, where 1 − y from the rounded y would lose 7e-12).
    if (u_root < detail::ZERO_DOUBLE) {
        const double e = std::exp(u_root);
        return upper ? detail::ONE / (detail::ONE + e) : e / (detail::ONE + e);
    }
    const double e = std::exp(-u_root);
    return upper ? e / (detail::ONE + e) : detail::ONE / (detail::ONE + e);
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

void vector_erf(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size() || input.empty()) {
        return;
    }

    const std::size_t size = input.size();

    // Use SIMD VectorOps for optimal performance
    if (arch::simd::SIMDPolicy::shouldUseSIMD(size)) {
        arch::simd::VectorOps::vector_erf(input.data(), output.data(), size);
    } else {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = erf(input[i]);
        }
    }
}

void vector_gamma_p(double a, std::span<const double> x_values, std::span<double> output) noexcept {
    if (x_values.size() != output.size() || x_values.empty()) {
        return;
    }

    const std::size_t size = x_values.size();

    // For now, use scalar implementation
    // Future enhancement: SIMD optimization of the series expansion
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = gamma_p(a, x_values[i]);
    }
}

void vector_gamma_q(double a, std::span<const double> x_values, std::span<double> output) noexcept {
    if (x_values.size() != output.size() || x_values.empty()) {
        return;
    }

    const std::size_t size = x_values.size();

    // For now, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = gamma_q(a, x_values[i]);
    }
}

void vector_beta_i(std::span<const double> x_values, double a, double b,
                   std::span<double> output) noexcept {
    if (x_values.size() != output.size() || x_values.empty()) {
        return;
    }

    const std::size_t size = x_values.size();

    // Hoist the prefactor constant: fixed across all elements for fixed (a, b).
    const double log_prefix = beta_prefactor_constant(a, b);
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = beta_i(x_values[i], a, b, log_prefix);
    }
}

void vector_lgamma(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size() || input.empty()) {
        return;
    }

    const std::size_t size = input.size();

    // Use SIMD log operations if available
    if (arch::simd::SIMDPolicy::shouldUseSIMD(size)) {
        // For now, use scalar loop in SIMD-sized chunks for cache efficiency
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = lgamma(input[i]);
        }
    } else {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = lgamma(input[i]);
        }
    }
}

void vector_lbeta(std::span<const double> a_values, std::span<const double> b_values,
                  std::span<double> output) noexcept {
    if (a_values.size() != b_values.size() || a_values.size() != output.size() ||
        a_values.empty()) {
        return;
    }

    const std::size_t size = a_values.size();

    // For now, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = lbeta(a_values[i], b_values[i]);
    }
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
    // Inverse standard normal CDF using inverse error function
    if (p <= detail::ZERO_DOUBLE || p >= detail::ONE) {
        if (p == detail::ZERO_DOUBLE)
            return -std::numeric_limits<double>::infinity();
        if (p == detail::ONE)
            return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Tails (|p − ½| > 0.425, AS 241's split): the survival form on the small side (#158).
    // 2p − 1 rounds to −1 for p < 2^-54 and drops p's low bits progressively above that, so
    // √2·erf_inv(2p − 1) returned −inf at p = 1e-300 and 0.44 relative error at 1e-15. p, and
    // 1 − p for p ≥ ½ (exact, Sterbenz), carry full information into the erfc-domain solver.
    if (p < 0.075)
        return -inv_survival_normal(p);
    if (p > 0.925)
        return inv_survival_normal(detail::ONE - p);

    // Centre: √2·erf_inv(2p − 1), where 2p − 1 loses at most an ulp of p, then a Newton polish
    // on the erf residual. The polish removes erf_inv's relative floor near 0 (its Halley loop
    // stops at an absolute 1e-12) and is well conditioned here: |u| ≤ erf_inv(0.85) ≈ 1.02, so
    // exp(u²) ≤ 2.9. The same polish as HalfNormal's central quantile.
    const double erf_arg = detail::TWO * p - detail::ONE;
    double u = erf_inv(erf_arg);
    for (int i = 0; i < 2; ++i) {
        const double r = std::erf(u) - erf_arg;
        u -= r * (detail::SQRT_PI * detail::HALF) * std::exp(u * u);
    }
    return detail::SQRT_2 * u;
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
// vs the true 5.46. Until v2.4.2 this helper was duplicated in half_normal.cpp and
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
    if (s >= detail::HALF)
        return detail::ZERO_DOUBLE;
    if (s < std::numeric_limits<double>::min())
        return inv_survival_normal_log(std::log(s));

    const double t = std::sqrt(-detail::TWO * std::log(s));
    // AS 26.2.23 coefficients (the set erf_inv uses for its moderate-tail branch).
    double u = t - (2.515517 + t * (0.802853 + t * 0.010328)) /
                       (detail::ONE + t * (1.432788 + t * (0.189269 + t * 0.001308)));
    for (int i = 0; i < 4; ++i) {
        const double pdf = detail::INV_SQRT_2PI * std::exp(-detail::HALF * u * u);
        if (!(pdf > detail::ZERO_DOUBLE))
            break;  // deeper than φ's underflow: keep the (law-limited) seed
        const double r = detail::HALF * std::erfc(u * detail::INV_SQRT_2) - s;
        const double step = r / pdf;
        u += step;
        if (std::fabs(step) <= 1e-15 * (detail::ONE + std::fabs(u)))
            break;
    }
    return u;
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
        log_x = std::log(df) - detail::TWO * std::log(abs_t) + log_y;
    }
    TTails r{};
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

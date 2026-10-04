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

namespace stats {
namespace detail {

// Forward declarations
static double beta_continued_fraction(double x, double a, double b) noexcept;
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

double log_gamma_prefactor(double a, double x, double shape_constant) noexcept {
    return a * log1pmx((x - a) / a) + shape_constant;
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
static double log_beta_prefactor(double x, double a, double b, double shape_constant) noexcept {
    if (a < kStirlingPrefactorShape || b < kStirlingPrefactorShape)
        return shape_constant + a * std::log(x) + b * std::log(detail::ONE - x);
    const double sum = a + b;
    const double x0 = a / sum;
    const double one_minus_x0 = b / sum;
    const double u = (x - x0) / x0;
    const double v = (x0 - x) / one_minus_x0;
    return a * log1pmx(u) + b * log1pmx(v) + shape_constant;
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
    return std::lgamma(m + detail::ONE) -
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
        return k * std::log(lambda) - lambda - std::lgamma(k + detail::ONE);
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
        return std::lgamma(n.hi + detail::ONE) - std::lgamma(xa + detail::ONE) -
               std::lgamma(xb + detail::ONE) + xa * std::log(pa) + xb * std::log1p(-pa);

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

double lgamma(double x) noexcept {
    return std::lgamma(x);
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

    if (x > a + detail::ONE) {
        // For large x, use the complementary function for better convergence
        return detail::ONE - gamma_q(a, x);
    }

    // Use the dedicated series function that has the correct formula
    return gamma_p_series(a, x);
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

    if (x <= a + detail::ONE) {
        // For small x, use the series expansion of P(a,x) and compute 1-P; for small a, Q
        // directly, which 1 − P = 1 − (1 − O(a)) cancels.
        if (a < kSmallShapeQ)
            return gamma_q_small_shape(a, x);
        return detail::ONE - gamma_p_series(a, x);
    }

    // For large x, use continued fraction expansion for Q(a,x)
    // Guard b before dividing: when x = a-1 exactly, b = 0 and d would be ±inf
    // before the abs(d)<ZERO clamp inside the loop executes. Mirror the pattern
    // used in beta_continued_fraction.
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

    double gamma_cf = std::exp(log_gamma_prefactor(a, x)) * h;
    return gamma_cf;
}

// Newton on a residual f(t) that is monotone and concave in t, from any start in (lo, hi) (#160,
// #159). Concavity makes the iteration globally convergent: at most one step overshoots onto the
// f < 0 side, and from there the iterates approach the root monotonically, so a positive residual
// after a negative one is rounding noise. `residual(t, slope)` returns f(t) and sets slope to
// |f'(t)|; `rising` gives the sign of f'. A non-finite f (an underflowed tail) bisects. Returns
// the root in t.
template <typename Residual>
static double solve_concave(double t, double lo, double hi, bool rising,
                            Residual residual) noexcept {
    bool seen_negative = false;
    int noise_flips = 0;
    double best_t = t;
    double best_f = std::numeric_limits<double>::infinity();
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
        // Once on the f < 0 side, exact iterates stay there; a return to f > 0 is rounding noise.
        // The second one ends a ping-pong the step test below may not catch at small slope.
        if (f > detail::ZERO_DOUBLE && seen_negative && ++noise_flips == 2)
            return best_t;
        if (f < detail::ZERO_DOUBLE)
            seen_negative = true;

        double next = rising ? t - f / slope : t + f / slope;
        // Converged before the bracket test: a step below one ulp leaves next == t, which is now
        // a bracket end.
        if (std::abs(next - t) <= detail::TWO * std::numeric_limits<double>::epsilon() *
                                      std::max(detail::ONE, std::abs(t)))
            return next;
        if (!std::isfinite(next) || next <= lo || next >= hi)
            next = detail::HALF * (lo + hi);  // underflowed tail or a step out of the bracket
        t = next;
    }
    return best_t;
}

// x with P(a, x) = p (#160), by Newton in t = log x on the small side of the probability scale:
// f(t) = log P(a, eᵗ) − log p below the median, log Q(a, eᵗ) − log q above it. The density of
// log X, exp(a·t − eᵗ)/Γ(a), is log-concave, so both residuals are concave in t and
// solve_concave applies. |f'(t)| = x·pdf(x)/P is the prefactor over P (or Q).
double gamma_p_inv(double a, double p) noexcept {
    if (std::isnan(a) || std::isnan(p) || a <= detail::ZERO_DOUBLE)
        return std::numeric_limits<double>::quiet_NaN();
    if (p <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p >= detail::ONE)
        return std::numeric_limits<double>::infinity();

    const bool upper = p > detail::HALF;
    const double log_target = std::log(upper ? detail::ONE - p : p);  // 1 − p exact for p ≥ ½

    // Bracket in t: e^-745 is below the smallest subnormal, e^709.78 near the largest double.
    double lo = -745.0;
    double hi = 709.78;
    if (!upper) {
        // P(a, x) ≤ x^a/Γ(a + 1), so the root is at or above that bound's root. Below t = −700
        // the bound equals P to a relative a·x/(a + 1) < 1e-304, so it is the answer — and the
        // only route to it when x underflows.
        const double t_bound = (log_target + std::lgamma(a + detail::ONE)) / a;
        if (t_bound < -700.0)
            return std::exp(t_bound);
        lo = t_bound;
        hi = std::log(a);  // P(a, a) > ½: the median is below the mean
    }

    // Wilson-Hilferty seed, x ≈ a·(1 − h + z·√h)³ with h = 1/(9a); the lower bound where it fails.
    const double h = detail::ONE / (detail::NINE * a);
    const double c = detail::ONE - h + inverse_normal_cdf(p) * std::sqrt(h);
    double t = c > detail::ZERO_DOUBLE ? std::log(a) + detail::THREE * std::log(c) : lo;
    if (!(t > lo && t < hi))
        t = upper ? detail::HALF * (lo + hi) : lo;

    return std::exp(solve_concave(t, lo, hi, !upper, [&](double tt, double& slope) {
        const double x = std::exp(tt);
        const double log_tail =
            std::log(upper ? gamma_q(a, x) : gamma_p(a, x));  // −inf on underflow
        slope = std::exp(log_gamma_prefactor(a, x) - log_tail);
        return log_tail - log_target;
    }));
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

    // Use continued fraction approximation
    double bt = std::exp(log_beta_prefactor(x, a, b, beta_prefactor_constant(a, b)));

    if (x < (a + detail::ONE) / (a + b + detail::TWO)) {
        return bt * beta_continued_fraction(x, a, b);
    } else {
        return detail::ONE - bt * beta_continued_fraction(detail::ONE - x, b, a);
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

    // log_beta_prefix is the caller's hoisted beta_prefactor_constant(a, b), in the direct or the
    // Stirling form as the shapes select (#166).
    double bt = std::exp(log_beta_prefactor(x, a, b, log_beta_prefix));

    if (x < (a + detail::ONE) / (a + b + detail::TWO)) {
        return bt * beta_continued_fraction(x, a, b);
    } else {
        return detail::ONE - bt * beta_continued_fraction(detail::ONE - x, b, a);
    }
}

// Helper function for beta incomplete function continued fraction
// Based on Numerical Recipes algorithm
static double beta_continued_fraction(double x, double a, double b) noexcept {
    const int max_iterations = expansion_iteration_cap(std::max(a, b));
    const double tolerance = detail::SPECIAL_FUNCTION_TOLERANCE;

    double qab = a + b;
    double qap = a + detail::ONE;
    double qam = a - detail::ONE;

    // Initial values for continued fraction
    double c = detail::ONE;
    double d = detail::ONE - qab * x / qap;

    if (std::abs(d) < detail::ZERO) {
        d = detail::ZERO;
    }

    d = detail::ONE / d;
    double h = d;

    for (int m = 1; m <= max_iterations; ++m) {
        int m2 = detail::TWO_INT * m;

        // Even step (positive): aa = m * (b - m) * x / [(a + m2 - 1) * (a + m2)]
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));

        // Update d and c
        d = detail::ONE + aa * d;
        if (std::abs(d) < detail::ZERO) {
            d = detail::ZERO;
        }
        c = detail::ONE + aa / c;
        if (std::abs(c) < detail::ZERO) {
            c = detail::ZERO;
        }

        d = detail::ONE / d;
        h *= d * c;

        // Odd step (negative): aa = -(a + m) * (qab + m) * x / [(a + m2) * (qap + m2)]
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));

        // Update d and c
        d = detail::ONE + aa * d;
        if (std::abs(d) < detail::ZERO) {
            d = detail::ZERO;
        }
        c = detail::ONE + aa / c;
        if (std::abs(c) < detail::ZERO) {
            c = detail::ZERO;
        }

        d = detail::ONE / d;
        double delta = d * c;
        h *= delta;

        // Check convergence
        if (std::abs(delta - detail::ONE) < tolerance) {
            break;
        }
    }

    // Return the continued fraction value multiplied by 1/a
    // This is part of the standard algorithm for regularized incomplete beta
    return h / a;
}

static double gamma_p_series(double a, double x) noexcept {
    // Compute the series expansion of the regularized incomplete gamma function
    // Based on Numerical Recipes algorithm
    if (x == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    // Standard series: P(a,x) = exp(-x + a*ln(x) - ln(Gamma(a))) * sum
    // where sum = 1/a * (1 + x/(a+1) + x^2/((a+1)*(a+2)) + ...)
    // This is equivalent to: sum = sum(n=0 to inf) [x^n / (a * (a+1) * ... * (a+n))]

    double ap = a;          // Start with 'a'
    double sum = 1.0 / ap;  // First term: 1/a
    double term = sum;      // Current term

    const double tolerance = detail::SPECIAL_FUNCTION_TOLERANCE;
    const int max_iterations = expansion_iteration_cap(a);

    for (int n = 1; n < max_iterations; ++n) {
        ap += 1.0;       // ap = a + n
        term *= x / ap;  // term *= x / (a + n)
        sum += term;     // accumulate sum
        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }

    // The result is exp(-x + a*ln(x) - lgamma(a)) * sum
    double result = std::exp(log_gamma_prefactor(a, x)) * sum;
    return std::min(1.0, std::max(0.0, result));  // Clamp to [0,1]
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
        return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double s = x + y;
    return std::lgamma(y) - (x - detail::HALF) * std::log1p(y / x) - y * std::log(s) + y +
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

double inverse_beta_i(double p, double a, double b) noexcept {
    // Inverse regularized incomplete beta I_x(a,b) = p  =>  solve for x in (0,1).
    if (p <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p >= detail::ONE)
        return detail::ONE;
    if (a <= detail::ZERO_DOUBLE || b <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Initial estimate.
    // The normal approximation N(a/(a+b), sqrt(ab/(a+b)^2/(a+b+1))) is accurate
    // in the middle of [0,1] but can give x <= 0 or x >= 1 in the tails, after
    // which Newton oscillates between the clamp boundaries and never converges.
    //
    // Tail asymptotic: I_x(a,b) ~ x^a / (a*B(a,b)) for small x
    //   => x ~ (p * a * B(a,b))^(1/a)
    // Symmetry for large p: use (1-p) and reversed parameters.
    const double lb = lbeta(a, b);
    double x;
    {
        const double mu = a / (a + b);
        const double sigma = std::sqrt(a * b / ((a + b) * (a + b) * (a + b + detail::ONE)));
        x = mu + sigma * inverse_normal_cdf(p);
    }
    // Blend normal approximation with the tail asymptotic.
    // For small p the normal approximation can give x slightly above 0 (e.g. 4e-4
    // instead of the true ~0.06 for Beta(2,3) at p=0.023).  Clamping a very small
    // positive x to max(1e-8,...) leaves Newton too far from the root and the first
    // step diverges.  Taking max(normal, asymptotic) for p<0.1 avoids this.
    if (x <= detail::ZERO_DOUBLE) {
        x = std::pow(p * a * std::exp(lb), 1.0 / a);
    } else if (x >= detail::ONE) {
        x = 1.0 - std::pow((1.0 - p) * b * std::exp(lb), 1.0 / b);
    } else {
        if (p < 0.1) {
            const double x_asymp = std::pow(p * a * std::exp(lb), 1.0 / a);
            x = std::max(x, x_asymp);  // never start below the asymptotic estimate
        } else if (p > 0.9) {
            const double x_asymp = 1.0 - std::pow((1.0 - p) * b * std::exp(lb), 1.0 / b);
            x = std::min(x, x_asymp);
        }
    }
    x = std::max(1e-8, std::min(1.0 - 1e-8, x));  // clamp to (0,1)

    // Newton-Raphson: x_{n+1} = x_n - (I_{x_n}(a,b) - p) / f(x_n)
    // where f(x) = x^(a-1)(1-x)^(b-1)/B(a,b) is the Beta PDF.
    const int max_iter = detail::MAX_NEWTON_ITERATIONS;
    const double tol = detail::DEFAULT_TOLERANCE;
    const double log_norm = -lbeta(a, b);  // -ln B(a,b)

    for (int i = 0; i < max_iter; ++i) {
        const double cdf_val = beta_i(x, a, b);
        const double error = cdf_val - p;

        if (std::abs(error) < tol)
            break;

        // PDF = exp((a-1)*log(x) + (b-1)*log(1-x) + log_norm)
        const double log_pdf = (a - detail::ONE) * std::log(x) +
                               (b - detail::ONE) * std::log(detail::ONE - x) + log_norm;
        const double pdf_val = std::exp(log_pdf);

        if (pdf_val <= detail::ZERO_DOUBLE)
            break;

        x -= error / pdf_val;
        x = std::max(1e-10, std::min(detail::ONE - 1e-10, x));
    }
    return x;
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
// returned Φ⁻¹(DBL_MIN) for every subnormal s, 2.5% off at 5e-324.
//
// Not delegated to erf_inv: its extreme-tail branch (|x| ≥ ERF_INV_TAIL_CUTOFF) seeds with a
// Φ⁻¹-domain formula that is off by ~√2 in the erf domain, and its Halley refinement cannot
// recover once std::erf saturates to 1 — measured during #57 bring-up: erf_inv(1−1e-14) ≈ 7.59
// vs the true 5.46. Until v2.4.2 this helper was duplicated in half_normal.cpp and
// truncated_normal.cpp (#158 promoted it here).
double inv_survival_normal(double s) noexcept {
    if (s >= detail::HALF)
        return detail::ZERO_DOUBLE;
    if (s < std::numeric_limits<double>::min()) {
        const double log_s = std::log(s);
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
    const double lead = std::exp(detail::HALF * detail::LN_PI - lbeta(a, detail::HALF) -
                                 detail::HALF * std::log(T));  // Γ(a + ½)/(Γ(a)·√T)
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
    double x, y, log_x, log_y;
    if (abs_t * abs_t < df) {
        const double u = abs_t * abs_t / df;
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
    r.log_t_pdf = a * log_x + detail::HALF * log_y - lbeta_a_half;
    const double prefactor = std::exp(r.log_t_pdf);
    if (x < (a + detail::ONE) / (a + 2.5)) {
        // BGRAT where the continued fraction is slow (large a, x near 1); its domain follows
        // Boost's use of it (a ≥ 15, x ≥ 0.3).
        r.tail = (a >= kStirlingPrefactorShape && x >= detail::HALF)
                     ? detail::HALF * beta_i_large_a_half(a, log_x)
                     : detail::HALF * prefactor * beta_continued_fraction(x, a, detail::HALF);
        r.central = detail::ONE - detail::TWO * r.tail;
    } else {
        r.central = prefactor * beta_continued_fraction(y, detail::HALF, a);
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

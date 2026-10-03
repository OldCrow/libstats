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

#include <algorithm>
#include <array>
#include <cmath>
#include <corvus/corvus.h>
#include <span>
#include <stdexcept>

namespace stats {
namespace detail {

// =============================================================================
// SPECIAL MATHEMATICAL FUNCTIONS — corvus is the engine (v2.5.0)
// =============================================================================
//
// Every special function here is a span-of-1 call into corvus, whose kernels
// are SIMD-vectorised and validated per tier against an mpmath oracle (corvus
// docs/ACCURACY.md). A scalar call pays the dispatch hop; hot loops use the
// vector_* adapters further down, which hand corvus whole blocks.
//
// Domain semantics follow corvus, not the retired local cores: a negative or
// out-of-range argument returns NaN rather than a clamped 0 or 1. Distribution
// validators reject such parameters before any call reaches here.

namespace {

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

template <class Fn>
[[nodiscard]] inline double corvus_scalar(Fn fn, double a, double b, double c) noexcept {
    double out;
    fn(std::span<const double>{&a, 1}, std::span<const double>{&b, 1},
       std::span<const double>{&c, 1}, std::span<double>{&out, 1});
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

double lgamma(double x) noexcept {
    return corvus_scalar(corvus::lgamma, x);
}

double gamma_p(double a, double x) noexcept {
    return corvus_scalar(corvus::gamma_p, a, x);
}

double gamma_q(double a, double x) noexcept {
    return corvus_scalar(corvus::gamma_q, a, x);
}

double beta_i(double x, double a, double b) noexcept {
    return corvus_scalar(corvus::beta_p, a, b, x);
}

double beta_i(double x, double a, double b, double /*log_beta_prefix*/) noexcept {
    // The prefix was a hoisted lgamma triple for the retired local core; corvus
    // forms its own prefactor in double-double. Kept for source compatibility;
    // batch callers belong on vector_beta_i.
    return beta_i(x, a, b);
}

double lbeta(double a, double b) noexcept {
    return corvus_scalar(corvus::lbeta, a, b);
}

double digamma(double x) noexcept {
    return corvus_scalar(corvus::digamma, x);
}

double trigamma(double x) noexcept {
    return corvus_scalar(corvus::trigamma, x);
}

double inverse_beta_i(double p, double a, double b) noexcept {
    return corvus_scalar(corvus::beta_p_inv, a, b, p);
}

namespace {
// corvus-order alias for local use alongside beta_q_inv.
[[nodiscard]] inline double beta_p_inv_(double a, double b, double p) noexcept {
    return inverse_beta_i(p, a, b);
}
}  // namespace

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
    // |x| + log i0e(|x|): i0e = e^-|x| I0 never underflows on a finite double,
    // so one form serves the whole axis and there is no seam to value-match
    // (#92). Absolute error ~ulp(|x|), which is what the von Mises
    // normaliser LN_2PI + log I0 needs; the relative error at small x is the
    // x^2/4 cancellation's, and no consumer reads it relatively.
    const double ax = std::fabs(x);
    if (std::isinf(ax)) {
        return ax;
    }
    return ax + std::log(corvus_scalar(corvus::i0e, ax));
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
    // Past the cut the complement is small, so 1 − complement is the stable
    // direction; below it the scaled ratio is direct (1 − complement would
    // cancel as κ → 0). i1e/i0e never forms the overflowing I0/I1 (#93).
    if (x >= kBesselRatioAsymptoticCut) {
        return 1.0 - bessel_i1_i0_complement(x);
    }
    return corvus_scalar(corvus::i1e, x) / corvus_scalar(corvus::i0e, x);
}

double erfc_inv(double y) noexcept {
    return corvus_scalar(corvus::erfcinv, y);
}

double gamma_p_inv(double a, double p) noexcept {
    return corvus_scalar(corvus::gamma_p_inv, a, p);
}

double gamma_q_inv(double a, double q) noexcept {
    return corvus_scalar(corvus::gamma_q_inv, a, q);
}

double beta_q_inv(double a, double b, double q) noexcept {
    return corvus_scalar(corvus::beta_q_inv, a, b, q);
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
    // Inverse standard normal CDF using inverse error function
    if (p <= detail::ZERO_DOUBLE || p >= detail::ONE) {
        if (p == detail::ZERO_DOUBLE)
            return -std::numeric_limits<double>::infinity();
        if (p == detail::ONE)
            return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Phi^-1(p) = -sqrt(2) * erfc_inv(2p). The erf_inv(2p - 1) form loses the
    // whole tail: 2p - 1 rounds to -1 for p < 2^-54. Both 2p and 1 - p (for
    // p >= 1/2, Sterbenz) are exact, so each side is solved from a small,
    // exact argument.
    return p <= detail::HALF ? -detail::SQRT_2 * erfc_inv(detail::TWO * p)
                             : detail::SQRT_2 * erfc_inv(detail::TWO * (detail::ONE - p));
}

double t_cdf(double t, double df) noexcept {
    // Student's t-distribution CDF using regularized incomplete beta function
    if (df <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (std::isinf(t)) {
        return (t > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
    }

    if (t == detail::ZERO_DOUBLE) {
        return detail::HALF;
    }

    // Use relationship with incomplete beta function:
    // t_cdf(t, df) = 1/2 + (t/sqrt(df)) * B(1/2, df/2) / B(1/2, df/2)
    // This is simplified using the symmetry of t-distribution

    // Past |t| ~ 1e154 t^2 overflows and x underflows; there I_x(a, 1/2) ~
    // x^a / (a B(a, 1/2)) with x = df / t^2, formed in the log domain so the
    // tail stays finite down to its true underflow.
    const double a = detail::HALF * df;
    const double x = df / (df + t * t);
    const double result = (x >= std::numeric_limits<double>::min())
                              ? beta_i(x, a, detail::HALF)
                              : std::exp(a * (std::log(df) - detail::TWO * std::log(std::fabs(t))) -
                                         std::log(a) - lbeta(a, detail::HALF));

    if (t > detail::ZERO_DOUBLE) {
        return detail::ONE - detail::HALF * result;
    } else {
        return detail::HALF * result;
    }
}

double inverse_t_cdf(double p, double df) noexcept {
    // Inverse t-distribution CDF using iterative methods
    if (p <= detail::ZERO_DOUBLE || p >= detail::ONE || df <= detail::ZERO_DOUBLE) {
        if (p == detail::ZERO_DOUBLE)
            return -std::numeric_limits<double>::infinity();
        if (p == detail::ONE)
            return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (p == detail::HALF) {
        return detail::ZERO_DOUBLE;
    }

    // Two-sided tail q = 2 min(p, 1 - p) satisfies I_x(df/2, 1/2) = q with
    // x = df / (df + t^2), so t^2 = df (1 - x) / x. Take 1 - x from
    // beta_q_inv (the swap identity). Where 1 - x <= 1/2 (|t| <= sqrt(df), at
    // least half of all p and nearly all of them at large df), x = 1 - (1 - x)
    // is exact to rounding and one inverse call suffices. In the tail x is
    // the small side and comes from its own beta_p_inv call, so neither is
    // formed by a cancelling subtraction. 1 - p is exact for p >= 1/2
    // (Sterbenz).
    const bool upper = p > detail::HALF;
    const double q = detail::TWO * (upper ? detail::ONE - p : p);
    const double a = df * detail::HALF;
    const double one_minus_x = beta_q_inv(detail::HALF, a, q);
    double t;
    if (one_minus_x <= detail::HALF) {
        t = std::sqrt(df * one_minus_x / (detail::ONE - one_minus_x));
    } else if (const double x = beta_p_inv_(a, detail::HALF, q);
               x >= std::numeric_limits<double>::min()) {
        t = std::sqrt(df * one_minus_x / x);
    } else {
        // x underflowed (small df, deep tail: the Cauchy-like quantile is
        // ~1e299 at df = 1, p = 1e-300, well inside double range, so #104
        // wants it finite). I_x(a, 1/2) ~ x^a / (a B(a, 1/2)) as x -> 0, so
        // ln x = (ln q + ln a + lbeta(a, 1/2)) / a and t = sqrt(df / x),
        // formed in the log domain; exp overflows only on true overflow.
        const double log_x = (std::log(q) + std::log(a) + lbeta(a, detail::HALF)) / a;
        t = std::exp(detail::HALF * (std::log(df) - log_x));
    }
    return upper ? t : -t;
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
    // Inverse chi-squared CDF using iterative methods
    if (p < detail::ZERO_DOUBLE || p > detail::ONE || df <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    // Chi-squared(df) is Gamma(df/2, 2).
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
    // Inverse gamma distribution CDF using iterative methods
    if (p < detail::ZERO_DOUBLE || p > detail::ONE || shape <= detail::ZERO_DOUBLE ||
        scale <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
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

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

    // Use relationship: inverse_normal_cdf(p) = sqrt(2) * erf_inv(2*p - 1)
    double erf_arg = detail::TWO * p - detail::ONE;
    return detail::SQRT_2 * erf_inv(erf_arg);
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

    // For very large degrees of freedom, use normal approximation for better accuracy
    if (df >= 1000.0) {
        return normal_cdf(t);
    }

    // Use relationship with incomplete beta function:
    // t_cdf(t, df) = 1/2 + (t/sqrt(df)) * B(1/2, df/2) / B(1/2, df/2)
    // This is simplified using the symmetry of t-distribution

    double x = df / (df + t * t);
    double result = beta_i(x, detail::HALF * df, detail::HALF);

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

    // Use approximate initial guess from normal distribution
    double z = inverse_normal_cdf(p);

    // For large degrees of freedom, t-distribution approaches normal.
    // Use 1000 as the cutoff (consistent with t_cdf) — at df=120 the
    // normal approximation still has ~0.02 error in the tails.
    if (df > detail::THOUSAND) {
        return z;
    }

    // Newton-Raphson iteration to refine the estimate
    double t = z;  // Initial guess
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = t_cdf(t, df);
        double error = cdf_val - p;

        if (std::abs(error) < tolerance) {
            break;
        }

        // Calculate derivative (PDF)
        double pdf_val =
            std::exp(lgamma((df + detail::ONE) * detail::HALF) - lgamma(df * detail::HALF) -
                     detail::HALF * std::log(df * detail::PI)) *
            std::pow(detail::ONE + t * t / df, -(df + detail::ONE) * detail::HALF);

        if (pdf_val <= detail::ZERO_DOUBLE) {
            break;  // Avoid division by zero
        }

        t -= error / pdf_val;
    }

    return t;
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

    // For very small p, use bisection to avoid Newton-Raphson instability
    if (p < 0.1 || p > 0.9) {
        // Use bisection method which is more stable for extreme probabilities
        double low = detail::ZERO_DOUBLE;
        double high = df + 10.0 * std::sqrt(df);
        // Expand upper bound until it actually brackets p (handles p > 0.9999)
        while (chi_squared_cdf(high, df) < p) {
            high *= 2.0;
            if (high > 1e15)
                break;  // safety cap
        }
        const double tolerance = detail::DEFAULT_TOLERANCE;
        const int max_iterations = detail::MAX_NEWTON_ITERATIONS;

        for (int i = 0; i < max_iterations; ++i) {
            double mid = (low + high) * detail::HALF;
            double cdf_val = chi_squared_cdf(mid, df);

            if (std::abs(cdf_val - p) < tolerance) {
                return mid;
            }

            if (cdf_val < p) {
                low = mid;
            } else {
                high = mid;
            }

            if (high - low < tolerance) {
                return (low + high) * detail::HALF;
            }
        }
        return (low + high) * detail::HALF;
    }

    // Initial guess using Wilson-Hilferty approximation
    double h = detail::TWO / (detail::NINE * df);
    double z = inverse_normal_cdf(p);
    double initial_guess = df * std::pow(detail::ONE - h + z * std::sqrt(h), 3);

    // Ensure initial guess is positive
    if (initial_guess <= detail::ZERO_DOUBLE) {
        initial_guess = df;  // Use mean as fallback
    }

    // Newton-Raphson iteration for moderate probabilities
    double x = initial_guess;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = chi_squared_cdf(x, df);
        double error = cdf_val - p;

        if (std::abs(error) < tolerance) {
            break;
        }

        // Calculate derivative (PDF)
        double pdf_val =
            std::exp((df * detail::HALF - detail::ONE) * std::log(x) - x * detail::HALF -
                     lgamma(df * detail::HALF) - df * detail::HALF * detail::LN2);

        if (pdf_val <= detail::ZERO_DOUBLE) {
            break;  // Avoid division by zero
        }

        double delta = error / pdf_val;
        x = std::max(detail::ZERO, x - delta);  // Ensure x stays positive

        // Check for divergence and fall back to bisection if needed
        if (!std::isfinite(x) || x > 1e15) {
            // Fall back to bisection method
            double low = detail::ZERO_DOUBLE;
            double high = df + 10.0 * std::sqrt(df);
            while (chi_squared_cdf(high, df) < p) {
                high *= 2.0;
                if (high > 1e15)
                    break;  // safety cap
            }

            for (int j = 0; j < max_iterations; ++j) {
                double mid = (low + high) * detail::HALF;
                double mid_cdf = chi_squared_cdf(mid, df);

                if (std::abs(mid_cdf - p) < tolerance) {
                    return mid;
                }

                if (mid_cdf < p) {
                    low = mid;
                } else {
                    high = mid;
                }

                if (high - low < tolerance) {
                    return (low + high) * detail::HALF;
                }
            }
            return (low + high) * detail::HALF;
        }
    }

    return x;
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

    // Initial guess using approximation
    // For gamma distribution, mean = shape * scale, variance = shape * scale^2
    double mean = shape * scale;
    double variance = shape * scale * scale;

    // Wilson-Hilferty approximation for initial guess
    double h = detail::TWO / (detail::NINE * shape);
    double z = inverse_normal_cdf(p);
    double initial_guess = mean * std::pow(detail::ONE - h + z * std::sqrt(h), 3);

    // Ensure initial guess is positive
    if (initial_guess <= detail::ZERO_DOUBLE) {
        initial_guess = mean;  // Use mean as fallback
    }

    // For extreme probabilities, use bisection method for stability
    if (p < 0.1 || p > 0.9) {
        double low = detail::ZERO_DOUBLE;
        double high = mean + 10.0 * std::sqrt(variance);  // Conservative upper bound
        const double tolerance = detail::DEFAULT_TOLERANCE;
        const int max_iterations = detail::MAX_NEWTON_ITERATIONS;

        for (int i = 0; i < max_iterations; ++i) {
            double mid = (low + high) * detail::HALF;
            double cdf_val = gamma_cdf(mid, shape, scale);

            if (std::abs(cdf_val - p) < tolerance) {
                return mid;
            }

            if (cdf_val < p) {
                low = mid;
            } else {
                high = mid;
            }

            if (high - low < tolerance) {
                return (low + high) * detail::HALF;
            }
        }
        return (low + high) * detail::HALF;
    }

    // Newton-Raphson iteration for moderate probabilities
    double x = initial_guess;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = gamma_cdf(x, shape, scale);
        double error = cdf_val - p;

        if (std::abs(error) < tolerance) {
            break;
        }

        // Calculate derivative (PDF)
        // Gamma PDF: f(x; α, β) = (1/β^α Γ(α)) * x^(α-1) * e^(-x/β)
        double log_pdf = (shape - detail::ONE) * std::log(x) - x / scale - shape * std::log(scale) -
                         lgamma(shape);
        double pdf_val = std::exp(log_pdf);

        if (pdf_val <= detail::ZERO_DOUBLE) {
            break;  // Avoid division by zero
        }

        double delta = error / pdf_val;
        x = std::max(detail::ZERO, x - delta);  // Ensure x stays positive

        // Check for divergence and fall back to bisection if needed
        if (x > mean + 10.0 * std::sqrt(variance) || !std::isfinite(x)) {
            // Fall back to bisection method
            double low = detail::ZERO_DOUBLE;
            double high = mean + 10.0 * std::sqrt(variance);

            for (int j = 0; j < max_iterations; ++j) {
                double mid = (low + high) * detail::HALF;
                double mid_cdf = gamma_cdf(mid, shape, scale);

                if (std::abs(mid_cdf - p) < tolerance) {
                    return mid;
                }

                if (mid_cdf < p) {
                    low = mid;
                } else {
                    high = mid;
                }

                if (high - low < tolerance) {
                    return (low + high) * detail::HALF;
                }
            }
            return (low + high) * detail::HALF;
        }
    }

    return x;
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

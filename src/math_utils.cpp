#include "../include/core/math_utils.h"

#include "../include/core/distribution_base.h"
#include "../include/core/mathematical_constants.h"
#include "../include/core/precision_constants.h"
#include "../include/core/safety.h"
#include "../include/core/threshold_constants.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd_policy.h"

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

    // Rational approximation constants (Moro's method)
    static const double a0 = 2.50662823884;
    static const double a1 = -18.61500062529;
    static const double a2 = 41.39119773534;
    static const double a3 = -25.44106049637;

    static const double b0 = -8.47351093090;
    static const double b1 = 23.08336743743;
    static const double b2 = -21.06224101826;
    static const double b3 = 3.13082909833;

    // Note: c0-c8 constants removed since we now use Acklam's method for moderate tail

    double result;

    if (a <= detail::STRONG_CORRELATION) {
        // Central region: use rational approximation
        double z = a * a;
        result = a * (((a3 * z + a2) * z + a1) * z + a0) /
                 ((((b3 * z + b2) * z + b1) * z + b0) * z + detail::ONE);
    } else if (a < detail::CONFIDENCE_99) {
        // Moderate tail region: use improved asymptotic expansion with better coefficients
        double z = std::sqrt(-std::log((detail::ONE - a) * detail::HALF));

        // More accurate coefficients for moderate tail (based on Acklam's method)
        static const double d0 = 2.515517;
        static const double d1 = 0.802853;
        static const double d2 = 0.010328;
        static const double e1 = 1.432788;
        static const double e2 = 0.189269;
        static const double e3 = 0.001308;

        result =
            z - (d0 + d1 * z + d2 * z * z) / (detail::ONE + e1 * z + e2 * z * z + e3 * z * z * z);
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

            // More accurate coefficients for extreme tail
            static const double d0 = 2.515517;
            static const double d1 = 0.802853;
            static const double d2 = 0.010328;
            static const double e0 = 1.432788;
            static const double e1 = 0.189269;
            static const double e2 = 0.001308;

            result = t - (d0 + d1 * t + d2 * t * t) /
                             (detail::ONE + e0 * t + e1 * t * t + e2 * t * t * t);

            // Additional correction term for better accuracy
            double correction = std::log(t * detail::SQRT_PI * detail::HALF) / (detail::TWO * t);
            result -=
                correction * detail::AD_THRESHOLD_1;  // Damped correction to avoid overcorrection
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

double gamma_p(double a, double x) noexcept {
    // Regularized incomplete gamma function P(a,x) = γ(a,x) / Γ(a)
    // where γ(a,x) is the lower incomplete gamma function
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
    if (x < detail::ZERO_DOUBLE || a <= detail::ZERO_DOUBLE) {
        return detail::ONE;
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ONE;
    }

    if (x <= a + detail::ONE) {
        // For small x, use the series expansion of P(a,x) and compute 1-P
        return detail::ONE - gamma_p_series(a, x);
    }

    // For large x, use continued fraction expansion for Q(a,x)
    double b = x + detail::ONE - a;
    double c = detail::LARGE_CONTINUED_FRACTION_VALUE;
    double d = detail::ONE / b;
    double h = d;

    const int max_iterations = detail::MAX_GAMMA_SERIES_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

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

    double gamma_cf = std::exp(-x + a * std::log(x) - lgamma(a)) * h;
    return gamma_cf;
}

double beta_i(double x, double a, double b) noexcept {
    // Regularized incomplete beta function I_x(a,b)
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

    // Use continued fraction approximation
    double bt = std::exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * std::log(x) +
                         b * std::log(detail::ONE - x));

    if (x < (a + detail::ONE) / (a + b + detail::TWO)) {
        return bt * beta_continued_fraction(x, a, b);
    } else {
        return detail::ONE - bt * beta_continued_fraction(detail::ONE - x, b, a);
    }
}

// Helper function for beta incomplete function continued fraction
// Based on Numerical Recipes algorithm
static double beta_continued_fraction(double x, double a, double b) noexcept {
    const int max_iterations = detail::MAX_BETA_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

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

    const double tolerance = detail::DEFAULT_TOLERANCE;
    const int max_iterations = detail::MAX_GAMMA_SERIES_ITERATIONS;

    for (int n = 1; n < max_iterations; ++n) {
        ap += 1.0;       // ap = a + n
        term *= x / ap;  // term *= x / (a + n)
        sum += term;     // accumulate sum
        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }

    // The result is exp(-x + a*ln(x) - lgamma(a)) * sum
    double log_result = -x + a * std::log(x) - lgamma(a);
    double result = std::exp(log_result) * sum;
    return std::min(1.0, std::max(0.0, result));  // Clamp to [0,1]
}

double lbeta(double a, double b) noexcept {
    return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
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
        double empirical_cdf = static_cast<double>(i + 1) / n;
        double theoretical_cdf = dist.getCumulativeProbability(sorted_data[i]);

        // Check both F_n(x) - F(x) and F(x) - F_{n-1}(x)
        double diff1 = std::abs(empirical_cdf - theoretical_cdf);
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

void vector_erfc(std::span<const double> input, std::span<double> output) noexcept {
    if (input.size() != output.size() || input.empty()) {
        return;
    }

    const std::size_t size = input.size();

    // Use scalar loop for now - SIMD erfc would require implementation
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = erfc(input[i]);
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

    // For now, use scalar implementation
    // Future enhancement: SIMD optimization of the continued fraction
    for (std::size_t i = 0; i < size; ++i) {
        output[i] = beta_i(x_values[i], a, b);
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
    // Standard normal CDF using error function
    return detail::HALF * (detail::ONE + erf(z * detail::INV_SQRT_2));
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

    // For large degrees of freedom, t-distribution approaches normal
    if (df > detail::HUNDRED) {
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
        double high = df + 10.0 * std::sqrt(df);  // Conservative upper bound
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
        if (x > df + 10.0 * std::sqrt(df) || !std::isfinite(x)) {
            // Fall back to bisection method
            double low = detail::ZERO_DOUBLE;
            double high = df + 10.0 * std::sqrt(df);

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

double f_cdf(double x, double df1, double df2) noexcept {
    // F-distribution CDF using regularized incomplete beta function
    if (x < detail::ZERO_DOUBLE || df1 <= detail::ZERO_DOUBLE || df2 <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (x == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // F-distribution relationship with beta function:
    // F_cdf(x, df1, df2) = I_y(df1/2, df2/2) where y = (df1*x)/(df1*x + df2)
    double y = (df1 * x) / (df1 * x + df2);
    return beta_i(y, df1 * detail::HALF, df2 * detail::HALF);
}

double inverse_f_cdf(double p, double df1, double df2) noexcept {
    // Inverse F-distribution CDF using iterative methods
    if (p < detail::ZERO_DOUBLE || p > detail::ONE || df1 <= detail::ZERO_DOUBLE ||
        df2 <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    // Initial guess using approximation
    double z = inverse_normal_cdf(p);
    double initial_guess = std::max(detail::ONE, detail::ONE + z * std::sqrt(detail::TWO / df1));

    // Newton-Raphson iteration
    double x = initial_guess;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = f_cdf(x, df1, df2);
        double error = cdf_val - p;

        if (std::abs(error) < tolerance) {
            break;
        }

        // Calculate derivative (PDF)
        double log_pdf = lgamma((df1 + df2) * detail::HALF) - lgamma(df1 * detail::HALF) -
                         lgamma(df2 * detail::HALF) +
                         (df1 * detail::HALF - detail::ONE) * std::log(x) -
                         (df1 + df2) * detail::HALF * std::log(detail::ONE + df1 * x / df2) +
                         df1 * detail::HALF * std::log(df1 / df2);

        double pdf_val = std::exp(log_pdf);

        if (pdf_val <= detail::ZERO_DOUBLE) {
            break;  // Avoid division by zero
        }

        double delta = error / pdf_val;
        x = std::max(detail::ZERO, x - delta);  // Ensure x stays positive
    }

    return x;
}

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

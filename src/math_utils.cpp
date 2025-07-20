#include "math_utils.h"
#include "distribution_base.h"
#include "safety.h"
#include "cpu_detection.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace libstats {
namespace math {

// Forward declarations
static double beta_continued_fraction(double x, double a, double b) noexcept;

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
    // Fast inverse error function using Winitzki's approximation
    // Accurate to about 10^-4, much faster than iterative methods
    
    if (x < -1.0 || x > 1.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    if (x == 0.0) return 0.0;
    if (x >= 1.0) return std::numeric_limits<double>::infinity();
    if (x <= -1.0) return -std::numeric_limits<double>::infinity();
    
    // Winitzki's approximation
    double sign = (x < 0) ? -1.0 : 1.0;
    x = std::abs(x);
    
    double ln_term = std::log(1.0 - x * x);
    double term1 = (2.0 / (constants::math::PI * constants::math::WINITZKI_A)) + ln_term / 2.0;
    double term2 = ln_term / constants::math::WINITZKI_A;
    
    double result = sign * std::sqrt(std::sqrt(term1 * term1 - term2) - term1);
    return result;
}

double lgamma(double x) noexcept {
    return std::lgamma(x);
}

double gamma_p(double a, double x) noexcept {
    // Regularized incomplete gamma function using series expansion
    if (x < 0.0 || a <= 0.0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    // Series expansion approach
    double sum = 1.0;
    double term = 1.0;
    double n = 1.0;
    
    while (std::abs(term) > constants::precision::DEFAULT_TOLERANCE && n < constants::precision::MAX_GAMMA_SERIES_ITERATIONS) {
        term *= x / (a + n - 1.0);
        sum += term;
        n += 1.0;
    }
    
    double result = std::exp(-x + a * std::log(x) - lgamma(a)) * sum;
    return std::min(1.0, std::max(0.0, result)); // Clamp to [0,1]
}

double gamma_q(double a, double x) noexcept {
    // Regularized complementary incomplete gamma function
    double p_value = gamma_p(a, x);
    return 1.0 - p_value;
}

double beta_i(double x, double a, double b) noexcept {
    // Regularized incomplete beta function I_x(a,b)
    if (x < 0.0 || x > 1.0 || a <= 0.0 || b <= 0.0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    if (x == 1.0) {
        return 1.0;
    }
    
    // Use continued fraction approximation
    double bt = std::exp(lgamma(a + b) - lgamma(a) - lgamma(b) + 
                        a * std::log(x) + b * std::log(1.0 - x));
    
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * beta_continued_fraction(x, a, b) / a;
    } else {
        return 1.0 - bt * beta_continued_fraction(1.0 - x, b, a) / b;
    }
}

// Helper function for beta incomplete function continued fraction
static double beta_continued_fraction(double x, double a, double b) noexcept {
    const int max_iterations = constants::precision::MAX_BETA_ITERATIONS;
    const double tolerance = constants::precision::DEFAULT_TOLERANCE;
    
    double qab = a + b;
    double qap = a + 1.0;
    double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    
    if (std::abs(d) < constants::precision::ZERO) {
        d = constants::precision::ZERO;
    }
    
    d = 1.0 / d;
    double h = d;
    
    for (int m = 1; m <= max_iterations; ++m) {
        int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        
        if (std::abs(d) < constants::precision::ZERO) {
            d = constants::precision::ZERO;
        }
        
        c = 1.0 + aa / c;
        
        if (std::abs(c) < constants::precision::ZERO) {
            c = constants::precision::ZERO;
        }
        
        d = 1.0 / d;
        h *= d * c;
        
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        
        if (std::abs(d) < constants::precision::ZERO) {
            d = constants::precision::ZERO;
        }
        
        c = 1.0 + aa / c;
        
        if (std::abs(c) < constants::precision::ZERO) {
            c = constants::precision::ZERO;
        }
        
        d = 1.0 / d;
        double del = d * c;
        h *= del;
        
        if (std::abs(del - 1.0) < tolerance) {
            break;
        }
    }
    
    return h;
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
        cdf[i] = static_cast<double>(i + 1) / sorted_data.size();
    }
    return cdf;
}

std::vector<double> calculate_quantiles(std::span<const double> data, std::span<const double> quantiles) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot calculate quantiles from empty data");
    }
    
    // Sort data for quantile calculation
    std::vector<double> sorted_data(data.begin(), data.end());
    std::sort(sorted_data.begin(), sorted_data.end());
    
    std::vector<double> result;
    result.reserve(quantiles.size());
    
    for (double q : quantiles) {
        if (q < 0.0 || q > 1.0) {
            throw std::invalid_argument("Quantile values must be in [0, 1]");
        }
        
        if (q == 0.0) {
            result.push_back(sorted_data.front());
        } else if (q == 1.0) {
            result.push_back(sorted_data.back());
        } else {
            // Use linear interpolation between data points
            double pos = q * (sorted_data.size() - 1);
            std::size_t lower_idx = static_cast<std::size_t>(std::floor(pos));
            std::size_t upper_idx = static_cast<std::size_t>(std::ceil(pos));
            
            if (lower_idx == upper_idx) {
                result.push_back(sorted_data[lower_idx]);
            } else {
                double weight = pos - lower_idx;
                double interpolated = sorted_data[lower_idx] * (1.0 - weight) + 
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
    double sum = 0.0;
    for (double x : data) {
        if (!std::isfinite(x)) {
            throw std::invalid_argument("Data contains non-finite values");
        }
        sum += x;
    }
    double mean = sum / n;
    
    // Calculate central moments
    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (double x : data) {
        double diff = x - mean;
        double diff2 = diff * diff;
        double diff3 = diff2 * diff;
        double diff4 = diff3 * diff;
        
        m2 += diff2;
        m3 += diff3;
        m4 += diff4;
    }
    
    m2 /= n;
    m3 /= n;
    m4 /= n;
    
    // Calculate variance (sample variance with Bessel's correction)
    double variance = (n > 1) ? (m2 * n) / (n - 1) : m2;
    
    // Calculate skewness and kurtosis
    double skewness = std::numeric_limits<double>::quiet_NaN();
    double kurtosis = std::numeric_limits<double>::quiet_NaN();
    
    if (m2 > constants::precision::ZERO) {
        double sigma = std::sqrt(m2);
        double sigma3 = sigma * sigma * sigma;
        double sigma4 = sigma3 * sigma;
        
        skewness = m3 / sigma3;
        kurtosis = (m4 / sigma4) - constants::statistical::thresholds::EXCESS_KURTOSIS_OFFSET; // Excess kurtosis
    }
    
    return {mean, variance, skewness, kurtosis};
}

bool validate_fitting_data(std::span<const double> data) noexcept {
    return std::all_of(data.begin(), data.end(), [](double x) { return std::isfinite(x); });
}

// =============================================================================
// GOODNESS-OF-FIT TESTING
// =============================================================================

double calculate_ks_statistic(const std::vector<double>& data, const DistributionBase& dist) noexcept {
    if (data.empty()) {
        return 0.0;
    }
    
    // Create a copy of the data for sorting
    std::vector<double> sorted_data(data);
    std::sort(sorted_data.begin(), sorted_data.end());
    
    const auto n = static_cast<double>(sorted_data.size());
    double max_diff = 0.0;
    
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

double calculate_ad_statistic(const std::vector<double>& data, const DistributionBase& dist) noexcept {
    if (data.empty()) {
        return 0.0;
    }
    
    // Create a copy of the data for sorting
    std::vector<double> sorted_data(data);
    std::sort(sorted_data.begin(), sorted_data.end());
    
    const auto n = static_cast<double>(sorted_data.size());
    double ad_sum = 0.0;
    
    // Calculate Anderson-Darling statistic
    for (std::size_t i = 0; i < sorted_data.size(); ++i) {
        double F_xi = dist.getCumulativeProbability(sorted_data[i]);
        double F_xi_rev = dist.getCumulativeProbability(sorted_data[sorted_data.size() - 1 - i]);
        
        // Clamp F values to avoid log(0)
        F_xi = std::max(constants::precision::ZERO, std::min(1.0 - constants::precision::ZERO, F_xi));
        F_xi_rev = std::max(constants::precision::ZERO, std::min(1.0 - constants::precision::ZERO, F_xi_rev));
        
        double weight = 2.0 * static_cast<double>(i + 1) - 1.0;
        ad_sum += weight * (std::log(F_xi) + std::log(1.0 - F_xi_rev));
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
    if (simd::VectorOps::should_use_simd(size)) {
        simd::VectorOps::vector_erf(input.data(), output.data(), size);
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

void vector_beta_i(std::span<const double> x_values, double a, double b, std::span<double> output) noexcept {
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
    if (simd::VectorOps::should_use_simd(size)) {
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

void vector_lbeta(std::span<const double> a_values, std::span<const double> b_values, std::span<double> output) noexcept {
    if (a_values.size() != b_values.size() || a_values.size() != output.size() || a_values.empty()) {
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
    return simd::VectorOps::should_use_simd(size);
}

std::size_t vectorized_math_threshold() noexcept {
    // Return the minimum size for SIMD operations
    return simd::VectorOps::min_simd_size();
}

// =============================================================================
// STATISTICAL DISTRIBUTION FUNCTIONS
// =============================================================================

double normal_cdf(double z) noexcept {
    // Standard normal CDF using error function
    return constants::math::HALF * (constants::math::ONE + erf(z * constants::math::INV_SQRT_2));
}

double inverse_normal_cdf(double p) noexcept {
    // Inverse standard normal CDF using inverse error function
    if (p <= 0.0 || p >= 1.0) {
        if (p == 0.0) return -std::numeric_limits<double>::infinity();
        if (p == 1.0) return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // Use relationship: inverse_normal_cdf(p) = sqrt(2) * erf_inv(2*p - 1)
    double erf_arg = constants::math::TWO * p - constants::math::ONE;
    return constants::math::SQRT_2 * erf_inv(erf_arg);
}

double t_cdf(double t, double df) noexcept {
    // Student's t-distribution CDF using regularized incomplete beta function
    if (df <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    if (std::isinf(t)) {
        return (t > 0.0) ? 1.0 : 0.0;
    }
    
    if (t == 0.0) {
        return constants::math::HALF;
    }
    
    // Use relationship with incomplete beta function:
    // t_cdf(t, df) = 1/2 + (t/sqrt(df)) * B(1/2, df/2) / B(1/2, df/2)
    // This is simplified using the symmetry of t-distribution
    
    double x = df / (df + t * t);
    double result = beta_i(x, constants::math::HALF * df, constants::math::HALF);
    
    if (t > 0.0) {
        return constants::math::ONE - constants::math::HALF * result;
    } else {
        return constants::math::HALF * result;
    }
}

double inverse_t_cdf(double p, double df) noexcept {
    // Inverse t-distribution CDF using iterative methods
    if (p <= 0.0 || p >= 1.0 || df <= 0.0) {
        if (p == 0.0) return -std::numeric_limits<double>::infinity();
        if (p == 1.0) return std::numeric_limits<double>::infinity();
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    if (p == constants::math::HALF) {
        return 0.0;
    }
    
    // Use approximate initial guess from normal distribution
    double z = inverse_normal_cdf(p);
    
    // For large degrees of freedom, t-distribution approaches normal
    if (df > 100.0) {
        return z;
    }
    
    // Newton-Raphson iteration to refine the estimate
    double t = z; // Initial guess
    const int max_iterations = 100;
    const double tolerance = constants::precision::DEFAULT_TOLERANCE;
    
    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = t_cdf(t, df);
        double error = cdf_val - p;
        
        if (std::abs(error) < tolerance) {
            break;
        }
        
        // Calculate derivative (PDF)
        double pdf_val = std::exp(lgamma((df + 1.0) * constants::math::HALF) - 
                                 lgamma(df * constants::math::HALF) - 
                                 constants::math::HALF * std::log(df * constants::math::PI)) *
                        std::pow(1.0 + t * t / df, -(df + 1.0) * constants::math::HALF);
        
        if (pdf_val <= 0.0) {
            break; // Avoid division by zero
        }
        
        t -= error / pdf_val;
    }
    
    return t;
}

double chi_squared_cdf(double x, double df) noexcept {
    // Chi-squared CDF using regularized incomplete gamma function
    if (x < 0.0 || df <= 0.0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    // Chi-squared with df degrees of freedom is Gamma(df/2, 2)
    // CDF = P(df/2, x/2) = regularized incomplete gamma function
    return gamma_p(df * constants::math::HALF, x * constants::math::HALF);
}

double inverse_chi_squared_cdf(double p, double df) noexcept {
    // Inverse chi-squared CDF using iterative methods
    if (p < 0.0 || p > 1.0 || df <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    if (p == 0.0) {
        return 0.0;
    }
    
    if (p == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    // Initial guess using Wilson-Hilferty approximation
    double h = constants::math::TWO / (9.0 * df);
    double z = inverse_normal_cdf(p);
    double initial_guess = df * std::pow(1.0 - h + z * std::sqrt(h), 3);
    
    // Ensure initial guess is positive
    if (initial_guess <= 0.0) {
        initial_guess = df; // Use mean as fallback
    }
    
    // Newton-Raphson iteration
    double x = initial_guess;
    const int max_iterations = 100;
    const double tolerance = constants::precision::DEFAULT_TOLERANCE;
    
    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = chi_squared_cdf(x, df);
        double error = cdf_val - p;
        
        if (std::abs(error) < tolerance) {
            break;
        }
        
        // Calculate derivative (PDF)
        double pdf_val = std::exp((df * constants::math::HALF - 1.0) * std::log(x) - 
                                 x * constants::math::HALF - 
                                 lgamma(df * constants::math::HALF) - 
                                 df * constants::math::HALF * constants::math::LN2);
        
        if (pdf_val <= 0.0) {
            break; // Avoid division by zero
        }
        
        double delta = error / pdf_val;
        x = std::max(constants::precision::ZERO, x - delta); // Ensure x stays positive
    }
    
    return x;
}

double f_cdf(double x, double df1, double df2) noexcept {
    // F-distribution CDF using regularized incomplete beta function
    if (x < 0.0 || df1 <= 0.0 || df2 <= 0.0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    // F-distribution relationship with beta function:
    // F_cdf(x, df1, df2) = I_y(df1/2, df2/2) where y = (df1*x)/(df1*x + df2)
    double y = (df1 * x) / (df1 * x + df2);
    return beta_i(y, df1 * constants::math::HALF, df2 * constants::math::HALF);
}

double inverse_f_cdf(double p, double df1, double df2) noexcept {
    // Inverse F-distribution CDF using iterative methods
    if (p < 0.0 || p > 1.0 || df1 <= 0.0 || df2 <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    if (p == 0.0) {
        return 0.0;
    }
    
    if (p == 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    // Initial guess using approximation
    double z = inverse_normal_cdf(p);
    double initial_guess = std::max(1.0, 1.0 + z * std::sqrt(2.0 / df1));
    
    // Newton-Raphson iteration
    double x = initial_guess;
    const int max_iterations = 100;
    const double tolerance = constants::precision::DEFAULT_TOLERANCE;
    
    for (int i = 0; i < max_iterations; ++i) {
        double cdf_val = f_cdf(x, df1, df2);
        double error = cdf_val - p;
        
        if (std::abs(error) < tolerance) {
            break;
        }
        
        // Calculate derivative (PDF)
        double log_pdf = lgamma((df1 + df2) * constants::math::HALF) - 
                        lgamma(df1 * constants::math::HALF) - 
                        lgamma(df2 * constants::math::HALF) +
                        (df1 * constants::math::HALF - 1.0) * std::log(x) -
                        (df1 + df2) * constants::math::HALF * std::log(1.0 + df1 * x / df2) +
                        df1 * constants::math::HALF * std::log(df1 / df2);
        
        double pdf_val = std::exp(log_pdf);
        
        if (pdf_val <= 0.0) {
            break; // Avoid division by zero
        }
        
        double delta = error / pdf_val;
        x = std::max(constants::precision::ZERO, x - delta); // Ensure x stays positive
    }
    
    return x;
}

} // namespace math
} // namespace libstats

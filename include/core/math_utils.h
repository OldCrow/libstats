#pragma once

#include <cmath>
#include <functional>
#include <span>
#include <concepts>
#include <algorithm>
#include "constants.h"
#include "safety.h"
#include "../platform/simd.h"

/**
 * @file math_utils.h
 * @brief Mathematical utilities and special functions for libstats
 * 
 * This header provides common mathematical functions used across 
 * different distributions, including special functions, numerical
 * integration, and optimization routines.
 */

namespace libstats {
namespace math {

// =============================================================================
// C++20 CONCEPTS FOR TYPE SAFETY
// =============================================================================

/**
 * @brief Concept for floating-point types suitable for mathematical operations
 */
template<typename T>
concept FloatingPoint = std::floating_point<T> && requires(T t) {
    std::isfinite(t);
    std::isnan(t);
    std::isinf(t);
};

/**
 * @brief Concept for callable objects that can be used as mathematical functions
 */
template<typename F, typename T>
concept MathFunction = std::invocable<F, T> && 
                       std::convertible_to<std::invoke_result_t<F, T>, double>;

// =============================================================================
// SPECIAL MATHEMATICAL FUNCTIONS
// =============================================================================

/**
 * @brief Error function erf(x) with high precision
 * @param x Input value
 * @return erf(x)
 */
[[nodiscard]] double erf(double x) noexcept;

/**
 * @brief Complementary error function erfc(x) with high precision
 * @param x Input value
 * @return erfc(x) = 1 - erf(x)
 */
[[nodiscard]] double erfc(double x) noexcept;

/**
 * @brief Inverse error function erf^-1(x)
 * @param x Input value in (-1, 1)
 * @return erf^-1(x)
 */
[[nodiscard]] double erf_inv(double x) noexcept;

/**
 * @brief Natural logarithm of the gamma function ln(Γ(x))
 * @param x Input value (x > 0)
 * @return ln(Γ(x))
 */
[[nodiscard]] double lgamma(double x) noexcept;

/**
 * @brief Regularized incomplete gamma function P(a,x) = γ(a,x)/Γ(a)
 * @param a Shape parameter (a > 0)
 * @param x Input value (x >= 0)
 * @return P(a,x)
 */
[[nodiscard]] double gamma_p(double a, double x) noexcept;

/**
 * @brief Regularized incomplete gamma function Q(a,x) = 1 - P(a,x)
 * @param a Shape parameter (a > 0)
 * @param x Input value (x >= 0)
 * @return Q(a,x)
 */
[[nodiscard]] double gamma_q(double a, double x) noexcept;

/**
 * @brief Regularized incomplete beta function I_x(a,b)
 * @param x Input value in [0,1]
 * @param a First shape parameter (a > 0)
 * @param b Second shape parameter (b > 0)
 * @return I_x(a,b)
 */
[[nodiscard]] double beta_i(double x, double a, double b) noexcept;

/**
 * @brief Natural logarithm of the beta function ln(B(a,b))
 * @param a First parameter (a > 0)
 * @param b Second parameter (b > 0)
 * @return ln(B(a,b))
 */
[[nodiscard]] double lbeta(double a, double b) noexcept;

// =============================================================================
// SIMD VECTORIZED SPECIAL FUNCTIONS
// =============================================================================

/**
 * @brief Vectorized error function computation using SIMD optimization
 * @param input Input values
 * @param output Output array for erf(input[i])
 * @param size Number of elements to process
 * @note Automatically selects optimal SIMD implementation based on CPU capabilities
 * @note For small arrays, falls back to scalar implementation to avoid overhead
 */
void vector_erf(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized complementary error function computation using SIMD optimization
 * @param input Input values
 * @param output Output array for erfc(input[i])
 * @param size Number of elements to process
 */
void vector_erfc(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized regularized incomplete gamma function P(a,x) using SIMD
 * @param a Shape parameter (constant for all x values)
 * @param x_values Input x values
 * @param output Output array for gamma_p(a, x_values[i])
 * @note Uses SIMD optimization for the series expansion when beneficial
 */
void vector_gamma_p(double a, std::span<const double> x_values, std::span<double> output) noexcept;

/**
 * @brief Vectorized regularized incomplete gamma function Q(a,x) using SIMD
 * @param a Shape parameter (constant for all x values)
 * @param x_values Input x values
 * @param output Output array for gamma_q(a, x_values[i])
 */
void vector_gamma_q(double a, std::span<const double> x_values, std::span<double> output) noexcept;

/**
 * @brief Vectorized regularized incomplete beta function I_x(a,b) using SIMD
 * @param x_values Input x values in [0,1]
 * @param a First shape parameter (constant for all x values)
 * @param b Second shape parameter (constant for all x values)
 * @param output Output array for beta_i(x_values[i], a, b)
 */
void vector_beta_i(std::span<const double> x_values, double a, double b, std::span<double> output) noexcept;

/**
 * @brief Vectorized natural logarithm of gamma function using SIMD
 * @param input Input values (x > 0)
 * @param output Output array for lgamma(input[i])
 */
void vector_lgamma(std::span<const double> input, std::span<double> output) noexcept;

/**
 * @brief Vectorized natural logarithm of beta function using SIMD
 * @param a_values First parameter values (a > 0)
 * @param b_values Second parameter values (b > 0)
 * @param output Output array for lbeta(a_values[i], b_values[i])
 * @note Requires a_values.size() == b_values.size() == output.size()
 */
void vector_lbeta(std::span<const double> a_values, std::span<const double> b_values, std::span<double> output) noexcept;

/**
 * @brief Check if vectorized operations should be used for given array size
 * @param size Number of elements to process
 * @return true if SIMD vectorization is beneficial for this size
 */
[[nodiscard]] bool should_use_vectorized_math(std::size_t size) noexcept;

/**
 * @brief Get minimum array size threshold for vectorized mathematical operations
 * @return Minimum number of elements where vectorization becomes beneficial
 */
[[nodiscard]] std::size_t vectorized_math_threshold() noexcept;

// =============================================================================
// NUMERICAL INTEGRATION
// =============================================================================

/**
 * @brief Adaptive Simpson's rule for numerical integration
 * @param func Function to integrate
 * @param lower_bound Lower integration bound
 * @param upper_bound Upper integration bound
 * @param tolerance Relative tolerance for convergence
 * @param max_depth Maximum recursion depth
 * @return Integral approximation
 */
template<MathFunction<double> F>
[[nodiscard]] double adaptive_simpson(
    F&& func, 
    double lower_bound, 
    double upper_bound, 
    double tolerance = constants::precision::DEFAULT_TOLERANCE,
    int max_depth = 20
) noexcept;

/**
 * @brief Gauss-Legendre quadrature for smooth functions
 * @param func Function to integrate
 * @param lower_bound Lower integration bound
 * @param upper_bound Upper integration bound
 * @param n_points Number of quadrature points (8, 16, 32, 64)
 * @return Integral approximation
 */
template<MathFunction<double> F>
[[nodiscard]] double gauss_legendre(
    F&& func,
    double lower_bound,
    double upper_bound,
    int n_points = 16
) noexcept;

// =============================================================================
// ROOT FINDING AND OPTIMIZATION
// =============================================================================

/**
 * @brief Newton-Raphson method for root finding
 * @param func Function for which to find root
 * @param derivative Derivative of the function
 * @param initial_guess Initial guess for the root
 * @param tolerance Convergence tolerance
 * @param max_iterations Maximum number of iterations
 * @return Root approximation
 */
template<MathFunction<double> F, MathFunction<double> DF>
[[nodiscard]] double newton_raphson(
    F&& func,
    DF&& derivative,
    double initial_guess,
    double tolerance = constants::precision::DEFAULT_TOLERANCE,
    int max_iterations = 100
) noexcept;

/**
 * @brief Brent's method for root finding (robust bracketing method)
 * @param func Function for which to find root
 * @param lower_bound Lower bracket (func(lower_bound) and func(upper_bound) must have opposite signs)
 * @param upper_bound Upper bracket
 * @param tolerance Convergence tolerance
 * @param max_iterations Maximum number of iterations
 * @return Root approximation
 */
template<MathFunction<double> F>
[[nodiscard]] double brent_root(
    F&& func,
    double lower_bound,
    double upper_bound,
    double tolerance = constants::precision::DEFAULT_TOLERANCE,
    int max_iterations = 100
) noexcept;

/**
 * @brief Golden section search for univariate optimization
 * @param func Function to minimize
 * @param lower_bound Lower search bound
 * @param upper_bound Upper search bound
 * @param tolerance Convergence tolerance
 * @return Minimum location
 */
template<MathFunction<double> F>
[[nodiscard]] double golden_section_search(
    F&& func,
    double lower_bound,
    double upper_bound,
    double tolerance = constants::precision::DEFAULT_TOLERANCE
) noexcept;

// =============================================================================
// STATISTICAL DISTRIBUTION FUNCTIONS
// =============================================================================

/**
 * @brief Student's t-distribution CDF
 * @param t t-statistic value
 * @param df degrees of freedom (df > 0)
 * @return P(T <= t) where T ~ t(df)
 */
[[nodiscard]] double t_cdf(double t, double df) noexcept;

/**
 * @brief Inverse Student's t-distribution CDF (quantile function)
 * @param p probability value in (0, 1)
 * @param df degrees of freedom (df > 0)
 * @return t such that P(T <= t) = p where T ~ t(df)
 */
[[nodiscard]] double inverse_t_cdf(double p, double df) noexcept;

/**
 * @brief Chi-squared distribution CDF
 * @param x value (x >= 0)
 * @param df degrees of freedom (df > 0)
 * @return P(X <= x) where X ~ χ²(df)
 */
[[nodiscard]] double chi_squared_cdf(double x, double df) noexcept;

/**
 * @brief Inverse chi-squared distribution CDF (quantile function)
 * @param p probability value in (0, 1)
 * @param df degrees of freedom (df > 0)
 * @return x such that P(X <= x) = p where X ~ χ²(df)
 */
[[nodiscard]] double inverse_chi_squared_cdf(double p, double df) noexcept;

/**
 * @brief Standard normal distribution CDF
 * @param z z-score value
 * @return P(Z <= z) where Z ~ N(0,1)
 */
[[nodiscard]] double normal_cdf(double z) noexcept;

/**
 * @brief Inverse standard normal distribution CDF (quantile function)
 * @param p probability value in (0, 1)
 * @return z such that P(Z <= z) = p where Z ~ N(0,1)
 */
[[nodiscard]] double inverse_normal_cdf(double p) noexcept;

/**
 * @brief F-distribution CDF
 * @param x value (x >= 0)
 * @param df1 numerator degrees of freedom (df1 > 0)
 * @param df2 denominator degrees of freedom (df2 > 0)
 * @return P(F <= x) where F ~ F(df1, df2)
 */
[[nodiscard]] double f_cdf(double x, double df1, double df2) noexcept;

/**
 * @brief Inverse F-distribution CDF (quantile function)
 * @param p probability value in (0, 1)
 * @param df1 numerator degrees of freedom (df1 > 0)
 * @param df2 denominator degrees of freedom (df2 > 0)
 * @return x such that P(F <= x) = p where F ~ F(df1, df2)
 */
[[nodiscard]] double inverse_f_cdf(double p, double df1, double df2) noexcept;

/**
 * @brief Gamma distribution CDF
 * @param x value (x >= 0)
 * @param shape shape parameter (alpha > 0)
 * @param scale scale parameter (beta > 0) 
 * @return P(X <= x) where X ~ Gamma(shape, scale)
 */
[[nodiscard]] double gamma_cdf(double x, double shape, double scale) noexcept;

/**
 * @brief Inverse gamma distribution CDF (quantile function)
 * @param p probability value in (0, 1)
 * @param shape shape parameter (alpha > 0)
 * @param scale scale parameter (beta > 0) 
 * @return x such that P(X <= x) = p where X ~ Gamma(shape, scale)
 */
[[nodiscard]] double gamma_inverse_cdf(double p, double shape, double scale) noexcept;

// =============================================================================
// STATISTICAL UTILITIES
// =============================================================================

/**
 * @brief Calculate empirical CDF from sorted data
 * @param data Sorted data vector
 * @return Vector of empirical CDF values
 */
[[nodiscard]] std::vector<double> empirical_cdf(std::span<const double> data);

/**
 * @brief Calculate quantiles from sorted data
 * @param data Sorted data vector  
 * @param quantiles Quantile levels to calculate
 * @return Vector of quantile values
 */
[[nodiscard]] std::vector<double> calculate_quantiles(
    std::span<const double> data,
    std::span<const double> quantiles
);

/**
 * @brief Calculate sample moments (mean, variance, skewness, kurtosis)
 * @param data Data vector
 * @return Array [mean, variance, skewness, excess_kurtosis]
 */
[[nodiscard]] std::array<double, 4> sample_moments(std::span<const double> data);

/**
 * @brief Validate data for statistical fitting
 * @param data Data to validate
 * @return true if data is valid for fitting
 */
[[nodiscard]] bool validate_fitting_data(std::span<const double> data) noexcept;

// =============================================================================
// GOODNESS-OF-FIT TESTING
// =============================================================================

} // namespace math
} // namespace libstats

// Forward declaration
namespace libstats {
    class DistributionBase;
}

namespace libstats {
namespace math {

/**
 * @brief Kolmogorov-Smirnov test statistic calculation
 * @param data Vector of data to test
 * @param dist Distribution to test against
 * @return KS test statistic
 */
[[nodiscard]] double calculate_ks_statistic(const std::vector<double>& data, const DistributionBase& dist) noexcept;

/**
 * @brief Anderson-Darling test statistic calculation
 * @param data Vector of data to test
 * @param dist Distribution to test against
 * @return AD test statistic
 */
[[nodiscard]] double calculate_ad_statistic(const std::vector<double>& data, const DistributionBase& dist) noexcept;

// =============================================================================
// NUMERICAL STABILITY UTILITIES
// =============================================================================

/**
 * @brief Compute log(1 + exp(x)) with numerical stability
 * @param x Input value
 * @return log(1 + exp(x))
 */
[[nodiscard]] inline double log1pexp(double x) noexcept {
    if (x > constants::thresholds::LOG1PEXP_LARGE_THRESHOLD) [[likely]] {
        return x;  // exp(x) dominates
    } else if (x > constants::thresholds::LOG1PEXP_SMALL_THRESHOLD) {
        return std::log1p(std::exp(x));
    } else [[unlikely]] {
        return std::exp(x);  // 1 + exp(x) ≈ 1
    }
}

/**
 * @brief Compute log(exp(x) - 1) with numerical stability
 * @param x Input value (x > 0)
 * @return log(exp(x) - 1)
 */
[[nodiscard]] inline double logexpm1(double x) noexcept {
    if (x > constants::thresholds::LOG1PEXP_LARGE_THRESHOLD) [[likely]] {
        return x;  // exp(x) dominates
    } else {
        return std::log(std::expm1(x));
    }
}

/**
 * @brief Compute log(x + y) given log(x) and log(y) with numerical stability
 * @param log_x log(x)
 * @param log_y log(y)
 * @return log(x + y)
 */
[[nodiscard]] inline double log_sum_exp(double log_x, double log_y) noexcept {
    if (log_x > log_y) [[likely]] {
        return log_x + std::log1p(std::exp(log_y - log_x));
    } else {
        return log_y + std::log1p(std::exp(log_x - log_y));
    }
}

/**
 * @brief Check if a floating-point value is finite and safe for computation
 * @param x Value to check
 * @return true if x is finite and safe
 */
template<typename T> requires FloatingPoint<T>
[[nodiscard]] constexpr bool is_safe_float(T x) noexcept {
    return std::isfinite(x) && 
           std::abs(x) < constants::thresholds::MAX_DISTRIBUTION_PARAMETER;
}

/**
 * @brief Clamp value to safe range for floating-point computation
 * @param x Value to clamp
 * @param min_val Minimum allowed value
 * @param max_val Maximum allowed value
 * @return Clamped value
 */
template<typename T> requires FloatingPoint<T>
[[nodiscard]] constexpr T clamp_safe(T x, T min_val, T max_val) noexcept {
    if (std::isnan(x)) [[unlikely]] {
        return min_val;
    }
    return std::clamp(x, min_val, max_val);
}

/**
 * @brief Safe division with zero checking
 * @param numerator Numerator
 * @param denominator Denominator
 * @param default_value Value to return if denominator is zero
 * @return Safe division result
 */
template<typename T> requires FloatingPoint<T>
[[nodiscard]] constexpr T safe_divide(T numerator, T denominator, T default_value = T{0}) noexcept {
    if (std::abs(denominator) < constants::precision::ZERO || std::isnan(denominator)) {
        return default_value;
    }
    return numerator / denominator;
}

/**
 * @brief Compute log(sum(exp(x_i))) with numerical stability (LogSumExp)
 * @param values Vector of log values
 * @return log(sum(exp(x_i)))
 */
[[nodiscard]] inline double log_sum_exp(std::span<const double> values) noexcept {
    if (values.empty()) return constants::probability::MIN_LOG_PROBABILITY;
    
    // Find maximum value for numerical stability
    double max_val = *std::max_element(values.begin(), values.end());
    
    // Handle edge case where all values are -inf
    if (std::isinf(max_val) && max_val < 0) {
        return constants::probability::MIN_LOG_PROBABILITY;
    }
    
    // Compute log(sum(exp(x_i - max_val))) + max_val
    double sum_exp = 0.0;
    for (double val : values) {
        if (std::isfinite(val)) {
            sum_exp += std::exp(val - max_val);
        }
    }
    
    return safety::safe_log(sum_exp) + max_val;
}

/**
 * @brief Compute stable weighted average in log space
 * @param log_values Vector of log values
 * @param log_weights Vector of log weights
 * @return Weighted average in log space
 */
[[nodiscard]] inline double log_weighted_average(std::span<const double> log_values, 
                                                 std::span<const double> log_weights) noexcept {
    if (log_values.size() != log_weights.size() || log_values.empty()) {
        return constants::probability::MIN_LOG_PROBABILITY;
    }
    
    // Compute log(sum(w_i * v_i)) and log(sum(w_i))
    std::vector<double> log_weighted_values(log_values.size());
    for (std::size_t i = 0; i < log_values.size(); ++i) {
        log_weighted_values[i] = log_values[i] + log_weights[i];
    }
    
    double log_sum_weighted = log_sum_exp(log_weighted_values);
    double log_sum_weights = log_sum_exp(log_weights);
    
    return log_sum_weighted - log_sum_weights;
}

/**
 * @brief Check numerical condition of a computation
 * @param value Result value to check
 * @param context Description of the computation
 * @return True if value is numerically sound
 */
[[nodiscard]] inline bool check_numerical_condition(double value, const std::string& context = "computation") noexcept {
    if (std::isnan(value)) {
        // In production, we might want to log this instead of throwing
        // For now, the context could be used for debugging/logging purposes
        [[maybe_unused]] auto _ = context; // Acknowledge parameter to avoid warning in non-debug builds
        return false;
    }
    if (std::isinf(value)) {
        [[maybe_unused]] auto _ = context;
        return false;
    }
    if (std::abs(value) > constants::thresholds::MAX_DISTRIBUTION_PARAMETER) {
        [[maybe_unused]] auto _ = context;
        return false;
    }
    return true;
}

/**
 * @brief Adaptive precision scaling based on data characteristics
 * @param base_tolerance Base tolerance
 * @param data_range Range of data values
 * @param problem_size Size of the problem
 * @return Scaled tolerance
 */
[[nodiscard]] inline double adaptive_tolerance(double base_tolerance, 
                                              double data_range, 
                                              std::size_t problem_size) noexcept {
    // Scale tolerance based on data range
    double range_factor = std::max(constants::math::ONE, std::log10(std::max(constants::math::ONE, data_range)));
    
    // Scale tolerance based on problem size
    double size_factor = std::max(constants::math::ONE, std::log10(static_cast<double>(problem_size)));
    
    return base_tolerance * range_factor * size_factor;
}

//==============================================================================
// NUMERICAL DIAGNOSTICS
//==============================================================================

/**
 * @brief Diagnostic information about numerical computations
 */
struct NumericalDiagnostics {
    bool has_nan = false;
    bool has_inf = false;
    bool has_underflow = false;
    bool has_overflow = false;
    double min_value = std::numeric_limits<double>::max();
    double max_value = std::numeric_limits<double>::lowest();
    double condition_estimate = constants::math::ONE;
    std::size_t problem_size = 0;
    std::string recommendations;
};

/**
 * @brief Analyze numerical properties of a data vector
 * @param data Vector to analyze
 * @param name Name for reporting
 * @return Diagnostic report
 */
[[nodiscard]] inline NumericalDiagnostics analyze_vector(std::span<const double> data, 
                                                         const std::string& name = "vector") {
    NumericalDiagnostics diag;
    diag.problem_size = data.size();
    
    if (data.empty()) {
        diag.recommendations = "Empty " + name + " - no analysis possible";
        return diag;
    }
    
    for (double value : data) {
        if (std::isnan(value)) {
            diag.has_nan = true;
        } else if (std::isinf(value)) {
            diag.has_inf = true;
        } else {
            diag.min_value = std::min(diag.min_value, value);
            diag.max_value = std::max(diag.max_value, value);
            
            if (std::abs(value) < constants::precision::ZERO) {
                diag.has_underflow = true;
            }
            if (std::abs(value) > constants::thresholds::MAX_DISTRIBUTION_PARAMETER) {
                diag.has_overflow = true;
            }
        }
    }
    
    // Estimate condition number (simplified)
    if (diag.min_value > 0 && diag.max_value > 0) {
        diag.condition_estimate = diag.max_value / diag.min_value;
    }
    
    // Generate recommendations
    if (diag.has_nan) {
        diag.recommendations += "NaN values detected - check input data; ";
    }
    if (diag.has_inf) {
        diag.recommendations += "Infinite values detected - potential overflow; ";
    }
    if (diag.condition_estimate > constants::thresholds::HIGH_CONDITION_NUMBER_THRESHOLD) {
        diag.recommendations += "High condition number - numerical instability likely; ";
    }
    if (diag.has_underflow) {
        diag.recommendations += "Underflow detected - consider log-space computation; ";
    }
    if (diag.has_overflow) {
        diag.recommendations += "Overflow detected - consider scaling or log-space computation; ";
    }
    
    if (diag.recommendations.empty()) {
        diag.recommendations = "Numerical properties appear healthy";
    }
    
    return diag;
}

} // namespace math
} // namespace libstats

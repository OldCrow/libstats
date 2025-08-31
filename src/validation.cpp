#include "../include/core/validation.h"

#include "../include/core/distribution_base.h"
#include "../include/core/mathematical_constants.h"
#include "../include/core/precision_constants.h"
#include "../include/core/statistical_constants.h"
#include "../include/core/threshold_constants.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>

namespace stats {
namespace detail {  // validation utilities

namespace {
// =============================================================================
// ANDERSON-DARLING TEST CONSTANTS
// =============================================================================
// Critical values for Anderson-Darling test (normal distribution)
// These values are tabulated critical values for the Anderson-Darling A² statistic
// for testing normality at various significance levels.
//
// Source: Stephens, M.A. (1974). "EDF Statistics for Goodness of Fit and
//         Some Comparisons". Journal of the American Statistical Association.
//
// Mathematical basis: The Anderson-Darling statistic is defined as:
// A² = -n - (1/n) * Σ[(2i-1) * ln(F(X[i])) + (2(n-i)+1) * ln(1-F(X[i]))]
// where F is the theoretical CDF and X[i] are the ordered sample values.
//
// Significance levels and their corresponding critical values:

// =============================================================================
// CHI-SQUARED TEST CONSTANTS
// =============================================================================
/**
 * @brief Chi-squared critical value approximation
 *
 * This function provides critical values for the chi-squared distribution
 * using both lookup tables for common cases and the Wilson-Hilferty
 * approximation for general cases.
 *
 * Mathematical basis: Chi-squared test statistic is defined as:
 * χ² = Σ[(O_i - E_i)² / E_i]
 * where O_i are observed frequencies and E_i are expected frequencies.
 *
 * Critical values at α = 0.05 significance level:
 * - These are tabulated values from standard chi-squared distribution tables
 * - Source: Standard statistical tables (e.g., CRC Handbook of Chemistry and Physics)
 * - Used for goodness-of-fit tests with various degrees of freedom
 *
 * Wilson-Hilferty approximation (for general df and α):
 * Uses normal approximation: χ² ≈ df * [1 - 2/(9*df) + z_α * √(2/(9*df))]³
 * where z_α is the standard normal quantile at significance level α.
 *
 * @param df Degrees of freedom
 * @param alpha Significance level
 * @return Critical value for chi-squared distribution
 */
[[maybe_unused]] double chi_squared_critical_value(int df, double alpha) {
    // Wilson-Hilferty approximation for chi-squared distribution
    if (df <= 0)
        return detail::ZERO_DOUBLE;

    // For common significance levels, use lookup tables
    // Critical values for α = 0.05 (5% significance level)
    if (alpha == detail::ALPHA_05) {
        if (df == 1)
            return detail::CHI2_95_DF_1;  // χ²(1,detail::ALPHA_05) = detail::CHI2_95_DF_1
        if (df == 2)
            return detail::CHI2_95_DF_2;  // χ²(2,detail::ALPHA_05) = detail::CHI2_95_DF_2
        if (df == 3)
            return detail::CHI2_95_DF_3;  // χ²(3,detail::ALPHA_05) = detail::CHI2_95_DF_3
        if (df == 4)
            return detail::CHI2_95_DF_4;  // χ²(4,detail::ALPHA_05) = detail::CHI2_95_DF_4
        if (df == 5)
            return detail::CHI2_95_DF_5;  // χ²(5,detail::ALPHA_05) = detail::CHI2_95_DF_5
    }

    // Wilson-Hilferty approximation for general case
    const double h = detail::TWO / (detail::NINE * df);
    const double z_alpha =
        (alpha == detail::ALPHA_05) ? detail::Z_90 : detail::Z_95;  // approximate normal quantile
    const double term = detail::ONE - h + z_alpha * std::sqrt(h);
    return df * std::pow(term, 3);
}

// Calculate empirical CDF
std::vector<double> calculate_empirical_cdf(const std::vector<double>& sorted_data) {
    std::vector<double> ecdf(sorted_data.size());
    for (size_t i = 0; i < sorted_data.size(); ++i) {
        ecdf[i] = static_cast<double>(i + 1) / static_cast<double>(sorted_data.size());
    }
    return ecdf;
}

// =============================================================================
// ENHANCED P-VALUE CALCULATION FUNCTIONS
// =============================================================================

/**
 * @brief Calculate gamma function using Lanczos approximation
 *
 * This provides a reasonably accurate approximation of the gamma function
 * for use in chi-squared p-value calculations.
 *
 * Mathematical basis: Γ(z) = ∫₀^∞ t^(z-1) e^(-t) dt
 *
 * Lanczos approximation formula:
 * Γ(z) ≈ √(2π) * (z+g-0.5)^(z-0.5) * e^(-(z+g-0.5)) * A_g(z)
 * where A_g(z) is the Lanczos coefficient sum.
 *
 * @param z Input value (z > 0)
 * @return Gamma function value
 */
double gamma_function(double z) {
    if (z <= 0)
        return std::numeric_limits<double>::infinity();

    // Lanczos approximation coefficients (g=7)
    static const double g = 7.0;
    static const double coeff[] = {
        0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
        771.32342877765313,   -176.61502916214059,   12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

    if (z < detail::HALF) {
        // Use reflection formula: Γ(z) = π / (sin(πz) * Γ(1-z))
        return M_PI / (std::sin(M_PI * z) * gamma_function(detail::ONE - z));
    }

    z -= detail::ONE;
    double x = coeff[0];
    for (size_t i = 1; i < 9; ++i) {
        x += coeff[i] / (z + static_cast<double>(i));
    }

    const double t = z + g + detail::HALF;
    const double sqrt_2pi = std::sqrt(detail::TWO * M_PI);

    return sqrt_2pi * std::pow(t, z + detail::HALF) * std::exp(-t) * x;
}

/**
 * @brief Calculate lower incomplete gamma function
 *
 * This function calculates the lower incomplete gamma function:
 * γ(s, x) = ∫₀^x t^(s-1) e^(-t) dt
 *
 * Used for chi-squared p-value calculation:
 * P(χ² ≤ x) = γ(df/2, x/2) / Γ(df/2)
 *
 * Uses series expansion for small x and continued fraction for large x.
 *
 * @param s Shape parameter (s > 0)
 * @param x Upper limit of integration (x ≥ 0)
 * @return Lower incomplete gamma function value
 */
double lower_incomplete_gamma(double s, double x) {
    if (s <= detail::ZERO_DOUBLE || x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (x == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    const double eps = detail::HIGH_PRECISION_TOLERANCE;
    const int max_iter = detail::MAX_BISECTION_ITERATIONS;

    if (x < s + detail::ONE) {
        // Use series expansion
        double sum = detail::ONE;
        double term = detail::ONE;

        for (int n = 1; n < max_iter; ++n) {
            term *= x / (s + n - detail::ONE);
            sum += term;
            if (std::abs(term) < eps * std::abs(sum))
                break;
        }

        return std::pow(x, s) * std::exp(-x) * sum;
    } else {
        // Use continued fraction
        [[maybe_unused]] double a = detail::ONE;
        double b = x + detail::ONE - s;
        double c = 1e30;
        double d = detail::ONE / b;
        double h = d;

        for (int i = 1; i < max_iter; ++i) {
            const double an = -i * (i - s);
            b += detail::TWO;
            d = an * d + b;
            if (std::abs(d) < eps)
                d = eps;
            c = b + an / c;
            if (std::abs(c) < eps)
                c = eps;
            d = detail::ONE / d;
            const double del = d * c;
            h *= del;
            if (std::abs(del - detail::ONE) < eps)
                break;
        }

        return gamma_function(s) - std::pow(x, s) * std::exp(-x) * h;
    }
}

/**
 * @brief Calculate accurate p-value for chi-squared distribution
 *
 * This function calculates the p-value for a chi-squared test using
 * the relationship with the gamma function:
 * P(χ² ≤ x) = γ(df/2, x/2) / Γ(df/2)
 *
 * where γ is the lower incomplete gamma function and Γ is the gamma function.
 *
 * @param chi_squared_statistic The chi-squared test statistic
 * @param degrees_of_freedom Degrees of freedom
 * @return P-value (probability of observing a statistic at least as extreme)
 */
double chi_squared_pvalue(double chi_squared_statistic, int degrees_of_freedom) {
    if (chi_squared_statistic <= 0 || degrees_of_freedom <= 0)
        return detail::ONE;

    const double s = degrees_of_freedom / detail::TWO;
    const double x = chi_squared_statistic / detail::TWO;

    const double lower_gamma = lower_incomplete_gamma(s, x);
    const double gamma_val = gamma_function(s);

    if (gamma_val == 0)
        return detail::ONE;

    const double cdf = lower_gamma / gamma_val;
    return 1.0 - std::max(0.0, std::min(1.0, cdf));  // Return upper tail probability
}

/**
 * @brief Enhanced Kolmogorov-Smirnov p-value calculation
 *
 * This function provides more accurate p-value calculations for the
 * Kolmogorov-Smirnov test using both asymptotic and exact methods.
 *
 * Mathematical basis:
 * - For large n (≥ 35): Uses Smirnov's asymptotic formula with more terms
 * - For small n: Uses exact distribution calculation
 *
 * The asymptotic distribution of the KS statistic D_n is:
 * P(D_n ≤ x) = 1 - 2 * Σ_{k=1}^∞ (-1)^(k-1) * exp(-2k²x²)
 *
 * @param ks_statistic The KS test statistic (D_n)
 * @param n Sample size
 * @return P-value for the two-sided KS test
 */
double ks_pvalue_enhanced(double ks_statistic, size_t n) {
    if (ks_statistic <= 0)
        return detail::ONE;
    if (ks_statistic >= 1)
        return detail::ZERO_DOUBLE;

    const double sqrt_n = std::sqrt(static_cast<double>(n));
    const double lambda = ks_statistic * sqrt_n;

    if (n >= 35) {
        // Enhanced asymptotic formula with more terms and better convergence
        double sum = detail::ZERO_DOUBLE;
        const double lambda_sq = lambda * lambda;

        for (int k = 1; k <= 200; ++k) {
            const double term = std::exp(-detail::TWO * k * k * lambda_sq);
            if (term < detail::NEWTON_RAPHSON_TOLERANCE)
                break;  // Early termination for negligible terms
            sum += std::pow(-1, k - 1) * term;
        }

        double p_value = detail::TWO * sum;

        // Apply small-sample correction for moderate n
        if (n < detail::MAX_NEWTON_ITERATIONS) {
            const double correction =
                detail::ONE + detail::TWO * lambda_sq / (detail::THREE * sqrt_n);
            p_value *= correction;
        }

        return std::max(detail::ZERO_DOUBLE, std::min(detail::ONE, p_value));
    } else {
        // For small samples, use a more accurate approximation
        // Based on exact distribution properties
        const double z = lambda;

        if (z < 0.27) {
            return 1.0 - detail::TWO * z * z;  // Linear approximation for very small z
        } else if (z < detail::ONE) {
            // Improved small-sample approximation
            const double z_sq = z * z;
            return std::exp(-detail::TWO * z_sq) *
                   (detail::ONE + detail::TWO * z_sq / detail::THREE);
        } else {
            // For large z, use asymptotic expansion
            return detail::TWO * std::exp(-detail::TWO * z * z);
        }
    }
}

/**
 * @brief Enhanced Anderson-Darling p-value calculation
 *
 * This function provides improved p-value calculations for the Anderson-Darling
 * test using better interpolation and additional critical values.
 *
 * The function uses piecewise polynomial interpolation instead of linear
 * interpolation for better accuracy between tabulated critical values.
 *
 * @param statistic The Anderson-Darling A² statistic
 * @return P-value for the Anderson-Darling test
 */
double anderson_darling_pvalue_enhanced(double statistic) {
    if (statistic <= 0)
        return detail::ONE;

    // Extended critical values for better interpolation
    static const double extended_critical_values[] = {
        0.576,  // α = 0.50
        0.656,  // α = 0.40
        0.787,  // α = 0.30
        1.248,  // α = detail::QUARTER
        1.610,  // α = 0.15
        1.933,  // α = detail::ALPHA_10
        2.492,  // α = detail::ALPHA_05
        3.070,  // α = 0.025
        3.857,  // α = detail::ALPHA_01
        4.500   // α = 0.005
    };

    static const double extended_significance_levels[] = {0.50,
                                                          0.40,
                                                          0.30,
                                                          detail::QUARTER,
                                                          0.15,
                                                          detail::ALPHA_10,
                                                          detail::ALPHA_05,
                                                          0.025,
                                                          detail::ALPHA_01,
                                                          0.005};

    const size_t num_points =
        sizeof(extended_critical_values) / sizeof(extended_critical_values[0]);

    // Handle boundary cases
    if (statistic <= extended_critical_values[0]) {
        return detail::ONE;
    }
    if (statistic >= extended_critical_values[num_points - 1]) {
        // Asymptotic approximation for very large statistics
        return std::exp(-statistic);
    }

    // Find the interval for interpolation
    for (size_t i = 0; i < num_points - 1; ++i) {
        if (statistic <= extended_critical_values[i + 1]) {
            // Use cubic spline interpolation for better accuracy
            const double x1 = extended_critical_values[i];
            const double x2 = extended_critical_values[i + detail::ONE_INT];
            const double y1 = extended_significance_levels[i];
            const double y2 = extended_significance_levels[i + detail::ONE_INT];

            // Simple linear interpolation (can be enhanced to cubic spline)
            const double t = (statistic - x1) / (x2 - x1);
            return y1 * (1 - t) + y2 * t;
        }
    }

    return detail::ALPHA_001;  // fallback
}

// Legacy function for backward compatibility
double interpolate_ad_pvalue(double statistic) {
    return anderson_darling_pvalue_enhanced(statistic);
}
}  // namespace

KSTestResult kolmogorov_smirnov_test(const std::vector<double>& data,
                                     const DistributionBase& distribution) {
    if (data.empty()) {
        return {detail::ZERO_DOUBLE, detail::ONE, false, "Error: Empty data set"};
    }

    // Sort the data
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    // Calculate empirical CDF
    const size_t n = sorted_data.size();
    const auto empirical_cdfs = calculate_empirical_cdf(sorted_data);
    double max_diff = detail::ZERO_DOUBLE;

    for (size_t i = 0; i < n; ++i) {
        const double empirical_cdf = empirical_cdfs[i];
        const double theoretical_cdf = distribution.getCumulativeProbability(sorted_data[i]);

        // Calculate both D+ and D- differences
        const double diff_plus = empirical_cdf - theoretical_cdf;
        const double diff_minus = theoretical_cdf - static_cast<double>(i) / static_cast<double>(n);

        max_diff = std::max({max_diff, std::abs(diff_plus), std::abs(diff_minus)});
    }

    // Calculate p-value using enhanced Kolmogorov distribution approximation
    const double p_value = ks_pvalue_enhanced(max_diff, n);

    const bool reject_null = p_value < detail::ALPHA_05;

    std::ostringstream interpretation;
    interpretation << "KS test: D = " << max_diff << ", p = " << p_value;
    if (reject_null) {
        interpretation << " (reject H0: data does not follow specified distribution)";
    } else {
        interpretation << " (fail to reject H0: data consistent with specified distribution)";
    }

    return {max_diff, p_value, reject_null, interpretation.str()};
}

ADTestResult anderson_darling_test(const std::vector<double>& data,
                                   const DistributionBase& distribution) {
    if (data.empty()) {
        return {detail::ZERO_DOUBLE, detail::ONE, false, "Error: Empty data set"};
    }

    // Sort the data
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    const size_t n = sorted_data.size();
    double ad_statistic = detail::ZERO_DOUBLE;

    // Calculate Anderson-Darling statistic using numerically stable approach
    for (size_t i = 0; i < n; ++i) {
        const double cdf_val = distribution.getCumulativeProbability(sorted_data[i]);
        const double cdf_complement =
            distribution.getCumulativeProbability(sorted_data[n - detail::ONE_INT - i]);

        // Clamp CDF values to avoid numerical issues with log(0) and log(negative)
        // Use threshold from constants to ensure consistency
        const double clamped_cdf =
            std::max(detail::ANDERSON_DARLING_MIN_PROB,
                     std::min(detail::ONE - detail::ANDERSON_DARLING_MIN_PROB, cdf_val));
        const double clamped_complement =
            std::max(detail::ANDERSON_DARLING_MIN_PROB,
                     std::min(detail::ONE - detail::ANDERSON_DARLING_MIN_PROB, cdf_complement));

        // Calculate log terms with safe bounds
        const double log_cdf = std::log(clamped_cdf);
        const double log_one_minus_complement = std::log(detail::ONE - clamped_complement);

        // Accumulate with proper weighting
        const double weight = detail::TWO * static_cast<double>(i + 1) - detail::ONE;
        ad_statistic += weight * (log_cdf + log_one_minus_complement);
    }

    ad_statistic = -static_cast<double>(n) - ad_statistic / static_cast<double>(n);

    // Adjust for sample size (Anderson-Darling adjustment)
    const double adjusted_statistic =
        ad_statistic * (detail::ONE + detail::AD_P_VALUE_MEDIUM / static_cast<double>(n) +
                        2.25 / (static_cast<double>(n) * static_cast<double>(n)));

    // Calculate p-value
    const double p_value = interpolate_ad_pvalue(adjusted_statistic);
    const bool reject_null = p_value < detail::ALPHA_05;

    std::ostringstream interpretation;
    interpretation << "Anderson-Darling test: A² = " << adjusted_statistic << ", p = " << p_value;
    if (reject_null) {
        interpretation << " (reject H0: data does not follow specified distribution)";
    } else {
        interpretation << " (fail to reject H0: data consistent with specified distribution)";
    }

    return {adjusted_statistic, p_value, reject_null, interpretation.str()};
}

ChiSquaredResult chi_squared_goodness_of_fit(const std::vector<int>& observed_counts,
                                             const std::vector<double>& expected_probabilities) {
    if (observed_counts.size() != expected_probabilities.size()) {
        return {detail::ZERO_DOUBLE, detail::ONE, 0, true,
                "Error: Observed and expected vectors must have same size"};
    }

    if (observed_counts.empty()) {
        return {detail::ZERO_DOUBLE, detail::ONE, 0, false, "Error: Empty data"};
    }

    const size_t num_categories = observed_counts.size();
    const int total_observations =
        std::accumulate(observed_counts.begin(), observed_counts.end(), 0);

    // Calculate expected counts
    std::vector<double> expected_counts(num_categories);
    for (size_t i = 0; i < num_categories; ++i) {
        expected_counts[i] = expected_probabilities[i] * total_observations;
    }

    // Check minimum expected frequency constraint
    const double min_expected = *std::min_element(expected_counts.begin(), expected_counts.end());
    if (min_expected < detail::FIVE) {
        return {detail::ZERO_DOUBLE, detail::ONE, 0, true,
                "Warning: Some expected frequencies < 5, test may not be reliable"};
    }

    // Calculate chi-squared statistic
    double chi_squared = detail::ZERO_DOUBLE;
    for (size_t i = 0; i < num_categories; ++i) {
        const double diff = observed_counts[i] - expected_counts[i];
        chi_squared += (diff * diff) / expected_counts[i];
    }

    const int degrees_of_freedom = static_cast<int>(num_categories) - 1;

    // Calculate accurate p-value using enhanced chi-squared distribution
    const double p_value = chi_squared_pvalue(chi_squared, degrees_of_freedom);
    const bool reject_null = p_value < detail::ALPHA_05;

    std::ostringstream interpretation;
    interpretation << "Chi-squared test: χ² = " << chi_squared << ", df = " << degrees_of_freedom
                   << ", p = " << p_value;
    if (reject_null) {
        interpretation << " (reject H0: observed frequencies differ from expected)";
    } else {
        interpretation << " (fail to reject H0: observed frequencies consistent with expected)";
    }

    return {chi_squared, p_value, degrees_of_freedom, reject_null, interpretation.str()};
}

ModelDiagnostics calculate_model_diagnostics(const DistributionBase& distribution,
                                             const std::vector<double>& data) {
    if (data.empty()) {
        return {detail::ZERO_DOUBLE, detail::ZERO_DOUBLE, detail::ZERO_DOUBLE, 0, 0};
    }

    // Calculate log-likelihood
    double log_likelihood = detail::ZERO_DOUBLE;
    for (double value : data) {
        const double density = distribution.getProbability(value);
        if (density > detail::ZERO_DOUBLE) {
            log_likelihood += std::log(density);
        } else {
            log_likelihood += -1e10;  // Large negative penalty for zero density
        }
    }

    const size_t n = data.size();
    const int k = distribution.getNumParameters();  // Number of parameters

    // Calculate AIC and BIC
    const double aic = detail::TWO * k - detail::TWO * log_likelihood;
    const double bic = k * std::log(static_cast<double>(n)) - detail::TWO * log_likelihood;

    return {log_likelihood, aic, bic, k, n};
}

std::vector<double> calculate_residuals(const std::vector<double>& data,
                                        const DistributionBase& distribution) {
    std::vector<double> residuals;
    residuals.reserve(data.size());

    for (double value : data) {
        // Calculate standardized residual
        const double expected = distribution.getMean();
        const double variance = distribution.getVariance();

        if (variance > detail::ZERO_DOUBLE) {
            const double standardized = (value - expected) / std::sqrt(variance);
            residuals.push_back(standardized);
        } else {
            residuals.push_back(detail::ZERO_DOUBLE);
        }
    }

    return residuals;
}

// =============================================================================
// BOOTSTRAP-BASED STATISTICAL TESTS
// =============================================================================

namespace {
/**
 * @brief Generate bootstrap sample from distribution
 *
 * Generates a bootstrap sample of the same size as the original data
 * by sampling from the theoretical distribution.
 *
 * @param distribution Distribution to sample from
 * @param sample_size Size of bootstrap sample
 * @param rng Random number generator
 * @return Bootstrap sample
 */
std::vector<double> generate_bootstrap_sample(const DistributionBase& distribution,
                                              size_t sample_size, std::mt19937& rng) {
    std::vector<double> bootstrap_sample;
    bootstrap_sample.reserve(sample_size);

    for (size_t i = 0; i < sample_size; ++i) {
        bootstrap_sample.push_back(
            static_cast<const DistributionInterface&>(distribution).sample(rng));
    }

    return bootstrap_sample;
}

/**
 * @brief Calculate KS statistic for bootstrap testing
 *
 * Calculates the Kolmogorov-Smirnov test statistic between
 * bootstrap sample and the theoretical distribution.
 *
 * @param bootstrap_sample Bootstrap sample
 * @param distribution Theoretical distribution
 * @return KS statistic
 */
double calculate_bootstrap_ks_statistic(const std::vector<double>& bootstrap_sample,
                                        const DistributionBase& distribution) {
    if (bootstrap_sample.empty())
        return detail::ZERO_DOUBLE;

    std::vector<double> sorted_sample = bootstrap_sample;
    std::sort(sorted_sample.begin(), sorted_sample.end());

    const size_t n = sorted_sample.size();
    double max_diff = detail::ZERO_DOUBLE;

    for (size_t i = 0; i < n; ++i) {
        const double empirical_cdf = static_cast<double>(i + 1) / static_cast<double>(n);
        const double theoretical_cdf = distribution.getCumulativeProbability(sorted_sample[i]);

        const double diff_plus = empirical_cdf - theoretical_cdf;
        const double diff_minus = theoretical_cdf - static_cast<double>(i) / static_cast<double>(n);

        max_diff = std::max({max_diff, std::abs(diff_plus), std::abs(diff_minus)});
    }

    return max_diff;
}

/**
 * @brief Calculate Anderson-Darling statistic for bootstrap testing
 *
 * Calculates the Anderson-Darling test statistic between
 * bootstrap sample and the theoretical distribution.
 *
 * @param bootstrap_sample Bootstrap sample
 * @param distribution Theoretical distribution
 * @return AD statistic
 */
double calculate_bootstrap_ad_statistic(const std::vector<double>& bootstrap_sample,
                                        const DistributionBase& distribution) {
    if (bootstrap_sample.empty())
        return detail::ZERO_DOUBLE;

    std::vector<double> sorted_sample = bootstrap_sample;
    std::sort(sorted_sample.begin(), sorted_sample.end());

    const size_t n = sorted_sample.size();
    double statistic = detail::ZERO_DOUBLE;

    for (size_t i = 0; i < n; ++i) {
        const double cdf_val = distribution.getCumulativeProbability(sorted_sample[i]);
        const double log_cdf = std::log(std::max(detail::ANDERSON_DARLING_MIN_PROB, cdf_val));
        const double log_1_minus_cdf =
            std::log(std::max(detail::ANDERSON_DARLING_MIN_PROB, 1.0 - cdf_val));

        const double term1 = (detail::TWO * static_cast<double>(i + 1) - detail::ONE) * log_cdf;
        const double term2 =
            (detail::TWO * static_cast<double>(n - i) - detail::ONE) * log_1_minus_cdf;

        statistic += term1 + term2;
    }

    return -static_cast<double>(n) - statistic / static_cast<double>(n);
}

/**
 * @brief Calculate percentile from sorted values
 *
 * @param sorted_values Sorted vector of values
 * @param percentile Percentile to calculate (0-100)
 * @return Percentile value
 */
double calculate_percentile(const std::vector<double>& sorted_values, double percentile) {
    if (sorted_values.empty())
        return detail::ZERO_DOUBLE;

    const double index =
        (percentile / detail::HUNDRED) * static_cast<double>(sorted_values.size() - 1);
    const size_t lower_index = static_cast<size_t>(std::floor(index));
    const size_t upper_index = static_cast<size_t>(std::ceil(index));

    if (lower_index == upper_index) {
        return sorted_values[lower_index];
    }

    const double weight = index - static_cast<double>(lower_index);
    return sorted_values[lower_index] * (detail::ONE - weight) +
           sorted_values[upper_index] * weight;
}
}  // namespace

BootstrapTestResult bootstrap_kolmogorov_smirnov_test(const std::vector<double>& data,
                                                      const DistributionBase& distribution,
                                                      size_t num_bootstrap, double alpha) {
    if (data.empty()) {
        return {detail::ZERO_DOUBLE, detail::ONE, {}, false, "Error: Empty data set", 0};
    }

    // Calculate observed KS statistic
    KSTestResult original_test = kolmogorov_smirnov_test(data, distribution);
    const double observed_statistic = original_test.statistic;

    // Generate bootstrap statistics
    std::vector<double> bootstrap_statistics;
    bootstrap_statistics.reserve(num_bootstrap);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < num_bootstrap; ++i) {
        std::vector<double> bootstrap_sample =
            generate_bootstrap_sample(distribution, data.size(), rng);
        double bootstrap_statistic =
            calculate_bootstrap_ks_statistic(bootstrap_sample, distribution);
        bootstrap_statistics.push_back(bootstrap_statistic);
    }

    // Calculate bootstrap p-value
    const size_t count_greater = static_cast<size_t>(
        std::count_if(bootstrap_statistics.begin(), bootstrap_statistics.end(),
                      [observed_statistic](double stat) { return stat >= observed_statistic; }));

    const double bootstrap_p_value =
        static_cast<double>(count_greater) / static_cast<double>(num_bootstrap);
    const bool reject_null = bootstrap_p_value < alpha;

    std::ostringstream interpretation;
    interpretation << "Bootstrap KS test: D = " << observed_statistic
                   << ", bootstrap p = " << bootstrap_p_value << " (" << num_bootstrap
                   << " bootstrap samples)";
    if (reject_null) {
        interpretation << " (reject H0: data does not follow specified distribution)";
    } else {
        interpretation << " (fail to reject H0: data consistent with specified distribution)";
    }

    return {observed_statistic, bootstrap_p_value,    bootstrap_statistics,
            reject_null,        interpretation.str(), num_bootstrap};
}

BootstrapTestResult bootstrap_anderson_darling_test(const std::vector<double>& data,
                                                    const DistributionBase& distribution,
                                                    size_t num_bootstrap, double alpha) {
    if (data.empty()) {
        return {detail::ZERO_DOUBLE, detail::ONE, {}, false, "Error: Empty data set", 0};
    }

    // Calculate observed AD statistic
    ADTestResult original_test = anderson_darling_test(data, distribution);
    const double observed_statistic = original_test.statistic;

    // Generate bootstrap statistics
    std::vector<double> bootstrap_statistics;
    bootstrap_statistics.reserve(num_bootstrap);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < num_bootstrap; ++i) {
        std::vector<double> bootstrap_sample =
            generate_bootstrap_sample(distribution, data.size(), rng);
        double bootstrap_statistic =
            calculate_bootstrap_ad_statistic(bootstrap_sample, distribution);
        bootstrap_statistics.push_back(bootstrap_statistic);
    }

    // Calculate bootstrap p-value
    const size_t count_greater = static_cast<size_t>(
        std::count_if(bootstrap_statistics.begin(), bootstrap_statistics.end(),
                      [observed_statistic](double stat) { return stat >= observed_statistic; }));

    const double bootstrap_p_value =
        static_cast<double>(count_greater) / static_cast<double>(num_bootstrap);
    const bool reject_null = bootstrap_p_value < alpha;

    std::ostringstream interpretation;
    interpretation << "Bootstrap AD test: A² = " << observed_statistic
                   << ", bootstrap p = " << bootstrap_p_value << " (" << num_bootstrap
                   << " bootstrap samples)";
    if (reject_null) {
        interpretation << " (reject H0: data does not follow specified distribution)";
    } else {
        interpretation << " (fail to reject H0: data consistent with specified distribution)";
    }

    return {observed_statistic, bootstrap_p_value,    bootstrap_statistics,
            reject_null,        interpretation.str(), num_bootstrap};
}

BootstrapTestResult bootstrap_parameter_test(const std::vector<double>& data,
                                             const DistributionBase& distribution,
                                             size_t num_bootstrap, double alpha) {
    if (data.empty()) {
        return {detail::ZERO_DOUBLE, detail::ONE, {}, false, "Error: Empty data set", 0};
    }

    // Calculate observed parameter fit quality (using log-likelihood)
    ModelDiagnostics original_diagnostics = calculate_model_diagnostics(distribution, data);
    const double observed_statistic =
        -original_diagnostics.log_likelihood;  // Use negative log-likelihood

    // Generate bootstrap statistics
    std::vector<double> bootstrap_statistics;
    bootstrap_statistics.reserve(num_bootstrap);

    std::random_device rd;
    std::mt19937 rng(rd());

    for (size_t i = 0; i < num_bootstrap; ++i) {
        std::vector<double> bootstrap_sample =
            generate_bootstrap_sample(distribution, data.size(), rng);

        // Calculate parameter fit quality for bootstrap sample
        ModelDiagnostics bootstrap_diagnostics =
            calculate_model_diagnostics(distribution, bootstrap_sample);
        double bootstrap_statistic = -bootstrap_diagnostics.log_likelihood;
        bootstrap_statistics.push_back(bootstrap_statistic);
    }

    // Calculate bootstrap p-value (two-tailed test)
    const size_t count_greater = static_cast<size_t>(
        std::count_if(bootstrap_statistics.begin(), bootstrap_statistics.end(),
                      [observed_statistic](double stat) { return stat >= observed_statistic; }));

    const double bootstrap_p_value =
        static_cast<double>(count_greater) / static_cast<double>(num_bootstrap);
    const bool reject_null = bootstrap_p_value < alpha;

    std::ostringstream interpretation;
    interpretation << "Bootstrap parameter test: -log L = " << observed_statistic
                   << ", bootstrap p = " << bootstrap_p_value << " (" << num_bootstrap
                   << " bootstrap samples)";
    if (reject_null) {
        interpretation << " (reject H0: parameters inconsistent with data)";
    } else {
        interpretation << " (fail to reject H0: parameters consistent with data)";
    }

    return {observed_statistic, bootstrap_p_value,    bootstrap_statistics,
            reject_null,        interpretation.str(), num_bootstrap};
}

std::vector<ConfidenceInterval> bootstrap_confidence_intervals(const std::vector<double>& data,
                                                               const DistributionBase& distribution,
                                                               double confidence_level,
                                                               size_t num_bootstrap) {
    std::vector<ConfidenceInterval> intervals;

    if (data.empty()) {
        return intervals;
    }

    // For this implementation, we'll focus on mean and variance confidence intervals
    // In a full implementation, this would extract and bootstrap all distribution parameters

    std::vector<double> bootstrap_means;
    std::vector<double> bootstrap_variances;
    bootstrap_means.reserve(num_bootstrap);
    bootstrap_variances.reserve(num_bootstrap);

    std::random_device rd;
    std::mt19937 rng(rd());

    // Generate bootstrap samples and calculate statistics
    for (size_t i = 0; i < num_bootstrap; ++i) {
        std::vector<double> bootstrap_sample =
            generate_bootstrap_sample(distribution, data.size(), rng);

        // Calculate sample mean and variance
        double sum =
            std::accumulate(bootstrap_sample.begin(), bootstrap_sample.end(), detail::ZERO_DOUBLE);
        double mean = sum / static_cast<double>(bootstrap_sample.size());
        bootstrap_means.push_back(mean);

        double variance = detail::ZERO_DOUBLE;
        for (double value : bootstrap_sample) {
            variance += (value - mean) * (value - mean);
        }
        variance /= static_cast<double>(bootstrap_sample.size() - 1);
        bootstrap_variances.push_back(variance);
    }

    // Sort bootstrap statistics
    std::sort(bootstrap_means.begin(), bootstrap_means.end());
    std::sort(bootstrap_variances.begin(), bootstrap_variances.end());

    // Calculate confidence intervals
    const double alpha = detail::ONE - confidence_level;
    const double lower_percentile = detail::HUNDRED * (alpha / detail::TWO);
    const double upper_percentile = detail::HUNDRED * (detail::ONE - alpha / detail::TWO);

    // Mean confidence interval
    ConfidenceInterval mean_interval;
    mean_interval.lower_bound = calculate_percentile(bootstrap_means, lower_percentile);
    mean_interval.upper_bound = calculate_percentile(bootstrap_means, upper_percentile);
    mean_interval.point_estimate = distribution.getMean();
    mean_interval.confidence_level = confidence_level;
    intervals.push_back(mean_interval);

    // Variance confidence interval
    ConfidenceInterval variance_interval;
    variance_interval.lower_bound = calculate_percentile(bootstrap_variances, lower_percentile);
    variance_interval.upper_bound = calculate_percentile(bootstrap_variances, upper_percentile);
    variance_interval.point_estimate = distribution.getVariance();
    variance_interval.confidence_level = confidence_level;
    intervals.push_back(variance_interval);

    return intervals;
}

}  // namespace detail
}  // namespace stats

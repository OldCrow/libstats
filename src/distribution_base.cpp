#include "../include/core/distribution_base.h"
#include "../include/core/constants.h"
#include "../include/core/math_utils.h"
#include "../include/core/safety.h"
#include "../include/platform/parallel_execution.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <sstream>

namespace libstats {

// =============================================================================
// RULE OF FIVE IMPLEMENTATION
// =============================================================================

DistributionBase::DistributionBase(const DistributionBase& /* other */) {
    // Copy constructor - deliberately do not copy cache state
    // Each instance maintains its own cache that must be recomputed
}

DistributionBase::DistributionBase(DistributionBase&& /* other */) noexcept {
    // Move constructor - deliberately do not move cache state
    // Moving is complete, but cache must be recomputed for safety
}

DistributionBase& DistributionBase::operator=(const DistributionBase& other) {
    if (this != &other) {
        // Copy assignment - don't copy cache state, each instance manages its own cache
        // The mutex and cache are not copyable, so we just invalidate our cache
        invalidateCache();
    }
    return *this;
}

DistributionBase& DistributionBase::operator=(DistributionBase&& other) noexcept {
    if (this != &other) {
        // Move assignment - don't move cache state, each instance manages its own cache
        // The mutex and cache are not movable, so we just invalidate our cache
        invalidateCache();
    }
    return *this;
}

// =============================================================================
// PARALLEL-OPTIMIZED BATCH OPERATIONS - WITH AUTOMATIC PARALLEL EXECUTION
// =============================================================================

std::vector<double> DistributionBase::getBatchProbabilities(const std::vector<double>& x_values) const {
    std::vector<double> results(x_values.size());
    
    // Use parallel transform for large datasets
    parallel::safe_transform(x_values.begin(), x_values.end(), results.begin(),
        [this](double x) { return this->getProbability(x); });
    
    return results;
}

std::vector<double> DistributionBase::getBatchLogProbabilities(const std::vector<double>& x_values) const {
    std::vector<double> results(x_values.size());
    
    // Use parallel transform for large datasets
    parallel::safe_transform(x_values.begin(), x_values.end(), results.begin(),
        [this](double x) { return this->getLogProbability(x); });
    
    return results;
}

std::vector<double> DistributionBase::getBatchCumulativeProbabilities(const std::vector<double>& x_values) const {
    std::vector<double> results(x_values.size());
    
    // Use parallel transform for large datasets
    parallel::safe_transform(x_values.begin(), x_values.end(), results.begin(),
        [this](double x) { return this->getCumulativeProbability(x); });
    
    return results;
}

std::vector<double> DistributionBase::getBatchQuantiles(const std::vector<double>& p_values) const {
    // Validate input probabilities first
    for (double p : p_values) {
        if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
            throw std::invalid_argument("Quantile probability must be in [0,1], got: " + std::to_string(p));
        }
    }
    
    std::vector<double> results(p_values.size());
    
    // Use parallel transform for large datasets after validation
    parallel::safe_transform(p_values.begin(), p_values.end(), results.begin(),
        [this](double p) { return this->getQuantile(p); });
    
    return results;
}

// =============================================================================
// STATISTICAL VALIDATION AND DIAGNOSTICS
// =============================================================================

FitResults DistributionBase::fitWithDiagnostics(const std::vector<double>& data) {
    FitResults results;
    
    // Validate input data
    try {
        validateFittingData(data);
        
        // Perform the fit
        fit(data);
        results.fit_successful = true;
        results.fit_diagnostics = "Fit completed successfully";
        
        // Calculate log-likelihood
        results.log_likelihood = constants::math::ZERO_DOUBLE;
        for (double x : data) {
            double log_prob = getLogProbability(x);
            if (std::isfinite(log_prob)) {
                results.log_likelihood += log_prob;
            }
        }
        
        // Calculate AIC and BIC
        int k = getNumParameters();
        int n = static_cast<int>(data.size());
        results.aic = constants::math::TWO * k - constants::math::TWO * results.log_likelihood;
        results.bic = k * std::log(n) - constants::math::TWO * results.log_likelihood;
        
        // Calculate residuals
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());
        
        results.residuals.reserve(data.size());
        for (size_t i = 0; i < sorted_data.size(); ++i) {
            double empirical_cdf = (i + constants::math::ONE) / (data.size() + constants::math::ONE);
            double theoretical_cdf = getCumulativeProbability(sorted_data[i]);
            double residual = empirical_cdf - theoretical_cdf;
            results.residuals.push_back(residual);
        }
        
        // Perform validation
        results.validation = validate(data);
        
    } catch (const std::exception& e) {
        results.fit_successful = false;
        results.fit_diagnostics = std::string("Fit failed: ") + e.what();
        results.log_likelihood = std::numeric_limits<double>::quiet_NaN();
        results.aic = std::numeric_limits<double>::quiet_NaN();
        results.bic = std::numeric_limits<double>::quiet_NaN();
        
        // Set default validation results
        results.validation.ks_statistic = std::numeric_limits<double>::quiet_NaN();
        results.validation.ks_p_value = constants::math::ZERO_DOUBLE;
        results.validation.ad_statistic = std::numeric_limits<double>::quiet_NaN();
        results.validation.ad_p_value = constants::math::ZERO_DOUBLE;
        results.validation.distribution_adequate = false;
        results.validation.recommendations = "Unable to validate due to fit failure";
    }
    
    return results;
}

ValidationResult DistributionBase::validate(const std::vector<double>& data) const {
    ValidationResult result;
    
    try {
        validateFittingData(data);
        
        // Kolmogorov-Smirnov test
        result.ks_statistic = math::calculate_ks_statistic(data, *this);
        
        // Simple p-value approximation for KS test
        double n = static_cast<double>(data.size());
        double lambda = (std::sqrt(n) + constants::thresholds::KS_APPROX_COEFF_1 + constants::thresholds::KS_APPROX_COEFF_2 / std::sqrt(n)) * result.ks_statistic;
        result.ks_p_value = constants::math::TWO * std::exp(constants::math::NEG_TWO * lambda * lambda);
        result.ks_p_value = safety::clamp_probability(result.ks_p_value);
        
        // Anderson-Darling test
        result.ad_statistic = math::calculate_ad_statistic(data, *this);
        
        // Simple p-value approximation for AD test
        // This is a simplified approximation - in practice, you'd use lookup tables
        if (result.ad_statistic < constants::thresholds::AD_THRESHOLD_1) {
            result.ad_p_value = constants::thresholds::AD_P_VALUE_HIGH;
        } else if (result.ad_statistic < constants::math::ONE) {
            result.ad_p_value = constants::thresholds::AD_P_VALUE_MEDIUM;
        } else if (result.ad_statistic < constants::math::TWO) {
            result.ad_p_value = constants::thresholds::ALPHA_10;
        } else {
            result.ad_p_value = constants::thresholds::ALPHA_01;
        }
        
        // Overall assessment
        result.distribution_adequate = (result.ks_p_value > constants::thresholds::ALPHA_05) && (result.ad_p_value > constants::thresholds::ALPHA_05);
        
        // Generate recommendations
        std::ostringstream recommendations;
        if (!result.distribution_adequate) {
            recommendations << "Distribution fit may be inadequate. ";
            if (result.ks_p_value <= constants::thresholds::ALPHA_05) {
                recommendations << "KS test suggests poor fit. ";
            }
            if (result.ad_p_value <= constants::thresholds::ALPHA_05) {
                recommendations << "AD test suggests poor fit. ";
            }
            recommendations << "Consider alternative distributions or data transformations.";
        } else {
            recommendations << "Distribution fit appears adequate based on goodness-of-fit tests.";
        }
        
        result.recommendations = recommendations.str();
        
    } catch (const std::exception& e) {
        result.ks_statistic = std::numeric_limits<double>::quiet_NaN();
        result.ks_p_value = 0.0;
        result.ad_statistic = std::numeric_limits<double>::quiet_NaN();
        result.ad_p_value = 0.0;
        result.distribution_adequate = false;
        result.recommendations = std::string("Validation failed: ") + e.what();
    }
    
    return result;
}

// =============================================================================
// INFORMATION THEORY METRICS
// =============================================================================

double DistributionBase::getKLDivergence(const DistributionBase& other) const {
    // Numerical approximation using integration
    // In practice, this would use adaptive quadrature
    double lower = std::max(getSupportLowerBound(), other.getSupportLowerBound());
    double upper = std::min(getSupportUpperBound(), other.getSupportUpperBound());
    
    if (lower >= upper) {
        return std::numeric_limits<double>::infinity();
    }
    
    // Simple integration using trapezoidal rule
    const int n_points = constants::thresholds::DEFAULT_INTEGRATION_POINTS;
    double step = (upper - lower) / n_points;
    double divergence = 0.0;
    
    for (int i = 0; i < n_points; ++i) {
        double x = lower + i * step;
        double p = getProbability(x);
        double q = other.getProbability(x);
        
        if (p > constants::probability::MIN_PROBABILITY && q > constants::probability::MIN_PROBABILITY) {
            divergence += p * std::log(p / q) * step;
        }
    }
    
    return divergence;
}

// =============================================================================
// DISTRIBUTION COMPARISON
// =============================================================================

bool DistributionBase::isApproximatelyEqual(const DistributionBase& other, double tolerance) const {
    // Compare basic properties
    if (getDistributionName() != other.getDistributionName()) {
        return false;
    }
    
    if (getNumParameters() != other.getNumParameters()) {
        return false;
    }
    
    // Compare statistical moments
    if (std::abs(getMean() - other.getMean()) > tolerance) {
        return false;
    }
    
    if (std::abs(getVariance() - other.getVariance()) > tolerance) {
        return false;
    }
    
    // Compare support bounds
    if (std::abs(getSupportLowerBound() - other.getSupportLowerBound()) > tolerance) {
        return false;
    }
    
    if (std::abs(getSupportUpperBound() - other.getSupportUpperBound()) > tolerance) {
        return false;
    }
    
    return true;
}

// Cache management is handled by ThreadSafeCacheManager base class

// =============================================================================
// NUMERICAL UTILITIES
// =============================================================================

double DistributionBase::numericalIntegration(std::function<double(double)> pdf_func,
                                            double lower_bound, double upper_bound,
                                            double tolerance) {
    // Adaptive Simpson's rule implementation
    return adaptiveSimpsonIntegration(pdf_func, lower_bound, upper_bound, tolerance, 0, constants::precision::MAX_ADAPTIVE_SIMPSON_DEPTH);
}

// Helper function for adaptive Simpson's rule
double DistributionBase::adaptiveSimpsonIntegration(std::function<double(double)> func,
                                                   double a, double b, double tolerance,
                                                   int depth, int max_depth) {
    if (depth > max_depth) {
        // Fallback to simple Simpson's rule if max depth exceeded
        double mid = (a + b) / 2.0;
        double fa = func(a);
        double fb = func(b);
        double fmid = func(mid);
        return (b - a) / 6.0 * (fa + 4.0 * fmid + fb);
    }
    
    double mid = (a + b) / 2.0;
    double left_mid = (a + mid) / 2.0;
    double right_mid = (mid + b) / 2.0;
    
    // Evaluate function at all points
    double fa = func(a);
    double fb = func(b);
    double fmid = func(mid);
    double fleft_mid = func(left_mid);
    double fright_mid = func(right_mid);
    
    // Compute Simpson's rule for whole interval
    double whole = (b - a) / 6.0 * (fa + 4.0 * fmid + fb);
    
    // Compute Simpson's rule for left and right halves
    double left = (mid - a) / 6.0 * (fa + 4.0 * fleft_mid + fmid);
    double right = (b - mid) / 6.0 * (fmid + 4.0 * fright_mid + fb);
    
    double combined = left + right;
    
    // Check if the error is within tolerance
    if (std::abs(combined - whole) < 15.0 * tolerance) {
        return combined + (combined - whole) / 15.0; // Richardson extrapolation
    }
    
    // Recursively refine both halves with half the tolerance
    return adaptiveSimpsonIntegration(func, a, mid, tolerance / 2.0, depth + 1, max_depth) +
           adaptiveSimpsonIntegration(func, mid, b, tolerance / 2.0, depth + 1, max_depth);
}

double DistributionBase::newtonRaphsonQuantile(std::function<double(double)> cdf_func,
                                             double target_probability,
                                             double initial_guess,
                                             double tolerance) {
    safety::check_probability(target_probability, "target_probability");
    
    double x = initial_guess;
    const int max_iterations = constants::precision::MAX_NEWTON_ITERATIONS;
    const double h = constants::precision::FORWARD_DIFF_STEP;  // For numerical derivative
    
    for (int i = 0; i < max_iterations; ++i) {
        double fx = cdf_func(x) - target_probability;
        
        if (std::abs(fx) < tolerance) {
            return x;
        }
        
        // Numerical derivative
        double fpx = (cdf_func(x + h) - cdf_func(x - h)) / (constants::math::TWO * h);
        
        if (std::abs(fpx) < constants::precision::ZERO) {
            break;  // Derivative too small
        }
        
        x = x - fx / fpx;
    }
    
    return x;
}

void DistributionBase::validateFittingData(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    
    if (data.size() < constants::thresholds::MIN_DATA_POINTS_FOR_FITTING) {
        throw std::invalid_argument("Need at least 2 data points for fitting");
    }
    
    // Check for finite values
    for (size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            throw std::invalid_argument("Data contains non-finite values at index " + 
                                      std::to_string(i));
        }
    }
}

std::vector<double> DistributionBase::calculateEmpiricalCDF(const std::vector<double>& data) {
    if (data.empty()) {
        return {};
    }
    
    std::vector<double> cdf_values;
    cdf_values.reserve(data.size());
    
    for (size_t i = 0; i < data.size(); ++i) {
        double empirical_cdf = (i + constants::math::ONE) / (data.size() + constants::math::ONE);
        cdf_values.push_back(empirical_cdf);
    }
    
    return cdf_values;
}

// =============================================================================
// SPECIAL MATHEMATICAL FUNCTIONS
// =============================================================================

double DistributionBase::erf(double x) noexcept {
    // Use math_utils implementation if available, otherwise use std::erf
    return std::erf(x);
}

double DistributionBase::erfc(double x) noexcept {
    return std::erfc(x);
}

double DistributionBase::lgamma(double x) noexcept {
    return std::lgamma(x);
}

double DistributionBase::gammaP(double a, double x) noexcept {
    // Simplified implementation - in practice, use specialized numerical algorithms
    if (x < 0.0 || a <= 0.0) {
        return 0.0;
    }
    
    if (x == 0.0) {
        return 0.0;
    }
    
    // Use series expansion for small x, continued fraction for large x
    if (x < a + 1.0) {
        // Series expansion
        double sum = 1.0;
        double term = 1.0;
        double n = 1.0;
        
        while (std::abs(term) > constants::precision::DEFAULT_TOLERANCE && n < constants::precision::MAX_GAMMA_SERIES_ITERATIONS) {
            term *= x / (a + n - constants::math::ONE);
            sum += term;
            n += constants::math::ONE;
        }
        
        return std::exp(-x + a * std::log(x) - lgamma(a)) * sum;
    } else {
        // Use complementary function
        return constants::math::ONE - gammaQ(a, x);
    }
}

double DistributionBase::gammaQ(double a, double x) noexcept {
    return constants::math::ONE - gammaP(a, x);
}

double DistributionBase::betaI(double x, double a, double b) noexcept {
    // Simplified implementation of incomplete beta function
    if (x < constants::math::ZERO_DOUBLE || x > constants::math::ONE) {
        return constants::math::ZERO_DOUBLE;
    }
    
    if (x == constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
    }
    
    if (x == constants::math::ONE) {
        return constants::math::ONE;
    }
    
    // Use continued fraction approximation
    double bt = std::exp(lgamma(a + b) - lgamma(a) - lgamma(b) + 
                        a * std::log(x) + b * std::log(1.0 - x));
    
    if (x < (a + 1.0) / (a + b + 2.0)) {
        return bt * betaI_continued_fraction(x, a, b) / a;
    } else {
        return 1.0 - bt * betaI_continued_fraction(1.0 - x, b, a) / b;
    }
}

// =============================================================================
// INTERNAL IMPLEMENTATION DETAILS
// =============================================================================


double DistributionBase::betaI_continued_fraction(double x, double a, double b) noexcept {
    // Continued fraction for incomplete beta function
    const int max_iterations = 100;
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

} // namespace libstats

#include "libstats/core/distribution_base.h"

#include "libstats/core/math_constants.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/performance_dispatcher.h"
#include "libstats/core/safety.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/platform/parallel_execution.h"  // Parallel execution (full implementation)

#include <algorithm>   // for std::sort, std::abs
#include <cmath>       // for std::log, std::exp, std::sqrt, std::isfinite
#include <cstddef>     // for size_t
#include <functional>  // for std::function
#include <numeric>     // for numerical algorithms
#include <sstream>     // for std::ostringstream
#include <stdexcept>   // for std::invalid_argument
#include <string>      // for std::string, std::to_string
#include <vector>      // for std::vector

namespace stats {

// =============================================================================
// RULE OF FIVE IMPLEMENTATION
// =============================================================================

DistributionBase::DistributionBase() {
    // Initialize all critical system components during base class construction
    // This ensures one-time initialization overhead happens during object creation,
    // not during the first method call, providing predictable performance.

    // Initialize SystemCapabilities (thread_local singleton)
    detail::SystemCapabilities::current();

    // Initialize PerformanceHistory (static global singleton)
    // This ensures the performance tracking system is ready
    [[maybe_unused]] auto& history = detail::PerformanceDispatcher::getPerformanceHistory();
}

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
        results.log_likelihood = detail::ZERO_DOUBLE;
        for (double x : data) {
            double log_prob = getLogProbability(x);
            if (std::isfinite(log_prob)) {
                results.log_likelihood += log_prob;
            }
        }

        // Calculate AIC and BIC
        int k = getNumParameters();
        int n = static_cast<int>(data.size());
        results.aic = detail::TWO * k - detail::TWO * results.log_likelihood;
        results.bic = k * std::log(n) - detail::TWO * results.log_likelihood;

        // Calculate residuals
        std::vector<double> sorted_data = data;
        std::sort(sorted_data.begin(), sorted_data.end());

        results.residuals.reserve(data.size());
        for (size_t i = 0; i < sorted_data.size(); ++i) {
            double empirical_cdf = (static_cast<double>(i) + detail::ONE) /
                                   (static_cast<double>(data.size()) + detail::ONE);
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
        results.validation.ks_p_value = detail::ZERO_DOUBLE;
        results.validation.ad_statistic = std::numeric_limits<double>::quiet_NaN();
        results.validation.ad_p_value = detail::ZERO_DOUBLE;
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
        result.ks_statistic = detail::calculate_ks_statistic(data, *this);

        // Simple p-value approximation for KS test
        double n = static_cast<double>(data.size());
        double lambda =
            (std::sqrt(n) + detail::KS_APPROX_COEFF_1 + detail::KS_APPROX_COEFF_2 / std::sqrt(n)) *
            result.ks_statistic;
        result.ks_p_value = detail::TWO * std::exp(detail::NEG_TWO * lambda * lambda);
        result.ks_p_value = detail::clamp_probability(result.ks_p_value);

        // Anderson-Darling test
        result.ad_statistic = detail::calculate_ad_statistic(data, *this);

        // LP-14: Continuous exponential p-value approximation (same formula as
        // stats::analysis::andersonDarlingTest in goodness_of_fit.h) — replaces the
        // 4-bucket step function that mapped very different statistic values to the same p-value.
        if (result.ad_statistic >= 13.0) {
            result.ad_p_value = 0.0;
        } else if (result.ad_statistic >= 6.0) {
            result.ad_p_value = std::exp(-1.28 * result.ad_statistic);
        } else {
            result.ad_p_value = std::exp(-1.8 * result.ad_statistic + 1.5);
        }
        result.ad_p_value = std::min(1.0, std::max(0.0, result.ad_p_value));

        // Overall assessment
        result.distribution_adequate =
            (result.ks_p_value > detail::ALPHA_05) && (result.ad_p_value > detail::ALPHA_05);

        // Generate recommendations
        std::ostringstream recommendations;
        if (!result.distribution_adequate) {
            recommendations << "Distribution fit may be inadequate. ";
            if (result.ks_p_value <= detail::ALPHA_05) {
                recommendations << "KS test suggests poor fit. ";
            }
            if (result.ad_p_value <= detail::ALPHA_05) {
                recommendations << "AD test suggests poor fit. ";
            }
            recommendations << "Consider alternative distributions or data transformations.";
        } else {
            recommendations << "Distribution fit appears adequate based on goodness-of-fit tests.";
        }

        result.recommendations = recommendations.str();

    } catch (const std::exception& e) {
        result.ks_statistic = std::numeric_limits<double>::quiet_NaN();
        result.ks_p_value = detail::ZERO_DOUBLE;
        result.ad_statistic = std::numeric_limits<double>::quiet_NaN();
        result.ad_p_value = detail::ZERO_DOUBLE;
        result.distribution_adequate = false;
        result.recommendations = std::string("Validation failed: ") + e.what();
    }

    return result;
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

    // LP-7: guard against NaN mean/variance — NaN comparisons always return false,
    // which would cause NaN distributions to be treated as approximately equal.
    const double myMean = getMean();
    const double otherMean = other.getMean();
    if (std::isnan(myMean) || std::isnan(otherMean) ||
        std::abs(myMean - otherMean) > tolerance) {
        return false;
    }

    const double myVar = getVariance();
    const double otherVar = other.getVariance();
    if (std::isnan(myVar) || std::isnan(otherVar) ||
        std::abs(myVar - otherVar) > tolerance) {
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
    return adaptiveSimpsonIntegration(pdf_func, lower_bound, upper_bound, tolerance, 0,
                                      detail::MAX_ADAPTIVE_SIMPSON_DEPTH);
}

// Helper function for adaptive Simpson's rule
double DistributionBase::adaptiveSimpsonIntegration(std::function<double(double)> func, double a,
                                                    double b, double tolerance, int depth,
                                                    int max_depth) {
    if (depth > max_depth) {
        // Fallback to simple Simpson's rule if max depth exceeded
        double mid = (a + b) / detail::TWO;
        double fa = func(a);
        double fb = func(b);
        double fmid = func(mid);
        return (b - a) / detail::SIX * (fa + detail::FOUR * fmid + fb);
    }

    double mid = (a + b) / detail::TWO;
    double left_mid = (a + mid) / detail::TWO;
    double right_mid = (mid + b) / detail::TWO;

    // Evaluate function at all points
    double fa = func(a);
    double fb = func(b);
    double fmid = func(mid);
    double fleft_mid = func(left_mid);
    double fright_mid = func(right_mid);

    // Compute Simpson's rule for whole interval
    double whole = (b - a) / detail::SIX * (fa + detail::FOUR * fmid + fb);

    // Compute Simpson's rule for left and right halves
    double left = (mid - a) / detail::SIX * (fa + detail::FOUR * fleft_mid + fmid);
    double right = (b - mid) / detail::SIX * (fmid + detail::FOUR * fright_mid + fb);

    double combined = left + right;

    // Check if the error is within tolerance
    if (std::abs(combined - whole) < 15.0 * tolerance) {
        return combined + (combined - whole) / 15.0;  // Richardson extrapolation
    }

    // Recursively refine both halves with half the tolerance
    return adaptiveSimpsonIntegration(func, a, mid, tolerance / detail::TWO,
                                      depth + detail::ONE_INT, max_depth) +
           adaptiveSimpsonIntegration(func, mid, b, tolerance / detail::TWO,
                                      depth + detail::ONE_INT, max_depth);
}

double DistributionBase::newtonRaphsonQuantile(std::function<double(double)> cdf_func,
                                               double target_probability, double initial_guess,
                                               double tolerance) {
    detail::check_probability(target_probability, "target_probability");

    double x = initial_guess;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;
    const double h = detail::FORWARD_DIFF_STEP;  // For numerical derivative
    // Q-2: capture initial derivative magnitude for relative guard (first iteration).
    double fpx0 = 0.0;

    for (int i = 0; i < max_iterations; ++i) {
        double fx = cdf_func(x) - target_probability;

        if (std::abs(fx) < tolerance) {
            return x;
        }

        // Numerical derivative
        double fpx = (cdf_func(x + h) - cdf_func(x - h)) / (detail::TWO * h);
        if (i == 0) fpx0 = std::abs(fpx);  // Latch initial magnitude once

        // Q-2: relative derivative guard — guards both absolute zero and collapse
        // relative to the initial derivative magnitude (avoids divergent steps).
        if (std::abs(fpx) < detail::ZERO ||
            (fpx0 > detail::ZERO && std::abs(fpx) < 1e-12 * fpx0)) {
            return x;  // Hard stop: current estimate is the best available
        }

        x = x - fx / fpx;
    }

    return x;
}

void DistributionBase::validateFittingData(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    if (data.size() < detail::MIN_DATA_POINTS_FOR_FITTING) {
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
        double empirical_cdf = (static_cast<double>(i) + detail::ONE) /
                               (static_cast<double>(data.size()) + detail::ONE);
        cdf_values.push_back(empirical_cdf);
    }

    return cdf_values;
}

// =============================================================================
// INTERNAL IMPLEMENTATION DETAILS
// =============================================================================
// NOTE: The erf/erfc/lgamma/gammaP/gammaQ/gammaQuantile/betaI protected
// wrappers that were here have been removed in v2.0.0 (AQ-5). Subclasses
// should call stats::detail:: functions from math_utils.h directly.
//
// betaI_continued_fraction is still defined below as a private helper
// used by distribution_base's own validate/fitWithDiagnostics paths.

double DistributionBase::betaI_continued_fraction(double x, double a, double b) noexcept {
    // Continued fraction for incomplete beta function
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;
    const double tolerance = detail::DEFAULT_TOLERANCE;

    double qab = a + b;
    double qap = a + detail::ONE;
    double qam = a - detail::ONE;
    double c = detail::ONE;
    double d = detail::ONE - qab * x / qap;

    if (std::abs(d) < detail::ZERO) {
        d = detail::ZERO;
    }

    d = detail::ONE / d;
    double h = d;

    for (int m = 1; m <= max_iterations; ++m) {
        int m2 = detail::TWO_INT * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = detail::ONE + aa * d;
        if (std::abs(d) < detail::ZERO) d = detail::ZERO;
        // Guard c BEFORE dividing by it: a near-zero c from the previous
        // iteration would otherwise produce aa/c ≈ 1e+30 before being caught.
        if (std::abs(c) < detail::ZERO) c = detail::ZERO;
        c = detail::ONE + aa / c;

        d = detail::ONE / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = detail::ONE + aa * d;
        if (std::abs(d) < detail::ZERO) d = detail::ZERO;
        if (std::abs(c) < detail::ZERO) c = detail::ZERO;
        c = detail::ONE + aa / c;

        d = detail::ONE / d;
        double del = d * c;
        h *= del;

        if (std::abs(del - detail::ONE) < tolerance) {
            break;
        }
    }

    return h;
}

}  // namespace stats

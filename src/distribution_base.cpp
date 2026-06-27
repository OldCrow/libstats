#include "libstats/core/distribution_base.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
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
// INTERNAL HELPERS
// =============================================================================

/**
 * Computes log-likelihood, AIC and BIC for a fitted DistributionBase.
 *
 * Implements the same formula as stats::analysis::informationCriteria<D>().
 * A file-local function is used here because DistributionBase cannot satisfy
 * the AnyDistribution concept (it lacks static kDistributionType / kIsDiscrete
 * members, which are only meaningful on concrete distribution types).
 *
 * Formula:
 *   log_likelihood = Σ log P(x_i | θ)
 *   AIC  = 2k − 2ℓ
 *   BIC  = k ln n − 2ℓ
 *
 * Sync note: if the formula in information_criteria.h ever changes, update here.
 */
static void compute_fit_ic(const std::vector<double>& data, const DistributionBase& dist,
                            double& log_likelihood, double& aic, double& bic) noexcept {
    const int k = dist.getNumParameters();
    const double n = static_cast<double>(data.size());
    log_likelihood = 0.0;
    for (double x : data)
        log_likelihood += dist.getLogProbability(x);
    aic = 2.0 * k - 2.0 * log_likelihood;
    bic = k * std::log(n) - 2.0 * log_likelihood;
}

// =============================================================================
// RULE OF FIVE IMPLEMENTATION
// =============================================================================

DistributionBase::DistributionBase() {
    // Warm up SystemCapabilities on first construction so the SIMD detection
    // singleton is initialised before any batch call.
    detail::SystemCapabilities::current();
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

        // Compute log-likelihood, AIC, BIC via shared helper (same formula as
        // stats::analysis::informationCriteria; see compute_fit_ic comment above).
        // The previous isfinite guard is removed: silently skipping -inf
        // log-probs was incorrect (it inflated the log-likelihood for
        // out-of-support data). Letting -inf propagate is mathematically right.
        compute_fit_ic(data, *this,
                       results.log_likelihood, results.aic, results.bic);

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
    if (std::isnan(myMean) || std::isnan(otherMean) || std::abs(myMean - otherMean) > tolerance) {
        return false;
    }

    const double myVar = getVariance();
    const double otherVar = other.getVariance();
    if (std::isnan(myVar) || std::isnan(otherVar) || std::abs(myVar - otherVar) > tolerance) {
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

// numericalIntegration, adaptiveSimpsonIntegration, newtonRaphsonQuantile removed
// in v2.0.0 (Step 3B). Use detail::adaptive_simpson() and detail::newton_raphson()
// from math_utils.h directly. No derived distribution called these through the
// protected interface.

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

// erf/erfc/lgamma/gammaP/gammaQ protected wrappers removed in v2.0.0 (AQ-5).
// betaI_continued_fraction removed in v2.0.0 (Step 3B): defined but never
// called. Use detail::beta_i() from math_utils.h if needed.

}  // namespace stats

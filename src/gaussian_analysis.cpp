#include "libstats/stats/analysis/gaussian_analysis.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)

#include "libstats/core/math_utils.h"
#include "libstats/core/math_constants.h"
#include "libstats/core/statistical_constants.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>

namespace stats::analysis::gaussian {

// ---------------------------------------------------------------------------
// Normality tests
// ---------------------------------------------------------------------------

std::tuple<double, double, bool>
shapiroWilkTest(const std::vector<double>& data, double alpha) {
    if (data.size() < 3 || data.size() > detail::MAX_DATA_POINTS_FOR_SW_TEST)
        throw std::invalid_argument("Shapiro-Wilk test requires 3 to 5000 data points");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be between 0 and 1");

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();

    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
    double ss = 0.0;
    for (double x : sorted) ss += (x - mean) * (x - mean);

    // Shapiro-Wilk approximate coefficients via Blom (1958) expected order statistics.
    // Normalization factor m_norm_sq ensures W ∈ [0, 1] (Cauchy-Schwarz bound).
    double numerator = 0.0;
    double m_norm_sq = 0.0;
    for (std::size_t i = 0; i < n / 2; ++i) {
        // Blom plotting position (i is 0-indexed): p = (i + 0.625) / (n + 0.25)
        const double p_i = (static_cast<double>(i) + 1.0 - 0.375) /
                           (static_cast<double>(n) + 0.25);
        const double m_i = detail::inverse_normal_cdf(p_i);
        m_norm_sq += 2.0 * m_i * m_i;  // symmetric pair contributes 2*m_i^2
        numerator += m_i * (sorted[n - 1 - i] - sorted[i]);
    }

    const double w = (m_norm_sq > detail::ZERO && ss > detail::ZERO)
        ? std::min(1.0, (numerator * numerator) / (ss * m_norm_sq))
        : 0.0;

    // p-value: asymptotic approximation n·(1−W) ~ χ²(1) under H₀ (MC-13).
    // This formula is crude and known to over-reject for small n or extreme W.
    // Royston (1992) provides a more accurate normalising transformation of W;
    // implementing it requires distribution-specific coefficient tables. For
    // better accuracy on small samples (n < 50) prefer a table-lookup or Monte
    // Carlo p-value; this approximation is retained for its zero-dependency
    // property and is adequate for large samples and exploratory analysis.
    const double chi2_approx = static_cast<double>(n) * (1.0 - w);
    const double p_value = std::min(1.0, std::max(0.0, std::exp(-chi2_approx / 2.0)));

    return {w, p_value, p_value < alpha};
}

std::tuple<double, double, bool>
jarqueBeraTest(const std::vector<double>& data, double alpha) {
    if (data.size() < 8)
        throw std::invalid_argument("At least 8 data points required for Jarque-Bera test");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be between 0 and 1");

    const std::size_t n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;

    double m2 = 0.0, m3 = 0.0, m4 = 0.0;
    for (double x : data) {
        const double d = x - mean;
        const double d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
        m4 += d2 * d2;
    }
    m2 /= n; m3 /= n; m4 /= n;

    const double skewness = m3 / std::pow(m2, 1.5);
    const double kurtosis = m4 / (m2 * m2) - detail::EXCESS_KURTOSIS_OFFSET;

    const double jb = static_cast<double>(n) * (
        skewness * skewness / detail::SIX +
        kurtosis * kurtosis / detail::TWO_TWENTY_FIVE);

    const double p_value = 1.0 - detail::chi_squared_cdf(jb, detail::TWO);
    return {jb, p_value, p_value < alpha};
}

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

std::pair<double, double>
confidenceIntervalMean(const std::vector<double>& data,
                       double confidence_level,
                       bool population_variance_known) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be between 0 and 1");

    const std::size_t n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    const double alpha = 1.0 - confidence_level;
    double margin;

    // MC-7: Use Bessel-corrected (n-1) variance whenever the population variance
    // is not known, regardless of sample size. The biased estimator (n denominator)
    // is reserved for the truly-known-population case.
    if (population_variance_known) {
        // Population variance known: plug-in biased estimator, z critical value.
        const double var = std::inner_product(data.begin(), data.end(),
            data.begin(), 0.0) / n - mean * mean;
        const double z = detail::inverse_normal_cdf(1.0 - alpha * 0.5);
        margin = z * std::sqrt(var) / std::sqrt(static_cast<double>(n));
    } else {
        // Population variance unknown: Bessel-corrected estimator throughout;
        // switch between z (n >= 30, CLT) and t (n < 30) critical values.
        if (n < 2) {
            // Single observation: variance undefined; return degenerate interval.
            return {mean, mean};
        }
        const double var = std::inner_product(data.begin(), data.end(), data.begin(),
            0.0, std::plus<>(),
            [mean](double x, double y){ return (x - mean) * (y - mean); }) /
            static_cast<double>(n - 1);
        const double std_err = std::sqrt(var) / std::sqrt(static_cast<double>(n));
        if (n >= 30) {
            const double z = detail::inverse_normal_cdf(1.0 - alpha * 0.5);
            margin = z * std_err;
        } else {
            const double t = detail::inverse_t_cdf(1.0 - alpha * 0.5,
                                                   static_cast<double>(n - 1));
            margin = t * std_err;
        }
    }
    return {mean - margin, mean + margin};
}

std::pair<double, double>
confidenceIntervalVariance(const std::vector<double>& data,
                           double confidence_level) {
    if (data.size() < 2)
        throw std::invalid_argument(
            "At least 2 data points required for variance confidence interval");
    if (confidence_level <= 0.0 || confidence_level >= 1.0)
        throw std::invalid_argument("Confidence level must be between 0 and 1");

    const std::size_t n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    const double var = std::inner_product(data.begin(), data.end(), data.begin(),
        0.0, std::plus<>(),
        [mean](double x, double y){ return (x - mean) * (y - mean); }) /
        static_cast<double>(n - 1);

    const double alpha = 1.0 - confidence_level;
    const double df = static_cast<double>(n - 1);
    const double chi2_lo = detail::inverse_chi_squared_cdf(alpha * 0.5, df);
    const double chi2_hi = detail::inverse_chi_squared_cdf(1.0 - alpha * 0.5, df);

    return {df * var / chi2_hi, df * var / chi2_lo};
}

// ---------------------------------------------------------------------------
// T-tests
// ---------------------------------------------------------------------------

std::tuple<double, double, bool>
oneSampleTTest(const std::vector<double>& data,
               double hypothesized_mean,
               double alpha) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be between 0 and 1");

    const std::size_t n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    const double var = std::inner_product(data.begin(), data.end(), data.begin(),
        0.0, std::plus<>(),
        [mean](double x, double y){ return (x - mean) * (y - mean); }) /
        static_cast<double>(n - 1);

    const double t = (mean - hypothesized_mean) /
                     (std::sqrt(var) / std::sqrt(static_cast<double>(n)));
    const double p = 2.0 * (1.0 - detail::t_cdf(std::abs(t),
                                                  static_cast<double>(n - 1)));
    return {t, p, p < alpha};
}

std::tuple<double, double, bool>
twoSampleTTest(const std::vector<double>& data1,
               const std::vector<double>& data2,
               bool equal_variances,
               double alpha) {
    if (data1.empty() || data2.empty())
        throw std::invalid_argument("Both data vectors must be non-empty");
    if (alpha <= 0.0 || alpha >= 1.0)
        throw std::invalid_argument("Alpha must be between 0 and 1");

    const std::size_t n1 = data1.size(), n2 = data2.size();
    const double m1 = std::accumulate(data1.begin(), data1.end(), 0.0) / n1;
    const double m2 = std::accumulate(data2.begin(), data2.end(), 0.0) / n2;
    const double v1 = std::inner_product(data1.begin(), data1.end(), data1.begin(),
        0.0, std::plus<>(), [m1](double x, double y){ return (x-m1)*(y-m1); }) /
        static_cast<double>(n1 - 1);
    const double v2 = std::inner_product(data2.begin(), data2.end(), data2.begin(),
        0.0, std::plus<>(), [m2](double x, double y){ return (x-m2)*(y-m2); }) /
        static_cast<double>(n2 - 1);

    double t, df;
    if (equal_variances) {
        const double sp = ((n1-1)*v1 + (n2-1)*v2) / static_cast<double>(n1+n2-2);
        t = (m1 - m2) / std::sqrt(sp * (1.0/n1 + 1.0/n2));
        df = static_cast<double>(n1 + n2 - 2);
    } else {
        const double se = std::sqrt(v1/n1 + v2/n2);
        t = (m1 - m2) / se;
        const double num = std::pow(v1/n1 + v2/n2, 2.0);
        const double den = std::pow(v1/n1, 2.0)/(n1-1) + std::pow(v2/n2, 2.0)/(n2-1);
        df = num / den;
    }

    const double p = 2.0 * (1.0 - detail::t_cdf(std::abs(t), df));
    return {t, p, p < alpha};
}

std::tuple<double, double, bool>
pairedTTest(const std::vector<double>& data1,
            const std::vector<double>& data2,
            double alpha) {
    if (data1.size() != data2.size())
        throw std::invalid_argument("Data vectors must have the same size for paired t-test");
    if (data1.empty())
        throw std::invalid_argument("Data vectors cannot be empty");

    std::vector<double> diff(data1.size());
    std::transform(data1.begin(), data1.end(), data2.begin(), diff.begin(),
                   [](double a, double b){ return a - b; });
    return oneSampleTTest(diff, 0.0, alpha);
}

// ---------------------------------------------------------------------------
// Bayesian inference
// ---------------------------------------------------------------------------

std::tuple<double, double, double, double>
bayesianEstimation(const std::vector<double>& data,
                   double prior_mean, double prior_precision,
                   double prior_shape, double prior_rate) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    const std::size_t n = data.size();
    const double sm = std::accumulate(data.begin(), data.end(), 0.0) / n;
    const double ssq = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);

    const double post_prec = prior_precision + n;
    const double post_mean = (prior_precision * prior_mean + n * sm) / post_prec;
    const double post_shape = prior_shape + n / 2.0;
    const double sum_sq_dev = ssq - n * sm * sm;
    const double d = sm - prior_mean;
    const double post_rate = prior_rate + 0.5 * sum_sq_dev
        + 0.5 * (prior_precision * n * d * d) / post_prec;

    return {post_mean, post_prec, post_shape, post_rate};
}

std::pair<double, double>
bayesianCredibleInterval(const std::vector<double>& data,
                         double credibility_level,
                         double prior_mean, double prior_precision,
                         double prior_shape, double prior_rate) {
    auto [pm, pp, ps, pr] = bayesianEstimation(
        data, prior_mean, prior_precision, prior_shape, prior_rate);

    const double df    = 2.0 * ps;
    const double scale = std::sqrt(pr / (pp * ps));
    const double alpha = 1.0 - credibility_level;
    const double t_crit = detail::inverse_t_cdf(1.0 - alpha * 0.5, df);
    const double margin = t_crit * scale;

    return {pm - margin, pm + margin};
}

// ---------------------------------------------------------------------------
// Robust estimation
// ---------------------------------------------------------------------------

/// Compute IRLS weight for a standardised residual sr.
/// Throws std::invalid_argument on an unrecognised estimator_type.
static double compute_robust_weight(double sr, const std::string& type, double c) {
    const double asr = std::abs(sr);
    if (type == "huber")
        return (asr <= c) ? 1.0 : c / asr;
    if (type == "tukey")
        return (asr <= c) ? std::pow(1.0 - std::pow(sr / c, 2.0), 2.0) : 0.0;
    if (type == "hampel") {
        if (asr <= c)         return 1.0;
        if (asr <= 2.0 * c)   return c / asr;
        if (asr <= 3.0 * c)   return c * (3.0 - asr / c) / (2.0 * asr);
        return 0.0;
    }
    throw std::invalid_argument("Unknown estimator type. Use 'huber', 'tukey', or 'hampel'");
}

std::pair<double, double>
robustEstimation(const std::vector<double>& data,
                 const std::string& estimator_type,
                 double tuning_constant) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    // Initial location/scale from median + MAD
    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();

    const double median = (n % 2 == 0)
        ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0
        : sorted[n/2];

    std::vector<double> abs_dev(n);
    std::transform(data.begin(), data.end(), abs_dev.begin(),
                   [median](double x){ return std::abs(x - median); });
    std::sort(abs_dev.begin(), abs_dev.end());
    const double mad = (n % 2 == 0)
        ? (abs_dev[n/2 - 1] + abs_dev[n/2]) / 2.0
        : abs_dev[n/2];

    double loc = median;
    double scale = mad * detail::MAD_SCALING_FACTOR;

    // IRLS M-estimator loop (MC-8 documentation):
    // Scale update uses weighted RMS without Fisher-consistency factor;
    // Hampel uses equal weights in scale update (see MC-8 comment above).
    for (int iter = 0; iter < 50; ++iter) {
        double sum_w = 0.0, sum_wx = 0.0;
        for (double x : data) {
            const double w = compute_robust_weight(
                (x - loc) / scale, estimator_type, tuning_constant);
            sum_w += w;
            sum_wx += w * x;
        }
        const double new_loc = sum_wx / sum_w;

        // Scale update: Huber/Tukey use their weights; Hampel uses w=1 (see MC-8).
        double wsq = 0.0;
        for (double x : data) {
            const double r  = x - new_loc;
            const double sr = r / scale;
            const double w  = (estimator_type == "hampel")
                ? 1.0
                : compute_robust_weight(sr, estimator_type, tuning_constant);
            wsq += w * r * r;
        }
        const double new_scale = std::sqrt(wsq / sum_w);

        const bool converged = (std::abs(new_loc - loc) < detail::MIN_STD_DEV &&
                                std::abs(new_scale - scale) < detail::MIN_STD_DEV);
        loc = new_loc; scale = new_scale;
        if (converged) break;
    }
    return {loc, scale};
}

// ---------------------------------------------------------------------------
// Alternative estimators
// ---------------------------------------------------------------------------

std::pair<double, double>
methodOfMomentsEstimation(const std::vector<double>& data) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    const std::size_t n = data.size();
    const double mean = std::accumulate(data.begin(), data.end(), 0.0) / n;
    const double var  = std::inner_product(data.begin(), data.end(), data.begin(),
        0.0, std::plus<>(), [mean](double x, double y){ return (x-mean)*(y-mean); }) / n;
    return {mean, std::sqrt(var)};
}

std::pair<double, double>
lMomentsEstimation(const std::vector<double>& data) {
    if (data.size() < 2)
        throw std::invalid_argument("At least 2 data points required for L-moments estimation");

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t n = sorted.size();

    const double l1 = std::accumulate(sorted.begin(), sorted.end(), 0.0) / n;
    // Hosking (1990) unbiased PWM estimator for L2:
    //   b1 = (1/n) * sum_{i=0}^{n-1} (i / (n-1)) * sorted[i]
    //   l2 = 2*b1 - l1
    // The naive weighted sum gives (n-1)*l2, not l2/2, so divide by (n-1).
    double l2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double w = (2.0 * static_cast<double>(i) + 1.0 - static_cast<double>(n)) /
                         static_cast<double>(n);
        l2 += w * sorted[i];
    }
    // SA-3: The throw above guarantees n >= 2, so n - 1 is always positive.
    l2 /= static_cast<double>(n - 1);

    // For Gaussian: λ₂ = σ/√π, so σ = λ₂·√π
    return {l1, l2 * std::sqrt(detail::PI)};
}

std::vector<double>
calculateHigherMoments(const std::vector<double>& data, bool center_on_mean) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    const std::size_t n = data.size();
    const double mean = center_on_mean
        ? std::accumulate(data.begin(), data.end(), 0.0) / n
        : 0.0;

    std::vector<double> moments(6, 0.0);
    for (double x : data) {
        const double d = center_on_mean ? (x - mean) : x;
        for (int k = 0; k < 6; ++k)
            moments[static_cast<std::size_t>(k)] += std::pow(d, k + 1);
    }
    for (double& m : moments) m /= n;
    return moments;
}

}  // namespace stats::analysis::gaussian

#include "../include/distributions/gamma.h"

// Core functionality - lightweight headers
#include "../include/core/dispatch_utils.h"
#include "../include/core/log_space_ops.h"
#include "../include/core/math_utils.h"
#include "../include/core/mathematical_constants.h"
#include "../include/core/precision_constants.h"
#include "../include/core/safety.h"
#include "../include/core/statistical_constants.h"
#include "../include/core/threshold_constants.h"
#include "../include/core/validation.h"

// Platform headers - use forward declarations where available
#include "../include/common/cpu_detection_fwd.h"  // Lightweight CPU detection
// Note: parallel_execution.h is transitively included via dispatch_utils.h
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stats {

//==========================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==========================================================================

GammaDistribution::GammaDistribution(double alpha, double beta) {
    auto validation = validateGammaParameters(alpha, beta);
    if (validation.isError()) {
        throw std::invalid_argument(validation.message);
    }
    alpha_ = alpha;
    beta_ = beta;
    updateCacheUnsafe();
}

GammaDistribution::GammaDistribution(const GammaDistribution& other) {
    std::unique_lock lock(other.cache_mutex_);
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    cache_valid_ = other.cache_valid_;
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

GammaDistribution& GammaDistribution::operator=(const GammaDistribution& other) {
    if (this != &other) {
        std::scoped_lock lock(cache_mutex_, other.cache_mutex_);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        updateCacheUnsafe();
    }
    return *this;
}

GammaDistribution::GammaDistribution(GammaDistribution&& other) {
    std::scoped_lock lock(other.cache_mutex_);
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    cache_valid_ = other.cache_valid_;
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

GammaDistribution& GammaDistribution::operator=(GammaDistribution&& other) {
    if (this != &other) {
        std::scoped_lock lock(cache_mutex_, other.cache_mutex_);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        updateCacheUnsafe();
    }
    return *this;
}

// Destructor is explicitly defaulted in header - no definition needed here

//==========================================================================
// 2. SAFE FACTORY METHODS (Exception-free construction)
//==========================================================================

// Note: Safe factory methods are implemented inline in header for performance
// All create() and createWithScale() methods are header-only implementations

//==========================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==========================================================================

double GammaDistribution::getScale() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return scale_;
}

double GammaDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return mean_;
}

double GammaDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return variance_;
}

double GammaDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return detail::TWO / sqrtAlpha_;
}

double GammaDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return detail::SIX / alpha_;  // Direct computation is safe
}

double GammaDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (alpha_ < detail::ONE) {
        return detail::ZERO_DOUBLE;
    }
    return (alpha_ - detail::ONE) / beta_;
}

void GammaDistribution::setAlpha(double alpha) {
    // Copy current beta for validation (thread-safe)
    double currentBeta;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentBeta = beta_;
    }

    // Validate parameters
    validateParameters(alpha, currentBeta);

    // Update with unique lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::setBeta(double beta) {
    // Copy current alpha for validation (thread-safe)
    double currentAlpha;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentAlpha = alpha_;
    }

    // Validate parameters
    validateParameters(currentAlpha, beta);

    // Update with unique lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::setParameters(double alpha, double beta) {
    // Validate parameters
    validateParameters(alpha, beta);

    // Update with unique lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==========================================================================
// 4. RESULT-BASED SETTERS
//==========================================================================

VoidResult GammaDistribution::trySetAlpha(double alpha) noexcept {
    // Copy current beta for validation (thread-safe)
    double currentBeta;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentBeta = beta_;
    }

    auto validation = validateGammaParameters(alpha, currentBeta);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok(true);
}

VoidResult GammaDistribution::trySetBeta(double beta) noexcept {
    // Copy current alpha for validation (thread-safe)
    double currentAlpha;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentAlpha = alpha_;
    }

    auto validation = validateGammaParameters(currentAlpha, beta);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok(true);
}

VoidResult GammaDistribution::trySetParameters(double alpha, double beta) noexcept {
    auto validation = validateGammaParameters(alpha, beta);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok(true);
}

//==========================================================================
// 5. CORE PROBABILITY METHODS
//==========================================================================

double GammaDistribution::getProbability(double x) const {
    if (x < detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Handle special case x = 0
    if (x == detail::ZERO_DOUBLE) {
        return (alpha_ < detail::ONE)    ? std::numeric_limits<double>::infinity()
               : (alpha_ == detail::ONE) ? beta_
                                         : detail::ZERO_DOUBLE;
    }

    // Use log-space computation for numerical stability
    const double log_pdf = getLogProbability(x);
    return std::exp(log_pdf);
}

double GammaDistribution::getLogProbability(double x) const noexcept {
    if (x < detail::ZERO_DOUBLE) {
        return detail::NEGATIVE_INFINITY;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Handle special case x = 0
    if (x == detail::ZERO_DOUBLE) {
        if (alpha_ < detail::ONE) {
            return std::numeric_limits<double>::infinity();
        } else if (alpha_ == detail::ONE) {
            return logBeta_;  // log(β)
        } else {
            return detail::MIN_LOG_PROBABILITY;
        }
    }

    // General case: log(f(x)) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - βx
    return alphaLogBeta_ - logGammaAlpha_ + alphaMinusOne_ * std::log(x) - beta_ * x;
}

double GammaDistribution::getCumulativeProbability(double x) const {
    if (x <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Use regularized incomplete gamma function P(α, βx)
    return detail::gamma_p(alpha_, beta_ * x);
}

double GammaDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    // Use Newton-Raphson iteration with bracketing
    return computeQuantile(p);
}

double GammaDistribution::sample(std::mt19937& rng) const {
    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Choose sampling method based on α
    if (alpha_ >= detail::ONE) {
        return sampleMarsagliaTsang(rng);
    } else {
        return sampleAhrensDieter(rng);
    }
}

std::vector<double> GammaDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        samples.push_back(sample(rng));
    }

    return samples;
}

//==========================================================================
// 6. DISTRIBUTION MANAGEMENT
//==========================================================================

void GammaDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for non-positive values
    for (double value : values) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    // Start with method of moments as initial guess
    fitMethodOfMoments(values);

    // Refine with maximum likelihood estimation
    fitMaximumLikelihood(values);
}

void GammaDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                         std::vector<GammaDistribution>& results) {
    if (datasets.empty()) {
        // Handle empty datasets gracefully
        results.clear();
        return;
    }

    // Ensure results vector has correct size
    if (results.size() != datasets.size()) {
        results.resize(datasets.size());
    }

    const std::size_t num_datasets = datasets.size();

    // Use distribution-specific parallel thresholds for optimal work distribution
    if (arch::shouldUseDistributionParallel("gamma", "batch_fit", num_datasets)) {
        // Direct parallel execution without internal thresholds - bypass ParallelUtils limitation
        ThreadPool& pool = ParallelUtils::getGlobalThreadPool();
        const std::size_t optimal_grain_size = std::max(std::size_t{1}, num_datasets / 8);
        std::vector<std::future<void>> futures;
        futures.reserve((num_datasets + optimal_grain_size - 1) / optimal_grain_size);

        for (std::size_t i = 0; i < num_datasets; i += optimal_grain_size) {
            const std::size_t chunk_end = std::min(i + optimal_grain_size, num_datasets);

            auto future = pool.submit([&datasets, &results, i, chunk_end]() {
                for (std::size_t j = i; j < chunk_end; ++j) {
                    results[j].fit(datasets[j]);
                }
            });

            futures.push_back(std::move(future));
        }

        // Wait for all chunks to complete
        for (auto& future : futures) {
            future.wait();
        }

    } else {
        // Serial processing for small numbers of datasets
        for (std::size_t i = 0; i < num_datasets; ++i) {
            results[i].fit(datasets[i]);
        }
    }
}

void GammaDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = detail::ONE;
    beta_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string GammaDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "GammaDistribution(alpha=" << alpha_ << ", beta=" << beta_ << ")";
    return oss.str();
}

//==========================================================================
// 7. ADVANCED STATISTICAL METHODS
//==========================================================================

std::pair<double, double> GammaDistribution::confidenceIntervalShape(
    const std::vector<double>& data, double confidence_level) {
    if (data.empty() || confidence_level <= detail::ZERO_DOUBLE ||
        confidence_level >= detail::ONE) {
        throw std::invalid_argument("Invalid data or confidence level");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    // Get MLE estimates as starting point
    auto mle_estimates = methodOfMomentsEstimation(data);
    double alpha_hat = mle_estimates.first;

    // Calculate log-likelihood at MLE
    size_t n = data.size();
    [[maybe_unused]] double sum_log_x = detail::ZERO_DOUBLE;
    [[maybe_unused]] double sum_x = detail::ZERO_DOUBLE;
    for (double x : data) {
        sum_log_x += std::log(x);
        sum_x += x;
    }

    // Chi-square critical value
    [[maybe_unused]] double alpha_level = detail::ONE - confidence_level;
    double chi2_critical =
        detail::CHI2_95_DF_1;  // χ²(1, detail::ALPHA_05) ≈ detail::CHI2_95_DF_1 for 95% CI
    if (confidence_level == detail::CONFIDENCE_99) {
        chi2_critical = detail::CHI2_99_DF_1;
    }
    if (confidence_level == detail::CONFIDENCE_90) {
        chi2_critical = 2.706;
    }

    // Profile likelihood bounds (simplified approximation)
    // For large samples, use asymptotic normality
    double se_alpha = alpha_hat / std::sqrt(n);  // Approximate standard error
    double margin = std::sqrt(chi2_critical / detail::TWO) * se_alpha;

    double lower_bound = std::max(detail::ALPHA_001, alpha_hat - margin);
    double upper_bound = alpha_hat + margin;

    return {lower_bound, upper_bound};
}

std::pair<double, double> GammaDistribution::confidenceIntervalRate(const std::vector<double>& data,
                                                                    double confidence_level) {
    if (data.empty() || confidence_level <= detail::ZERO_DOUBLE ||
        confidence_level >= detail::ONE) {
        throw std::invalid_argument("Invalid data or confidence level");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    // Get MLE estimates as starting point
    auto mle_estimates = methodOfMomentsEstimation(data);
    double beta_hat = mle_estimates.second;

    size_t n = data.size();
    [[maybe_unused]] double sum_x = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);

    // Chi-square critical value
    [[maybe_unused]] double alpha_level = detail::ONE - confidence_level;
    double chi2_critical =
        detail::CHI2_95_DF_1;  // χ²(1, detail::ALPHA_05) ≈ detail::CHI2_95_DF_1 for 95% CI
    if (confidence_level == detail::CONFIDENCE_99) {
        chi2_critical = detail::CHI2_99_DF_1;
    }
    if (confidence_level == detail::CONFIDENCE_90) {
        chi2_critical = 2.706;
    }

    // For rate parameter β, use asymptotic normality
    // Variance of MLE for β is approximately β²/(n*α)
    double se_beta = beta_hat / std::sqrt(static_cast<double>(n) * mle_estimates.first);
    double margin = std::sqrt(chi2_critical / detail::TWO) * se_beta;

    double lower_bound = std::max(detail::ALPHA_001, beta_hat - margin);
    double upper_bound = beta_hat + margin;

    return {lower_bound, upper_bound};
}

std::tuple<double, double, bool> GammaDistribution::likelihoodRatioTest(
    const std::vector<double>& data, double null_shape, double null_rate,
    double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= detail::ZERO_DOUBLE || significance_level >= detail::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    if (null_shape <= detail::ZERO_DOUBLE || null_rate <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Null hypothesis parameters must be positive");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    size_t n = data.size();

    // Calculate sufficient statistics
    double sum_x = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);
    double sum_log_x = detail::ZERO_DOUBLE;
    for (double x : data) {
        sum_log_x += std::log(x);
    }

    // Log-likelihood under null hypothesis H0: (α₀, β₀)
    double log_likelihood_null = static_cast<double>(n) * null_shape * std::log(null_rate) -
                                 static_cast<double>(n) * std::lgamma(null_shape) +
                                 (null_shape - detail::ONE) * sum_log_x - null_rate * sum_x;

    // MLE estimates under alternative hypothesis H1
    auto mle_estimates = methodOfMomentsEstimation(data);
    double alpha_hat = mle_estimates.first;
    double beta_hat = mle_estimates.second;

    // Log-likelihood under alternative hypothesis H1: (α̂, β̂)
    double log_likelihood_alt = static_cast<double>(n) * alpha_hat * std::log(beta_hat) -
                                static_cast<double>(n) * std::lgamma(alpha_hat) +
                                (alpha_hat - detail::ONE) * sum_log_x - beta_hat * sum_x;

    // Likelihood ratio test statistic: -2 * ln(L₀/L₁) = 2 * (ln(L₁) - ln(L₀))
    double lr_statistic = detail::TWO * (log_likelihood_alt - log_likelihood_null);

    // Under H0, LR statistic follows χ²(2) distribution asymptotically
    // (2 degrees of freedom for 2 parameters: α and β)

    // Chi-square critical values for df=2
    double chi2_critical =
        detail::CHI2_95_DF_2;  // χ²(2, detail::ALPHA_05) ≈ detail::CHI2_95_DF_2 for 95% confidence
    if (significance_level == detail::ALPHA_01) {
        chi2_critical = detail::CHI2_99_DF_2;
    }
    if (significance_level == detail::ALPHA_10) {
        chi2_critical = 4.605;
    }

    // Approximate p-value using chi-square distribution
    // For a more accurate p-value, we would use the complementary gamma function
    double p_value = 1.0 - detail::gamma_p(1.0, lr_statistic / detail::TWO);  // Approximation

    // Reject null hypothesis if LR statistic > critical value
    bool reject_null = (lr_statistic > chi2_critical);

    return std::make_tuple(lr_statistic, p_value, reject_null);
}

std::tuple<double, double, double, double> GammaDistribution::bayesianEstimation(
    const std::vector<double>& data, double prior_shape_shape, double prior_shape_rate,
    double prior_rate_shape, double prior_rate_rate) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (prior_shape_shape <= detail::ZERO_DOUBLE || prior_shape_rate <= detail::ZERO_DOUBLE ||
        prior_rate_shape <= detail::ZERO_DOUBLE || prior_rate_rate <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Prior parameters must be positive");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    size_t n = data.size();

    // Calculate sufficient statistics
    [[maybe_unused]] double sum_x = std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE);
    [[maybe_unused]] double sum_log_x = detail::ZERO_DOUBLE;
    for (double x : data) {
        sum_log_x += std::log(x);
    }

    // For Gamma distribution with conjugate priors:
    // Shape parameter α ~ Gamma(prior_shape_shape, prior_shape_rate)
    // Rate parameter β ~ Gamma(prior_rate_shape, prior_rate_rate)
    //
    // However, these are not conjugate for both parameters simultaneously.
    // We use an approximation where we update each parameter separately
    // assuming the other is known (pseudo-conjugate approach)

    // Update for shape parameter α (assuming β is fixed at its prior mean)
    [[maybe_unused]] double prior_beta_mean = prior_rate_shape / prior_rate_rate;

    // Pseudo-conjugate update for α:
    // The likelihood contribution for α given β is proportional to
    // α^n * Γ(α)^(-n) * α^(sum_log_x)
    // This is not exactly conjugate, so we use method of moments approximation

    auto mom_estimates = methodOfMomentsEstimation(data);
    double alpha_estimate = mom_estimates.first;
    double beta_estimate = mom_estimates.second;

    // Combine prior and data information using weighted averages
    // Weight by effective sample sizes
    [[maybe_unused]] double effective_n_alpha = static_cast<double>(n) + prior_shape_shape;
    [[maybe_unused]] double effective_n_beta = static_cast<double>(n) + prior_rate_shape;

    // Posterior shape parameter (α) - approximate Bayesian update
    double posterior_alpha_shape = prior_shape_shape + static_cast<double>(n) * alpha_estimate;
    double posterior_alpha_rate = prior_shape_rate + static_cast<double>(n);

    // Posterior rate parameter (β) - approximate Bayesian update
    double posterior_beta_shape = prior_rate_shape + static_cast<double>(n) * beta_estimate;
    double posterior_beta_rate = prior_rate_rate + static_cast<double>(n);

    // Return posterior hyperparameters: (α_shape, α_rate, β_shape, β_rate)
    return std::make_tuple(posterior_alpha_shape, posterior_alpha_rate, posterior_beta_shape,
                           posterior_beta_rate);
}

std::pair<double, double> GammaDistribution::robustEstimation(const std::vector<double>& data,
                                                              const std::string& estimator_type,
                                                              double trim_proportion) {
    if (data.empty() || trim_proportion < detail::ZERO_DOUBLE || trim_proportion > detail::HALF) {
        throw std::invalid_argument("Invalid data or trim proportion");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    std::vector<double> processed_data;

    if (estimator_type == "trimmed") {
        // Trimmed estimator: remove extreme values
        size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(data.size()));
        if (trim_count * detail::TWO_INT >= data.size()) {
            throw std::invalid_argument("Trim proportion too large");
        }

        processed_data.assign(sorted_data.begin() + static_cast<std::ptrdiff_t>(trim_count),
                              sorted_data.end() - static_cast<std::ptrdiff_t>(trim_count));
    } else if (estimator_type == "winsorized") {
        // Winsorized estimator: replace extreme values with less extreme ones
        processed_data = sorted_data;
        size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(data.size()));

        if (trim_count > 0 && trim_count * detail::TWO_INT < data.size()) {
            double lower_bound = sorted_data[trim_count];
            double upper_bound = sorted_data[data.size() - trim_count - detail::ONE_INT];

            for (size_t i = 0; i < trim_count; ++i) {
                processed_data[i] = lower_bound;
                processed_data[data.size() - 1 - i] = upper_bound;
            }
        }
    } else if (estimator_type == "quantile") {
        // Quantile-based estimator: use interquartile range
        processed_data = sorted_data;
        size_t q1_idx = static_cast<size_t>(detail::QUARTER * static_cast<double>(data.size()));
        size_t q3_idx =
            static_cast<size_t>(detail::AD_P_VALUE_MEDIUM * static_cast<double>(data.size()));

        if (q3_idx > q1_idx) {
            processed_data.assign(sorted_data.begin() + static_cast<std::ptrdiff_t>(q1_idx),
                                  sorted_data.begin() + static_cast<std::ptrdiff_t>(q3_idx + 1));
        }
    } else {
        throw std::invalid_argument("Unknown estimator type: " + estimator_type);
    }

    if (processed_data.empty()) {
        throw std::invalid_argument("No data remaining after robust processing");
    }

    // Apply method of moments to the processed data
    return methodOfMomentsEstimation(processed_data);
}

std::pair<double, double> GammaDistribution::methodOfMomentsEstimation(
    const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    size_t n = data.size();

    // Calculate sample mean
    double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);

    // Calculate sample variance
    double sum_sq_diff = detail::ZERO_DOUBLE;
    for (double value : data) {
        double diff = value - sample_mean;
        sum_sq_diff += diff * diff;
    }
    double sample_variance = sum_sq_diff / static_cast<double>(n - 1);

    if (sample_variance <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Sample variance must be positive");
    }

    // Method of moments estimators:
    // α = (sample_mean)² / sample_variance
    // β = sample_mean / sample_variance
    double alpha_estimate = (sample_mean * sample_mean) / sample_variance;
    double beta_estimate = sample_mean / sample_variance;

    return {alpha_estimate, beta_estimate};
}

std::tuple<std::pair<double, double>, std::pair<double, double>>
GammaDistribution::bayesianCredibleInterval(const std::vector<double>& data,
                                            double credibility_level, double prior_shape_shape,
                                            double prior_shape_rate, double prior_rate_shape,
                                            double prior_rate_rate) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (credibility_level <= detail::ZERO_DOUBLE || credibility_level >= detail::ONE) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }
    if (prior_shape_shape <= detail::ZERO_DOUBLE || prior_shape_rate <= detail::ZERO_DOUBLE ||
        prior_rate_shape <= detail::ZERO_DOUBLE || prior_rate_rate <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("Prior parameters must be positive");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    // Get posterior hyperparameters from Bayesian estimation
    auto posterior_params = bayesianEstimation(data, prior_shape_shape, prior_shape_rate,
                                               prior_rate_shape, prior_rate_rate);

    double post_alpha_shape = std::get<0>(posterior_params);
    double post_alpha_rate = std::get<1>(posterior_params);
    double post_beta_shape = std::get<2>(posterior_params);
    double post_beta_rate = std::get<3>(posterior_params);

    // Calculate credible intervals for both parameters
    double alpha_tail = (detail::ONE - credibility_level) / detail::TWO;

    // For shape parameter α ~ Gamma(post_alpha_shape, post_alpha_rate)
    // Use gamma inverse CDF (quantile function)
    double alpha_lower =
        detail::gamma_inverse_cdf(alpha_tail, post_alpha_shape, detail::ONE / post_alpha_rate);
    double alpha_upper = detail::gamma_inverse_cdf(detail::ONE - alpha_tail, post_alpha_shape,
                                                   detail::ONE / post_alpha_rate);

    // For rate parameter β ~ Gamma(post_beta_shape, post_beta_rate)
    double beta_lower =
        detail::gamma_inverse_cdf(alpha_tail, post_beta_shape, detail::ONE / post_beta_rate);
    double beta_upper = detail::gamma_inverse_cdf(detail::ONE - alpha_tail, post_beta_shape,
                                                  detail::ONE / post_beta_rate);

    // Ensure bounds are positive and reasonable
    alpha_lower = std::max(alpha_lower, detail::MIN_STD_DEV);
    alpha_upper = std::max(alpha_upper, alpha_lower + detail::MIN_STD_DEV);
    beta_lower = std::max(beta_lower, detail::MIN_STD_DEV);
    beta_upper = std::max(beta_upper, beta_lower + detail::MIN_STD_DEV);

    return std::make_tuple(std::make_pair(alpha_lower, alpha_upper),
                           std::make_pair(beta_lower, beta_upper));
}

std::pair<double, double> GammaDistribution::lMomentsEstimation(const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for non-positive values
    for (double value : data) {
        if (value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }

    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    size_t n = data.size();

    // Calculate L-moments (first two)
    // L1 (L-mean) = mean of order statistics
    double L1 = std::accumulate(sorted_data.begin(), sorted_data.end(), detail::ZERO_DOUBLE) /
                static_cast<double>(n);

    // L2 (L-scale) = expectation of (X(2) - X(1)) for sample size 2
    double L2 = detail::ZERO_DOUBLE;
    for (size_t i = 0; i < n; ++i) {
        double weight = (detail::TWO * static_cast<double>(i) - static_cast<double>(n) + 1.0) /
                        static_cast<double>(n);
        L2 += weight * sorted_data[i];
    }
    L2 /= detail::TWO;

    if (L2 <= detail::ZERO_DOUBLE) {
        throw std::invalid_argument("L-scale must be positive");
    }

    // L-moment ratio
    double tau = L2 / L1;

    // For Gamma distribution, solve for parameters using L-moment relationships
    // This uses approximate relationships between L-moments and Gamma parameters
    // More robust than ordinary moments but requires iterative solution

    // Initial guess using method of moments approximation
    double cv = tau;  // Coefficient of variation approximation
    [[maybe_unused]] double alpha_estimate = detail::ONE / (cv * cv);
    [[maybe_unused]] double beta_estimate = alpha_estimate / L1;

    // Refine estimate (simplified - in practice would use iterative method)
    // For now, use method of moments on the sorted data as a robust approach
    return methodOfMomentsEstimation(data);
}

std::tuple<double, double, bool> GammaDistribution::normalApproximationTest(
    const std::vector<double>& data, double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= detail::ZERO_DOUBLE || significance_level >= detail::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    auto mle_estimates = methodOfMomentsEstimation(data);
    double alpha_hat = mle_estimates.first;
    [[maybe_unused]] double beta_hat = mle_estimates.second;

    size_t n = data.size();
    double normal_mean = alpha_hat;
    double normal_var = alpha_hat / static_cast<double>(n);
    double normal_sd = std::sqrt(normal_var);

    double threshold_z = detail::inverse_normal_cdf(detail::ONE - significance_level / detail::TWO);
    double lower_bound = normal_mean - threshold_z * normal_sd;
    double upper_bound = normal_mean + threshold_z * normal_sd;

    double sample_mean =
        std::accumulate(data.begin(), data.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);

    bool reject_null = (sample_mean < lower_bound || sample_mean > upper_bound);

    return std::make_tuple(lower_bound, upper_bound, reject_null);
}

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
//==========================================================================

std::tuple<double, double, bool> GammaDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data, const GammaDistribution& distribution,
    double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= detail::ZERO_DOUBLE || significance_level >= detail::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    // Use the overflow-safe KS statistic calculation from math_utils
    double ks_statistic = detail::calculate_ks_statistic(data, distribution);

    const size_t n = data.size();
    double critical_value = 1.36 / std::sqrt(n);  // Approximation for KS test critical value
    bool reject_null = ks_statistic > critical_value;

    // P-value calculation for KS test (improved asymptotic approximation)
    // Use Kolmogorov distribution approximation
    double lambda = std::sqrt(n) * ks_statistic;
    double p_value;

    if (lambda < 0.27) {
        p_value = detail::ONE;
    } else if (lambda < detail::ONE) {
        p_value = detail::ONE - detail::TWO * std::pow(lambda, 2) *
                                    (detail::ONE - detail::TWO * lambda * lambda / detail::THREE);
    } else {
        // Asymptotic series for large lambda
        p_value = detail::TWO * std::exp(-detail::TWO * lambda * lambda);
        // Add correction terms
        double correction = detail::ONE - detail::TWO * lambda * lambda / detail::THREE +
                            8.0 * std::pow(lambda, 4) / 15.0;
        p_value *= std::max(detail::ZERO_DOUBLE, correction);
    }

    // Ensure p-value is in valid range
    p_value = std::min(detail::ONE, std::max(detail::ZERO_DOUBLE, p_value));

    return std::make_tuple(ks_statistic, p_value, reject_null);
}

std::tuple<double, double, bool> GammaDistribution::andersonDarlingTest(
    const std::vector<double>& data, const GammaDistribution& distribution,
    double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= detail::ZERO_DOUBLE || significance_level >= detail::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    // Use the centralized AD statistic calculation from math_utils
    double ad_statistic = detail::calculate_ad_statistic(data, distribution);

    // Use the same p-value approximation as Gaussian distribution for consistency
    const double n = static_cast<double>(data.size());
    const double modified_stat =
        ad_statistic * (detail::ONE + detail::AD_P_VALUE_MEDIUM / n + 2.25 / (n * n));

    // Approximate p-value using exponential approximation
    double p_value;
    if (modified_stat >= 13.0) {
        p_value = detail::ZERO_DOUBLE;
    } else if (modified_stat >= detail::SIX) {
        p_value = std::exp(-1.28 * modified_stat);
    } else {
        p_value = std::exp(-1.8 * modified_stat + 1.5);
    }

    // Clamp p-value to [0, 1]
    p_value = std::min(detail::ONE, std::max(detail::ZERO_DOUBLE, p_value));

    const bool reject_null = p_value < significance_level;

    return std::make_tuple(ad_statistic, p_value, reject_null);
}

//==========================================================================
// 9. CROSS-VALIDATION METHODS
//==========================================================================

std::vector<std::tuple<double, double, double>> GammaDistribution::kFoldCrossValidation(
    const std::vector<double>& data, int k, unsigned int random_seed) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (k < 2 || k > static_cast<int>(data.size())) {
        throw std::invalid_argument("Invalid number of folds for k-fold cross-validation");
    }

    std::mt19937 rng(random_seed);
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<std::tuple<double, double, double>> results;

    size_t fold_size = data.size() / static_cast<size_t>(k);
    for (int fold = 0; fold < k; ++fold) {
        size_t start = static_cast<size_t>(fold) * fold_size;
        size_t end = static_cast<size_t>(fold + 1) * fold_size;

        std::vector<double> training_data;
        std::vector<double> validation_data;

        for (size_t i = 0; i < data.size(); ++i) {
            if (i >= start && i < end) {
                validation_data.push_back(data[indices[i]]);
            } else {
                training_data.push_back(data[indices[i]]);
            }
        }

        GammaDistribution trained_model;
        trained_model.fit(training_data);

        // Evaluate on validation data
        std::vector<double> absolute_errors;
        std::vector<double> squared_errors;
        double log_likelihood = detail::ZERO_DOUBLE;

        absolute_errors.reserve(validation_data.size());
        squared_errors.reserve(validation_data.size());

        // Calculate prediction errors and log-likelihood
        for (const auto& value : validation_data) {
            // For gamma distribution, the "prediction" is the mean
            const double predicted_mean = trained_model.getMean();

            const double absolute_error = std::abs(value - predicted_mean);
            const double squared_error = (value - predicted_mean) * (value - predicted_mean);

            absolute_errors.push_back(absolute_error);
            squared_errors.push_back(squared_error);

            log_likelihood += trained_model.getLogProbability(value);
        }

        // Calculate MAE and RMSE
        const double mae =
            std::accumulate(absolute_errors.begin(), absolute_errors.end(), detail::ZERO_DOUBLE) /
            static_cast<double>(absolute_errors.size());
        const double mse =
            std::accumulate(squared_errors.begin(), squared_errors.end(), detail::ZERO_DOUBLE) /
            static_cast<double>(squared_errors.size());
        const double rmse = std::sqrt(mse);

        results.emplace_back(mae, rmse, log_likelihood);
    }

    return results;
}

std::tuple<double, double, double> GammaDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    if (data.size() < 3) {
        throw std::invalid_argument("Insufficient data for leave-one-out cross-validation");
    }

    const size_t n = data.size();
    std::vector<double> absolute_errors;
    std::vector<double> squared_errors;
    double total_log_likelihood = detail::ZERO_DOUBLE;

    absolute_errors.reserve(n);
    squared_errors.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        std::vector<double> train_data = data;
        train_data.erase(train_data.begin() + static_cast<std::ptrdiff_t>(i));

        GammaDistribution model;
        model.fit(train_data);

        // Evaluate on left-out point
        // For gamma distribution, the "prediction" is the mean
        const double predicted_mean = model.getMean();
        const double actual_value = data[i];

        const double absolute_error = std::abs(actual_value - predicted_mean);
        const double squared_error =
            (actual_value - predicted_mean) * (actual_value - predicted_mean);

        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);

        total_log_likelihood += model.getLogProbability(actual_value);
    }

    // Calculate summary statistics
    const double mean_absolute_error =
        std::accumulate(absolute_errors.begin(), absolute_errors.end(), detail::ZERO_DOUBLE) /
        static_cast<double>(n);
    const double mean_squared_error =
        std::accumulate(squared_errors.begin(), squared_errors.end(), detail::ZERO_DOUBLE) /
        static_cast<double>(n);
    const double root_mean_squared_error = std::sqrt(mean_squared_error);

    return std::make_tuple(mean_absolute_error, root_mean_squared_error, total_log_likelihood);
}

//==========================================================================
// 10. INFORMATION CRITERIA
//==========================================================================

std::tuple<double, double, double, double> GammaDistribution::computeInformationCriteria(
    const std::vector<double>& data, const GammaDistribution& fitted_distribution) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    size_t n = data.size();
    int k = 2;  // Number of parameters (alpha and beta)

    // Calculate log-likelihood
    double log_likelihood = detail::ZERO_DOUBLE;
    for (double value : data) {
        log_likelihood += fitted_distribution.getLogProbability(value);
    }

    // AIC (Akaike Information Criterion)
    double aic = detail::TWO_INT * k - detail::TWO_INT * log_likelihood;

    // BIC (Bayesian Information Criterion)
    double bic = k * std::log(static_cast<double>(n)) - 2 * log_likelihood;

    // AICc (Corrected AIC for small sample sizes)
    double aicc = aic + (2 * k * (k + 1)) / static_cast<double>(n - static_cast<size_t>(k) - 1);

    return std::make_tuple(aic, bic, aicc, log_likelihood);
}

//==========================================================================
// 11. BOOTSTRAP METHODS
//==========================================================================

std::tuple<std::pair<double, double>, std::pair<double, double>>
GammaDistribution::bootstrapParameterConfidenceIntervals(const std::vector<double>& data,
                                                         double confidence_level, int n_bootstrap,
                                                         unsigned int random_seed) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= detail::ZERO_DOUBLE || confidence_level >= detail::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    if (n_bootstrap < 100) {
        throw std::invalid_argument("Number of bootstrap samples must be at least 100");
    }

    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);

    std::vector<double> bootstrap_alphas;
    std::vector<double> bootstrap_betas;
    bootstrap_alphas.reserve(static_cast<size_t>(n_bootstrap));
    bootstrap_betas.reserve(static_cast<size_t>(n_bootstrap));

    for (int i = 0; i < n_bootstrap; ++i) {
        // Generate bootstrap sample
        std::vector<double> bootstrap_sample;
        bootstrap_sample.reserve(data.size());

        for (size_t j = 0; j < data.size(); ++j) {
            bootstrap_sample.push_back(data[dist(rng)]);
        }

        // Fit distribution to bootstrap sample
        try {
            auto estimates = methodOfMomentsEstimation(bootstrap_sample);
            bootstrap_alphas.push_back(estimates.first);
            bootstrap_betas.push_back(estimates.second);
        } catch (const std::exception&) {
            // Skip invalid bootstrap samples
            --i;
        }
    }

    // Sort bootstrap estimates
    std::sort(bootstrap_alphas.begin(), bootstrap_alphas.end());
    std::sort(bootstrap_betas.begin(), bootstrap_betas.end());

    // Calculate confidence intervals using percentile method
    double alpha_level = (detail::ONE - confidence_level) / detail::TWO;
    size_t lower_idx =
        static_cast<size_t>(alpha_level * static_cast<double>(bootstrap_alphas.size()));
    size_t upper_idx =
        static_cast<size_t>((1.0 - alpha_level) * static_cast<double>(bootstrap_alphas.size())) - 1;

    double alpha_lower = bootstrap_alphas[lower_idx];
    double alpha_upper = bootstrap_alphas[upper_idx];
    double beta_lower = bootstrap_betas[lower_idx];
    double beta_upper = bootstrap_betas[upper_idx];

    return std::make_tuple(std::make_pair(alpha_lower, alpha_upper),
                           std::make_pair(beta_lower, beta_upper));
}

//==========================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==========================================================================

// Moved from inline methods in header for better compilation speed

double GammaDistribution::getAlphaAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicAlpha_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getAlpha();
}

double GammaDistribution::getBetaAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicBeta_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getBeta();
}

int GammaDistribution::getNumParameters() const noexcept {
    return 2;
}

std::string GammaDistribution::getDistributionName() const {
    return "Gamma";
}

bool GammaDistribution::isDiscrete() const noexcept {
    return false;
}

double GammaDistribution::getSupportLowerBound() const noexcept {
    return 0.0;
}

double GammaDistribution::getSupportUpperBound() const noexcept {
    return std::numeric_limits<double>::infinity();
}

VoidResult GammaDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateGammaParameters(alpha_, beta_);
}

double GammaDistribution::getMedian() const noexcept {
    return getQuantile(0.5);
}

bool GammaDistribution::operator!=(const GammaDistribution& other) const {
    return !(*this == other);
}

GammaDistribution GammaDistribution::createUnchecked(double alpha, double beta) noexcept {
    GammaDistribution dist(alpha, beta, true);  // bypass validation
    return dist;
}

GammaDistribution::GammaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept
    : DistributionBase(), alpha_(alpha), beta_(beta) {
    // Cache will be updated on first use
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Initialize atomic parameters to invalid state
    atomicAlpha_.store(alpha, std::memory_order_release);
    atomicBeta_.store(beta, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

bool GammaDistribution::isExponentialDistribution() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return isExponential_;
}

bool GammaDistribution::isChiSquaredDistribution() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return isChiSquared_;
}

double GammaDistribution::getDegreesOfFreedom() const {
    if (!isChiSquaredDistribution()) {
        throw std::logic_error(
            "Distribution is not a chi-squared distribution (beta != detail::HALF)");
    }
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return detail::TWO * alpha_;
}

double GammaDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // H(X) = α - log(β) + log(Γ(α)) + (1-α)ψ(α)
    return alpha_ - logBeta_ + logGammaAlpha_ + (detail::ONE - alpha_) * digammaAlpha_;
}

bool GammaDistribution::canUseNormalApproximation() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return isLargeAlpha_;
}

Result<GammaDistribution> GammaDistribution::createFromMoments(double mean,
                                                               double variance) noexcept {
    if (mean <= detail::ZERO_DOUBLE) {
        return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter,
                                                    "Mean must be positive");
    }
    if (variance <= detail::ZERO_DOUBLE) {
        return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter,
                                                    "Variance must be positive");
    }

    // Method of moments: α = mean²/variance, β = mean/variance
    double alpha = (mean * mean) / variance;
    double beta = mean / variance;

    return create(alpha, beta);
}

//==========================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS IMPLEMENTATION
//==========================================================================

void GammaDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                       const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<GammaDistribution>::distType(),
        detail::DistributionTraits<GammaDistribution>::complexity(),
        [](const GammaDistribution& dist, double value) { return dist.getProbability(value); },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha, cached_beta,
                                               cached_log_gamma_alpha, cached_alpha_log_beta,
                                               cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        res[i] =
                            std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                     cached_beta * x - cached_log_gamma_alpha);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        res[i] =
                            std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                     cached_beta * x - cached_log_gamma_alpha);
                    }
                }
            }
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                      cached_beta * x - cached_log_gamma_alpha);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Cache-Aware lambda: For continuous distributions, caching is counterproductive
            // Fallback to parallel execution which is faster and more predictable
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
            // This approach avoids the cache contention issues that caused performance regression
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                      cached_beta * x - cached_log_gamma_alpha);
                }
            });
        });
}

void GammaDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<GammaDistribution>::distType(),
        detail::DistributionTraits<GammaDistribution>::complexity(),
        [](const GammaDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, dist.alpha_, cached_beta,
                                                  cached_log_gamma_alpha, cached_alpha_log_beta,
                                                  cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else {
                        res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                                 cached_alpha_minus_one * std::log(x) - cached_beta * x;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else {
                        res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                                 cached_alpha_minus_one * std::log(x) - cached_beta * x;
                    }
                }
            }
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                             cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                             cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        });
}

void GammaDistribution::getCumulativeProbability(std::span<const double> values,
                                                 std::span<double> results,
                                                 const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<GammaDistribution>::distType(),
        detail::DistributionTraits<GammaDistribution>::complexity(),
        [](const GammaDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha,
                                                         cached_beta);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                    }
                }
            }
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        });
}

//==========================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==========================================================================

void GammaDistribution::getProbabilityWithStrategy(std::span<const double> values,
                                                   std::span<double> results,
                                                   detail::Strategy strategy) const {
    // GPU acceleration fallback - GPU implementation not yet available, use optimal CPU strategy
    if (strategy == detail::Strategy::GPU_ACCELERATED) {
        strategy = detail::Strategy::WORK_STEALING;
    }

    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
        [](const GammaDistribution& dist, double value) { return dist.getProbability(value); },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha, cached_beta,
                                               cached_log_gamma_alpha, cached_alpha_log_beta,
                                               cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                      cached_beta * x - cached_log_gamma_alpha);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                      cached_beta * x - cached_log_gamma_alpha);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe GPU-accelerated processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) -
                                      cached_beta * x - cached_log_gamma_alpha);
                }
            });
        });
}

void GammaDistribution::getLogProbabilityWithStrategy(std::span<const double> values,
                                                      std::span<double> results,
                                                      detail::Strategy strategy) const {
    // GPU acceleration fallback - GPU implementation not yet available, use optimal CPU strategy
    if (strategy == detail::Strategy::GPU_ACCELERATED) {
        strategy = detail::Strategy::WORK_STEALING;
    }

    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
        [](const GammaDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha, cached_beta,
                                                  cached_log_gamma_alpha, cached_alpha_log_beta,
                                                  cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                             cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                             cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe GPU-accelerated processing
            [[maybe_unused]] const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha +
                             cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        });
}

void GammaDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                                             std::span<double> results,
                                                             detail::Strategy strategy) const {
    // GPU acceleration fallback - GPU implementation not yet available, use optimal CPU strategy
    if (strategy == detail::Strategy::GPU_ACCELERATED) {
        strategy = detail::Strategy::WORK_STEALING;
    }

    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
        [](const GammaDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha,
                                                         cached_beta);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // GPU-Accelerated lambda: should use pool.parallelFor for dynamic load balancing
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<GammaDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe GPU-accelerated processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();

            // Use work-stealing pool for optimal load balancing and performance
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        });
}

//==========================================================================
// 15. COMPARISON OPERATORS
//==========================================================================

bool GammaDistribution::operator==(const GammaDistribution& other) const {
    // Use scoped_lock to prevent deadlock when comparing two distributions
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    // Compare parameters within tolerance
    return (std::abs(alpha_ - other.alpha_) <= detail::DEFAULT_TOLERANCE) &&
           (std::abs(beta_ - other.beta_) <= detail::DEFAULT_TOLERANCE);
}

//==========================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==========================================================================

std::ostream& operator<<(std::ostream& os, const GammaDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, GammaDistribution& dist) {
    std::string temp;
    double alpha, beta;

    // Expected format: "GammaDistribution(alpha=value, beta=value)"
    // Read "GammaDistribution(alpha="
    is >> temp;  // "GammaDistribution(alpha=value,"

    if (temp.find("GammaDistribution(alpha=") == 0) {
        // Extract alpha value
        size_t equals_pos = temp.find('=');
        size_t comma_pos = temp.find(',');
        if (equals_pos != std::string::npos && comma_pos != std::string::npos) {
            std::string alpha_str =
                temp.substr(equals_pos + detail::ONE_INT, comma_pos - equals_pos - detail::ONE_INT);
            alpha = std::stod(alpha_str);

            // Read "beta=value)"
            is >> temp;
            if (temp.find("beta=") == 0) {
                size_t beta_equals_pos = temp.find('=');
                size_t close_paren_pos = temp.find(')');
                if (beta_equals_pos != std::string::npos && close_paren_pos != std::string::npos) {
                    std::string beta_str =
                        temp.substr(beta_equals_pos + detail::ONE_INT,
                                    close_paren_pos - beta_equals_pos - detail::ONE_INT);
                    beta = std::stod(beta_str);

                    // Set parameters if valid
                    auto result = dist.trySetParameters(alpha, beta);
                    if (result.isError()) {
                        is.setstate(std::ios::failbit);
                    }
                } else {
                    is.setstate(std::ios::failbit);
                }
            } else {
                is.setstate(std::ios::failbit);
            }
        } else {
            is.setstate(std::ios::failbit);
        }
    } else {
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==============================================================================
// 17. PRIVATE FACTORY IMPLEMENTATION METHODS
//==============================================================================

// Note: Private factory implementation methods are currently inline in the header
// This section exists for standardization and documentation purposes

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void GammaDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                      std::size_t count,
                                                      [[maybe_unused]] double alpha, double beta,
                                                      double log_gamma_alpha, double alpha_log_beta,
                                                      double alpha_minus_one) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                results[i] = std::exp(alpha_log_beta - log_gamma_alpha +
                                      alpha_minus_one * std::log(values[i]) - beta * values[i]);
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, arch::simd::aligned_allocator<double>> log_values(count);
    std::vector<double, arch::simd::aligned_allocator<double>> exp_inputs(count);

    // Step 1: Handle negative values and compute log(values)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE) {
            log_values[i] = detail::MIN_LOG_PROBABILITY;  // Will be set to 0 later
        } else {
            log_values[i] = std::log(values[i]);
        }
    }

    // Step 2: Compute alpha_minus_one * log(values) using SIMD
    arch::simd::VectorOps::scalar_multiply(log_values.data(), alpha_minus_one, exp_inputs.data(),
                                           count);

    // Step 3: Add alpha_log_beta - log_gamma_alpha
    const double log_constant = alpha_log_beta - log_gamma_alpha;
    arch::simd::VectorOps::scalar_add(exp_inputs.data(), log_constant, exp_inputs.data(), count);

    // Step 4: Subtract beta * values
    arch::simd::VectorOps::scalar_multiply(values, -beta, log_values.data(), count);
    arch::simd::VectorOps::vector_add(exp_inputs.data(), log_values.data(), exp_inputs.data(),
                                      count);

    // Step 5: Apply vectorized exponential
    arch::simd::VectorOps::vector_exp(exp_inputs.data(), results, count);

    // Step 6: Handle negative input values (set to zero) - must be done after exp
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        }
    }
}

void GammaDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count,
                                                         [[maybe_unused]] double alpha, double beta,
                                                         double log_gamma_alpha,
                                                         double alpha_log_beta,
                                                         double alpha_minus_one) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
            } else {
                results[i] = alpha_log_beta - log_gamma_alpha +
                             alpha_minus_one * std::log(values[i]) - beta * values[i];
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, arch::simd::aligned_allocator<double>> log_values(count);
    std::vector<double, arch::simd::aligned_allocator<double>> beta_scaled_values(count);

    // Step 1: Handle negative values and compute log(values)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE) {
            log_values[i] = 0.0;  // Use 0 for now, will handle negative values at the end
        } else {
            log_values[i] = std::log(values[i]);
        }
    }

    // Step 2: Compute alpha_minus_one * log(values) using SIMD
    arch::simd::VectorOps::scalar_multiply(log_values.data(), alpha_minus_one, results, count);

    // Step 3: Add alpha_log_beta - log_gamma_alpha
    const double log_constant = alpha_log_beta - log_gamma_alpha;
    arch::simd::VectorOps::scalar_add(results, log_constant, results, count);

    // Step 4: Subtract beta * values using SIMD
    arch::simd::VectorOps::scalar_multiply(values, -beta, beta_scaled_values.data(), count);
    arch::simd::VectorOps::vector_add(results, beta_scaled_values.data(), results, count);

    // Step 5: Handle negative input values (set to MIN_LOG_PROBABILITY) - must be done after all
    // SIMD ops
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE) {
            results[i] = detail::MIN_LOG_PROBABILITY;
        }
    }
}

void GammaDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                double* results, std::size_t count,
                                                                double alpha,
                                                                double beta) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                results[i] = regularizedIncompleteGamma(alpha, beta * values[i]);
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary array for beta * values
    std::vector<double, arch::simd::aligned_allocator<double>> scaled_values(count);

    // Step 1: Compute beta * values using SIMD
    arch::simd::VectorOps::scalar_multiply(values, beta, scaled_values.data(), count);

    // Step 2: Apply regularized incomplete gamma function to each scaled value
    // Note: This function is not easily vectorizable, so we still use scalar loop
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else {
            results[i] = detail::gamma_p(alpha, scaled_values[i]);
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

double GammaDistribution::incompleteGamma(double a, double x) noexcept {
    // Lower incomplete gamma function γ(a,x) using series expansion or continued fractions
    if (x <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (a <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // For x < a+1, use series expansion
    if (x < a + detail::ONE) {
        // Series: γ(a,x) = e^(-x) * x^a * Σ(x^n / Γ(a+n+1)) for n=0 to ∞
        double sum = detail::ONE;
        double term = detail::ONE;
        double n = detail::ONE;

        // Continue series until convergence
        while (std::abs(term) > detail::ULTRA_HIGH_PRECISION_TOLERANCE * std::abs(sum) &&
               n < detail::MAX_BISECTION_ITERATIONS) {
            term *= x / (a + n - detail::ONE);
            sum += term;
            n += detail::ONE;
        }

        return std::exp(-x + a * std::log(x) - std::lgamma(a)) * sum;
    } else {
        // For x >= a+1, use continued fraction: γ(a,x) = Γ(a) - Γ(a,x)
        // where Γ(a,x) is computed using continued fraction
        double b = x + detail::ONE - a;
        double c = detail::LARGE_CONTINUED_FRACTION_VALUE;
        double d = detail::ONE / b;
        double h = d;

        for (std::size_t i = 1; i <= detail::MAX_CONTINUED_FRACTION_ITERATIONS; ++i) {
            double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
            b += detail::TWO;
            d = an * d + b;
            if (std::abs(d) < detail::ULTRA_SMALL_THRESHOLD) {
                d = detail::ULTRA_SMALL_THRESHOLD;
            }
            c = b + an / c;
            if (std::abs(c) < detail::ULTRA_SMALL_THRESHOLD) {
                c = detail::ULTRA_SMALL_THRESHOLD;
            }
            d = detail::ONE / d;
            double del = d * c;
            h *= del;
            if (std::abs(del - 1.0) < detail::ULTRA_HIGH_PRECISION_TOLERANCE) {
                break;
            }
        }

        double upper_incomplete = std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
        return std::tgamma(a) - upper_incomplete;
    }
}

double GammaDistribution::regularizedIncompleteGamma(double a, double x) noexcept {
    // Regularized lower incomplete gamma function P(a,x) = γ(a,x) / Γ(a)
    if (x <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (a <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return incompleteGamma(a, x) / std::tgamma(a);
}

double GammaDistribution::computeQuantile(double p) const noexcept {
    // Quantile function using Newton-Raphson iteration with initial guess
    if (p <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p >= detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    // Initial guess using Wilson-Hilferty transformation for large α
    double initial_guess;
    if (alpha_ > detail::ONE) {
        double h = detail::TWO / (detail::NINE * alpha_);
        double z = detail::inverse_normal_cdf(p);
        initial_guess = alpha_ * std::pow(detail::ONE - h + z * std::sqrt(h), 3) / beta_;
    } else {
        // For small α, use exponential approximation
        initial_guess = -std::log(detail::ONE - p) / beta_;
    }

    // Newton-Raphson iteration
    double x = std::max(initial_guess, detail::NEWTON_RAPHSON_TOLERANCE);
    const double tolerance = detail::HIGH_PRECISION_TOLERANCE;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;

    for (int i = 0; i < max_iterations; ++i) {
        double cdf = getCumulativeProbability(x);
        double pdf = getProbability(x);

        if (std::abs(cdf - p) < tolerance) {
            break;
        }

        if (pdf < detail::ULTRA_SMALL_THRESHOLD) {
            // If PDF is too small, use bisection method fallback
            break;
        }

        double delta = (cdf - p) / pdf;
        x = std::max(x - delta, x * 0.1);  // Ensure x stays positive

        if (std::abs(delta) < tolerance * x) {
            break;
        }
    }

    return x;
}

double GammaDistribution::sampleMarsagliaTsang(std::mt19937& rng) const noexcept {
    // Marsaglia-Tsang "squeeze" method for α ≥ 1
    // This is a fast rejection sampling method

    std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);
    std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);

    const double d = alpha_ - detail::ONE / detail::THREE;
    const double c = detail::ONE / std::sqrt(detail::NINE * d);

    while (true) {
        double x, v;

        do {
            x = normal(rng);
            v = detail::ONE + c * x;
        } while (v <= detail::ZERO_DOUBLE);

        v = v * v * v;
        double u = uniform(rng);

        // Quick accept
        if (u < detail::ONE - 0.0331 * (x * x) * (x * x)) {
            return d * v / beta_;
        }

        // Quick reject
        if (std::log(u) < detail::HALF * x * x + d * (detail::ONE - v + std::log(v))) {
            return d * v / beta_;
        }
    }
}

double GammaDistribution::sampleAhrensDieter(std::mt19937& rng) const noexcept {
    // Ahrens-Dieter acceptance-rejection method for α < 1
    std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);

    const double b = (detail::E + alpha_) / detail::E;

    while (true) {
        double u = uniform(rng);
        double p = b * u;

        if (p <= detail::ONE) {
            double x = std::pow(p, detail::ONE / alpha_);
            double u2 = uniform(rng);
            if (u2 <= std::exp(-x)) {
                return x / beta_;
            }
        } else {
            double x = -std::log((b - p) / alpha_);
            double u2 = uniform(rng);
            if (u2 <= std::pow(x, alpha_ - detail::ONE)) {
                return x / beta_;
            }
        }
    }
}

void GammaDistribution::fitMethodOfMoments(const std::vector<double>& values) {
    // Method of moments parameter estimation
    if (values.empty()) {
        return;
    }

    auto estimates = methodOfMomentsEstimation(values);

    // Update parameters using the estimates
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = estimates.first;
    beta_ = estimates.second;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::fitMaximumLikelihood(const std::vector<double>& values) {
    // Maximum likelihood estimation using Newton-Raphson iteration
    if (values.empty()) {
        return;
    }

    size_t n = values.size();
    double sum_x = std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE);
    double sum_log_x = detail::ZERO_DOUBLE;
    for (double x : values) {
        sum_log_x += std::log(x);
    }

    double mean_x = sum_x / static_cast<double>(n);
    double mean_log_x = sum_log_x / static_cast<double>(n);

    // Initial guess using method of moments
    double s = std::log(mean_x) - mean_log_x;
    double alpha_est =
        (detail::THREE - s + std::sqrt((s - detail::THREE) * (s - detail::THREE) + 24.0 * s)) /
        (12.0 * s);

    // Newton-Raphson iteration for α
    const double tolerance = detail::NEWTON_RAPHSON_TOLERANCE;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;

    for (int i = 0; i < max_iterations; ++i) {
        double digamma_alpha = GammaDistribution::computeDigamma(alpha_est);
        double trigamma_alpha = GammaDistribution::computeTrigamma(alpha_est);

        double f = std::log(alpha_est) - digamma_alpha - s;
        double df = detail::ONE / alpha_est - trigamma_alpha;

        if (std::abs(f) < tolerance) {
            break;
        }

        alpha_est = alpha_est - f / df;
        alpha_est = std::max(alpha_est, detail::NEWTON_RAPHSON_TOLERANCE);  // Ensure positive
    }

    double beta_est = alpha_est / mean_x;

    // Update parameters
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha_est;
    beta_ = beta_est;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==============================================================================
// 20. PRIVATE UTILITY METHODS
//==============================================================================

double GammaDistribution::computeDigamma(double x) noexcept {
    // Digamma function ψ(x) = d/dx log(Γ(x))
    // Uses asymptotic series for large x and reflection formula for small x

    if (x <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // For small x, use reflection formula: ψ(x) = ψ(1-x) - π*cot(π*x)
    // But this is complex, so we use recurrence relation: ψ(x+1) = ψ(x) + 1/x
    double result = detail::ZERO_DOUBLE;
    double z = x;

    // Use recurrence to get z >= 8 for asymptotic series
    while (z < 8.0) {
        result -= detail::ONE / z;
        z += detail::ONE;
    }

    // Asymptotic series for large z
    // ψ(z) ≈ log(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - 1/(252z⁶) + ...
    double z_inv = detail::ONE / z;
    double z_inv_sq = z_inv * z_inv;

    result += std::log(z) - detail::HALF * z_inv;
    result -= z_inv_sq / 12.0;                         // Bernoulli B₂/2
    result += z_inv_sq * z_inv_sq / 120.0;             // Bernoulli B₄/4
    result -= z_inv_sq * z_inv_sq * z_inv_sq / 252.0;  // Bernoulli B₆/6

    return result;
}

double GammaDistribution::computeTrigamma(double x) noexcept {
    // Trigamma function ψ'(x) = d²/dx² log(Γ(x))
    // Uses asymptotic series for large x and recurrence relation

    if (x <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double result = detail::ZERO_DOUBLE;
    double z = x;

    // Use recurrence: ψ'(x+1) = ψ'(x) - 1/x²
    while (z < 8.0) {
        result += detail::ONE / (z * z);
        z += detail::ONE;
    }

    // Asymptotic series for large z
    // ψ'(z) ≈ 1/z + 1/(2z²) + 1/(6z³) - 1/(30z⁵) + 1/(42z⁷) - ...
    double z_inv = detail::ONE / z;
    double z_inv_sq = z_inv * z_inv;

    result += z_inv + detail::HALF * z_inv_sq;
    result += z_inv_sq * z_inv / detail::SIX;                 // 1/(6z³)
    result -= z_inv_sq * z_inv_sq * z_inv / 30.0;             // -1/(30z⁵)
    result += z_inv_sq * z_inv_sq * z_inv_sq * z_inv / 42.0;  // 1/(42z⁷)

    return result;
}

//==============================================================================
// 21. DISTRIBUTION PARAMETERS
//==============================================================================

// Note: Distribution parameters are declared in the header as private member variables
// This section exists for standardization and documentation purposes

//==============================================================================
// 22. PERFORMANCE CACHE
//==============================================================================

// Note: Performance cache variables are declared in the header as mutable private members
// This section exists for standardization and documentation purposes

//==============================================================================
// 23. OPTIMIZATION FLAGS
//==============================================================================

// Note: Optimization flags are declared in the header as private member variables
// This section exists for standardization and documentation purposes

//==============================================================================
// 24. SPECIALIZED CACHES
//==============================================================================

// Note: Specialized caches are declared in the header as private member variables
// This section exists for standardization and documentation purposes

}  // namespace stats

#include "../include/distributions/gamma.h"
#include "../include/core/error_handling.h"
#include "../include/core/math_utils.h"
#include "../include/platform/parallel_execution.h"
#include "../include/platform/thread_pool.h"
#include "../include/platform/work_stealing_pool.h"
#include "../include/core/performance_dispatcher.h"
#include "../include/core/dispatch_utils.h" // For DispatchUtils::autoDispatch
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <iostream>
#include <sstream>
#include <chrono>

namespace libstats {

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
    return constants::math::TWO / sqrtAlpha_;
}

double GammaDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return 6.0 / alpha_;  // Direct computation is safe
}

double GammaDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (alpha_ < 1.0) {
        return 0.0;
    }
    return (alpha_ - 1.0) / beta_;
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
    if (x < 0.0) {
        return 0.0;
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
    if (x == 0.0) {
        return (alpha_ < 1.0) ? std::numeric_limits<double>::infinity() : 
               (alpha_ == 1.0) ? beta_ : 0.0;
    }
    
    // Use log-space computation for numerical stability
    const double log_pdf = getLogProbability(x);
    return std::exp(log_pdf);
}

double GammaDistribution::getLogProbability(double x) const noexcept {
    if (x < 0.0) {
        return constants::probability::NEGATIVE_INFINITY;
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
    if (x == 0.0) {
        if (alpha_ < 1.0) {
            return std::numeric_limits<double>::infinity();
        } else if (alpha_ == 1.0) {
            return logBeta_;  // log(β)
        } else {
            return constants::probability::MIN_LOG_PROBABILITY;
        }
    }
    
    // General case: log(f(x)) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - βx
    return alphaLogBeta_ - logGammaAlpha_ + alphaMinusOne_ * std::log(x) - beta_ * x;
}

double GammaDistribution::getCumulativeProbability(double x) const {
    if (x <= 0.0) {
        return 0.0;
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
    return math::gamma_p(alpha_, beta_ * x);
}

double GammaDistribution::getQuantile(double p) const {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    if (p == 0.0) {
        return 0.0;
    }
    if (p == 1.0) {
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
    if (alpha_ >= 1.0) {
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
        if (value <= 0.0) {
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
    if (parallel::shouldUseDistributionParallel("gamma", "batch_fit", num_datasets)) {
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
    alpha_ = constants::math::ONE;
    beta_ = constants::math::ONE;
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
    if (data.empty() || confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("Invalid data or confidence level");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    // Get MLE estimates as starting point
    auto mle_estimates = methodOfMomentsEstimation(data);
    double alpha_hat = mle_estimates.first;
    
    // Calculate log-likelihood at MLE
    size_t n = data.size();
    [[maybe_unused]] double sum_log_x = 0.0;
    [[maybe_unused]] double sum_x = 0.0;
    for (double x : data) {
        sum_log_x += std::log(x);
        sum_x += x;
    }
    
    // Chi-square critical value
    [[maybe_unused]] double alpha_level = 1.0 - confidence_level;
    double chi2_critical = 3.841;  // χ²(1, 0.05) ≈ 3.841 for 95% CI
    if (confidence_level == 0.99) {
        chi2_critical = 6.635;
    }
    if (confidence_level == 0.90) {
        chi2_critical = 2.706;
    }
    
    // Profile likelihood bounds (simplified approximation)
    // For large samples, use asymptotic normality
    double se_alpha = alpha_hat / std::sqrt(n);  // Approximate standard error
    double margin = std::sqrt(chi2_critical / 2.0) * se_alpha;
    
    double lower_bound = std::max(0.001, alpha_hat - margin);
    double upper_bound = alpha_hat + margin;
    
    return {lower_bound, upper_bound};
}

std::pair<double, double> GammaDistribution::confidenceIntervalRate(
    const std::vector<double>& data, double confidence_level) {
    if (data.empty() || confidence_level <= 0.0 || confidence_level >= 1.0) {
        throw std::invalid_argument("Invalid data or confidence level");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    // Get MLE estimates as starting point
    auto mle_estimates = methodOfMomentsEstimation(data);
    double beta_hat = mle_estimates.second;
    
    size_t n = data.size();
    [[maybe_unused]] double sum_x = std::accumulate(data.begin(), data.end(), 0.0);
    
    // Chi-square critical value
    [[maybe_unused]] double alpha_level = 1.0 - confidence_level;
    double chi2_critical = 3.841;  // χ²(1, 0.05) ≈ 3.841 for 95% CI
    if (confidence_level == 0.99) {
        chi2_critical = 6.635;
    }
    if (confidence_level == 0.90) {
        chi2_critical = 2.706;
    }
    
    // For rate parameter β, use asymptotic normality
    // Variance of MLE for β is approximately β²/(n*α)
    double se_beta = beta_hat / std::sqrt(static_cast<double>(n) * mle_estimates.first);
    double margin = std::sqrt(chi2_critical / 2.0) * se_beta;
    
    double lower_bound = std::max(0.001, beta_hat - margin);
    double upper_bound = beta_hat + margin;
    
    return {lower_bound, upper_bound};
}

std::tuple<double, double, bool> GammaDistribution::likelihoodRatioTest(
    const std::vector<double>& data, double null_shape, double null_rate,
    double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    if (null_shape <= 0.0 || null_rate <= 0.0) {
        throw std::invalid_argument("Null hypothesis parameters must be positive");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    size_t n = data.size();
    
    // Calculate sufficient statistics
    double sum_x = std::accumulate(data.begin(), data.end(), 0.0);
    double sum_log_x = 0.0;
    for (double x : data) {
        sum_log_x += std::log(x);
    }
    
    // Log-likelihood under null hypothesis H0: (α₀, β₀)
    double log_likelihood_null = static_cast<double>(n) * null_shape * std::log(null_rate) 
                               - static_cast<double>(n) * std::lgamma(null_shape)
                               + (null_shape - 1.0) * sum_log_x
                               - null_rate * sum_x;
    
    // MLE estimates under alternative hypothesis H1
    auto mle_estimates = methodOfMomentsEstimation(data);
    double alpha_hat = mle_estimates.first;
    double beta_hat = mle_estimates.second;
    
    // Log-likelihood under alternative hypothesis H1: (α̂, β̂)
    double log_likelihood_alt = static_cast<double>(n) * alpha_hat * std::log(beta_hat)
                              - static_cast<double>(n) * std::lgamma(alpha_hat)
                              + (alpha_hat - 1.0) * sum_log_x
                              - beta_hat * sum_x;
    
    // Likelihood ratio test statistic: -2 * ln(L₀/L₁) = 2 * (ln(L₁) - ln(L₀))
    double lr_statistic = 2.0 * (log_likelihood_alt - log_likelihood_null);
    
    // Under H0, LR statistic follows χ²(2) distribution asymptotically
    // (2 degrees of freedom for 2 parameters: α and β)
    
    // Chi-square critical values for df=2
    double chi2_critical = 5.991;  // χ²(2, 0.05) ≈ 5.991 for 95% confidence
    if (significance_level == 0.01) {
        chi2_critical = 9.210;
    }
    if (significance_level == 0.10) {
        chi2_critical = 4.605;
    }
    
    // Approximate p-value using chi-square distribution
    // For a more accurate p-value, we would use the complementary gamma function
    double p_value = 1.0 - math::gamma_p(1.0, lr_statistic / 2.0);  // Approximation
    
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
    if (prior_shape_shape <= 0.0 || prior_shape_rate <= 0.0 ||
        prior_rate_shape <= 0.0 || prior_rate_rate <= 0.0) {
        throw std::invalid_argument("Prior parameters must be positive");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    size_t n = data.size();
    
    // Calculate sufficient statistics
    [[maybe_unused]] double sum_x = std::accumulate(data.begin(), data.end(), 0.0);
    [[maybe_unused]] double sum_log_x = 0.0;
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
    return std::make_tuple(posterior_alpha_shape, posterior_alpha_rate,
                          posterior_beta_shape, posterior_beta_rate);
}

std::pair<double, double> GammaDistribution::robustEstimation(
    const std::vector<double>& data, const std::string& estimator_type,
    double trim_proportion) {
    if (data.empty() || trim_proportion < 0.0 || trim_proportion > 0.5) {
        throw std::invalid_argument("Invalid data or trim proportion");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    std::vector<double> processed_data;
    
    if (estimator_type == "trimmed") {
        // Trimmed estimator: remove extreme values
        size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(data.size()));
        if (trim_count * 2 >= data.size()) {
            throw std::invalid_argument("Trim proportion too large");
        }
        
        processed_data.assign(sorted_data.begin() + static_cast<std::ptrdiff_t>(trim_count), 
                             sorted_data.end() - static_cast<std::ptrdiff_t>(trim_count));
    } 
    else if (estimator_type == "winsorized") {
        // Winsorized estimator: replace extreme values with less extreme ones
        processed_data = sorted_data;
        size_t trim_count = static_cast<size_t>(trim_proportion * static_cast<double>(data.size()));
        
        if (trim_count > 0 && trim_count * 2 < data.size()) {
            double lower_bound = sorted_data[trim_count];
            double upper_bound = sorted_data[data.size() - trim_count - 1];
            
            for (size_t i = 0; i < trim_count; ++i) {
                processed_data[i] = lower_bound;
                processed_data[data.size() - 1 - i] = upper_bound;
            }
        }
    }
    else if (estimator_type == "quantile") {
        // Quantile-based estimator: use interquartile range
        processed_data = sorted_data;
        size_t q1_idx = static_cast<size_t>(0.25 * static_cast<double>(data.size()));
        size_t q3_idx = static_cast<size_t>(0.75 * static_cast<double>(data.size()));
        
        if (q3_idx > q1_idx) {
            processed_data.assign(sorted_data.begin() + static_cast<std::ptrdiff_t>(q1_idx), 
                                 sorted_data.begin() + static_cast<std::ptrdiff_t>(q3_idx + 1));
        }
    }
    else {
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
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    size_t n = data.size();
    
    // Calculate sample mean
    double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);
    
    // Calculate sample variance
    double sum_sq_diff = 0.0;
    for (double value : data) {
        double diff = value - sample_mean;
        sum_sq_diff += diff * diff;
    }
    double sample_variance = sum_sq_diff / static_cast<double>(n - 1);
    
    if (sample_variance <= 0.0) {
        throw std::invalid_argument("Sample variance must be positive");
    }
    
    // Method of moments estimators:
    // α = (sample_mean)² / sample_variance
    // β = sample_mean / sample_variance
    double alpha_estimate = (sample_mean * sample_mean) / sample_variance;
    double beta_estimate = sample_mean / sample_variance;
    
    return {alpha_estimate, beta_estimate};
}

std::tuple<std::pair<double, double>, std::pair<double, double>> GammaDistribution::bayesianCredibleInterval(
    const std::vector<double>& data, double credibility_level,
    double prior_shape_shape, double prior_shape_rate,
    double prior_rate_shape, double prior_rate_rate) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (credibility_level <= 0.0 || credibility_level >= 1.0) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }
    if (prior_shape_shape <= 0.0 || prior_shape_rate <= 0.0 ||
        prior_rate_shape <= 0.0 || prior_rate_rate <= 0.0) {
        throw std::invalid_argument("Prior parameters must be positive");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
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
    double alpha_tail = (1.0 - credibility_level) / 2.0;
    
    // For shape parameter α ~ Gamma(post_alpha_shape, post_alpha_rate)
    // Use gamma inverse CDF (quantile function)
    double alpha_lower = math::gamma_inverse_cdf(alpha_tail, post_alpha_shape, 1.0 / post_alpha_rate);
    double alpha_upper = math::gamma_inverse_cdf(1.0 - alpha_tail, post_alpha_shape, 1.0 / post_alpha_rate);
    
    // For rate parameter β ~ Gamma(post_beta_shape, post_beta_rate)
    double beta_lower = math::gamma_inverse_cdf(alpha_tail, post_beta_shape, 1.0 / post_beta_rate);
    double beta_upper = math::gamma_inverse_cdf(1.0 - alpha_tail, post_beta_shape, 1.0 / post_beta_rate);
    
    // Ensure bounds are positive and reasonable
    alpha_lower = std::max(alpha_lower, 1e-6);
    alpha_upper = std::max(alpha_upper, alpha_lower + 1e-6);
    beta_lower = std::max(beta_lower, 1e-6);
    beta_upper = std::max(beta_upper, beta_lower + 1e-6);
    
    return std::make_tuple(std::make_pair(alpha_lower, alpha_upper),
                          std::make_pair(beta_lower, beta_upper));
}

std::pair<double, double> GammaDistribution::lMomentsEstimation(
    const std::vector<double>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for non-positive values
    for (double value : data) {
        if (value <= 0.0) {
            throw std::invalid_argument("All values must be positive for Gamma distribution");
        }
    }
    
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t n = data.size();
    
    // Calculate L-moments (first two)
    // L1 (L-mean) = mean of order statistics
    double L1 = std::accumulate(sorted_data.begin(), sorted_data.end(), 0.0) / static_cast<double>(n);
    
    // L2 (L-scale) = expectation of (X(2) - X(1)) for sample size 2
    double L2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double weight = (2.0 * static_cast<double>(i) - static_cast<double>(n) + 1.0) / static_cast<double>(n);
        L2 += weight * sorted_data[i];
    }
    L2 /= 2.0;
    
    if (L2 <= 0.0) {
        throw std::invalid_argument("L-scale must be positive");
    }
    
    // L-moment ratio
    double tau = L2 / L1;
    
    // For Gamma distribution, solve for parameters using L-moment relationships
    // This uses approximate relationships between L-moments and Gamma parameters
    // More robust than ordinary moments but requires iterative solution
    
    // Initial guess using method of moments approximation
    double cv = tau;  // Coefficient of variation approximation
    [[maybe_unused]] double alpha_estimate = 1.0 / (cv * cv);
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
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }
    
    auto mle_estimates = methodOfMomentsEstimation(data);
    double alpha_hat = mle_estimates.first;
    [[maybe_unused]] double beta_hat = mle_estimates.second;

    size_t n = data.size();
    double normal_mean = alpha_hat;
    double normal_var = alpha_hat / static_cast<double>(n);
    double normal_sd = std::sqrt(normal_var);

    double threshold_z = math::inverse_normal_cdf(1.0 - significance_level / 2.0);
    double lower_bound = normal_mean - threshold_z * normal_sd;
    double upper_bound = normal_mean + threshold_z * normal_sd;

    double sample_mean = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(n);

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
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    // Use the overflow-safe KS statistic calculation from math_utils
    double ks_statistic = math::calculate_ks_statistic(data, distribution);
    
    const size_t n = data.size();
    double critical_value = 1.36 / std::sqrt(n); // Approximation for KS test critical value
    bool reject_null = ks_statistic > critical_value;

    // P-value calculation for KS test (improved asymptotic approximation)
    // Use Kolmogorov distribution approximation
    double lambda = std::sqrt(n) * ks_statistic;
    double p_value;
    
    if (lambda < 0.27) {
        p_value = 1.0;
    } else if (lambda < 1.0) {
        p_value = 1.0 - 2.0 * std::pow(lambda, 2) * (1.0 - 2.0 * lambda * lambda / 3.0);
    } else {
        // Asymptotic series for large lambda
        p_value = 2.0 * std::exp(-2.0 * lambda * lambda);
        // Add correction terms
        double correction = 1.0 - 2.0 * lambda * lambda / 3.0 + 8.0 * std::pow(lambda, 4) / 15.0;
        p_value *= std::max(0.0, correction);
    }
    
    // Ensure p-value is in valid range
    p_value = std::min(1.0, std::max(0.0, p_value));

    return std::make_tuple(ks_statistic, p_value, reject_null);
}

std::tuple<double, double, bool> GammaDistribution::andersonDarlingTest(
    const std::vector<double>& data, const GammaDistribution& distribution,
    double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (significance_level <= 0.0 || significance_level >= 1.0) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    // Use the centralized AD statistic calculation from math_utils
    double ad_statistic = math::calculate_ad_statistic(data, distribution);

    // Use the same p-value approximation as Gaussian distribution for consistency
    const double n = static_cast<double>(data.size());
    const double modified_stat = ad_statistic * (1.0 + 0.75 / n + 2.25 / (n * n));
    
    // Approximate p-value using exponential approximation
    double p_value;
    if (modified_stat >= 13.0) {
        p_value = 0.0;
    } else if (modified_stat >= 6.0) {
        p_value = std::exp(-1.28 * modified_stat);
    } else {
        p_value = std::exp(-1.8 * modified_stat + 1.5);
    }
    
    // Clamp p-value to [0, 1]
    p_value = std::min(1.0, std::max(0.0, p_value));
    
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
        double log_likelihood = 0.0;
        
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
        const double mae = std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / static_cast<double>(absolute_errors.size());
        const double mse = std::accumulate(squared_errors.begin(), squared_errors.end(), 0.0) / static_cast<double>(squared_errors.size());
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
    double total_log_likelihood = 0.0;
    
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
        const double squared_error = (actual_value - predicted_mean) * (actual_value - predicted_mean);
        
        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);
        
        total_log_likelihood += model.getLogProbability(actual_value);
    }

    // Calculate summary statistics
    const double mean_absolute_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), 0.0) / static_cast<double>(n);
    const double mean_squared_error = std::accumulate(squared_errors.begin(), squared_errors.end(), 0.0) / static_cast<double>(n);
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
    int k = 2; // Number of parameters (alpha and beta)

    // Calculate log-likelihood
    double log_likelihood = 0.0;
    for (double value : data) {
        log_likelihood += fitted_distribution.getLogProbability(value);
    }

    // AIC (Akaike Information Criterion)
    double aic = 2 * k - 2 * log_likelihood;

    // BIC (Bayesian Information Criterion)
    double bic = k * std::log(static_cast<double>(n)) - 2 * log_likelihood;

    // AICc (Corrected AIC for small sample sizes)
    double aicc = aic + (2 * k * (k + 1)) / static_cast<double>(n - static_cast<size_t>(k) - 1);

    return std::make_tuple(aic, bic, aicc, log_likelihood);
}

//==========================================================================
// 11. BOOTSTRAP METHODS
//==========================================================================

std::tuple<std::pair<double, double>, std::pair<double, double>> GammaDistribution::bootstrapParameterConfidenceIntervals(
    const std::vector<double>& data, double confidence_level, int n_bootstrap, unsigned int random_seed) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= 0.0 || confidence_level >= 1.0) {
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
    double alpha_level = (1.0 - confidence_level) / 2.0;
    size_t lower_idx = static_cast<size_t>(alpha_level * static_cast<double>(bootstrap_alphas.size()));
    size_t upper_idx = static_cast<size_t>((1.0 - alpha_level) * static_cast<double>(bootstrap_alphas.size())) - 1;

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
        throw std::logic_error("Distribution is not a chi-squared distribution (beta != 0.5)");
    }
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return 2.0 * alpha_;
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
    return alpha_ - logBeta_ + logGammaAlpha_ + (1.0 - alpha_) * digammaAlpha_;
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

Result<GammaDistribution> GammaDistribution::createFromMoments(double mean, double variance) noexcept {
    if (mean <= 0.0) {
        return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter, 
                                                   "Mean must be positive");
    }
    if (variance <= 0.0) {
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
                                       const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<GammaDistribution>::distType(),
        performance::DistributionTraits<GammaDistribution>::complexity(),
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
                                              cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            if (parallel::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= 0.0) {
                        res[i] = 0.0;
                    } else {
                        res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= 0.0) {
                        res[i] = 0.0;
                    } else {
                        res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                    }
                }
            }
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache) {
            // Cache-Aware lambda: For continuous distributions, caching is counterproductive
            // Fallback to parallel execution which is faster and more predictable
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            
            // Use parallel processing instead of caching for continuous distributions
            // Caching continuous values provides no benefit (near-zero hit rate) and severe performance penalty
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                }
            });
        }
    );
}

void GammaDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                          const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<GammaDistribution>::distType(),
        performance::DistributionTraits<GammaDistribution>::complexity(),
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
                                                cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            if (parallel::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= 0.0) {
                        res[i] = constants::probability::NEGATIVE_INFINITY;
                    } else {
                        res[i] = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= 0.0) {
                        res[i] = constants::probability::NEGATIVE_INFINITY;
                    } else {
                        res[i] = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                    }
                }
            }
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache) {
            // Cache-Aware lambda: For continuous distributions, caching is counterproductive
            // Fallback to parallel execution which is faster and more predictable
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();
            
            // Use parallel processing instead of caching for continuous distributions
            // Caching continuous values provides no benefit (near-zero hit rate) and severe performance penalty
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= 0.0) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        }
    );
}

void GammaDistribution::getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                                 const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<GammaDistribution>::distType(),
        performance::DistributionTraits<GammaDistribution>::complexity(),
        [](const GammaDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
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
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha, cached_beta);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            if (parallel::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= 0.0) {
                        res[i] = 0.0;
                    } else {
                        res[i] = math::gamma_p(cached_alpha, cached_beta * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= 0.0) {
                        res[i] = 0.0;
                    } else {
                        res[i] = math::gamma_p(cached_alpha, cached_beta * x);
                    }
                }
            }
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = math::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache) {
            // Cache-Aware lambda: For continuous distributions, caching is counterproductive
            // Fallback to parallel execution which is faster and more predictable
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            
            // Use parallel processing instead of caching for continuous distributions
            // Caching continuous values provides no benefit (near-zero hit rate) and severe performance penalty
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = math::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        }
    );
}

//==========================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==========================================================================

void GammaDistribution::getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                   performance::Strategy strategy) const {
    // Safety override for continuous distributions - cache-aware provides no benefit and severe performance penalty
    if (strategy == performance::Strategy::CACHE_AWARE) {
        strategy = performance::Strategy::PARALLEL_SIMD;
    }
    
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
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
                                              cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            // Cache-Aware lambda: should use cache.get and cache.put
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            
            // Cache parameters for thread-safe cache-aware processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();
            
            // Cache-aware processing: for gamma distribution, caching can be beneficial for complex computations
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "gamma_pdf_" << x << "_" << cached_alpha << "_" << cached_beta;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x <= 0.0) {
                        result = 0.0;
                    } else {
                        result = std::exp(cached_alpha_log_beta + cached_alpha_minus_one * std::log(x) - cached_beta * x - cached_log_gamma_alpha);
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

void GammaDistribution::getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                      performance::Strategy strategy) const {
    // Safety override for continuous distributions - cache-aware provides no benefit and severe performance penalty
    if (strategy == performance::Strategy::CACHE_AWARE) {
        strategy = performance::Strategy::PARALLEL_SIMD;
    }
    
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
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
                                                 cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else {
                    res[i] = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            // Cache-Aware lambda: should use cache.get and cache.put
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            
            // Cache parameters for thread-safe cache-aware processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            const double cached_log_gamma_alpha = dist.logGammaAlpha_;
            const double cached_alpha_log_beta = dist.alphaLogBeta_;
            const double cached_alpha_minus_one = dist.alphaMinusOne_;
            lock.unlock();
            
            // Cache-aware processing: for log probability, caching can be beneficial for repeated computations
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "gamma_logpdf_" << x << "_" << cached_alpha << "_" << cached_beta;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x <= 0.0) {
                        result = constants::probability::NEGATIVE_INFINITY;
                    } else {
                        result = cached_alpha_log_beta - cached_log_gamma_alpha + cached_alpha_minus_one * std::log(x) - cached_beta * x;
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

void GammaDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                             performance::Strategy strategy) const {
    // Safety override for continuous distributions - cache-aware provides no benefit and severe performance penalty
    if (strategy == performance::Strategy::CACHE_AWARE) {
        strategy = performance::Strategy::PARALLEL_SIMD;
    }
    
    performance::DispatchUtils::executeWithStrategy(
        *this,
        values,
        results,
        strategy,
        [](const GammaDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
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
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_alpha, cached_beta);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils::parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = math::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
                if (x <= 0.0) {
                    res[i] = 0.0;
                } else {
                    res[i] = math::gamma_p(cached_alpha, cached_beta * x);
                }
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
            // Cache-Aware lambda: should use cache.get and cache.put
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            
            const std::size_t count = vals.size();
            if (count == 0) return;
            
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
            
            // Cache parameters for thread-safe cache-aware processing
            const double cached_alpha = dist.alpha_;
            const double cached_beta = dist.beta_;
            lock.unlock();
            
            // Cache-aware processing: for CDF, caching can be beneficial for expensive computations
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "gamma_cdf_" << x << "_" << cached_alpha << "_" << cached_beta;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x <= 0.0) {
                        result = 0.0;
                    } else {
                        result = math::gamma_p(cached_alpha, cached_beta * x);
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
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
    return (std::abs(alpha_ - other.alpha_) <= constants::precision::DEFAULT_TOLERANCE) &&
           (std::abs(beta_ - other.beta_) <= constants::precision::DEFAULT_TOLERANCE);
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
    is >> temp; // "GammaDistribution(alpha=value,"
    
    if (temp.find("GammaDistribution(alpha=") == 0) {
        // Extract alpha value
        size_t equals_pos = temp.find('=');
        size_t comma_pos = temp.find(',');
        if (equals_pos != std::string::npos && comma_pos != std::string::npos) {
            std::string alpha_str = temp.substr(equals_pos + 1, comma_pos - equals_pos - 1);
            alpha = std::stod(alpha_str);
            
            // Read "beta=value)"
            is >> temp;
            if (temp.find("beta=") == 0) {
                size_t beta_equals_pos = temp.find('=');
                size_t close_paren_pos = temp.find(')');
                if (beta_equals_pos != std::string::npos && close_paren_pos != std::string::npos) {
                    std::string beta_str = temp.substr(beta_equals_pos + 1, close_paren_pos - beta_equals_pos - 1);
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

void GammaDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                      [[maybe_unused]] double alpha, double beta, double log_gamma_alpha,
                                                      double alpha_log_beta, double alpha_minus_one) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= 0.0) {
                results[i] = 0.0;
            } else {
                results[i] = std::exp(alpha_log_beta - log_gamma_alpha + alpha_minus_one * std::log(values[i]) - beta * values[i]);
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, simd::aligned_allocator<double>> log_values(count);
    std::vector<double, simd::aligned_allocator<double>> exp_inputs(count);
    
    // Step 1: Handle negative values and compute log(values)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0) {
            log_values[i] = constants::probability::MIN_LOG_PROBABILITY;  // Will be set to 0 later
        } else {
            log_values[i] = std::log(values[i]);
        }
    }
    
    // Step 2: Compute alpha_minus_one * log(values) using SIMD
    simd::VectorOps::scalar_multiply(log_values.data(), alpha_minus_one, exp_inputs.data(), count);
    
    // Step 3: Add alpha_log_beta - log_gamma_alpha
    const double log_constant = alpha_log_beta - log_gamma_alpha;
    simd::VectorOps::scalar_add(exp_inputs.data(), log_constant, exp_inputs.data(), count);
    
    // Step 4: Subtract beta * values
    simd::VectorOps::scalar_multiply(values, -beta, log_values.data(), count);
    simd::VectorOps::vector_add(exp_inputs.data(), log_values.data(), exp_inputs.data(), count);
    
    // Step 5: Apply vectorized exponential
    simd::VectorOps::vector_exp(exp_inputs.data(), results, count);
    
    // Step 6: Handle negative input values (set to zero) - must be done after exp
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0) {
            results[i] = 0.0;
        }
    }
}

void GammaDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                         [[maybe_unused]] double alpha, double beta, double log_gamma_alpha,
                                                         double alpha_log_beta, double alpha_minus_one) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= 0.0) {
                results[i] = constants::probability::NEGATIVE_INFINITY;
            } else {
                results[i] = alpha_log_beta - log_gamma_alpha + alpha_minus_one * std::log(values[i]) - beta * values[i];
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary arrays for vectorized operations
    std::vector<double, simd::aligned_allocator<double>> log_values(count);
    std::vector<double, simd::aligned_allocator<double>> beta_scaled_values(count);
    
    // Step 1: Handle negative values and compute log(values)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0) {
            log_values[i] = 0.0;  // Use 0 for now, will handle negative values at the end
        } else {
            log_values[i] = std::log(values[i]);
        }
    }
    
    // Step 2: Compute alpha_minus_one * log(values) using SIMD
    simd::VectorOps::scalar_multiply(log_values.data(), alpha_minus_one, results, count);
    
    // Step 3: Add alpha_log_beta - log_gamma_alpha
    const double log_constant = alpha_log_beta - log_gamma_alpha;
    simd::VectorOps::scalar_add(results, log_constant, results, count);
    
    // Step 4: Subtract beta * values using SIMD
    simd::VectorOps::scalar_multiply(values, -beta, beta_scaled_values.data(), count);
    simd::VectorOps::vector_add(results, beta_scaled_values.data(), results, count);
    
    // Step 5: Handle negative input values (set to MIN_LOG_PROBABILITY) - must be done after all SIMD ops
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0) {
            results[i] = constants::probability::MIN_LOG_PROBABILITY;
        }
    }
}

void GammaDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                                                double alpha, double beta) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = simd::SIMDPolicy::shouldUseSIMD(count);
    
    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= 0.0) {
                results[i] = 0.0;
            } else {
                results[i] = regularizedIncompleteGamma(alpha, beta * values[i]);
            }
        }
        return;
    }
    
    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary array for beta * values
    std::vector<double, simd::aligned_allocator<double>> scaled_values(count);
    
    // Step 1: Compute beta * values using SIMD
    simd::VectorOps::scalar_multiply(values, beta, scaled_values.data(), count);
    
    // Step 2: Apply regularized incomplete gamma function to each scaled value
    // Note: This function is not easily vectorizable, so we still use scalar loop
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= 0.0) {
            results[i] = 0.0;
        } else {
            results[i] = math::gamma_p(alpha, scaled_values[i]);
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

double GammaDistribution::incompleteGamma(double a, double x) noexcept {
    // Lower incomplete gamma function γ(a,x) using series expansion or continued fractions
    if (x <= 0.0) {
        return 0.0;
    }
    if (a <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // For x < a+1, use series expansion
    if (x < a + 1.0) {
        // Series: γ(a,x) = e^(-x) * x^a * Σ(x^n / Γ(a+n+1)) for n=0 to ∞
        double sum = 1.0;
        double term = 1.0;
        double n = 1.0;
        
        // Continue series until convergence
        while (std::abs(term) > 1e-15 * std::abs(sum) && n < 1000) {
            term *= x / (a + n - 1.0);
            sum += term;
            n += 1.0;
        }
        
        return std::exp(-x + a * std::log(x) - std::lgamma(a)) * sum;
    } else {
        // For x >= a+1, use continued fraction: γ(a,x) = Γ(a) - Γ(a,x)
        // where Γ(a,x) is computed using continued fraction
        double b = x + 1.0 - a;
        double c = 1e30;
        double d = 1.0 / b;
        double h = d;
        
        for (int i = 1; i <= 1000; ++i) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (std::abs(d) < 1e-30) {
                d = 1e-30;
            }
            c = b + an / c;
            if (std::abs(c) < 1e-30) {
                c = 1e-30;
            }
            d = 1.0 / d;
            double del = d * c;
            h *= del;
            if (std::abs(del - 1.0) < 1e-15) {
                break;
            }
        }
        
        double upper_incomplete = std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
        return std::tgamma(a) - upper_incomplete;
    }
}

double GammaDistribution::regularizedIncompleteGamma(double a, double x) noexcept {
    // Regularized lower incomplete gamma function P(a,x) = γ(a,x) / Γ(a)
    if (x <= 0.0) {
        return 0.0;
    }
    if (a <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    return incompleteGamma(a, x) / std::tgamma(a);
}

double GammaDistribution::computeQuantile(double p) const noexcept {
    // Quantile function using Newton-Raphson iteration with initial guess
    if (p <= 0.0) {
        return 0.0;
    }
    if (p >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    // Initial guess using Wilson-Hilferty transformation for large α
    double initial_guess;
    if (alpha_ > 1.0) {
        double h = 2.0 / (9.0 * alpha_);
        double z = math::inverse_normal_cdf(p);
        initial_guess = alpha_ * std::pow(1.0 - h + z * std::sqrt(h), 3) / beta_;
    } else {
        // For small α, use exponential approximation
        initial_guess = -std::log(1.0 - p) / beta_;
    }
    
    // Newton-Raphson iteration
    double x = std::max(initial_guess, 1e-10);
    const double tolerance = 1e-12;
    const int max_iterations = 100;
    
    for (int i = 0; i < max_iterations; ++i) {
        double cdf = getCumulativeProbability(x);
        double pdf = getProbability(x);
        
        if (std::abs(cdf - p) < tolerance) {
            break;
        }
        
        if (pdf < 1e-30) {
            // If PDF is too small, use bisection method fallback
            break;
        }
        
        double delta = (cdf - p) / pdf;
        x = std::max(x - delta, x * 0.1); // Ensure x stays positive
        
        if (std::abs(delta) < tolerance * x) {
            break;
        }
    }
    
    return x;
}

double GammaDistribution::sampleMarsagliaTsang(std::mt19937& rng) const noexcept {
    // Marsaglia-Tsang "squeeze" method for α ≥ 1
    // This is a fast rejection sampling method
    
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    
    const double d = alpha_ - 1.0/3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    
    while (true) {
        double x, v;
        
        do {
            x = normal(rng);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        
        v = v * v * v;
        double u = uniform(rng);
        
        // Quick accept
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return d * v / beta_;
        }
        
        // Quick reject
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) {
            return d * v / beta_;
        }
    }
}

double GammaDistribution::sampleAhrensDieter(std::mt19937& rng) const noexcept {
    // Ahrens-Dieter acceptance-rejection method for α < 1
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    const double b = (constants::math::E + alpha_) / constants::math::E;
    
    while (true) {
        double u = uniform(rng);
        double p = b * u;
        
        if (p <= 1.0) {
            double x = std::pow(p, 1.0 / alpha_);
            double u2 = uniform(rng);
            if (u2 <= std::exp(-x)) {
                return x / beta_;
            }
        } else {
            double x = -std::log((b - p) / alpha_);
            double u2 = uniform(rng);
            if (u2 <= std::pow(x, alpha_ - 1.0)) {
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
    double sum_x = std::accumulate(values.begin(), values.end(), 0.0);
    double sum_log_x = 0.0;
    for (double x : values) {
        sum_log_x += std::log(x);
    }
    
    double mean_x = sum_x / static_cast<double>(n);
    double mean_log_x = sum_log_x / static_cast<double>(n);
    
    // Initial guess using method of moments
    double s = std::log(mean_x) - mean_log_x;
    double alpha_est = (3.0 - s + std::sqrt((s - 3.0) * (s - 3.0) + 24.0 * s)) / (12.0 * s);
    
    // Newton-Raphson iteration for α
    const double tolerance = 1e-10;
    const int max_iterations = 100;
    
    for (int i = 0; i < max_iterations; ++i) {
        double digamma_alpha = GammaDistribution::computeDigamma(alpha_est);
        double trigamma_alpha = GammaDistribution::computeTrigamma(alpha_est);
        
        double f = std::log(alpha_est) - digamma_alpha - s;
        double df = 1.0 / alpha_est - trigamma_alpha;
        
        if (std::abs(f) < tolerance) {
            break;
        }
        
        alpha_est = alpha_est - f / df;
        alpha_est = std::max(alpha_est, 1e-10); // Ensure positive
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
    
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    // For small x, use reflection formula: ψ(x) = ψ(1-x) - π*cot(π*x)
    // But this is complex, so we use recurrence relation: ψ(x+1) = ψ(x) + 1/x
    double result = 0.0;
    double z = x;
    
    // Use recurrence to get z >= 8 for asymptotic series
    while (z < 8.0) {
        result -= 1.0 / z;
        z += 1.0;
    }
    
    // Asymptotic series for large z
    // ψ(z) ≈ log(z) - 1/(2z) - 1/(12z²) + 1/(120z⁴) - 1/(252z⁶) + ...
    double z_inv = 1.0 / z;
    double z_inv_sq = z_inv * z_inv;
    
    result += std::log(z) - 0.5 * z_inv;
    result -= z_inv_sq / 12.0;           // Bernoulli B₂/2
    result += z_inv_sq * z_inv_sq / 120.0; // Bernoulli B₄/4
    result -= z_inv_sq * z_inv_sq * z_inv_sq / 252.0; // Bernoulli B₆/6
    
    return result;
}

double GammaDistribution::computeTrigamma(double x) noexcept {
    // Trigamma function ψ'(x) = d²/dx² log(Γ(x))
    // Uses asymptotic series for large x and recurrence relation
    
    if (x <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    double result = 0.0;
    double z = x;
    
    // Use recurrence: ψ'(x+1) = ψ'(x) - 1/x²
    while (z < 8.0) {
        result += 1.0 / (z * z);
        z += 1.0;
    }
    
    // Asymptotic series for large z
    // ψ'(z) ≈ 1/z + 1/(2z²) + 1/(6z³) - 1/(30z⁵) + 1/(42z⁷) - ...
    double z_inv = 1.0 / z;
    double z_inv_sq = z_inv * z_inv;
    
    result += z_inv + 0.5 * z_inv_sq;
    result += z_inv_sq * z_inv / 6.0;           // 1/(6z³)
    result -= z_inv_sq * z_inv_sq * z_inv / 30.0; // -1/(30z⁵)
    result += z_inv_sq * z_inv_sq * z_inv_sq * z_inv / 42.0; // 1/(42z⁷)
    
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

} // namespace libstats

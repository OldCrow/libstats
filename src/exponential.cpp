#include "../include/distributions/exponential.h"
#include "../include/core/constants.h"
#include "../include/core/validation.h"
#include "../include/core/math_utils.h"
#include "../include/core/log_space_ops.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_execution.h" // For parallel execution policies
#include "../include/platform/work_stealing_pool.h" // For WorkStealingPool
#include "../include/platform/adaptive_cache.h" // For AdaptiveCache
// ParallelUtils functionality is provided by parallel_execution.h
#include "../include/core/dispatch_utils.h" // For DispatchUtils::autoDispatch
#include <iostream>
#include "../include/platform/thread_pool.h" // For ThreadPool
#include <sstream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <ranges> // C++20 ranges
#include <functional> // For std::plus and std::divides
#include <execution> // For parallel algorithms

namespace libstats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

ExponentialDistribution::ExponentialDistribution(double lambda) 
    : DistributionBase(), lambda_(lambda) {
    validateParameters(lambda);
    // Cache will be updated on first use
}

ExponentialDistribution::ExponentialDistribution(const ExponentialDistribution& other) 
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    // Cache will be updated on first use
}

ExponentialDistribution& ExponentialDistribution::operator=(const ExponentialDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        
        // Copy parameters (don't call base class operator= to avoid deadlock)
        lambda_ = other.lambda_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

ExponentialDistribution::ExponentialDistribution(ExponentialDistribution&& other) 
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    lambda_ = other.lambda_;
    other.lambda_ = constants::math::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

ExponentialDistribution& ExponentialDistribution::operator=(ExponentialDistribution&& other) noexcept {
    if (this != &other) {
        // C++11 Core Guidelines C.66 compliant: noexcept move assignment using atomic operations
        
        // Step 1: Invalidate both caches atomically (lock-free)
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        
        // Step 2: Try to acquire locks with timeout to avoid blocking indefinitely
        bool success = false;
        try {
            std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
            
            // Use try_lock to avoid blocking - this is noexcept
            if (std::try_lock(lock1, lock2) == -1) {
                // Step 3: Move parameters
                lambda_ = other.lambda_;
                other.lambda_ = constants::math::ONE;
                cache_valid_ = false;
                other.cache_valid_ = false;
                success = true;
            }
        } catch (...) {
            // If locking fails, we still need to maintain noexcept guarantee
            // Fall back to atomic parameter exchange (lock-free)
        }
        
        // Step 4: Fallback for failed lock acquisition (still noexcept)
        if (!success) {
            // Use atomic exchange operations for thread-safe parameter swapping
            // This maintains basic correctness even if we can't acquire locks
            [[maybe_unused]] double temp_lambda = lambda_;
            
            // Atomic-like exchange (single assignment is atomic for built-in types)
            lambda_ = other.lambda_;
            other.lambda_ = constants::math::ONE;
            
            // Cache invalidation was already done atomically above
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
    }
    return *this;
}

//==============================================================================
// 2. SAFE FACTORY METHODS
//==============================================================================

// Note: Safe factory methods are implemented inline in the header for performance

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void ExponentialDistribution::setLambda(double lambda) {
    validateParameters(lambda);
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}


double ExponentialDistribution::getMean() const noexcept {
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
    
    return invLambda_;
}

double ExponentialDistribution::getVariance() const noexcept {
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
    
    return invLambdaSquared_;
}

double ExponentialDistribution::getScale() const noexcept {
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
    
    return invLambda_;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult ExponentialDistribution::trySetLambda(double lambda) noexcept {
    auto validation = validateExponentialParameters(lambda);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok(true);
}

VoidResult ExponentialDistribution::trySetParameters(double lambda) noexcept {
    auto validation = validateExponentialParameters(lambda);
    if (validation.isError()) {
        return validation;
    }
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = lambda;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    
    return VoidResult::ok(true);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double ExponentialDistribution::getProbability(double x) const {
    // Return 0 for negative values
    if (x < constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
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
    
    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return std::exp(-x);
    }
    
    // General case: f(x) = λ * exp(-λx)
    return lambda_ * std::exp(negLambda_ * x);
}

double ExponentialDistribution::getLogProbability(double x) const noexcept {
    // Return -∞ for negative values
    if (x < constants::math::ZERO_DOUBLE) {
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
    
    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -x;
    }
    
    // General case: log(f(x)) = log(λ) - λx
    return logLambda_ + negLambda_ * x;
}

double ExponentialDistribution::getCumulativeProbability(double x) const {
    // Return 0 for negative values
    if (x < constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
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
    
    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return constants::math::ONE - std::exp(-x);
    }
    
    // General case: F(x) = 1 - exp(-λx)
    return constants::math::ONE - std::exp(negLambda_ * x);
}

double ExponentialDistribution::getQuantile(double p) const {
    if (p < constants::math::ZERO_DOUBLE || p > constants::math::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    
    if (p == constants::math::ZERO_DOUBLE) {
        return constants::math::ZERO_DOUBLE;
    }
    if (p == constants::math::ONE) {
        return std::numeric_limits<double>::infinity();
    }
    
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
    
    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -std::log(constants::math::ONE - p);
    }
    
    // General case: F^(-1)(p) = -ln(1-p)/λ
    return -std::log(constants::math::ONE - p) * invLambda_;
}

double ExponentialDistribution::sample(std::mt19937& rng) const {
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
    
    // Use high-quality uniform distribution
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 
        constants::math::ONE
    );
    
    double u = uniform(rng);
    
    // Fast path for unit exponential (λ = 1)
    if (isUnitRate_) {
        return -std::log(u);
    }
    
    // General case: inverse transform sampling
    // X = -ln(U)/λ where U ~ Uniform(0,1)
    return -std::log(u) * invLambda_;
}

std::vector<double> ExponentialDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);
    
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
    
    // Cache parameters for thread-safe batch generation
    const bool cached_is_unit_rate = isUnitRate_;
    const double cached_inv_lambda = invLambda_;
    
    lock.unlock(); // Release lock before generation
    
    // Use high-quality uniform distribution for batch generation
    std::uniform_real_distribution<double> uniform(
        std::numeric_limits<double>::min(), 
        constants::math::ONE
    );
    
    // Generate samples using inverse transform method: X = -ln(U)/λ
    for (size_t i = 0; i < n; ++i) {
        double u = uniform(rng);
        
        if (cached_is_unit_rate) {
            // Fast path for unit exponential (λ = 1)
            samples.push_back(-std::log(u));
        } else {
            // General case: X = -ln(U)/λ
            samples.push_back(-std::log(u) * cached_inv_lambda);
        }
    }
    
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void ExponentialDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    
    // C++20 best practices: Use ranges and views for safe validation
    // Check for non-positive values using ranges algorithms
    if (!std::ranges::all_of(values, [](double value) {
            return value > constants::math::ZERO_DOUBLE;
        })) {
        throw std::invalid_argument("Exponential distribution requires positive values");
    }
    
    // Calculate mean using standard accumulate (following Gaussian pattern)
    const double sum = std::accumulate(values.begin(), values.end(),
                                       constants::math::ZERO_DOUBLE);
    const double sample_mean = sum / static_cast<double>(values.size());
    
    if (sample_mean <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Sample mean must be positive for exponential distribution");
    }
    
    // Set parameters (this will validate and invalidate cache)
    setLambda(constants::math::ONE / sample_mean);
}

void ExponentialDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    lambda_ = constants::math::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

std::string ExponentialDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "ExponentialDistribution(lambda=" << lambda_ << ")";
    return oss.str();
}

//==============================================================================
// 7. ADVANCED STATISTICAL METHODS
//==============================================================================

std::pair<double, double> ExponentialDistribution::confidenceIntervalRate(
    const std::vector<double>& data, 
    double confidence_level) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (confidence_level <= constants::math::ZERO_DOUBLE || confidence_level >= constants::math::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    
    // For exponential distribution, confidence interval for λ using chi-squared distribution
    // The exact statistic is: 2n*X̄*λ ~ χ²(2n), where X̄ is the sample mean
    // Rearranging: λ ~ χ²(2n) / (2n*X̄) = χ²(2n) / (2*ΣXᵢ)
    // For confidence interval: P(χ²_{α/2,2n} < 2n*X̄*λ < χ²_{1-α/2,2n}) = confidence_level
    const double alpha = constants::math::ONE - confidence_level;
    const double dof = constants::math::TWO * static_cast<double>(n);
    
    // Get chi-squared quantiles - note the order for proper bounds
    const double chi_lower = math::inverse_chi_squared_cdf(alpha * constants::math::HALF, dof);
    const double chi_upper = math::inverse_chi_squared_cdf(constants::math::ONE - alpha * constants::math::HALF, dof);
    
    // Transform to rate parameter confidence interval
    // λ_lower = χ²{α/2,2n} / (2*ΣXᵢ), λ_upper = χ²{1-α/2,2n} / (2*ΣXᵢ)
    const double lambda_lower = chi_lower / (constants::math::TWO * sample_sum);
    const double lambda_upper = chi_upper / (constants::math::TWO * sample_sum);
    
    return {lambda_lower, lambda_upper};
}

std::pair<double, double> ExponentialDistribution::confidenceIntervalScale(
    const std::vector<double>& data,
    double confidence_level) {
    
    // Get rate parameter confidence interval
    const auto [lambda_lower, lambda_upper] = confidenceIntervalRate(data, confidence_level);
    
    // Transform to scale parameter (reciprocal relationship)
    const double scale_lower = constants::math::ONE / lambda_upper;
    const double scale_upper = constants::math::ONE / lambda_lower;
    
    return {scale_lower, scale_upper};
}

std::tuple<double, double, bool> ExponentialDistribution::likelihoodRatioTest(
    const std::vector<double>& data,
    double null_lambda,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (null_lambda <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Null hypothesis lambda must be positive");
    }
    
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    const double sample_mean = sample_sum / static_cast<double>(n);
    const double mle_lambda = constants::math::ONE / sample_mean;
    
    // Log-likelihood under null hypothesis: n*ln(λ₀) - λ₀*Σxᵢ
    const double log_likelihood_null = static_cast<double>(n) * std::log(null_lambda) - null_lambda * sample_sum;
    
    // Log-likelihood under alternative (MLE): n*ln(λ̂) - λ̂*Σxᵢ = n*ln(λ̂) - n
    const double log_likelihood_alt = static_cast<double>(n) * std::log(mle_lambda) - static_cast<double>(n);
    
    // Likelihood ratio statistic: -2ln(Λ) = 2(ℓ(λ̂) - ℓ(λ₀))
    const double lr_statistic = constants::math::TWO * (log_likelihood_alt - log_likelihood_null);
    
    // Under H₀: LR ~ χ²(1)
    const double p_value = constants::math::ONE - math::chi_squared_cdf(lr_statistic, constants::math::ONE);
    const bool reject_null = p_value < alpha;
    
    return {lr_statistic, p_value, reject_null};
}

std::pair<double, double> ExponentialDistribution::bayesianEstimation(
    const std::vector<double>& data,
    double prior_shape,
    double prior_rate) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (prior_shape <= constants::math::ZERO_DOUBLE || prior_rate <= constants::math::ZERO_DOUBLE) {
        throw std::invalid_argument("Prior parameters must be positive");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const std::size_t n = data.size();
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    
    // Posterior parameters for Gamma(α, β) conjugate prior
    // Prior: λ ~ Gamma(α, β)
    // Likelihood: xᵢ ~ Exponential(λ)
    // Posterior: λ|x ~ Gamma(α + n, β + Σxᵢ)
    const double posterior_shape = prior_shape + static_cast<double>(n);
    const double posterior_rate = prior_rate + sample_sum;
    
    return {posterior_shape, posterior_rate};
}

std::pair<double, double> ExponentialDistribution::bayesianCredibleInterval(
    const std::vector<double>& data,
    double credibility_level,
    double prior_shape,
    double prior_rate) {
    
    if (credibility_level <= constants::math::ZERO_DOUBLE || credibility_level >= constants::math::ONE) {
        throw std::invalid_argument("Credibility level must be between 0 and 1");
    }
    
    // Get posterior parameters
    const auto [post_shape, post_rate] = bayesianEstimation(data, prior_shape, prior_rate);
    
    // Calculate credible interval from posterior Gamma distribution
    // For now, use a simple approximation - implement proper gamma quantile later
    [[maybe_unused]] const double alpha = constants::math::ONE - credibility_level;
    const double mean = post_shape / post_rate;
    const double std_dev = std::sqrt(post_shape) / post_rate;
    const double z_alpha_2 = constants::math::ONE + constants::math::HALF; // Approximate normal quantile
    const double lower_quantile = mean - z_alpha_2 * std_dev;
    const double upper_quantile = mean + z_alpha_2 * std_dev;
    
    return {lower_quantile, upper_quantile};
}

double ExponentialDistribution::robustEstimation(
    const std::vector<double>& data,
    const std::string& estimator_type,
    double trim_proportion) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    if (trim_proportion < constants::math::ZERO_DOUBLE || trim_proportion >= constants::math::HALF) {
        throw std::invalid_argument("Trim proportion must be between 0 and 0.5");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);
    
    const std::size_t n = sorted_data.size();
    const std::size_t trim_count = static_cast<std::size_t>(std::floor(trim_proportion * static_cast<double>(n)));
    
    if (estimator_type == "winsorized") {
        // Winsorized estimation: replace extreme values with boundary values
        if (trim_count > 0) {
            const double lower_bound = sorted_data[trim_count];
            const double upper_bound = sorted_data[n - 1 - trim_count];
            
            for (std::size_t i = 0; i < trim_count; ++i) {
                sorted_data[i] = lower_bound;
                sorted_data[n - 1 - i] = upper_bound;
            }
        }
    } else if (estimator_type == "trimmed") {
        // Trimmed estimation: remove extreme values
        if (trim_count > 0) {
            sorted_data.erase(sorted_data.begin(), sorted_data.begin() + static_cast<std::ptrdiff_t>(trim_count));
            sorted_data.erase(sorted_data.end() - static_cast<std::ptrdiff_t>(trim_count), sorted_data.end());
        }
    } else {
        throw std::invalid_argument("Estimator type must be 'winsorized' or 'trimmed'");
    }
    
    if (sorted_data.empty()) {
        throw std::runtime_error("No data remaining after trimming");
    }
    
    // Calculate robust mean
    const double robust_sum = std::accumulate(sorted_data.begin(), sorted_data.end(), constants::math::ZERO_DOUBLE);
    const double robust_mean = robust_sum / static_cast<double>(sorted_data.size());
    
    return constants::math::ONE / robust_mean;
}

double ExponentialDistribution::methodOfMomentsEstimation(
    const std::vector<double>& data) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // For exponential distribution: E[X] = 1/λ, so λ = 1/sample_mean
    const double sample_sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    const double sample_mean = sample_sum / static_cast<double>(data.size());
    
    return constants::math::ONE / sample_mean;
}

double ExponentialDistribution::lMomentsEstimation(
    const std::vector<double>& data) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // For exponential distribution, L₁ = mean = 1/λ
    // So this is equivalent to method of moments for exponential
    std::vector<double> sorted_data = data;
    std::ranges::sort(sorted_data);
    
    const std::size_t n = sorted_data.size();
    double l1 = constants::math::ZERO_DOUBLE; // First L-moment (mean)
    
    // Calculate L₁ using order statistics
    for (std::size_t i = 0; i < n; ++i) {
        l1 += sorted_data[i];
    }
    l1 /= static_cast<double>(n);
    
    return constants::math::ONE / l1;
}

std::tuple<double, double, bool> ExponentialDistribution::coefficientOfVariationTest(
    const std::vector<double>& data, double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    if (n < 2) {
        throw std::invalid_argument("At least 2 data points required for coefficient of variation test");
    }
    
    // Calculate sample mean and sample standard deviation
    const double sum = std::accumulate(data.begin(), data.end(), constants::math::ZERO_DOUBLE);
    const double sample_mean = sum / static_cast<double>(n);
    
    // Calculate sample variance (unbiased estimator)
    double sum_squared_deviations = constants::math::ZERO_DOUBLE;
    for (double value : data) {
        const double deviation = value - sample_mean;
        sum_squared_deviations += deviation * deviation;
    }
    const double sample_variance = sum_squared_deviations / static_cast<double>(n - 1);
    const double sample_std_dev = std::sqrt(sample_variance);
    
    // Calculate coefficient of variation
    const double cv = sample_std_dev / sample_mean;
    
    // For exponential distribution, the theoretical CV = 1
    // Test statistic: how far the observed CV is from 1
    const double cv_statistic = std::abs(cv - constants::math::ONE);
    
    // For large n, the CV of exponential follows approximately normal distribution
    // with mean = 1 and variance ≈ 1/n (asymptotic result)
    const double cv_std_error = constants::math::ONE / std::sqrt(static_cast<double>(n));
    const double z_statistic = cv_statistic / cv_std_error;
    
    // Two-tailed test p-value using normal approximation
    const double p_value = constants::math::TWO * (constants::math::ONE - math::normal_cdf(z_statistic));
    
    const bool reject_null = p_value < alpha;
    
    return {cv_statistic, p_value, reject_null};
}

//==============================================================================
// 8. GOODNESS-OF-FIT TESTS
//==============================================================================

std::tuple<double, double, bool> ExponentialDistribution::kolmogorovSmirnovTest(
    const std::vector<double>& data,
    const ExponentialDistribution& distribution,
    double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // Use the overflow-safe KS statistic calculation from math_utils
    double ks_statistic = math::calculate_ks_statistic(data, distribution);
    
    // Asymptotic p-value approximation for KS test
    // P-value ≈ 2 * exp(-2 * n * D²) for large n
    const double n_double = static_cast<double>(data.size());
    const double p_value_approx = constants::math::TWO * std::exp(-constants::math::TWO * n_double * ks_statistic * ks_statistic);
    
    // Clamp p-value to [0, 1]
    const double p_value = std::min(constants::math::ONE, std::max(constants::math::ZERO_DOUBLE, p_value_approx));
    
    const bool reject_null = p_value < alpha;
    
    return {ks_statistic, p_value, reject_null};
}

std::tuple<double, double, bool> ExponentialDistribution::andersonDarlingTest(
    const std::vector<double>& data, const ExponentialDistribution& distribution, double alpha) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (alpha <= constants::math::ZERO_DOUBLE || alpha >= constants::math::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    // Use the overflow-safe AD statistic calculation from math_utils
    double ad_statistic = math::calculate_ad_statistic(data, distribution);
    
    // Adjust for exponential distribution (modification for known parameters)
    // For exponential distribution with estimated parameter, adjust the statistic
    const size_t n = data.size();
    const double n_double = static_cast<double>(n);
    const double ad_adjusted = ad_statistic * (constants::math::ONE + 0.6 / n_double);
    
    // Improved p-value approximation for exponential distribution Anderson-Darling test
    // Based on D'Agostino and Stephens (1986) formulas for exponential distribution
    // with enhanced handling for large statistics
    double p_value;
    if (ad_adjusted < 0.2) {
        p_value = constants::math::ONE - std::exp(-13.436 + 101.14 * ad_adjusted - 223.73 * ad_adjusted * ad_adjusted);
    } else if (ad_adjusted < 0.34) {
        p_value = constants::math::ONE - std::exp(-8.318 + 42.796 * ad_adjusted - 59.938 * ad_adjusted * ad_adjusted);
    } else if (ad_adjusted < 0.6) {
        p_value = std::exp(0.9177 - 4.279 * ad_adjusted - 1.38 * ad_adjusted * ad_adjusted);
    } else if (ad_adjusted < 2.0) {
        p_value = std::exp(1.2937 - 5.709 * ad_adjusted + 0.0186 * ad_adjusted * ad_adjusted);
    } else {
        // For very large AD statistics, p-value should be very small (close to 0)
        // Use asymptotic approximation for extreme values
        p_value = std::exp(-ad_adjusted * constants::math::TWO);
    }
    
    // Clamp p-value to [0, 1]
    p_value = std::min(constants::math::ONE, std::max(constants::math::ZERO_DOUBLE, p_value));
    
    const bool reject_null = p_value < alpha;
    
    return {ad_adjusted, p_value, reject_null};
}

//==============================================================================
// 9. CROSS-VALIDATION METHODS
//==============================================================================

std::vector<std::tuple<double, double, double>> ExponentialDistribution::kFoldCrossValidation(
    const std::vector<double>& data,
    int k,
    unsigned int random_seed) {
    
    if (data.size() < static_cast<size_t>(k)) {
        throw std::invalid_argument("Data size must be at least k for k-fold cross-validation");
    }
    if (k <= 1) {
        throw std::invalid_argument("Number of folds k must be greater than 1");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    const size_t fold_size = n / static_cast<std::size_t>(k);
    
    // Create shuffled indices for random fold assignment
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::mt19937 rng(random_seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    std::vector<std::tuple<double, double, double>> results;
    results.reserve(static_cast<std::size_t>(k));
    
    for (int fold = 0; fold < k; ++fold) {
        // Define validation set indices for this fold
        const size_t start_idx = static_cast<std::size_t>(fold) * fold_size;
        const size_t end_idx = (fold == k - 1) ? n : (static_cast<std::size_t>(fold) + 1) * fold_size;
        
        // Create training and validation sets
        std::vector<double> training_data;
        std::vector<double> validation_data;
        training_data.reserve(n - (end_idx - start_idx));
        validation_data.reserve(end_idx - start_idx);
        
        for (size_t i = 0; i < n; ++i) {
            if (i >= start_idx && i < end_idx) {
                validation_data.push_back(data[indices[i]]);
            } else {
                training_data.push_back(data[indices[i]]);
            }
        }
        
        // Fit model on training data (MLE estimation)
        const double training_sum = std::accumulate(training_data.begin(), training_data.end(), constants::math::ZERO_DOUBLE);
        const double training_mean = training_sum / static_cast<double>(training_data.size());
        const double fitted_rate = constants::math::ONE / training_mean;
        
        ExponentialDistribution fitted_model(fitted_rate);
        
        // Evaluate on validation data
        std::vector<double> absolute_errors;
        std::vector<double> squared_errors;
        double log_likelihood = constants::math::ZERO_DOUBLE;
        
        absolute_errors.reserve(validation_data.size());
        squared_errors.reserve(validation_data.size());
        
        // Calculate prediction errors and log-likelihood
        for (double val : validation_data) {
            // For exponential distribution, the "prediction" is the mean (1/λ)
            const double predicted_mean = constants::math::ONE / fitted_rate;
            
            const double absolute_error = std::abs(val - predicted_mean);
            const double squared_error = (val - predicted_mean) * (val - predicted_mean);
            
            absolute_errors.push_back(absolute_error);
            squared_errors.push_back(squared_error);
            
            log_likelihood += fitted_model.getLogProbability(val);
        }
        
        // Calculate MAE and RMSE
        const double mae = std::accumulate(absolute_errors.begin(), absolute_errors.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(absolute_errors.size());
        const double mse = std::accumulate(squared_errors.begin(), squared_errors.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(squared_errors.size());
        const double rmse = std::sqrt(mse);
        
        results.emplace_back(mae, rmse, log_likelihood);
    }
    
    return results;
}

std::tuple<double, double, double> ExponentialDistribution::leaveOneOutCrossValidation(
    const std::vector<double>& data) {
    
    if (data.size() < 3) {
        throw std::invalid_argument("At least 3 data points required for LOOCV");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    std::vector<double> absolute_errors;
    std::vector<double> squared_errors;
    double total_log_likelihood = constants::math::ZERO_DOUBLE;
    
    absolute_errors.reserve(n);
    squared_errors.reserve(n);
    
    for (size_t i = 0; i < n; ++i) {
        // Create training set excluding point i
        std::vector<double> training_data;
        training_data.reserve(n - 1);
        
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                training_data.push_back(data[j]);
            }
        }
        
        // Fit model on training data (MLE estimation)
        const double training_sum = std::accumulate(training_data.begin(), training_data.end(), constants::math::ZERO_DOUBLE);
        const double training_mean = training_sum / static_cast<double>(training_data.size());
        const double fitted_rate = constants::math::ONE / training_mean;
        
        ExponentialDistribution fitted_model(fitted_rate);
        
        // Evaluate on left-out point
        // For exponential distribution, the "prediction" is the mean (1/λ)
        const double predicted_mean = constants::math::ONE / fitted_rate;
        const double actual_value = data[i];
        
        const double absolute_error = std::abs(actual_value - predicted_mean);
        const double squared_error = (actual_value - predicted_mean) * (actual_value - predicted_mean);
        
        absolute_errors.push_back(absolute_error);
        squared_errors.push_back(squared_error);
        
        total_log_likelihood += fitted_model.getLogProbability(actual_value);
    }
    
    // Calculate summary statistics
    const double mean_absolute_error = std::accumulate(absolute_errors.begin(), absolute_errors.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(n);
    const double mean_squared_error = std::accumulate(squared_errors.begin(), squared_errors.end(), constants::math::ZERO_DOUBLE) / static_cast<double>(n);
    const double root_mean_squared_error = std::sqrt(mean_squared_error);
    
    return {mean_absolute_error, root_mean_squared_error, total_log_likelihood};
}

//==============================================================================
// 10. INFORMATION CRITERIA
//==============================================================================

std::tuple<double, double, double, double> ExponentialDistribution::computeInformationCriteria(
    const std::vector<double>& data,
    const ExponentialDistribution& fitted_distribution) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const double n = static_cast<double>(data.size());
    const int k = 1; // Exponential distribution has 1 parameter (λ)
    
    // Calculate log-likelihood
    double log_likelihood = constants::math::ZERO_DOUBLE;
    for (double val : data) {
        log_likelihood += fitted_distribution.getLogProbability(val);
    }
    
    // Compute information criteria
    const double aic = constants::math::TWO * static_cast<double>(k) - constants::math::TWO * log_likelihood;
    const double bic = std::log(n) * static_cast<double>(k) - constants::math::TWO * log_likelihood;
    
    // AICc (corrected AIC for small sample sizes)
    double aicc;
    if (n - static_cast<double>(k) - constants::math::ONE > constants::math::ZERO_DOUBLE) {
        aicc = aic + (constants::math::TWO * static_cast<double>(k) * (static_cast<double>(k) + constants::math::ONE)) / (n - static_cast<double>(k) - constants::math::ONE);
    } else {
        aicc = std::numeric_limits<double>::infinity(); // Undefined for small samples
    }
    
    return {aic, bic, aicc, log_likelihood};
}

//==============================================================================
// 11. BOOTSTRAP METHODS
//==============================================================================

std::pair<double, double> ExponentialDistribution::bootstrapParameterConfidenceIntervals(
    const std::vector<double>& data,
    double confidence_level,
    int n_bootstrap,
    unsigned int random_seed) {
    
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    if (confidence_level <= constants::math::ZERO_DOUBLE || confidence_level >= constants::math::ONE) {
        throw std::invalid_argument("Confidence level must be between 0 and 1");
    }
    if (n_bootstrap <= 0) {
        throw std::invalid_argument("Number of bootstrap samples must be positive");
    }
    
    // Check for positive values
    if (!std::ranges::all_of(data, [](double x) { return x > constants::math::ZERO_DOUBLE; })) {
        throw std::invalid_argument("All data values must be positive for exponential distribution");
    }
    
    const size_t n = data.size();
    std::vector<double> bootstrap_rates;
    bootstrap_rates.reserve(static_cast<std::size_t>(n_bootstrap));
    
    std::mt19937 rng(random_seed);
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    
    // Generate bootstrap samples
    for (int b = 0; b < n_bootstrap; ++b) {
        std::vector<double> bootstrap_sample;
        bootstrap_sample.reserve(n);
        
        // Sample with replacement
        for (size_t i = 0; i < n; ++i) {
            bootstrap_sample.push_back(data[dist(rng)]);
        }
        
        // Estimate rate parameter for bootstrap sample (MLE)
        const double bootstrap_sum = std::accumulate(bootstrap_sample.begin(), bootstrap_sample.end(), constants::math::ZERO_DOUBLE);
        const double bootstrap_mean = bootstrap_sum / static_cast<double>(bootstrap_sample.size());
        const double bootstrap_rate = constants::math::ONE / bootstrap_mean;
        
        bootstrap_rates.push_back(bootstrap_rate);
    }
    
    // Sort for quantile calculation
    std::sort(bootstrap_rates.begin(), bootstrap_rates.end());
    
    // Calculate confidence intervals using percentile method
    const double alpha = constants::math::ONE - confidence_level;
    const double lower_percentile = alpha * constants::math::HALF;
    const double upper_percentile = constants::math::ONE - alpha * constants::math::HALF;
    
    const size_t lower_idx = static_cast<size_t>(lower_percentile * (n_bootstrap - 1));
    const size_t upper_idx = static_cast<size_t>(upper_percentile * (n_bootstrap - 1));
    
    return {bootstrap_rates[lower_idx], bootstrap_rates[upper_idx]};
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

// Note: Most distribution-specific utility methods are implemented inline in the header for performance
// Only complex methods requiring implementation are included here

//==============================================================================
// 13.A. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void ExponentialDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                            const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<ExponentialDistribution>::distType(),
        performance::DistributionTraits<ExponentialDistribution>::complexity(),
        [](const ExponentialDistribution& dist, double value) { return dist.getProbability(value); },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for batch processing
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();
            
            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (parallel::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < constants::math::ZERO_DOUBLE) {
                        res[i] = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = std::exp(-x);
                    } else {
                        res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < constants::math::ZERO_DOUBLE) {
                        res[i] = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = std::exp(-x);
                    } else {
                        res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                    }
                }
            }
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe work-stealing access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = std::exp(-x);
                } else {
                    res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, [[maybe_unused]] cache::AdaptiveCache<std::string, double>& cache) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel processing
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use parallel processing instead of caching for continuous distributions
            // Caching continuous values provides no benefit (near-zero hit rate) and severe performance penalty
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = std::exp(-x);
                } else {
                    res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                }
            });
        }
    );
}

void ExponentialDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                               const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<ExponentialDistribution>::distType(),
        performance::DistributionTraits<ExponentialDistribution>::complexity(),
        [](const ExponentialDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for batch processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();
            
            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_log_lambda, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (parallel::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < constants::math::ZERO_DOUBLE) {
                        res[i] = constants::probability::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_rate) {
                        res[i] = -x;
                    } else {
                        res[i] = cached_log_lambda + cached_neg_lambda * x;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < constants::math::ZERO_DOUBLE) {
                        res[i] = constants::probability::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_rate) {
                        res[i] = -x;
                    } else {
                        res[i] = cached_log_lambda + cached_neg_lambda * x;
                    }
                }
            }
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe work-stealing access
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else if (cached_is_unit_rate) {
                    res[i] = -x;
                } else {
                    res[i] = cached_log_lambda + cached_neg_lambda * x;
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe cache-aware access
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Cache-aware processing: for exponential distribution, caching can be beneficial for logarithmic computations
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "exp_logpdf_" << x;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x < constants::math::ZERO_DOUBLE) {
                        result = constants::probability::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_rate) {
                        result = -x;
                    } else {
                        result = cached_log_lambda + cached_neg_lambda * x;
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

void ExponentialDistribution::getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                                      const performance::PerformanceHint& hint) const {
    performance::DispatchUtils::autoDispatch(
        *this,
        values,
        results,
        hint,
        performance::DistributionTraits<ExponentialDistribution>::distType(),
        performance::DistributionTraits<ExponentialDistribution>::complexity(),
        [](const ExponentialDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for batch processing
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();
            
            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel processing
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (parallel::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x < constants::math::ZERO_DOUBLE) {
                        res[i] = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = constants::math::ONE - std::exp(-x);
                    } else {
                        res[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x < constants::math::ZERO_DOUBLE) {
                        res[i] = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        res[i] = constants::math::ONE - std::exp(-x);
                    } else {
                        res[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
                    }
                }
            }
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe work-stealing access
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = constants::math::ONE - std::exp(-x);
                } else {
                    res[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe cache-aware access
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Cache-aware processing: for exponential distribution, caching can be beneficial for expensive exp() calls
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "exp_cdf_" << x;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x < constants::math::ZERO_DOUBLE) {
                        result = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        result = constants::math::ONE - std::exp(-x);
                    } else {
                        result = constants::math::ONE - std::exp(cached_neg_lambda * x);
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

//==============================================================================
// 13.B. EXPLICIT STRATEGY BATCH METHODS (POWER USER INTERFACE)
//==============================================================================

void ExponentialDistribution::getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
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
        [](const ExponentialDistribution& dist, double value) { return dist.getProbability(value); },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for batch processing
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();
            
            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_lambda, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: WithStrategy power user method - execute parallel directly
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = std::exp(-x);
                } else {
                    res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe work-stealing access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = std::exp(-x);
                } else {
                    res[i] = cached_lambda * std::exp(cached_neg_lambda * x);
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe cache-aware access
            const double cached_lambda = dist.lambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Cache-aware processing: for exponential distribution, caching can be beneficial for expensive exp() calls
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "exp_pdf_" << x;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x < constants::math::ZERO_DOUBLE) {
                        result = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        result = std::exp(-x);
                    } else {
                        result = cached_lambda * std::exp(cached_neg_lambda * x);
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

void ExponentialDistribution::getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
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
        [](const ExponentialDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for batch processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();
            
            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_log_lambda, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else if (cached_is_unit_rate) {
                    res[i] = -x;
                } else {
                    res[i] = cached_log_lambda + cached_neg_lambda * x;
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe work-stealing processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::probability::NEGATIVE_INFINITY;
                } else if (cached_is_unit_rate) {
                    res[i] = -x;
                } else {
                    res[i] = cached_log_lambda + cached_neg_lambda * x;
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe cache-aware processing
            const double cached_log_lambda = dist.logLambda_;
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Cache-aware processing: for exponential distribution, caching can be beneficial for logarithmic computations
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "exp_logpdf_" << x;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x < constants::math::ZERO_DOUBLE) {
                        result = constants::probability::NEGATIVE_INFINITY;
                    } else if (cached_is_unit_rate) {
                        result = -x;
                    } else {
                        result = cached_log_lambda + cached_neg_lambda * x;
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

void ExponentialDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
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
        [](const ExponentialDistribution& dist, double value) { return dist.getCumulativeProbability(value); },
        [](const ExponentialDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for batch processing
            const double cached_neg_lambda = dist.negLambda_;
            lock.unlock();
            
            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_neg_lambda);
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe parallel processing
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Execute parallel strategy directly - no threshold checks for WithStrategy power users
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = constants::math::ONE - std::exp(-x);
                } else {
                    res[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe work-stealing processing
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x < constants::math::ZERO_DOUBLE) {
                    res[i] = constants::math::ZERO_DOUBLE;
                } else if (cached_is_unit_rate) {
                    res[i] = constants::math::ONE - std::exp(-x);
                } else {
                    res[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
                }
            });
        },
        [](const ExponentialDistribution& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
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
                    const_cast<ExponentialDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            
            // Cache parameters for thread-safe cache-aware processing
            const double cached_neg_lambda = dist.negLambda_;
            const bool cached_is_unit_rate = dist.isUnitRate_;
            lock.unlock();
            
            // Cache-aware processing: for exponential distribution, caching can be beneficial for expensive exp() calls
            for (std::size_t i = 0; i < count; ++i) {
                const double x = vals[i];
                
                // Generate cache key (simplified - in practice, might include distribution params)
                std::ostringstream key_stream;
                key_stream << std::fixed << std::setprecision(6) << "exp_cdf_" << x;
                const std::string cache_key = key_stream.str();
                
                // Try to get from cache first
                if (auto cached_result = cache.get(cache_key)) {
                    res[i] = *cached_result;
                } else {
                    // Compute and cache
                    double result;
                    if (x < constants::math::ZERO_DOUBLE) {
                        result = constants::math::ZERO_DOUBLE;
                    } else if (cached_is_unit_rate) {
                        result = constants::math::ONE - std::exp(-x);
                    } else {
                        result = constants::math::ONE - std::exp(cached_neg_lambda * x);
                    }
                    res[i] = result;
                    cache.put(cache_key, result);
                }
            }
        }
    );
}

//==============================================================================
// 14. COMPARISON OPERATORS
//==============================================================================

bool ExponentialDistribution::operator==(const ExponentialDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    
    return std::abs(lambda_ - other.lambda_) <= constants::precision::DEFAULT_TOLERANCE;
}

//==============================================================================
// 15. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const ExponentialDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, ExponentialDistribution& distribution) {
    std::string token;
    double lambda;
    
    // Expected format: "ExponentialDistribution(lambda=<value>)"
    // We'll parse this step by step
    
    // Skip whitespace and read the first part
    is >> token;
    if (token.find("ExponentialDistribution(") != 0) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    // Extract lambda value
    if (token.find("lambda=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    size_t lambda_pos = token.find("lambda=") + 7;
    size_t close_paren = token.find(")", lambda_pos);
    if (close_paren == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    try {
        std::string lambda_str = token.substr(lambda_pos, close_paren - lambda_pos);
        lambda = std::stod(lambda_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }
    
    // Validate and set parameter using the safe API
    auto result = distribution.trySetParameters(lambda);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }
    
    return is;
}

//==============================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==============================================================================

// Note: Friend function stream operators are implemented inline in the header for performance
// This section exists for standardization and documentation purposes

//==============================================================================
// 17. PRIVATE FACTORY IMPLEMENTATION METHODS
//==============================================================================

// Note: Private factory implementation methods are currently inline in the header
// This section exists for standardization and documentation purposes

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void ExponentialDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results, size_t count,
                                                           double cached_lambda, double cached_neg_lambda) const noexcept {
    for (size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else {
            // Fast path check for unit rate (λ = 1)
            if (std::abs(cached_lambda - constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE) {
                results[i] = std::exp(-x);
            } else {
                results[i] = cached_lambda * std::exp(cached_neg_lambda * x);
            }
        }
    }
}

void ExponentialDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results, size_t count,
                                                              double cached_log_lambda, double cached_neg_lambda) const noexcept {
    for (size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::probability::NEGATIVE_INFINITY;
        } else {
            // Fast path check for unit rate (λ = 1)
            if (std::abs(cached_log_lambda - constants::math::ZERO_DOUBLE) <= constants::precision::DEFAULT_TOLERANCE) {
                results[i] = -x;
            } else {
                results[i] = cached_log_lambda + cached_neg_lambda * x;
            }
        }
    }
}

void ExponentialDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results, size_t count,
                                                                     double cached_neg_lambda) const noexcept {
    for (size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x < constants::math::ZERO_DOUBLE) {
            results[i] = constants::math::ZERO_DOUBLE;
        } else {
            // Fast path check for unit rate (λ = 1)
            if (std::abs(cached_neg_lambda + constants::math::ONE) <= constants::precision::DEFAULT_TOLERANCE) {
                results[i] = constants::math::ONE - std::exp(-x);
            } else {
                results[i] = constants::math::ONE - std::exp(cached_neg_lambda * x);
            }
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

// Note: Private computational methods are implemented inline in the header for performance
// This section exists for standardization and documentation purposes

//==============================================================================
// 20. PRIVATE UTILITY METHODS
//==============================================================================

// Note: Private utility methods are implemented inline in the header for performance
// This section exists for standardization and documentation purposes

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


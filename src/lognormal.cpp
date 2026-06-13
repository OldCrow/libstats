#include "libstats/distributions/lognormal.h"

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"
#include "libstats/core/validation.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

LogNormalDistribution::LogNormalDistribution(double mu, double sigma)
    : DistributionBase(), mu_(mu), sigma_(sigma) {
    validateParameters(mu, sigma);
    updateCacheUnsafe();
}

LogNormalDistribution::LogNormalDistribution(const LogNormalDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    sigma_ = other.sigma_;
    logSigma_ = other.logSigma_;
    negInv2SigmaSquared_ = other.negInv2SigmaSquared_;
    invSigmaSqrt2_ = other.invSigmaSqrt2_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    isStandard_ = other.isStandard_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
}

LogNormalDistribution& LogNormalDistribution::operator=(const LogNormalDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_ = other.mu_;
        sigma_ = other.sigma_;
        logSigma_ = other.logSigma_;
        negInv2SigmaSquared_ = other.negInv2SigmaSquared_;
        invSigmaSqrt2_ = other.invSigmaSqrt2_;
        logNormConst_ = other.logNormConst_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        isStandard_ = other.isStandard_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicSigma_.store(sigma_, std::memory_order_release);
    }
    return *this;
}

LogNormalDistribution::LogNormalDistribution(LogNormalDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    sigma_ = other.sigma_;
    logSigma_ = other.logSigma_;
    negInv2SigmaSquared_ = other.negInv2SigmaSquared_;
    invSigmaSqrt2_ = other.invSigmaSqrt2_;
    logNormConst_ = other.logNormConst_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    isStandard_ = other.isStandard_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.sigma_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
}

LogNormalDistribution& LogNormalDistribution::operator=(LogNormalDistribution&& other) noexcept {
    if (this != &other) {
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);

        bool success = false;
        try {
            std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
            if (std::try_lock(lock1, lock2) == -1) {
                mu_ = other.mu_;
                sigma_ = other.sigma_;
                logSigma_ = other.logSigma_;
                negInv2SigmaSquared_ = other.negInv2SigmaSquared_;
                invSigmaSqrt2_ = other.invSigmaSqrt2_;
                logNormConst_ = other.logNormConst_;
                mean_ = other.mean_;
                variance_ = other.variance_;
                isStandard_ = other.isStandard_;
                other.mu_ = detail::ZERO_DOUBLE;
                other.sigma_ = detail::ONE;
                cache_valid_ = false;
                other.cache_valid_ = false;
                atomicMu_.store(mu_, std::memory_order_release);
                atomicSigma_.store(sigma_, std::memory_order_release);
                success = true;
            }
        } catch (...) {
        }

        if (!success) {
            mu_ = other.mu_;
            sigma_ = other.sigma_;
            other.mu_ = detail::ZERO_DOUBLE;
            other.sigma_ = detail::ONE;
            cache_valid_ = false;
            other.cache_valid_ = false;
            atomicMu_.store(mu_, std::memory_order_release);
            atomicSigma_.store(sigma_, std::memory_order_release);
        }
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

LogNormalDistribution LogNormalDistribution::createUnchecked(double mu, double sigma) noexcept {
    return LogNormalDistribution(mu, sigma, true);
}

LogNormalDistribution::LogNormalDistribution(double mu, double sigma,
                                             bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), sigma_(sigma) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void LogNormalDistribution::setMu(double mu) {
    validateParameters(mu, getSigma());
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void LogNormalDistribution::setSigma(double sigma) {
    validateParameters(getMu(), sigma);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void LogNormalDistribution::setParameters(double mu, double sigma) {
    validateParameters(mu, sigma);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double LogNormalDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return mean_;
}

double LogNormalDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return variance_;
}

double LogNormalDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double esigma2 = std::exp(sigma_ * sigma_);
    return (esigma2 + detail::TWO) * std::sqrt(esigma2 - detail::ONE);
}

double LogNormalDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double s2 = sigma_ * sigma_;
    return std::exp(detail::FOUR * s2) + detail::TWO * std::exp(detail::THREE * s2) +
           detail::THREE * std::exp(detail::TWO * s2) - detail::SIX;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult LogNormalDistribution::trySetMu(double mu) noexcept {
    auto v = validateLogNormalParameters(mu, getSigma());
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult LogNormalDistribution::trySetSigma(double sigma) noexcept {
    auto v = validateLogNormalParameters(getMu(), sigma);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult LogNormalDistribution::trySetParameters(double mu, double sigma) noexcept {
    auto v = validateLogNormalParameters(mu, sigma);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult LogNormalDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateLogNormalParameters(mu_, sigma_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double LogNormalDistribution::getProbability(double x) const {
    if (x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return std::exp(getLogProbability(x));
}

double LogNormalDistribution::getLogProbability(double x) const noexcept {
    if (x <= detail::ZERO_DOUBLE)
        return detail::NEGATIVE_INFINITY;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double z = std::log(x) - mu_;
    return negInv2SigmaSquared_ * z * z - std::log(x) + logNormConst_;
}

double LogNormalDistribution::getCumulativeProbability(double x) const {
    if (x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // Φ((log x − μ)/σ) via standard normal CDF
    return detail::normal_cdf((std::log(x) - mu_) / sigma_);
}

double LogNormalDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return std::exp(mu_ + sigma_ * detail::inverse_normal_cdf(p));
}

double LogNormalDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double cached_mu = mu_;
    const double cached_sigma = sigma_;
    lock.unlock();

    std::normal_distribution<double> norm(cached_mu, cached_sigma);
    return std::exp(norm(rng));
}

std::vector<double> LogNormalDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double cached_mu = mu_;
    const double cached_sigma = sigma_;
    lock.unlock();

    std::normal_distribution<double> norm(cached_mu, cached_sigma);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(std::exp(norm(rng)));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void LogNormalDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    for (double v : values) {
        if (v <= detail::ZERO_DOUBLE || !std::isfinite(v)) {
            throw std::invalid_argument(
                "Log-normal distribution requires strictly positive finite values");
        }
    }

    // Closed-form MLE: μ̂ = mean(log xᵢ), σ̂² = population variance of log xᵢ.
    // Use the MLE (divide by n, not n-1) for σ̂ to match standard MLE definition.
    const double n = static_cast<double>(values.size());

    double sum_log = detail::ZERO_DOUBLE;
    for (double v : values)
        sum_log += std::log(v);
    const double mu_hat = sum_log / n;

    double sum_sq = detail::ZERO_DOUBLE;
    for (double v : values) {
        const double diff = std::log(v) - mu_hat;
        sum_sq += diff * diff;
    }
    const double sigma_hat = std::sqrt(sum_sq / n);

    if (sigma_hat <= detail::ZERO_DOUBLE) {
        // All values are identical — degenerate case; preserve mu, reset sigma.
        setParameters(mu_hat, detail::MIN_STD_DEV);
    } else {
        setParameters(mu_hat, sigma_hat);
    }
}

void LogNormalDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                             std::vector<LogNormalDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void LogNormalDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    sigma_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string LogNormalDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "LogNormalDistribution(mu=" << mu_ << ",sigma=" << sigma_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double LogNormalDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicMu_.load(std::memory_order_acquire);
    }
    return getMu();
}

double LogNormalDistribution::getSigmaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicSigma_.load(std::memory_order_acquire);
    }
    return getSigma();
}

double LogNormalDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return std::exp(mu_ - sigma_ * sigma_);
}

double LogNormalDistribution::getMedian() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return std::exp(mu_);
}

double LogNormalDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = log(σ) + μ + ½(1 + log(2π))
    return logSigma_ + mu_ + detail::HALF * (detail::ONE + detail::LN_2PI);
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void LogNormalDistribution::getProbability(std::span<const double> values,
                                           std::span<double> results,
                                           const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<LogNormalDistribution>::distType(),
        detail::OperationType::PDF,
        [](const LogNormalDistribution& d, double x) { return d.getProbability(x); },
        [](const LogNormalDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double neg_inv_2sigma2 = d.negInv2SigmaSquared_;
            const double log_norm_const = d.logNormConst_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, mu, neg_inv_2sigma2, log_norm_const);
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double neg_inv_2sigma2 = d.negInv2SigmaSquared_;
            const double log_norm_const = d.logNormConst_;
            lock.unlock();

            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        const double z = std::log(x) - mu;
                        res[i] = std::exp(neg_inv_2sigma2 * z * z - std::log(x) + log_norm_const);
                    }
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        const double z = std::log(x) - mu;
                        res[i] = std::exp(neg_inv_2sigma2 * z * z - std::log(x) + log_norm_const);
                    }
                }
            }
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double neg_inv_2sigma2 = d.negInv2SigmaSquared_;
            const double log_norm_const = d.logNormConst_;
            lock.unlock();

            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    const double z = std::log(x) - mu;
                    res[i] = std::exp(neg_inv_2sigma2 * z * z - std::log(x) + log_norm_const);
                }
            });
            pool.waitForAll();
        });
}

void LogNormalDistribution::getLogProbability(std::span<const double> values,
                                              std::span<double> results,
                                              const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<LogNormalDistribution>::distType(),
        detail::OperationType::LOG_PDF,
        [](const LogNormalDistribution& d, double x) { return d.getLogProbability(x); },
        [](const LogNormalDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double neg_inv_2sigma2 = d.negInv2SigmaSquared_;
            const double log_norm_const = d.logNormConst_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, mu, neg_inv_2sigma2,
                                               log_norm_const);
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double neg_inv_2sigma2 = d.negInv2SigmaSquared_;
            const double log_norm_const = d.logNormConst_;
            lock.unlock();

            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else {
                        const double z = std::log(x) - mu;
                        res[i] = neg_inv_2sigma2 * z * z - std::log(x) + log_norm_const;
                    }
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::NEGATIVE_INFINITY;
                    } else {
                        const double z = std::log(x) - mu;
                        res[i] = neg_inv_2sigma2 * z * z - std::log(x) + log_norm_const;
                    }
                }
            }
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double neg_inv_2sigma2 = d.negInv2SigmaSquared_;
            const double log_norm_const = d.logNormConst_;
            lock.unlock();

            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                } else {
                    const double z = std::log(x) - mu;
                    res[i] = neg_inv_2sigma2 * z * z - std::log(x) + log_norm_const;
                }
            });
            pool.waitForAll();
        });
}

void LogNormalDistribution::getCumulativeProbability(std::span<const double> values,
                                                     std::span<double> results,
                                                     const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<LogNormalDistribution>::distType(),
        detail::OperationType::CDF,
        [](const LogNormalDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const LogNormalDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double inv_sigma_sqrt2 = d.invSigmaSqrt2_;
            lock.unlock();
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, mu, inv_sigma_sqrt2);
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double sigma = d.sigma_;
            lock.unlock();

            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE)
                                 ? detail::ZERO_DOUBLE
                                 : detail::normal_cdf((std::log(x) - mu) / sigma);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = (x <= detail::ZERO_DOUBLE)
                                 ? detail::ZERO_DOUBLE
                                 : detail::normal_cdf((std::log(x) - mu) / sigma);
                }
            }
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;

            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_;
            const double sigma = d.sigma_;
            lock.unlock();

            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE)
                             ? detail::ZERO_DOUBLE
                             : detail::normal_cdf((std::log(x) - mu) / sigma);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

void LogNormalDistribution::getProbabilityWithStrategy(std::span<const double> values,
                                                       std::span<double> results,
                                                       detail::Strategy strategy) const {
    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
        [](const LogNormalDistribution& d, double x) { return d.getProbability(x); },
        [](const LogNormalDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, neg_inv_2sigma2 = d.negInv2SigmaSquared_,
                         lnc = d.logNormConst_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, mu, neg_inv_2sigma2, lnc);
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, neg_inv_2sigma2 = d.negInv2SigmaSquared_,
                         lnc = d.logNormConst_;
            lock.unlock();
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                    return;
                }
                const double z = std::log(x) - mu;
                res[i] = std::exp(neg_inv_2sigma2 * z * z - std::log(x) + lnc);
            });
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, neg_inv_2sigma2 = d.negInv2SigmaSquared_,
                         lnc = d.logNormConst_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                    return;
                }
                const double z = std::log(x) - mu;
                res[i] = std::exp(neg_inv_2sigma2 * z * z - std::log(x) + lnc);
            });
            pool.waitForAll();
        });
}

void LogNormalDistribution::getLogProbabilityWithStrategy(std::span<const double> values,
                                                          std::span<double> results,
                                                          detail::Strategy strategy) const {
    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
        [](const LogNormalDistribution& d, double x) { return d.getLogProbability(x); },
        [](const LogNormalDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, neg_inv_2sigma2 = d.negInv2SigmaSquared_,
                         lnc = d.logNormConst_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, mu, neg_inv_2sigma2, lnc);
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, neg_inv_2sigma2 = d.negInv2SigmaSquared_,
                         lnc = d.logNormConst_;
            lock.unlock();
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                    return;
                }
                const double z = std::log(x) - mu;
                res[i] = neg_inv_2sigma2 * z * z - std::log(x) + lnc;
            });
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, neg_inv_2sigma2 = d.negInv2SigmaSquared_,
                         lnc = d.logNormConst_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::NEGATIVE_INFINITY;
                    return;
                }
                const double z = std::log(x) - mu;
                res[i] = neg_inv_2sigma2 * z * z - std::log(x) + lnc;
            });
            pool.waitForAll();
        });
}

void LogNormalDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                                                 std::span<double> results,
                                                                 detail::Strategy strategy) const {
    detail::DispatchUtils::executeWithStrategy(
        *this, values, results, strategy,
        [](const LogNormalDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const LogNormalDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, inv_sigma_sqrt2 = d.invSigmaSqrt2_;
            lock.unlock();
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, mu, inv_sigma_sqrt2);
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, sigma = d.sigma_;
            lock.unlock();
            ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE)
                             ? detail::ZERO_DOUBLE
                             : detail::normal_cdf((std::log(x) - mu) / sigma);
            });
        },
        [](const LogNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<LogNormalDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double mu = d.mu_, sigma = d.sigma_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] = (x <= detail::ZERO_DOUBLE)
                             ? detail::ZERO_DOUBLE
                             : detail::normal_cdf((std::log(x) - mu) / sigma);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool LogNormalDistribution::operator==(const LogNormalDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(sigma_ - other.sigma_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool LogNormalDistribution::operator!=(const LogNormalDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const LogNormalDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, LogNormalDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("LogNormalDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t mu_pos = token.find("mu=");
    const size_t comma = token.find(",", mu_pos);
    const size_t sigma_pos = token.find("sigma=");
    const size_t close = token.find(")", sigma_pos);
    if (mu_pos == std::string::npos || comma == std::string::npos ||
        sigma_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double mu = std::stod(token.substr(mu_pos + 3, comma - mu_pos - 3));
        const double sigma = std::stod(token.substr(sigma_pos + 6, close - sigma_pos - 6));
        auto result = d.trySetParameters(mu, sigma);
        if (result.isError())
            is.setstate(std::ios::failbit);
    } catch (...) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// Compute+fixup pattern: SIMD runs over the full array assuming in-support
// inputs (x > 0), then a scalar fixup zeroes/neginfs any x <= 0 elements.
// Callers passing log-normal data rarely supply x <= 0, so the fixup is
// almost always a no-op.
//
// LogPDF pipeline (six steps):
//   temp    = log(x)                 [vector_log]
//   results = log(x) − μ  (= z)      [scalar_add(temp, −mu)]
//   results = z²                      [vector_multiply(results, results)]
//   results = −z²/(2σ²)              [scalar_multiply(neg_inv_2sigma2)]
//   results −= log(x)                 [vector_subtract(results, temp)]
//   results += log_norm_const         [scalar_add]
//
// PDF: identical then vector_exp.
// CDF: vector_log → scalar_add(−μ) → scalar_multiply(inv_sigma_sqrt2)
//      → vector_erf → scalar_add(1) → scalar_multiply(0.5)
//==============================================================================

void LogNormalDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_neg_inv_2sigma2, double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                const double lx = std::log(x);
                const double z = lx - cached_mu;
                results[i] = std::exp(cached_neg_inv_2sigma2 * z * z - lx + cached_log_norm_const);
            }
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    // Step 2: results = log(x) − μ  (z)
    arch::simd::VectorOps::scalar_add(temp.data(), -cached_mu, results, count);
    // Step 3: results = z²
    arch::simd::VectorOps::vector_multiply(results, results, results, count);
    // Step 4: results = −z²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_inv_2sigma2, results, count);
    // Step 5: results = −z²/(2σ²) − log(x)
    arch::simd::VectorOps::vector_subtract(results, temp.data(), results, count);
    // Step 6: results += log_norm_const
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);
    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: x <= 0 is outside support; PDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

void LogNormalDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_neg_inv_2sigma2, double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
            } else {
                const double lx = std::log(x);
                const double z = lx - cached_mu;
                results[i] = cached_neg_inv_2sigma2 * z * z - lx + cached_log_norm_const;
            }
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    // Step 2: results = log(x) − μ  (z)
    arch::simd::VectorOps::scalar_add(temp.data(), -cached_mu, results, count);
    // Step 3: results = z²
    arch::simd::VectorOps::vector_multiply(results, results, results, count);
    // Step 4: results = −z²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_inv_2sigma2, results, count);
    // Step 5: results = −z²/(2σ²) − log(x)
    arch::simd::VectorOps::vector_subtract(results, temp.data(), results, count);
    // Step 6: results += log_norm_const
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);

    // Fixup: x <= 0 is outside support; LogPDF = −∞.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void LogNormalDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_inv_sigma_sqrt2) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                // erf argument: (log(x) − μ) / (σ√2) = (log(x) − μ) * inv_sigma_sqrt2
                const double z = (std::log(x) - cached_mu) * cached_inv_sigma_sqrt2;
                results[i] = detail::HALF * (detail::ONE + std::erf(z));
            }
        }
        return;
    }

    std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

    // Step 1: temp = log(x)
    arch::simd::VectorOps::vector_log(values, temp.data(), count);
    // Step 2: results = log(x) − μ
    arch::simd::VectorOps::scalar_add(temp.data(), -cached_mu, results, count);
    // Step 2b: results = (log(x) − μ) / (σ√2)
    arch::simd::VectorOps::scalar_multiply(results, cached_inv_sigma_sqrt2, results, count);
    // Step 3: results = erf(...)
    arch::simd::VectorOps::vector_erf(results, results, count);
    // Step 4: results = 1 + erf(...)
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    // Step 5: results = 0.5·(1 + erf(...))
    arch::simd::VectorOps::scalar_multiply(results, detail::HALF, results, count);

    // Fixup: x <= 0 is outside support; CDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void LogNormalDistribution::updateCacheUnsafe() const noexcept {
    // Core derived quantities
    logSigma_ = std::log(sigma_);
    const double s2 = sigma_ * sigma_;
    negInv2SigmaSquared_ = -detail::HALF / s2;
    invSigmaSqrt2_ = detail::INV_SQRT_2 / sigma_;  // 1/(σ√2)

    // LogPDF normalisation constant: −log(σ) − ½ log(2π)
    logNormConst_ = -logSigma_ - detail::HALF_LN_2PI;

    // Moments
    mean_ = std::exp(mu_ + detail::HALF * s2);
    variance_ = std::expm1(s2) * std::exp(detail::TWO * mu_ + s2);

    // Optimization flag
    isStandard_ = (std::fabs(mu_) <= detail::DEFAULT_TOLERANCE &&
                   std::fabs(sigma_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

#include "libstats/distributions/binomial.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;
using stats::detail::validateNonNegativeParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

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

BinomialDistribution::BinomialDistribution(int n, double p) : DistributionBase(), n_(n), p_(p) {
    validateParameters(n, p);
    updateCacheUnsafe();
}

BinomialDistribution::BinomialDistribution(const BinomialDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    n_ = other.n_;
    p_ = other.p_;
    logNFact_ = other.logNFact_;
    logP_ = other.logP_;
    log1mP_ = other.log1mP_;
    atomicN_.store(n_, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
}

BinomialDistribution& BinomialDistribution::operator=(const BinomialDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        n_ = other.n_;
        p_ = other.p_;
        logNFact_ = other.logNFact_;
        logP_ = other.logP_;
        log1mP_ = other.log1mP_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicN_.store(n_, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
    }
    return *this;
}

BinomialDistribution::BinomialDistribution(BinomialDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    n_ = other.n_;
    p_ = other.p_;
    logNFact_ = other.logNFact_;
    logP_ = other.logP_;
    log1mP_ = other.log1mP_;
    other.n_ = 10;
    other.p_ = detail::HALF;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicN_.store(n_, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
}

BinomialDistribution& BinomialDistribution::operator=(BinomialDistribution&& other) noexcept {
    if (this != &other) {

        n_ = other.n_;
        p_ = other.p_;
        logNFact_ = other.logNFact_;
        logP_ = other.logP_;
        log1mP_ = other.log1mP_;
        other.n_ = 10;
        other.p_ = detail::HALF;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicN_.store(n_, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

BinomialDistribution BinomialDistribution::createUnchecked(int n, double p) noexcept {
    return BinomialDistribution(n, p, true);
}

BinomialDistribution::BinomialDistribution(int n, double p, bool /*bypassValidation*/) noexcept
    : DistributionBase(), n_(n), p_(p) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void BinomialDistribution::setN(int n) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(n, p_);
    n_ = n;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void BinomialDistribution::setP(double p) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(n_, p);
    p_ = p;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void BinomialDistribution::setParameters(int n, double p) {
    validateParameters(n, p);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    n_ = n;
    p_ = p;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double BinomialDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return static_cast<double>(n_) * p_;
}

double BinomialDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return static_cast<double>(n_) * p_ * (detail::ONE - p_);
}

double BinomialDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double var = static_cast<double>(n_) * p_ * (detail::ONE - p_);
    if (var <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    return (detail::ONE - detail::TWO * p_) / std::sqrt(var);
}

double BinomialDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double var = static_cast<double>(n_) * p_ * (detail::ONE - p_);
    if (var <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    return (detail::ONE - detail::SIX * p_ * (detail::ONE - p_)) / var;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult BinomialDistribution::trySetN(int n) noexcept {
    try {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto v = validateBinomialParameters(n, p_);
        if (v.isError())
            return v;
        n_ = n;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        updateCacheUnsafe();
        return VoidResult::ok({});
    } catch (const std::exception& e) {
        return VoidResult::makeError(ValidationError::UnknownError, e.what());
    }
}

VoidResult BinomialDistribution::trySetP(double p) noexcept {
    try {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto v = validateBinomialParameters(n_, p);
        if (v.isError())
            return v;
        p_ = p;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        updateCacheUnsafe();
        return VoidResult::ok({});
    } catch (const std::exception& e) {
        return VoidResult::makeError(ValidationError::UnknownError, e.what());
    }
}

VoidResult BinomialDistribution::trySetParameters(int n, double p) noexcept {
    try {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto v = validateBinomialParameters(n, p);
        if (v.isError())
            return v;
        n_ = n;
        p_ = p;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        updateCacheUnsafe();
        return VoidResult::ok({});
    } catch (const std::exception& e) {
        return VoidResult::makeError(ValidationError::UnknownError, e.what());
    }
}

VoidResult BinomialDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateBinomialParameters(n_, p_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double BinomialDistribution::logBinomCoeff(int k) const noexcept {
    if (k < 0 || k > n_)
        return detail::NEGATIVE_INFINITY;
    return logNFact_ - std::lgamma(static_cast<double>(k + 1)) -
           std::lgamma(static_cast<double>(n_ - k + 1));
}

double BinomialDistribution::getProbability(double x) const {
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;
    const int k = static_cast<int>(std::round(x));
    if (k < 0 || k > n_)
        return detail::ZERO_DOUBLE;
    if (p_ == detail::ZERO_DOUBLE)
        return (k == 0) ? detail::ONE : detail::ZERO_DOUBLE;
    if (p_ == detail::ONE)
        return (k == n_) ? detail::ONE : detail::ZERO_DOUBLE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double lp =
        logBinomCoeff(k) + static_cast<double>(k) * logP_ + static_cast<double>(n_ - k) * log1mP_;
    return std::clamp(std::exp(lp), detail::ZERO_DOUBLE, detail::ONE);
}

double BinomialDistribution::getLogProbability(double x) const noexcept {
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;
    const int k = static_cast<int>(std::round(x));
    if (k < 0 || k > n_)
        return detail::NEGATIVE_INFINITY;
    if (p_ == detail::ZERO_DOUBLE)
        return (k == 0) ? detail::ZERO_DOUBLE : detail::NEGATIVE_INFINITY;
    if (p_ == detail::ONE)
        return (k == n_) ? detail::ZERO_DOUBLE : detail::NEGATIVE_INFINITY;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return logBinomCoeff(k) + static_cast<double>(k) * logP_ +
           static_cast<double>(n_ - k) * log1mP_;
}

double BinomialDistribution::getCumulativeProbability(double x) const {
    // EDGE-3: CDF(-inf) must be 0, not 1. Three-way branch on non-finite inputs.
    if (!std::isfinite(x))
        return std::isnan(x)  ? detail::ZERO_DOUBLE
               : (x < 0)      ? detail::ZERO_DOUBLE   // -inf
                               : detail::ONE;           // +inf
    const int k = static_cast<int>(std::floor(x));
    if (k < 0)
        return detail::ZERO_DOUBLE;
    if (k >= n_)
        return detail::ONE;

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    if (p_ == detail::ZERO_DOUBLE)
        return detail::ONE;  // all mass at k=0
    if (p_ == detail::ONE)
        return detail::ZERO_DOUBLE;  // all mass at k=n

    // CDF(k; n, p) = I_{1-p}(n-k, k+1)  — regularized incomplete beta
    return detail::beta_i(detail::ONE - p_, static_cast<double>(n_ - k),
                          static_cast<double>(k + 1));
}

double BinomialDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return static_cast<double>(n_);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const int n = n_;
    lock.unlock();

    // Scan from 0 — for large n use bisection
    if (n <= 500) {
        for (int k = 0; k <= n; ++k) {
            if (getCumulativeProbability(static_cast<double>(k)) >= p)
                return static_cast<double>(k);
        }
        return static_cast<double>(n);
    }
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (getCumulativeProbability(static_cast<double>(mid)) < p)
            lo = mid + 1;
        else
            hi = mid;
    }
    return static_cast<double>(lo);
}

double BinomialDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const int n = n_;
    const double p = p_;
    lock.unlock();
    std::binomial_distribution<int> dist(n, p);
    return static_cast<double>(dist(rng));
}

std::vector<double> BinomialDistribution::sample(std::mt19937& rng, size_t count) const {
    std::vector<double> samples;
    samples.reserve(count);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const int n = n_;
    const double p = p_;
    lock.unlock();
    std::binomial_distribution<int> dist(n, p);
    for (size_t i = 0; i < count; ++i)
        samples.push_back(static_cast<double>(dist(rng)));
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void BinomialDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Cannot fit distribution to empty data");

    int maxObs = 0;
    double sum = detail::ZERO_DOUBLE;
    double sum_sq = detail::ZERO_DOUBLE;
    std::size_t count = 0;
    for (double v : values) {
        if (v >= detail::ZERO_DOUBLE && std::isfinite(v)) {
            const int k = static_cast<int>(std::round(v));
            maxObs = std::max(maxObs, k);
            const double kd = static_cast<double>(k);
            sum    += kd;
            sum_sq += kd * kd;
            ++count;
        }
    }
    if (count == 0 || maxObs == 0) {
        reset();
        return;
    }

    const double n_d  = static_cast<double>(count);
    const double xbar = sum / n_d;

    // Method-of-moments (MC-10): n̂ = x̄² / (x̄ − s²),  p̂ = x̄ / n̂
    // Valid when s² < x̄ (underdispersion consistent with Binomial).
    // Falls back to n̂ = max(obs) when MoM is inapplicable (overdispersed or
    // all observations equal).
    const double var = sum_sq / n_d - xbar * xbar;  // biased MLE sample variance

    int n_hat;
    double p_hat;
    if (count >= 2 && var > detail::ZERO_DOUBLE && var < xbar) {
        const double n_mom = xbar * xbar / (xbar - var);
        n_hat = std::max(maxObs, static_cast<int>(std::round(n_mom)));
        p_hat = std::clamp(xbar / static_cast<double>(n_hat),
                           detail::ZERO_DOUBLE, detail::ONE);
    } else {
        // Fallback: max(obs) is a lower bound for n; MLE p̂ = x̄ / n given n.
        n_hat = maxObs;
        p_hat = std::clamp(xbar / static_cast<double>(n_hat),
                           detail::ZERO_DOUBLE, detail::ONE);
    }
    setParameters(n_hat, p_hat);
}

void BinomialDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<BinomialDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void BinomialDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    n_ = 10;
    p_ = detail::HALF;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string BinomialDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "BinomialDistribution(n=" << n_ << ",p=" << p_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

int BinomialDistribution::getNAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicN_.load(std::memory_order_acquire);
    return getN();
}

double BinomialDistribution::getPAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicP_.load(std::memory_order_acquire);
    return getP();
}

double BinomialDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double np1p = static_cast<double>(n_ + 1) * p_;
    const int mode = static_cast<int>(std::floor(np1p));
    // If (n+1)p is exactly integer, mode is np1p-1; otherwise floor(np1p)
    if (std::fabs(np1p - std::round(np1p)) < 1e-12 && np1p > detail::ZERO_DOUBLE)
        return np1p - detail::ONE;
    return static_cast<double>(mode);
}

double BinomialDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const int n = n_;
    const double p = p_;
    const double lnf = logNFact_;
    lock.unlock();

    // Entropy is returned in nats (natural units; std::log = log base e).
    //
    // For n ≤ 1000: exact Pearson-Shannon entropy via the PMF.
    //   H = -Σ_{k=0}^{n} P(k) · ln P(k)
    //   ln P(k) = ln C(n,k) + k·ln(p) + (n-k)·ln(1-p)
    //           = (logNFact - lgamma(k+1) - lgamma(n-k+1)) + k*lp + (n-k)*l1mp
    // This is exact up to floating-point rounding for any p in [0,1].
    //
    // For n > 1000: Gaussian approximation H ≈ ½ ln(2πe·npq), which
    // has <0.1% relative error for n·1000 and any non-degenerate p.
    constexpr int kExactThreshold = 1000;
    if (n <= kExactThreshold) {
        if (p <= detail::ZERO_DOUBLE || p >= detail::ONE)
            return detail::ZERO_DOUBLE;  // degenerate: all mass at one point
        const double lp   = std::log(p);
        const double l1mp = std::log(detail::ONE - p);
        double h = detail::ZERO_DOUBLE;
        for (int k = 0; k <= n; ++k) {
            // log P(k): log-binomial coefficient + log p^k (1-p)^(n-k)
            const double log_pmf = lnf
                - std::lgamma(static_cast<double>(k + 1))
                - std::lgamma(static_cast<double>(n - k + 1))
                + static_cast<double>(k) * lp
                + static_cast<double>(n - k) * l1mp;
            // P(k) * log P(k); guard against log_pmf = -inf when P(k) is tiny
            if (std::isfinite(log_pmf))
                h -= std::exp(log_pmf) * log_pmf;
        }
        return h;
    }
    // Large n: Gaussian approximation (MC-14)
    const double var = static_cast<double>(n) * p * (detail::ONE - p);
    if (var <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    return detail::HALF * std::log(detail::TWO * detail::PI * detail::E * var);
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void BinomialDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const BinomialDistribution& d, double x) { return d.getProbability(x); },
        [](const BinomialDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<BinomialDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const int n = d.n_;
            const double lnf = d.logNFact_, lp = d.logP_, l1mp = d.log1mP_;
            lock.unlock();
            d.getProbabilityBatchImpl(vals, res, count, n, lnf, lp, l1mp);
        },
        [](const BinomialDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<BinomialDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const int n = d.n_;
            const double lnf = d.logNFact_, lp = d.logP_, l1mp = d.log1mP_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = d.getProbability(vals[i]);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = d.getProbability(vals[i]);
            }
            (void)n;
            (void)lnf;
            (void)lp;
            (void)l1mp;
        },
        [](const BinomialDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getProbability(vals[i]); });
            pool.waitForAll();
        });
}

void BinomialDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const BinomialDistribution& d, double x) { return d.getLogProbability(x); },
        [](const BinomialDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<BinomialDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const int n = d.n_;
            const double lnf = d.logNFact_, lp = d.logP_, l1mp = d.log1mP_;
            lock.unlock();
            d.getLogProbabilityBatchImpl(vals, res, count, n, lnf, lp, l1mp);
        },
        [](const BinomialDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = d.getLogProbability(vals[i]);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = d.getLogProbability(vals[i]);
            }
        },
        [](const BinomialDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getLogProbability(vals[i]); });
            pool.waitForAll();
        });
}

void BinomialDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const BinomialDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const BinomialDistribution& d, const double* vals, double* res, size_t count) {
            d.getCumulativeProbabilityBatchImpl(vals, res, count);
        },
        [](const BinomialDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = d.getCumulativeProbability(vals[i]);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = d.getCumulativeProbability(vals[i]);
            }
        },
        [](const BinomialDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getCumulativeProbability(vals[i]); });
            pool.waitForAll();
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool BinomialDistribution::operator==(const BinomialDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return (n_ == other.n_) && (std::fabs(p_ - other.p_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE);
}

bool BinomialDistribution::operator!=(const BinomialDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const BinomialDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, BinomialDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("BinomialDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t n_pos = token.find("n=");
    const size_t comma = token.find(",", n_pos);
    const size_t p_pos = token.find("p=");
    const size_t close = token.find(")", p_pos);
    if (n_pos == std::string::npos || comma == std::string::npos || p_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const int n = static_cast<int>(std::stod(token.substr(n_pos + 2, comma - n_pos - 2)));
        const double p = std::stod(token.substr(p_pos + 2, close - p_pos - 2));
        auto result = d.trySetParameters(n, p);
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
// Scalar loop with cached logNFact_, logP_, log1mP_ — identical pattern
// to Von Mises and Poisson: no SIMD because lgamma per element is not
// in VectorOps.  The caching eliminates the dominant repeated computations.
//==============================================================================

void BinomialDistribution::getLogProbabilityBatchImpl(const double* values, double* results,
                                                      std::size_t count, int cached_n,
                                                      double cached_logNFact, double cached_logP,
                                                      double cached_log1mP) const noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (!std::isfinite(x)) {
            results[i] = detail::NEGATIVE_INFINITY;
            continue;
        }
        const int k = static_cast<int>(std::round(x));
        if (k < 0 || k > cached_n) {
            results[i] = detail::NEGATIVE_INFINITY;
            continue;
        }
        if (cached_logP == detail::NEGATIVE_INFINITY) {
            results[i] = (k == 0) ? 0.0 : detail::NEGATIVE_INFINITY;
            continue;
        }
        if (cached_log1mP == detail::NEGATIVE_INFINITY) {
            results[i] = (k == cached_n) ? 0.0 : detail::NEGATIVE_INFINITY;
            continue;
        }
        const double lc = cached_logNFact - std::lgamma(static_cast<double>(k + 1)) -
                          std::lgamma(static_cast<double>(cached_n - k + 1));
        results[i] = lc + static_cast<double>(k) * cached_logP +
                     static_cast<double>(cached_n - k) * cached_log1mP;
    }
}

void BinomialDistribution::getProbabilityBatchImpl(const double* values, double* results,
                                                   std::size_t count, int cached_n,
                                                   double cached_logNFact, double cached_logP,
                                                   double cached_log1mP) const noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (!std::isfinite(x)) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }
        const int k = static_cast<int>(std::round(x));
        if (k < 0 || k > cached_n) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }
        if (cached_logP == detail::NEGATIVE_INFINITY) {
            results[i] = (k == 0) ? detail::ONE : detail::ZERO_DOUBLE;
            continue;
        }
        if (cached_log1mP == detail::NEGATIVE_INFINITY) {
            results[i] = (k == cached_n) ? detail::ONE : detail::ZERO_DOUBLE;
            continue;
        }
        const double lc = cached_logNFact - std::lgamma(static_cast<double>(k + 1)) -
                          std::lgamma(static_cast<double>(cached_n - k + 1));
        const double lp = lc + static_cast<double>(k) * cached_logP +
                          static_cast<double>(cached_n - k) * cached_log1mP;
        results[i] = std::clamp(std::exp(lp), detail::ZERO_DOUBLE, detail::ONE);
    }
}

void BinomialDistribution::getCumulativeProbabilityBatchImpl(const double* values, double* results,
                                                             std::size_t count) const noexcept {
    for (std::size_t i = 0; i < count; ++i)
        results[i] = getCumulativeProbability(values[i]);
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void BinomialDistribution::updateCacheUnsafe() const noexcept {
    logNFact_ = std::lgamma(static_cast<double>(n_ + 1));
    logP_ = (p_ > detail::ZERO_DOUBLE) ? std::log(p_) : detail::NEGATIVE_INFINITY;
    log1mP_ = (p_ < detail::ONE) ? std::log(detail::ONE - p_) : detail::NEGATIVE_INFINITY;
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicN_.store(n_, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS
//==============================================================================

}  // namespace stats

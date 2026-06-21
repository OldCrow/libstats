#include "libstats/distributions/negative_binomial.h"
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

NegativeBinomialDistribution::NegativeBinomialDistribution(double r, double p)
    : DistributionBase(), r_(r), p_(p) {
    validateParameters(r, p);
    updateCacheUnsafe();
}

NegativeBinomialDistribution::NegativeBinomialDistribution(
    const NegativeBinomialDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    r_ = other.r_;
    p_ = other.p_;
    logGammaR_ = other.logGammaR_;
    logP_ = other.logP_;
    log1mP_ = other.log1mP_;
    atomicR_.store(r_, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
}

NegativeBinomialDistribution& NegativeBinomialDistribution::operator=(
    const NegativeBinomialDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        r_ = other.r_;
        p_ = other.p_;
        logGammaR_ = other.logGammaR_;
        logP_ = other.logP_;
        log1mP_ = other.log1mP_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicR_.store(r_, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
    }
    return *this;
}

NegativeBinomialDistribution::NegativeBinomialDistribution(
    NegativeBinomialDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    r_ = other.r_;
    p_ = other.p_;
    logGammaR_ = other.logGammaR_;
    logP_ = other.logP_;
    log1mP_ = other.log1mP_;
    other.r_ = detail::ONE;
    other.p_ = detail::HALF;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicR_.store(r_, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
}

NegativeBinomialDistribution& NegativeBinomialDistribution::operator=(
    NegativeBinomialDistribution&& other) noexcept {
    if (this != &other) {

        r_ = other.r_;
        p_ = other.p_;
        logGammaR_ = other.logGammaR_;
        logP_ = other.logP_;
        log1mP_ = other.log1mP_;
        other.r_ = detail::ONE;
        other.p_ = detail::HALF;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicR_.store(r_, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

NegativeBinomialDistribution NegativeBinomialDistribution::createUnchecked(double r,
                                                                           double p) noexcept {
    return NegativeBinomialDistribution(r, p, true);
}

NegativeBinomialDistribution::NegativeBinomialDistribution(double r, double p,
                                                           bool /*bypassValidation*/) noexcept
    : DistributionBase(), r_(r), p_(p) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void NegativeBinomialDistribution::setR(double r) {
    validateParameters(r, getP());
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    r_ = r;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void NegativeBinomialDistribution::setP(double p) {
    validateParameters(getR(), p);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    p_ = p;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void NegativeBinomialDistribution::setParameters(double r, double p) {
    validateParameters(r, p);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    r_ = r;
    p_ = p;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double NegativeBinomialDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return r_ * (detail::ONE - p_) / p_;
}

double NegativeBinomialDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return r_ * (detail::ONE - p_) / (p_ * p_);
}

double NegativeBinomialDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double denom = r_ * (detail::ONE - p_);
    if (denom <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    return (detail::TWO - p_) / std::sqrt(denom);
}

double NegativeBinomialDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double q = detail::ONE - p_;
    const double denom = r_ * q;
    if (denom <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    return detail::SIX / r_ + (p_ * p_) / denom;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult NegativeBinomialDistribution::trySetR(double r) noexcept {
    auto v = validateNegativeBinomialParameters(r, getP());
    if (v.isError())
        return v;
    setR(r);
    return VoidResult::ok({});
}

VoidResult NegativeBinomialDistribution::trySetP(double p) noexcept {
    auto v = validateNegativeBinomialParameters(getR(), p);
    if (v.isError())
        return v;
    setP(p);
    return VoidResult::ok({});
}

VoidResult NegativeBinomialDistribution::trySetParameters(double r, double p) noexcept {
    auto v = validateNegativeBinomialParameters(r, p);
    if (v.isError())
        return v;
    setParameters(r, p);
    return VoidResult::ok({});
}

VoidResult NegativeBinomialDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateNegativeBinomialParameters(r_, p_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double NegativeBinomialDistribution::getProbability(double x) const {
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;
    const int k = static_cast<int>(std::round(x));
    if (k < 0)
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
    // p=1: all probability mass at k=0
    if (p_ >= detail::ONE)
        return (k == 0) ? detail::ONE : detail::ZERO_DOUBLE;
    const double lp = std::lgamma(static_cast<double>(k) + r_) -
                      std::lgamma(static_cast<double>(k + 1)) - logGammaR_ + r_ * logP_ +
                      static_cast<double>(k) * log1mP_;
    return std::clamp(std::exp(lp), detail::ZERO_DOUBLE, detail::ONE);
}

double NegativeBinomialDistribution::getLogProbability(double x) const noexcept {
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;
    const int k = static_cast<int>(std::round(x));
    if (k < 0)
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
    // p=1: all probability mass at k=0
    if (p_ >= detail::ONE)
        return (k == 0) ? detail::ZERO_DOUBLE : detail::NEGATIVE_INFINITY;
    return std::lgamma(static_cast<double>(k) + r_) - std::lgamma(static_cast<double>(k + 1)) -
           logGammaR_ + r_ * logP_ + static_cast<double>(k) * log1mP_;
}

double NegativeBinomialDistribution::getCumulativeProbability(double x) const {
    if (!std::isfinite(x))
        return std::isnan(x) ? detail::ZERO_DOUBLE : detail::ONE;
    const int k = static_cast<int>(std::floor(x));
    if (k < 0)
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
    // P(X ≤ k; r, p) = I_p(r, k+1)  — regularized incomplete beta
    return detail::beta_i(p_, r_, static_cast<double>(k + 1));
}

double NegativeBinomialDistribution::getQuantile(double prob) const {
    if (prob < detail::ZERO_DOUBLE || prob > detail::ONE)
        throw std::invalid_argument("Probability must be in [0, 1]");
    if (prob == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (prob == detail::ONE)
        return std::numeric_limits<double>::infinity();

    // Estimate upper bound as mean + 10·stddev (ample headroom)
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double r = r_, p = p_;
    lock.unlock();
    const double mean = r * (detail::ONE - p) / p;
    const double stddev = std::sqrt(r * (detail::ONE - p) / (p * p));
    const int max_k = static_cast<int>(std::ceil(mean + detail::TEN * stddev + detail::HUNDRED));

    // Linear scan for small quantiles; bisection otherwise
    if (mean <= detail::FIFTY) {
        for (int k = 0; k <= max_k; ++k) {
            if (getCumulativeProbability(static_cast<double>(k)) >= prob)
                return static_cast<double>(k);
        }
        return static_cast<double>(max_k);
    }
    int lo = 0, hi = max_k;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (getCumulativeProbability(static_cast<double>(mid)) < prob)
            lo = mid + 1;
        else
            hi = mid;
    }
    return static_cast<double>(lo);
}

double NegativeBinomialDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double r = r_;
    const double p = p_;
    lock.unlock();
    // Gamma-Poisson mixture: λ ~ Gamma(r, (1-p)/p), then X ~ Poisson(λ)
    std::gamma_distribution<double> gamma_dist(r, (detail::ONE - p) / p);
    const double lambda = gamma_dist(rng);
    std::poisson_distribution<int> poisson_dist(lambda);
    return static_cast<double>(poisson_dist(rng));
}

std::vector<double> NegativeBinomialDistribution::sample(std::mt19937& rng, size_t count) const {
    std::vector<double> samples;
    samples.reserve(count);
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double r = r_;
    const double p = p_;
    lock.unlock();
    std::gamma_distribution<double> gamma_dist(r, (detail::ONE - p) / p);
    for (size_t i = 0; i < count; ++i) {
        const double lambda = gamma_dist(rng);
        std::poisson_distribution<int> poisson_dist(lambda);
        samples.push_back(static_cast<double>(poisson_dist(rng)));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void NegativeBinomialDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Cannot fit distribution to empty data");

    // Collect valid observations
    std::vector<double> obs;
    obs.reserve(values.size());
    for (double v : values) {
        if (v >= detail::ZERO_DOUBLE && std::isfinite(v))
            obs.push_back(std::round(v));
    }
    if (obs.empty()) {
        reset();
        return;
    }

    const double n = static_cast<double>(obs.size());
    double mean = detail::ZERO_DOUBLE;
    for (double k : obs)
        mean += k;
    mean /= n;

    if (mean <= detail::ZERO_DOUBLE) {
        // All zeros: r=1, p=1
        setParameters(detail::ONE, detail::ONE);
        return;
    }

    double var = detail::ZERO_DOUBLE;
    for (double k : obs) {
        const double d = k - mean;
        var += d * d;
    }
    var /= (n - detail::ONE > detail::ZERO ? n - detail::ONE : detail::ONE);

    // Under-dispersion: use MoM with fallback r̂ = 1
    double r_hat;
    if (var <= mean + detail::ZERO) {
        r_hat = detail::ONE;
    } else {
        r_hat = (mean * mean) / (var - mean);
        if (!std::isfinite(r_hat) || r_hat <= detail::ZERO_DOUBLE)
            r_hat = detail::ONE;
    }

    // Newton-Raphson on the profile score equation:
    //   f(r) = Σᵢ [ψ(kᵢ+r) − ψ(r)] + n·[log(r) − log(r+k̄)] = 0
    //   f'(r) = Σᵢ [ψ'(kᵢ+r) − ψ'(r)] + n·[1/r − 1/(r+k̄)]
    //
    // After convergence: p̂ = r / (r + k̄)
    //
    // Reference: Ported from libhmm's nb_mle_solve.
    constexpr int MAX_ITER = 200;
    constexpr double CONV_TOL = 1e-11;

    double r_curr = r_hat;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double score = detail::ZERO_DOUBLE;
        double d_score = detail::ZERO_DOUBLE;
        for (double k : obs) {
            score += detail::digamma(k + r_curr) - detail::digamma(r_curr);
            d_score += detail::trigamma(k + r_curr) - detail::trigamma(r_curr);
        }
        score += n * (std::log(r_curr) - std::log(r_curr + mean));
        d_score += n * (detail::ONE / r_curr - detail::ONE / (r_curr + mean));

        if (std::fabs(d_score) < 1e-15)
            break;
        const double dr = score / d_score;
        r_curr -= dr;
        if (r_curr <= detail::ZERO_DOUBLE) {
            r_curr = detail::HIGH_PRECISION_TOLERANCE;
        }
        if (std::fabs(dr) < CONV_TOL * r_curr)
            break;
    }

    if (!std::isfinite(r_curr) || r_curr <= detail::ZERO_DOUBLE)
        r_curr = r_hat;

    const double p_hat =
        std::clamp(r_curr / (r_curr + mean), detail::HIGH_PRECISION_TOLERANCE, detail::ONE);
    setParameters(r_curr, p_hat);
}

void NegativeBinomialDistribution::parallelBatchFit(
    const std::vector<std::vector<double>>& datasets,
    std::vector<NegativeBinomialDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void NegativeBinomialDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    r_ = detail::ONE;
    p_ = detail::HALF;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string NegativeBinomialDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "NegativeBinomialDistribution(r=" << r_ << ",p=" << p_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double NegativeBinomialDistribution::getRAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicR_.load(std::memory_order_acquire);
    return getR();
}

double NegativeBinomialDistribution::getPAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicP_.load(std::memory_order_acquire);
    return getP();
}

double NegativeBinomialDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (r_ <= detail::ONE)
        return detail::ZERO_DOUBLE;
    return std::floor((r_ - detail::ONE) * (detail::ONE - p_) / p_);
}

double NegativeBinomialDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // Stirling approximation: H ≈ ½ log(2πe · r(1-p)/p²)
    const double var = r_ * (detail::ONE - p_) / (p_ * p_);
    if (var <= detail::ZERO)
        return detail::ZERO_DOUBLE;
    return detail::HALF * std::log(detail::TWO * detail::PI * detail::E * var);
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void NegativeBinomialDistribution::getProbability(std::span<const double> values,
                                                  std::span<double> results,
                                                  const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint,
        detail::OperationType::PDF,
        [](const NegativeBinomialDistribution& d, double x) { return d.getProbability(x); },
        [](const NegativeBinomialDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<NegativeBinomialDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double r = d.r_, lgr = d.logGammaR_, lp = d.logP_, l1mp = d.log1mP_;
            lock.unlock();
            d.getProbabilityBatchImpl(vals, res, count, r, lgr, lp, l1mp);
        },
        [](const NegativeBinomialDistribution& d, std::span<const double> vals,
           std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = d.getProbability(vals[i]);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = d.getProbability(vals[i]);
            }
        },
        [](const NegativeBinomialDistribution& d, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getProbability(vals[i]); });
            pool.waitForAll();
        });
}

void NegativeBinomialDistribution::getLogProbability(std::span<const double> values,
                                                     std::span<double> results,
                                                     const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint,
        detail::OperationType::LOG_PDF,
        [](const NegativeBinomialDistribution& d, double x) { return d.getLogProbability(x); },
        [](const NegativeBinomialDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<NegativeBinomialDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double r = d.r_, lgr = d.logGammaR_, lp = d.logP_, l1mp = d.log1mP_;
            lock.unlock();
            d.getLogProbabilityBatchImpl(vals, res, count, r, lgr, lp, l1mp);
        },
        [](const NegativeBinomialDistribution& d, std::span<const double> vals,
           std::span<double> res) {
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
        [](const NegativeBinomialDistribution& d, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getLogProbability(vals[i]); });
            pool.waitForAll();
        });
}

void NegativeBinomialDistribution::getCumulativeProbability(
    std::span<const double> values, std::span<double> results,
    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint,
        detail::OperationType::CDF,
        [](const NegativeBinomialDistribution& d, double x) {
            return d.getCumulativeProbability(x);
        },
        [](const NegativeBinomialDistribution& d, const double* vals, double* res, size_t count) {
            d.getCumulativeProbabilityBatchImpl(vals, res, count);
        },
        [](const NegativeBinomialDistribution& d, std::span<const double> vals,
           std::span<double> res) {
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
        [](const NegativeBinomialDistribution& d, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
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

bool NegativeBinomialDistribution::operator==(const NegativeBinomialDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return (std::fabs(r_ - other.r_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE) &&
           (std::fabs(p_ - other.p_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE);
}

bool NegativeBinomialDistribution::operator!=(const NegativeBinomialDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const NegativeBinomialDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, NegativeBinomialDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("NegativeBinomialDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t r_pos = token.find("r=");
    const size_t comma = token.find(",", r_pos);
    const size_t p_pos = token.find("p=");
    const size_t close = token.find(")", p_pos);
    if (r_pos == std::string::npos || comma == std::string::npos || p_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double r = std::stod(token.substr(r_pos + 2, comma - r_pos - 2));
        const double p = std::stod(token.substr(p_pos + 2, close - p_pos - 2));
        auto result = d.trySetParameters(r, p);
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
// Scalar loop with cached logGammaR_, logP_, log1mP_ — same pattern as
// Binomial: no SIMD because lgamma(k+r) varies per element.
//==============================================================================

void NegativeBinomialDistribution::getLogProbabilityBatchImpl(const double* values, double* results,
                                                              std::size_t count, double cached_r,
                                                              double cached_logGammaR,
                                                              double cached_logP,
                                                              double cached_log1mP) const noexcept {
    // cached_log1mP == -inf when p=1; guard k*(-inf) = 0*(-inf) = NaN
    const bool p_is_one = !std::isfinite(cached_log1mP);
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (!std::isfinite(x)) {
            results[i] = detail::NEGATIVE_INFINITY;
            continue;
        }
        const int k = static_cast<int>(std::round(x));
        if (k < 0) {
            results[i] = detail::NEGATIVE_INFINITY;
            continue;
        }
        if (p_is_one) {
            results[i] = (k == 0) ? detail::ZERO_DOUBLE : detail::NEGATIVE_INFINITY;
            continue;
        }
        results[i] = std::lgamma(static_cast<double>(k) + cached_r) -
                     std::lgamma(static_cast<double>(k + 1)) - cached_logGammaR +
                     cached_r * cached_logP + static_cast<double>(k) * cached_log1mP;
    }
}

void NegativeBinomialDistribution::getProbabilityBatchImpl(const double* values, double* results,
                                                           std::size_t count, double cached_r,
                                                           double cached_logGammaR,
                                                           double cached_logP,
                                                           double cached_log1mP) const noexcept {
    const bool p_is_one = !std::isfinite(cached_log1mP);
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (!std::isfinite(x)) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }
        const int k = static_cast<int>(std::round(x));
        if (k < 0) {
            results[i] = detail::ZERO_DOUBLE;
            continue;
        }
        if (p_is_one) {
            results[i] = (k == 0) ? detail::ONE : detail::ZERO_DOUBLE;
            continue;
        }
        const double lp = std::lgamma(static_cast<double>(k) + cached_r) -
                          std::lgamma(static_cast<double>(k + 1)) - cached_logGammaR +
                          cached_r * cached_logP + static_cast<double>(k) * cached_log1mP;
        results[i] = std::clamp(std::exp(lp), detail::ZERO_DOUBLE, detail::ONE);
    }
}

void NegativeBinomialDistribution::getCumulativeProbabilityBatchImpl(
    const double* values, double* results, std::size_t count) const noexcept {
    for (std::size_t i = 0; i < count; ++i)
        results[i] = getCumulativeProbability(values[i]);
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void NegativeBinomialDistribution::updateCacheUnsafe() const noexcept {
    logGammaR_ = std::lgamma(r_);
    logP_ = std::log(p_);
    log1mP_ = (p_ < detail::ONE) ? std::log(detail::ONE - p_) : detail::NEGATIVE_INFINITY;
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicR_.store(r_, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS
//==============================================================================

}  // namespace stats

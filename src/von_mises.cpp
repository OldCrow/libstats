#include "libstats/distributions/von_mises.h"
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;
using stats::detail::validateNonNegativeParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/bessel.h"
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
// Private helper: angle wrapping
//==============================================================================

double VonMisesDistribution::wrapAngle(double x) noexcept {
    if (!std::isfinite(x))
        return x;
    x = std::fmod(x, detail::TWO_PI);
    if (x <= -detail::PI)
        x += detail::TWO_PI;
    if (x > detail::PI)
        x -= detail::TWO_PI;
    return x;
}

//==============================================================================
// Private helper: κ from mean resultant length R̄
//
// Mardia–Jupp approximation (Mardia & Jupp 2000, Directional Statistics §A.2),
// refined by Newton–Raphson on A(κ) = I₁(κ)/I₀(κ) = R̄.
// Derivative: A'(κ) = 1 − A(κ)² − A(κ)/κ.
// Always converges (function is monotone increasing); 3–5 Newton steps suffice.
//==============================================================================

namespace {

[[nodiscard]] double kappa_from_r_bar(double R_bar) noexcept {
    if (R_bar <= 0.0)
        return 0.0;
    if (R_bar >= 1.0)
        return 1.0e6;  // effectively point mass

    double kappa;
    if (R_bar < 0.53) {
        kappa = detail::TWO * R_bar + R_bar * R_bar * R_bar +
                (5.0 / 6.0) * R_bar * R_bar * R_bar * R_bar * R_bar;
    } else if (R_bar < 0.85) {
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (detail::ONE - R_bar);
    } else {
        const double r = R_bar;
        kappa = detail::ONE / (r * r * r - 4.0 * r * r + 3.0 * r);
    }
    if (kappa < 0.0)
        kappa = 0.0;

    for (int iter = 0; iter < 20 && kappa > 0.0; ++iter) {
        const double i0 = detail::bessel_i0(kappa);
        const double i1 = detail::bessel_i1(kappa);
        if (i0 <= 0.0)
            break;
        const double A = i1 / i0;
        const double Ap = detail::ONE - A * A - A / kappa;
        if (std::fabs(Ap) < 1e-15)
            break;
        const double dk = (A - R_bar) / Ap;
        kappa -= dk;
        if (kappa < 0.0) {
            kappa = 0.0;
            break;
        }
        if (std::fabs(dk) < 1e-12 * (detail::ONE + kappa))
            break;
    }
    return kappa;
}

}  // anonymous namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

VonMisesDistribution::VonMisesDistribution(double mu, double kappa)
    : DistributionBase(), mu_(wrapAngle(mu)), kappa_(kappa) {
    validateParameters(mu_, kappa_);
    updateCacheUnsafe();
}

VonMisesDistribution::VonMisesDistribution(const VonMisesDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    kappa_ = other.kappa_;
    logNormaliser_ = other.logNormaliser_;
    circularVariance_ = other.circularVariance_;
    isUniform_ = other.isUniform_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
}

VonMisesDistribution& VonMisesDistribution::operator=(const VonMisesDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_ = other.mu_;
        kappa_ = other.kappa_;
        logNormaliser_ = other.logNormaliser_;
        circularVariance_ = other.circularVariance_;
        isUniform_ = other.isUniform_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicKappa_.store(kappa_, std::memory_order_release);
    }
    return *this;
}

VonMisesDistribution::VonMisesDistribution(VonMisesDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mu_ = other.mu_;
    kappa_ = other.kappa_;
    logNormaliser_ = other.logNormaliser_;
    circularVariance_ = other.circularVariance_;
    isUniform_ = other.isUniform_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.kappa_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
}

VonMisesDistribution& VonMisesDistribution::operator=(VonMisesDistribution&& other) noexcept {
    if (this != &other) {

        mu_ = other.mu_;
        kappa_ = other.kappa_;
        logNormaliser_ = other.logNormaliser_;
        circularVariance_ = other.circularVariance_;
        isUniform_ = other.isUniform_;
        other.mu_ = detail::ZERO_DOUBLE;
        other.kappa_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicKappa_.store(kappa_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

VonMisesDistribution VonMisesDistribution::createUnchecked(double mu, double kappa) noexcept {
    return VonMisesDistribution(wrapAngle(mu), kappa, true);
}

VonMisesDistribution::VonMisesDistribution(double mu, double kappa,
                                           bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), kappa_(kappa) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void VonMisesDistribution::setMu(double mu) {
    validateParameters(mu, getKappa());
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    // Changing only μ doesn't affect logNormaliser_ or circularVariance_
    // (those depend only on κ), but cache_valid_ is reset for thread safety.
    updateCacheUnsafe();
}

void VonMisesDistribution::setKappa(double kappa) {
    validateParameters(getMu(), kappa);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void VonMisesDistribution::setParameters(double mu, double kappa) {
    validateParameters(mu, kappa);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double VonMisesDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mu_;
}

double VonMisesDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    return circularVariance_;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult VonMisesDistribution::trySetMu(double mu) noexcept {
    auto v = validateVonMisesParameters(mu, getKappa());
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult VonMisesDistribution::trySetKappa(double kappa) noexcept {
    auto v = validateVonMisesParameters(getMu(), kappa);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult VonMisesDistribution::trySetParameters(double mu, double kappa) noexcept {
    auto v = validateVonMisesParameters(mu, kappa);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult VonMisesDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateVonMisesParameters(mu_, kappa_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double VonMisesDistribution::getProbability(double x) const {
    if (!std::isfinite(x))
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
    return std::exp(kappa_ * std::cos(x - mu_) - logNormaliser_);
}

double VonMisesDistribution::getLogProbability(double x) const noexcept {
    if (!std::isfinite(x))
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
    return kappa_ * std::cos(x - mu_) - logNormaliser_;
}

double VonMisesDistribution::getCumulativeProbability(double x) const {
    if (!std::isfinite(x))
        return std::isnan(x) ? detail::ZERO_DOUBLE : detail::ONE;

    const double v = wrapAngle(x);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double kappa = kappa_;
    const double mu = mu_;
    const double lnorm = logNormaliser_;
    lock.unlock();

    // Trapezoidal rule: integrate f(t) from −π to v with 512 steps.
    constexpr int N = 512;
    const double a = -detail::PI;
    const double h = (v - a) / static_cast<double>(N);
    if (std::fabs(h) < 1e-15)
        return detail::ZERO_DOUBLE;

    auto pdf = [&](double t) { return std::exp(kappa * std::cos(t - mu) - lnorm); };

    double sum = detail::HALF * (pdf(a) + pdf(v));
    for (int i = 1; i < N; ++i)
        sum += pdf(a + static_cast<double>(i) * h);
    return std::clamp(sum * h, detail::ZERO_DOUBLE, detail::ONE);
}

double VonMisesDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    if (p == detail::ZERO_DOUBLE)
        return -detail::PI;
    if (p == detail::ONE)
        return detail::PI;

    // Bisection on CDF in (−π, π].
    double lo = -detail::PI, hi = detail::PI;
    for (int i = 0; i < 60; ++i) {
        const double mid = (lo + hi) * detail::HALF;
        if (getCumulativeProbability(mid) < p)
            lo = mid;
        else
            hi = mid;
        if ((hi - lo) < 1e-12)
            break;
    }
    return (lo + hi) * detail::HALF;
}

double VonMisesDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double kappa = kappa_;
    const double mu = mu_;
    lock.unlock();

    // Near-uniform case (κ ≈ 0): sample uniformly on the circle.
    if (kappa < 1e-9) {
        std::uniform_real_distribution<double> u(-detail::PI, detail::PI);
        return u(rng);
    }

    // Best (1979) rejection sampler for the Von Mises distribution.
    // Reference: D.J. Best and N.I. Fisher (1979). Efficient simulation of
    //            the von Mises distribution. Applied Statistics 28(2), 152–157.
    const double tau = detail::ONE + std::sqrt(detail::ONE + 4.0 * kappa * kappa);
    const double rho = (tau - std::sqrt(detail::TWO * tau)) / (detail::TWO * kappa);
    const double r = (detail::ONE + rho * rho) / (detail::TWO * rho);

    std::uniform_real_distribution<double> u01(detail::ZERO_DOUBLE, detail::ONE);

    for (;;) {
        const double u1 = u01(rng);
        const double z = std::cos(detail::PI * u1);
        const double f = (detail::ONE + r * z) / (r + z);
        const double c = kappa * (r - f);
        const double u2 = u01(rng);

        bool accept = false;
        if (c * (detail::TWO - c) > u2) {
            accept = true;
        } else if (c > detail::ZERO_DOUBLE) {
            accept = (std::log(c / u2) + detail::ONE - c >= detail::ZERO_DOUBLE);
        }

        if (accept) {
            const double u3 = u01(rng);
            const double angle = (u3 > detail::HALF) ? std::acos(f) : -std::acos(f);
            return wrapAngle(mu + angle);
        }
    }
}

std::vector<double> VonMisesDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; ++i)
        samples.push_back(sample(rng));
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void VonMisesDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    double S = detail::ZERO_DOUBLE, C = detail::ZERO_DOUBLE;
    for (double x : values) {
        S += std::sin(x);
        C += std::cos(x);
    }
    const double n = static_cast<double>(values.size());
    const double mu_hat = wrapAngle(std::atan2(S / n, C / n));
    const double R_bar = std::sqrt(S * S + C * C) / n;
    const double kappa_hat = kappa_from_r_bar(R_bar);

    setParameters(mu_hat, kappa_hat);
}

void VonMisesDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<VonMisesDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void VonMisesDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    kappa_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string VonMisesDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "VonMisesDistribution(mu=" << mu_ << ",kappa=" << kappa_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double VonMisesDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicMu_.load(std::memory_order_acquire);
    return getMu();
}

double VonMisesDistribution::getKappaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicKappa_.load(std::memory_order_acquire);
    return getKappa();
}

double VonMisesDistribution::getCircularVariance() const noexcept {
    return getVariance();
}

double VonMisesDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mu_;
}

double VonMisesDistribution::getEntropy() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = log(2π) − log I₀(κ) + κ·I₁(κ)/I₀(κ)
    // At κ=0: H = log(2π) − log(1) + 0 = log(2π) ✓ (uniform on the circle)
    if (isUniform_)
        return detail::LN_2PI;
    const double log_i0 = detail::log_bessel_i0(kappa_);
    const double i0 = detail::bessel_i0(kappa_);
    const double i1 = detail::bessel_i1(kappa_);
    const double A1 = (i0 > 0.0) ? i1 / i0 : 0.0;
    return detail::LN_2PI - log_i0 + kappa_ * A1;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void VonMisesDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const VonMisesDistribution& d, double x) { return d.getProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<VonMisesDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.kappa_, mu = d.mu_, lnorm = d.logNormaliser_;
            lock.unlock();
            d.getProbabilityBatchUnsafeImpl(vals, res, count, k, mu, lnorm);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<VonMisesDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.kappa_, mu = d.mu_, lnorm = d.logNormaliser_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] = std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm)
                                              : detail::ZERO_DOUBLE;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] = std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm)
                                              : detail::ZERO_DOUBLE;
                }
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<VonMisesDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.kappa_, mu = d.mu_, lnorm = d.logNormaliser_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] =
                    std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm) : detail::ZERO_DOUBLE;
            });
            pool.waitForAll();
        });
}

void VonMisesDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const VonMisesDistribution& d, double x) { return d.getLogProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<VonMisesDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.kappa_, mu = d.mu_, lnorm = d.logNormaliser_;
            lock.unlock();
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, k, mu, lnorm);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
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
                    const_cast<VonMisesDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.kappa_, mu = d.mu_, lnorm = d.logNormaliser_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    res[i] =
                        std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    res[i] =
                        std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
                }
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            std::shared_lock<std::shared_mutex> lock(d.cache_mutex_);
            if (!d.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(d.cache_mutex_);
                if (!d.cache_valid_)
                    const_cast<VonMisesDistribution&>(d).updateCacheUnsafe();
                ulock.unlock();
                lock.lock();
            }
            const double k = d.kappa_, mu = d.mu_, lnorm = d.logNormaliser_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                res[i] =
                    std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
            });
            pool.waitForAll();
        });
}

void VonMisesDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const VonMisesDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
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
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
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

bool VonMisesDistribution::operator==(const VonMisesDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(kappa_ - other.kappa_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool VonMisesDistribution::operator!=(const VonMisesDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const VonMisesDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, VonMisesDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("VonMisesDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t mu_pos = token.find("mu=");
    const size_t comma = token.find(",", mu_pos);
    const size_t kappa_pos = token.find("kappa=");
    const size_t close = token.find(")", kappa_pos);
    if (mu_pos == std::string::npos || comma == std::string::npos ||
        kappa_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double mu = std::stod(token.substr(mu_pos + 3, comma - mu_pos - 3));
        const double kappa = std::stod(token.substr(kappa_pos + 6, close - kappa_pos - 6));
        auto result = d.trySetParameters(mu, kappa);
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
// LogPDF batch:  z[i] = x[i] − μ  |  c[i] = vector_cos(z)  |  r[i] = κ·c[i] − ln Z
// PDF batch:     same as LogPDF then r[i] = vector_exp(r)
//
// Both paths use VectorOps::vector_cos (AVX/AVX2/SSE2/NEON/AVX-512) which
// replaced the earlier scalar std::cos loop. Non-finite inputs receive an
// exact sentinel value via a scalar fixup pass after the SIMD kernel.
//
// The primary performance gain over per-element calls remains avoiding the
// cache-validity check and lock acquisition on every element.
//==============================================================================

void VonMisesDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                      std::size_t count, double cached_kappa,
                                                      double cached_mu,
                                                      double cached_log_normaliser) const noexcept {
    // Step 1: z[i] = values[i] - mu  (scalar_add with -mu)
    arch::simd::VectorOps::scalar_add(values, -cached_mu, results, count);

    // Step 2: results[i] = cos(z[i])  (vectorised)
    arch::simd::VectorOps::vector_cos(results, results, count);

    // Step 3: results[i] = kappa * results[i] - log_normaliser
    arch::simd::VectorOps::scalar_multiply(results, cached_kappa, results, count);
    arch::simd::VectorOps::scalar_add(results, -cached_log_normaliser, results, count);

    // Fixup: non-finite inputs must produce -∞ regardless of the SIMD result
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void VonMisesDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                   std::size_t count, double cached_kappa,
                                                   double cached_mu,
                                                   double cached_log_normaliser) const noexcept {
    // Compute log-PDF then exponentiate
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_kappa, cached_mu,
                               cached_log_normaliser);

    // Step 4: results[i] = exp(results[i])
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: non-finite inputs must produce 0
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i]))
            results[i] = detail::ZERO_DOUBLE;
    }
}

void VonMisesDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                             std::size_t count) const noexcept {
    for (std::size_t i = 0; i < count; ++i)
        results[i] = getCumulativeProbability(values[i]);
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void VonMisesDistribution::updateCacheUnsafe() const noexcept {
    // logNormaliser = log(2π) + log I₀(κ)
    // When κ = 0: I₀(0) = 1, log I₀ = 0, logNormaliser = log(2π). ✓
    logNormaliser_ = detail::LN_2PI + detail::log_bessel_i0(kappa_);

    isUniform_ = (kappa_ < 1e-10);

    // Circular variance = 1 − I₁(κ)/I₀(κ)
    if (isUniform_) {
        circularVariance_ = detail::ONE;
    } else {
        const double i0 = detail::bessel_i0(kappa_);
        const double i1 = detail::bessel_i1(kappa_);
        circularVariance_ = (i0 > 0.0) ? detail::ONE - i1 / i0 : detail::ONE;
    }

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

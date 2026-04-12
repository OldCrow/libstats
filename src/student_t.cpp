#include "libstats/distributions/student_t.h"

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"  // provides detail::digamma, detail::t_cdf, detail::inverse_t_cdf
#include "libstats/core/validation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

static double requireValidNu(double nu) {
    if (nu <= 0.0 || !std::isfinite(nu)) {
        throw std::invalid_argument("Degrees of freedom nu must be a positive finite number");
    }
    return nu;
}

StudentTDistribution::StudentTDistribution(double nu)
    : DistributionBase(), nu_(requireValidNu(nu)) {
    updateCacheUnsafe();
}

StudentTDistribution::StudentTDistribution(const StudentTDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    nu_ = other.nu_;
    halfNu_ = other.halfNu_;
    halfNuPlusOne_ = other.halfNuPlusOne_;
    negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
    invNu_ = other.invNu_;
    logNormConst_ = other.logNormConst_;
    variance_ = other.variance_;
    kurtosis_ = other.kurtosis_;
    isCauchy_ = other.isCauchy_;
    isMeanDefined_ = other.isMeanDefined_;
    isVarianceDefined_ = other.isVarianceDefined_;
    atomicNu_.store(nu_, std::memory_order_release);
}

StudentTDistribution& StudentTDistribution::operator=(const StudentTDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        nu_ = other.nu_;
        halfNu_ = other.halfNu_;
        halfNuPlusOne_ = other.halfNuPlusOne_;
        negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
        invNu_ = other.invNu_;
        logNormConst_ = other.logNormConst_;
        variance_ = other.variance_;
        kurtosis_ = other.kurtosis_;
        isCauchy_ = other.isCauchy_;
        isMeanDefined_ = other.isMeanDefined_;
        isVarianceDefined_ = other.isVarianceDefined_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicNu_.store(nu_, std::memory_order_release);
    }
    return *this;
}

StudentTDistribution::StudentTDistribution(StudentTDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    nu_ = other.nu_;
    halfNu_ = other.halfNu_;
    halfNuPlusOne_ = other.halfNuPlusOne_;
    negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
    invNu_ = other.invNu_;
    logNormConst_ = other.logNormConst_;
    variance_ = other.variance_;
    kurtosis_ = other.kurtosis_;
    isCauchy_ = other.isCauchy_;
    isMeanDefined_ = other.isMeanDefined_;
    isVarianceDefined_ = other.isVarianceDefined_;
    other.nu_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicNu_.store(nu_, std::memory_order_release);
}

StudentTDistribution& StudentTDistribution::operator=(StudentTDistribution&& other) noexcept {
    if (this != &other) {
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);

        bool success = false;
        try {
            std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
            if (std::try_lock(lock1, lock2) == -1) {
                nu_ = other.nu_;
                halfNu_ = other.halfNu_;
                halfNuPlusOne_ = other.halfNuPlusOne_;
                negHalfNuPlusOne_ = other.negHalfNuPlusOne_;
                invNu_ = other.invNu_;
                logNormConst_ = other.logNormConst_;
                variance_ = other.variance_;
                kurtosis_ = other.kurtosis_;
                isCauchy_ = other.isCauchy_;
                isMeanDefined_ = other.isMeanDefined_;
                isVarianceDefined_ = other.isVarianceDefined_;
                other.nu_ = detail::ONE;
                cache_valid_ = false;
                other.cache_valid_ = false;
                atomicNu_.store(nu_, std::memory_order_release);
                success = true;
            }
        } catch (...) {
        }

        if (!success) {
            nu_ = other.nu_;
            other.nu_ = detail::ONE;
            cache_valid_ = false;
            other.cache_valid_ = false;
            atomicNu_.store(nu_, std::memory_order_release);
        }
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

StudentTDistribution StudentTDistribution::createUnchecked(double nu) noexcept {
    return StudentTDistribution(nu, true);
}

StudentTDistribution::StudentTDistribution(double nu, bool /*bypassValidation*/) noexcept
    : DistributionBase(), nu_(nu) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

void StudentTDistribution::setNu(double nu) {
    validateParameters(nu);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    nu_ = nu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

VoidResult StudentTDistribution::trySetNu(double nu) noexcept {
    auto validation = validateStudentTParameters(nu);
    if (validation.isError()) {
        return validation;
    }
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    nu_ = nu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok(true);
}

VoidResult StudentTDistribution::validateCurrentParameters() const noexcept {
    return validateStudentTParameters(getNu());
}

//==============================================================================
// 3. STATISTICAL MOMENTS
//==============================================================================

double StudentTDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!isMeanDefined_) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return detail::ZERO_DOUBLE;
}

double StudentTDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return variance_;
}

double StudentTDistribution::getSkewness() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (nu_ <= 3.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return detail::ZERO_DOUBLE;
}

double StudentTDistribution::getKurtosis() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return kurtosis_;
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double StudentTDistribution::getProbability(double x) const {
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
    return std::exp(logNormConst_ + negHalfNuPlusOne_ * std::log(detail::ONE + x * x * invNu_));
}

double StudentTDistribution::getLogProbability(double x) const noexcept {
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
    return logNormConst_ + negHalfNuPlusOne_ * std::log(detail::ONE + x * x * invNu_);
}

double StudentTDistribution::getCumulativeProbability(double x) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_nu = nu_;
    lock.unlock();
    return detail::t_cdf(x, cached_nu);
}

double StudentTDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_nu = nu_;
    lock.unlock();
    return detail::inverse_t_cdf(p, cached_nu);
}

double StudentTDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    const double cached_half_nu = halfNu_;
    const double cached_nu = nu_;
    lock.unlock();

    // t = Z / sqrt(chi2 / nu)  where chi2 ~ Gamma(nu/2, 2) = chi-squared(nu)
    std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);
    std::gamma_distribution<double> gamma_gen(cached_half_nu, detail::TWO);

    const double z = normal(rng);
    const double chi2 = gamma_gen(rng);
    return z / std::sqrt(chi2 / cached_nu);
}

std::vector<double> StudentTDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(sample(rng));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void StudentTDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    for (double v : values) {
        if (!std::isfinite(v)) {
            throw std::invalid_argument("Data must contain only finite values");
        }
    }

    const double n = static_cast<double>(values.size());

    // Upper bound: beyond NU_MAX the t-distribution is indistinguishable from
    // Gaussian, and the score function flattens (psi((nu+1)/2) - psi(nu/2) ~ 1/(2*nu)),
    // making Newton-Raphson steps unstable.
    constexpr double NU_MAX = 1000.0;

    // Initial estimate: method of moments using sample kurtosis.
    // Excess kurtosis = 6/(nu-4) for nu>4, so nu = 4 + 6/kurtosis.
    // For nu <= 4, or when sample kurtosis is unavailable, start at nu=5.
    // Clamp the initial estimate to keep the optimizer in a region with
    // meaningful gradient — starting above ~100 risks flat-tail divergence.
    double nu_est = 5.0;
    if (values.size() >= 4) {
        double mean = std::accumulate(values.begin(), values.end(), 0.0) / n;
        double m2 = 0.0, m4 = 0.0;
        for (double v : values) {
            double d = v - mean;
            double d2 = d * d;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n;
        m4 /= n;
        if (m2 > detail::ZERO_DOUBLE) {
            double excess_kurt = m4 / (m2 * m2) - 3.0;
            if (excess_kurt > detail::ZERO_DOUBLE) {
                double nu_from_kurt = 4.0 + 6.0 / excess_kurt;
                if (nu_from_kurt > detail::ONE && std::isfinite(nu_from_kurt)) {
                    nu_est = std::min(nu_from_kurt, 100.0);
                }
            }
        }
    }

    // Precompute per-observation xi^2 values
    std::vector<double> x2(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        x2[i] = values[i] * values[i];
    }

    // Newton-Raphson on the score equation:
    //   S(nu) = n*[psi((nu+1)/2) - psi(nu/2) - 1/nu]
    //           - sum(log(1 + xi^2/nu))
    //           + ((nu+1)/nu) * sum(xi^2 / (nu + xi^2))
    //
    // Derivative dS/dnu (for Newton step):
    //   S'(nu) = n/2 * [psi'((nu+1)/2) - psi'(nu/2)] + n/nu^2
    //            + (1/nu^2) * sum(xi^2 * (nu - xi^2) / (nu + xi^2)^2)
    //            ... approximated numerically for simplicity
    //
    // We use a simpler robust approach: numerical derivative via finite difference.

    const int max_iter = 50;
    const double tol = 1e-8;
    double nu = nu_est;

    auto score = [&](double v) -> double {
        double psi_plus = detail::digamma((v + detail::ONE) * detail::HALF);
        double psi_half = detail::digamma(v * detail::HALF);
        double s = n * (psi_plus - psi_half - detail::ONE / v);
        for (double xi2 : x2) {
            s -= std::log(detail::ONE + xi2 / v);
            s += (v + detail::ONE) / v * xi2 / (v + xi2);
        }
        return s;
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        const double s = score(nu);
        if (std::abs(s) < tol * n) {
            break;
        }

        // Numerical derivative
        const double h = nu * 1e-5;
        const double ds = (score(nu + h) - score(nu - h)) / (detail::TWO * h);

        if (std::abs(ds) < 1e-15) {
            break;  // Flat; can't iterate
        }

        double step = s / ds;
        // Clamp step to avoid moving outside the valid domain
        step = std::max(step, -(nu - 0.1));
        nu -= step;
        nu = std::clamp(nu, 0.1, NU_MAX);

        if (std::abs(step) < tol) {
            break;
        }
    }

    setNu(nu);
}

void StudentTDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    nu_ = detail::ONE;
    updateCacheUnsafe();
}

std::string StudentTDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "StudentTDistribution(nu=" << nu_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double StudentTDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // H = (nu+1)/2 * [psi((nu+1)/2) - psi(nu/2)] + log(sqrt(nu)*B(nu/2, 1/2))
    // where B is the beta function.
    // Simplified: H = halfNuPlusOne_*(psi(halfNuPlusOne_) - psi(halfNu_)) + lbeta(halfNu_, 0.5)
    //                 + 0.5*log(nu)
    const double psi_plus = detail::digamma(halfNuPlusOne_);
    const double psi_half = detail::digamma(halfNu_);
    return halfNuPlusOne_ * (psi_plus - psi_half) + detail::lbeta(halfNu_, detail::HALF) +
           detail::HALF * std::log(nu_);
}

//==============================================================================
// 13–14. SMART AUTO-DISPATCH AND EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

void StudentTDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<StudentTDistribution>::distType(),
        detail::OperationType::PDF,
        [](const StudentTDistribution& dist, double value) { return dist.getProbability(value); },
        [](const StudentTDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<StudentTDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, lnc, nhnpo, inv_nu);
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<StudentTDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = std::exp(lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu));
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    res[i] = std::exp(lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu));
                }
            }
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                res[i] = std::exp(lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu));
            });
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                res[i] = std::exp(lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu));
            });
        });
}

void StudentTDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<StudentTDistribution>::distType(),
        detail::OperationType::LOG_PDF,
        [](const StudentTDistribution& dist, double value) {
            return dist.getLogProbability(value);
        },
        [](const StudentTDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<StudentTDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, lnc, nhnpo, inv_nu);
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    res[i] = lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu);
                }
            }
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                res[i] = lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu);
            });
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double lnc = dist.logNormConst_;
            const double nhnpo = dist.negHalfNuPlusOne_;
            const double inv_nu = dist.invNu_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                res[i] = lnc + nhnpo * std::log(1.0 + vals[i] * vals[i] * inv_nu);
            });
        });
}

void StudentTDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::DistributionTraits<StudentTDistribution>::distType(),
        detail::OperationType::CDF,
        [](const StudentTDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const StudentTDistribution& dist, const double* vals, double* res, size_t count) {
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_nu);
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            for (std::size_t i = 0; i < count; ++i) {
                res[i] = detail::t_cdf(vals[i], cached_nu);
            }
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = detail::t_cdf(vals[i], cached_nu); });
        },
        [](const StudentTDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            const double cached_nu = dist.nu_;
            lock.unlock();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = detail::t_cdf(vals[i], cached_nu); });
        });
}

// Helper: map explicit Strategy to the nearest PerformanceHint::PreferredStrategy.
// FORCE_VECTORIZED triggers the BatchUnsafeImpl SIMD path (used by simd_verification).
static detail::PerformanceHint strategyToHint(detail::Strategy strategy) noexcept {
    detail::PerformanceHint hint;
    switch (strategy) {
        case detail::Strategy::SCALAR:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
            break;
        case detail::Strategy::VECTORIZED:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
            break;
        case detail::Strategy::PARALLEL:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
            break;
        case detail::Strategy::WORK_STEALING:
            hint.strategy = detail::PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT;
            break;
    }
    return hint;
}

void StudentTDistribution::getProbabilityWithStrategy(std::span<const double> values,
                                                      std::span<double> results,
                                                      detail::Strategy strategy) const {
    getProbability(values, results, strategyToHint(strategy));
}

void StudentTDistribution::getLogProbabilityWithStrategy(std::span<const double> values,
                                                         std::span<double> results,
                                                         detail::Strategy strategy) const {
    getLogProbability(values, results, strategyToHint(strategy));
}

void StudentTDistribution::getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                                                std::span<double> results,
                                                                detail::Strategy strategy) const {
    getCumulativeProbability(values, results, strategyToHint(strategy));
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool StudentTDistribution::operator==(const StudentTDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(nu_ - other.nu_) <= detail::DEFAULT_TOLERANCE;
}

bool StudentTDistribution::operator!=(const StudentTDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const StudentTDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, StudentTDistribution& dist) {
    std::string token;
    double nu;

    is >> token;
    if (token.find("StudentTDistribution(") != 0) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t nu_pos = token.find("nu=");
    if (nu_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t close = token.find(")", nu_pos);
    if (close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        nu = std::stod(token.substr(nu_pos + 3, close - nu_pos - 3));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }
    auto result = dist.trySetNu(nu);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// Log-space pipeline for PDF and LogPDF.
// No out-of-support fixup: Student's t is defined on all of ℝ.
//
// LogPDF:
//   Step 1: results = x²                     (vector_multiply)
//   Step 2: results = x²/ν                   (scalar_multiply by inv_nu)
//   Step 3: results = 1 + x²/ν              (scalar_add 1)
//   Step 4: results = log(1 + x²/ν)         (vector_log)
//   Step 5: results = −(ν+1)/2 · log(...)   (scalar_multiply by neg_half_nu_plus_one)
//   Step 6: results += log_norm_const        (scalar_add)
//
// PDF: steps 1–6 then vector_exp.
//
// CDF architecture: detail::t_cdf delegates to regularized incomplete beta
//   (beta_i) via the identity t_cdf(t, ν) = 1 - 0.5·I_{x(t)}(ν/2, 1/2).
//   beta_i uses a continued-fraction algorithm whose iteration count varies
//   per input: the same fundamental constraint as Gamma CDF (see gamma.cpp
//   section 18). PDF and LogPDF use a fixed 6-step pipeline and achieve
//   4–8x SIMD speedup; this CDF path is scalar for the same reason.
//==============================================================================

void StudentTDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count, double log_norm_const,
                                                         double neg_half_nu_plus_one,
                                                         double inv_nu) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            results[i] = std::exp(log_norm_const +
                                  neg_half_nu_plus_one *
                                      std::log(detail::ONE + values[i] * values[i] * inv_nu));
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = x²/ν
    arch::simd::VectorOps::scalar_multiply(results, inv_nu, results, count);
    // Step 3: results = 1 + x²/ν
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    // Step 4: results = log(1 + x²/ν)
    arch::simd::VectorOps::vector_log(results, results, count);
    // Step 5: results = −(ν+1)/2 · log(1 + x²/ν)
    arch::simd::VectorOps::scalar_multiply(results, neg_half_nu_plus_one, results, count);
    // Step 6: results += log_norm_const → full LogPDF
    arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);
    // PDF: exponentiate
    arch::simd::VectorOps::vector_exp(results, results, count);
}

void StudentTDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                            std::size_t count,
                                                            double log_norm_const,
                                                            double neg_half_nu_plus_one,
                                                            double inv_nu) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            results[i] =
                log_norm_const +
                neg_half_nu_plus_one * std::log(detail::ONE + values[i] * values[i] * inv_nu);
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = x²/ν
    arch::simd::VectorOps::scalar_multiply(results, inv_nu, results, count);
    // Step 3: results = 1 + x²/ν
    arch::simd::VectorOps::scalar_add(results, detail::ONE, results, count);
    // Step 4: results = log(1 + x²/ν)
    arch::simd::VectorOps::vector_log(results, results, count);
    // Step 5: results = −(ν+1)/2 · log(1 + x²/ν)
    arch::simd::VectorOps::scalar_multiply(results, neg_half_nu_plus_one, results, count);
    // Step 6: results += log_norm_const → full LogPDF
    arch::simd::VectorOps::scalar_add(results, log_norm_const, results, count);
}

void StudentTDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                   double* results,
                                                                   std::size_t count,
                                                                   double nu) const noexcept {
    // Scalar per element. See section 18 header for the explanation.
    for (std::size_t i = 0; i < count; ++i) {
        results[i] = detail::t_cdf(values[i], nu);
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

// computeDigamma removed: promoted to detail::digamma in math_utils.
//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void StudentTDistribution::updateCacheUnsafe() const noexcept {
    halfNu_ = nu_ * detail::HALF;
    halfNuPlusOne_ = (nu_ + detail::ONE) * detail::HALF;
    negHalfNuPlusOne_ = -halfNuPlusOne_;
    invNu_ = detail::ONE / nu_;

    // log normalization constant: lgamma((ν+1)/2) − 0.5·log(ν·π) − lgamma(ν/2)
    logNormConst_ = std::lgamma(halfNuPlusOne_) - detail::HALF * std::log(nu_ * detail::PI) -
                    std::lgamma(halfNu_);

    // Moments (conditional on ν)
    if (nu_ > detail::TWO) {
        variance_ = nu_ / (nu_ - detail::TWO);
    } else if (nu_ > detail::ONE) {
        variance_ = std::numeric_limits<double>::infinity();
    } else {
        variance_ = std::numeric_limits<double>::quiet_NaN();
    }

    if (nu_ > 4.0) {
        kurtosis_ = 6.0 / (nu_ - 4.0);
    } else {
        kurtosis_ = std::numeric_limits<double>::quiet_NaN();
    }

    // Optimization flags
    isCauchy_ = (std::abs(nu_ - detail::ONE) <= detail::DEFAULT_TOLERANCE);
    isMeanDefined_ = (nu_ > detail::ONE);
    isVarianceDefined_ = (nu_ > detail::TWO);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicNu_.store(nu_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

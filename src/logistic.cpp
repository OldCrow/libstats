#include "libstats/distributions/logistic.h"

#include "libstats/common/distribution_impl_common.h"
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

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

namespace {

//------------------------------------------------------------------------------
// Stable scalar kernels shared by the scalar API and the parallel fallbacks.
//
// Every one of these exponentiates only −|z| ∈ (−∞, 0]: e = exp(−|z|) ∈ (0, 1],
// which can neither overflow nor lose the sign of the tail it belongs to.
//------------------------------------------------------------------------------

/// LogPDF for a finite z:  −|z| − 2·log1p(e^(−|z|)).  The −log(s) term is added
/// by the caller (it is cached, and the batch path folds it in separately).
[[nodiscard]] inline double logisticLogKernel(double z) noexcept {
    const double az = std::fabs(z);
    return -az - detail::TWO * std::log1p(std::exp(-az));
}

/// CDF for a finite z, branching on the sign rather than the magnitude:
///   z ≥ 0 → 1/(1 + e^(−z));  z < 0 → e^(z)/(1 + e^(z)).
[[nodiscard]] inline double logisticCdfKernel(double z) noexcept {
    const double e = std::exp(-std::fabs(z));
    return (z >= detail::ZERO_DOUBLE) ? detail::ONE / (detail::ONE + e)
                                      : e / (detail::ONE + e);
}

/// tanh(z/2) = 2·F(z) − 1, evaluated from e = exp(−|z|) so that no overflow can
/// occur. Used by the MLE score equation.
[[nodiscard]] inline double logisticTanhHalf(double z) noexcept {
    const double e = std::exp(-std::fabs(z));
    const double t = (detail::ONE - e) / (detail::ONE + e);
    return (z >= detail::ZERO_DOUBLE) ? t : -t;
}

}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

LogisticDistribution::LogisticDistribution(double mu, double s)
    : DistributionBase(), mu_(mu), s_(s) {
    validateParameters(mu, s);
    updateCacheUnsafe();
}

LogisticDistribution::LogisticDistribution(const LogisticDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    s_ = other.s_;
    inv_s_ = other.inv_s_;
    neg_inv_s_ = other.neg_inv_s_;
    neg_log_s_ = other.neg_log_s_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicS_.store(s_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

LogisticDistribution& LogisticDistribution::operator=(const LogisticDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_ = other.mu_;
        s_ = other.s_;
        inv_s_ = other.inv_s_;
        neg_inv_s_ = other.neg_inv_s_;
        neg_log_s_ = other.neg_log_s_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicS_.store(s_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

LogisticDistribution::LogisticDistribution(LogisticDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mu_ = other.mu_;
    s_ = other.s_;
    inv_s_ = other.inv_s_;
    neg_inv_s_ = other.neg_inv_s_;
    neg_log_s_ = other.neg_log_s_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.s_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicS_.store(s_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
}

LogisticDistribution& LogisticDistribution::operator=(LogisticDistribution&& other) noexcept {
    if (this != &other) {
        mu_ = other.mu_;
        s_ = other.s_;
        inv_s_ = other.inv_s_;
        neg_inv_s_ = other.neg_inv_s_;
        neg_log_s_ = other.neg_log_s_;
        other.mu_ = detail::ZERO_DOUBLE;
        other.s_ = detail::ONE;
        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicS_.store(s_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        other.atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

LogisticDistribution LogisticDistribution::createUnchecked(double mu, double s) noexcept {
    return LogisticDistribution(mu, s, true);
}

LogisticDistribution::LogisticDistribution(double mu, double s, bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), s_(s) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

double LogisticDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicMu_.load(std::memory_order_acquire);
    return getMu();
}

double LogisticDistribution::getSAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicS_.load(std::memory_order_acquire);
    return getS();
}

void LogisticDistribution::setMu(double mu) {
    validateParameters(mu, s_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void LogisticDistribution::setS(double s) {
    validateParameters(mu_, s);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    s_ = s;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void LogisticDistribution::setParameters(double mu, double s) {
    validateParameters(mu, s);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    s_ = s;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult LogisticDistribution::trySetMu(double mu) noexcept {
    auto v = validateLogisticParameters(mu, s_);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult LogisticDistribution::trySetS(double s) noexcept {
    auto v = validateLogisticParameters(mu_, s);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    s_ = s;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult LogisticDistribution::trySetParameters(double mu, double s) noexcept {
    auto v = validateLogisticParameters(mu, s);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    s_ = s;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult LogisticDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateLogisticParameters(mu_, s_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double LogisticDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;  // #103: pdf(±inf) = 0

    double m, is, nls;
    withCacheSnapshot([&] {
        m = mu_;
        is = inv_s_;
        nls = neg_log_s_;
    });
    return std::exp(logisticLogKernel((x - m) * is) + nls);
}

double LogisticDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;  // #103: logpdf(±inf) = −inf

    double m, is, nls;
    withCacheSnapshot([&] {
        m = mu_;
        is = inv_s_;
        nls = neg_log_s_;
    });
    return logisticLogKernel((x - m) * is) + nls;
}

double LogisticDistribution::getCumulativeProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return (x > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;  // #103

    double m, is;
    withCacheSnapshot([&] {
        m = mu_;
        is = inv_s_;
    });
    return logisticCdfKernel((x - m) * is);
}

double LogisticDistribution::getQuantile(double p) const {
    if (std::isnan(p) || p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return -std::numeric_limits<double>::infinity();
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    double m, sv;
    withCacheSnapshot([&] {
        m = mu_;
        sv = s_;
    });
    // #104: log(p) and log1p(−p) are both finite for p ∈ (0,1), so the result is
    // never NaN; it can only reach ±inf through a genuine double overflow of
    // s · logit(p), which needs s ≳ 1e306.
    return m + sv * (std::log(p) - std::log1p(-p));
}

double LogisticDistribution::sample(std::mt19937& rng) const {
    double m, sv;
    withCacheSnapshot([&] {
        m = mu_;
        sv = s_;
    });
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    const double u = uniform(rng);
    return m + sv * (std::log(u) - std::log1p(-u));
}

std::vector<double> LogisticDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    double m, sv;
    withCacheSnapshot([&] {
        m = mu_;
        sv = s_;
    });
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    for (size_t i = 0; i < n; ++i) {
        const double u = uniform(rng);
        samples.push_back(m + sv * (std::log(u) - std::log1p(-u)));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void LogisticDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    for (double v : values) {
        if (!std::isfinite(v))
            throw std::invalid_argument("All values must be finite for Logistic MLE");
    }

    const std::size_t n = values.size();
    if (n < 2)
        throw std::invalid_argument("Logistic MLE requires at least two observations");

    // ---- μ̂ = median (the logistic median equals μ, so this is consistent) ----
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const double mu_hat =
        (n % 2 == 1) ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) * detail::HALF;

    if (sorted.front() == sorted.back())
        throw std::invalid_argument(
            "Logistic MLE is degenerate for constant data (no spread to estimate the scale)");

    // ---- Starting value: method of moments, Var = s²π²/3 → s = sd·√3/π ----
    const double mean =
        std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);
    double ss = detail::ZERO_DOUBLE;
    for (double v : values)
        ss += (v - mean) * (v - mean);
    const double sd = std::sqrt(ss / static_cast<double>(n - 1));
    double s0 = sd * std::sqrt(detail::THREE) / detail::PI;
    if (!std::isfinite(s0) || s0 <= detail::ZERO_DOUBLE)
        s0 = detail::ONE;

    // ---- Score equation for s with μ fixed at μ̂ ----
    //   ∂ℓ/∂s = 0  ⇔  g(s) := Σ zᵢ·tanh(zᵢ/2) − n = 0,   zᵢ = (xᵢ − μ̂)/s
    // g is strictly decreasing in s (each zᵢ·tanh(zᵢ/2) grows with |zᵢ|, and |zᵢ|
    // shrinks as s grows), so the root is unique.  Newton is run on u = log s,
    // where the derivative below is guaranteed negative and the iteration cannot
    // step to a non-positive scale.
    const auto score = [&](double u, double& g, double& gp) {
        const double inv_s = std::exp(-u);
        g = -static_cast<double>(n);
        gp = detail::ZERO_DOUBLE;
        for (double v : values) {
            const double z = (v - mu_hat) * inv_s;
            const double t = logisticTanhHalf(z);
            g += z * t;
            // d/du [z·tanh(z/2)] = −z·[tanh(z/2) + (z/2)·sech²(z/2)]
            gp -= z * (t + detail::HALF * z * (detail::ONE - t * t));
        }
    };

    double u = std::log(s0);
    double g, gp;

    // Bracket the root: g > 0 at u_lo, g < 0 at u_hi.
    double u_lo = u, u_hi = u;
    score(u_lo, g, gp);
    for (int i = 0; i < 200 && g <= detail::ZERO_DOUBLE; ++i) {
        u_lo -= detail::ONE;
        score(u_lo, g, gp);
    }
    score(u_hi, g, gp);
    for (int i = 0; i < 200 && g >= detail::ZERO_DOUBLE; ++i) {
        u_hi += detail::ONE;
        score(u_hi, g, gp);
    }
    if (!(u_lo < u_hi))
        throw std::invalid_argument("Logistic MLE failed to bracket the scale score equation");
    u = detail::HALF * (u_lo + u_hi);

    // Safeguarded Newton: fall back to bisection whenever the Newton step would
    // leave the bracket or is not finite.
    for (int iter = 0; iter < 100; ++iter) {
        score(u, g, gp);
        if (g > detail::ZERO_DOUBLE)
            u_lo = u;
        else
            u_hi = u;

        double u_next = (gp != detail::ZERO_DOUBLE) ? u - g / gp
                                                    : detail::HALF * (u_lo + u_hi);
        if (!std::isfinite(u_next) || u_next <= u_lo || u_next >= u_hi)
            u_next = detail::HALF * (u_lo + u_hi);

        const double step = std::fabs(u_next - u);
        u = u_next;
        if (step < 1.0e-14 * (detail::ONE + std::fabs(u)))
            break;
    }

    const double s_hat = std::exp(u);
    if (!std::isfinite(s_hat) || s_hat <= detail::ZERO_DOUBLE)
        throw std::invalid_argument("Logistic MLE produced a degenerate scale estimate (s ≤ 0)");

    setParameters(mu_hat, s_hat);
}

void LogisticDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<LogisticDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void LogisticDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    s_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string LogisticDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "LogisticDistribution(mu=" << mu_ << ",s=" << s_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double LogisticDistribution::getEntropy() const {
    double nls;
    withCacheSnapshot([&] { nls = neg_log_s_; });
    return detail::TWO - nls;  // log(s) + 2
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void LogisticDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        // Scalar element
        [](const LogisticDistribution& d, double x) { return d.getProbability(x); },
        // SIMD vectorised batch
        [](const LogisticDistribution& d, const double* vals, double* res, std::size_t count) {
            double m, nis, nls;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                nis = d.neg_inv_s_;
                nls = d.neg_log_s_;
            });
            d.getProbabilityBatchUnsafeImpl(vals, res, count, m, nis, nls);
        },
        // Parallel fallback
        [](const LogisticDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double m, is, nls;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
                nls = d.neg_log_s_;
            });
            const auto kernel = [=](double x) {
                if (std::isnan(x))
                    return x;
                if (!std::isfinite(x))
                    return detail::ZERO_DOUBLE;
                return std::exp(logisticLogKernel((x - m) * is) + nls);
            };
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count,
                                           [&](std::size_t i) { res[i] = kernel(vals[i]); });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = kernel(vals[i]);
            }
        },
        // Work-stealing
        [](const LogisticDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double m, is, nls;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
                nls = d.neg_log_s_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x))
                    res[i] = x;
                else if (!std::isfinite(x))
                    res[i] = detail::ZERO_DOUBLE;
                else
                    res[i] = std::exp(logisticLogKernel((x - m) * is) + nls);
            });
            pool.waitForAll();
        });
}

void LogisticDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const LogisticDistribution& d, double x) { return d.getLogProbability(x); },
        [](const LogisticDistribution& d, const double* vals, double* res, std::size_t count) {
            double m, nis, nls;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                nis = d.neg_inv_s_;
                nls = d.neg_log_s_;
            });
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, m, nis, nls);
        },
        [](const LogisticDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double m, is, nls;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
                nls = d.neg_log_s_;
            });
            const auto kernel = [=](double x) {
                if (std::isnan(x))
                    return x;
                if (!std::isfinite(x))
                    return detail::NEGATIVE_INFINITY;
                return logisticLogKernel((x - m) * is) + nls;
            };
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count,
                                           [&](std::size_t i) { res[i] = kernel(vals[i]); });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = kernel(vals[i]);
            }
        },
        [](const LogisticDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double m, is, nls;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
                nls = d.neg_log_s_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x))
                    res[i] = x;
                else if (!std::isfinite(x))
                    res[i] = detail::NEGATIVE_INFINITY;
                else
                    res[i] = logisticLogKernel((x - m) * is) + nls;
            });
            pool.waitForAll();
        });
}

void LogisticDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const LogisticDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const LogisticDistribution& d, const double* vals, double* res, std::size_t count) {
            double m, is;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
            });
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, m, is);
        },
        [](const LogisticDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double m, is;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
            });
            const auto kernel = [=](double x) {
                if (std::isnan(x))
                    return x;
                if (!std::isfinite(x))
                    return (x > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
                return logisticCdfKernel((x - m) * is);
            };
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count,
                                           [&](std::size_t i) { res[i] = kernel(vals[i]); });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = kernel(vals[i]);
            }
        },
        [](const LogisticDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double m, is;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                is = d.inv_s_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x))
                    res[i] = x;
                else if (!std::isfinite(x))
                    res[i] = (x > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
                else
                    res[i] = logisticCdfKernel((x - m) * is);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool LogisticDistribution::operator==(const LogisticDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) <= detail::DEFAULT_TOLERANCE &&
           std::fabs(s_ - other.s_) <= detail::DEFAULT_TOLERANCE;
}

bool LogisticDistribution::operator!=(const LogisticDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const LogisticDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, LogisticDistribution& dist) {
    std::string token;
    double mu, s;

    is >> token;
    if (!token.starts_with("LogisticDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t mu_pos = token.find("mu=");
    const size_t s_pos = token.find(",s=");
    const size_t close = token.find(")", s_pos != std::string::npos ? s_pos : 0);

    if (mu_pos == std::string::npos || s_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        mu = std::stod(token.substr(mu_pos + 3, s_pos - mu_pos - 3));
        s = std::stod(token.substr(s_pos + 3, close - s_pos - 3));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(mu, s);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================
//
// #112 discipline for all three kernels below: `values` is read exactly once,
// in Step 1, *before* anything is written to `results`.  Every later step —
// including the NaN/±inf fixups — reads only the local temp buffer `tmp` (which
// holds the transformed input) or `results` itself.  An aliased call therefore
// still cannot be relied upon (the span contract forbids it), but no step of
// this pipeline depends on re-reading caller memory.

void LogisticDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_neg_inv_s, double cached_neg_log_s) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    std::vector<double, arch::simd::aligned_allocator<double>> tmp(count);

    // Step 1: tmp = x − μ   (the only read of `values`)
    VectorOps::scalar_add(values, -cached_mu, tmp.data(), count);

    // Step 2: tmp = |x − μ| then tmp = −|z| = −|x − μ|/s
    for (std::size_t i = 0; i < count; ++i)
        tmp[i] = std::fabs(tmp[i]);
    VectorOps::scalar_multiply(tmp.data(), cached_neg_inv_s, tmp.data(), count);

    // Step 3: results = e^(−|z|) ∈ (0, 1]
    VectorOps::vector_exp(tmp.data(), results, count);

    // Steps 4–5: results = log(1 + e^(−|z|)).  The argument is in (1, 2], so the
    // plain log is as accurate here as log1p would be (used by the scalar path).
    VectorOps::scalar_add(results, detail::ONE, results, count);
    VectorOps::vector_log(results, results, count);

    // Steps 6–8: results = −|z| − 2·log(1 + e^(−|z|)) − log s
    VectorOps::scalar_multiply(results, detail::NEG_TWO, results, count);
    VectorOps::vector_add(tmp.data(), results, results, count);
    VectorOps::scalar_add(results, cached_neg_log_s, results, count);

    // Fixup from the local temp (never from `values`).  After Step 2 tmp holds
    // −|z|, so it is NaN iff x was NaN and −inf iff x was ±inf (μ finite, s
    // finite and positive).  Both ±inf inputs therefore land on LogPDF = −inf,
    // which is what the pipeline already computes — the loop only guarantees it
    // independently of how a given SIMD tier treats infinities.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(tmp[i]))
            results[i] = std::numeric_limits<double>::quiet_NaN();
        else if (!std::isfinite(tmp[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void LogisticDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count, double cached_mu,
                                                         double cached_neg_inv_s,
                                                         double cached_neg_log_s) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    // LogPDF into results (already fixed up: NaN stays NaN, ±inf → −inf)
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_mu, cached_neg_inv_s,
                                     cached_neg_log_s);

    // exp(−inf) = 0 and exp(NaN) = NaN, so the LogPDF fixups carry through the
    // final exponential unchanged — no second fixup pass is required.
    VectorOps::vector_exp(results, results, count);
}

void LogisticDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_inv_s) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    std::vector<double, arch::simd::aligned_allocator<double>> tmp(count);

    // Step 1: tmp = z = (x − μ)/s   (the only read of `values`; sign retained)
    VectorOps::scalar_add(values, -cached_mu, tmp.data(), count);
    VectorOps::scalar_multiply(tmp.data(), cached_inv_s, tmp.data(), count);

    // Step 2: results = −|z|, then results = e^(−|z|) ∈ (0, 1]
    for (std::size_t i = 0; i < count; ++i)
        results[i] = -std::fabs(tmp[i]);
    VectorOps::vector_exp(results, results, count);

    // Step 3: sign-selected sigmoid — 1/(1+e) above the location, e/(1+e) below.
    // Both branches are monotone to the exact limits and neither forms e^(+|z|).
    for (std::size_t i = 0; i < count; ++i) {
        const double z = tmp[i];
        if (std::isnan(z)) {
            results[i] = std::numeric_limits<double>::quiet_NaN();
        } else if (!std::isfinite(z)) {
            results[i] = (z > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
        } else {
            const double e = results[i];
            results[i] = (z >= detail::ZERO_DOUBLE) ? detail::ONE / (detail::ONE + e)
                                                    : e / (detail::ONE + e);
        }
    }
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void LogisticDistribution::updateCacheUnsafe() const noexcept {
    inv_s_ = detail::ONE / s_;
    neg_inv_s_ = -inv_s_;
    neg_log_s_ = -std::log(s_);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicS_.store(s_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

#include "libstats/distributions/gumbel.h"

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
// All take a *finite* z; the ±inf and NaN cases are decided by the caller,
// because −z − e^(−z) would form inf − inf at z = ±inf.
//------------------------------------------------------------------------------

/// LogPDF for a finite z, without the −log β term (added by the caller).
/// As z → −∞, e^(−z) overflows to +inf and the result collapses to −inf, which
/// is the correct limit; no intermediate NaN can appear because −z stays finite.
[[nodiscard]] inline double gumbelLogKernel(double z) noexcept {
    return -z - std::exp(-z);
}

/// CDF for a finite z: exp(−e^(−z)).  Both extremes land on exact endpoints —
/// e^(−z) = +inf → exp(−inf) = 0, and e^(−z) = 0 → exp(−0) = 1.
[[nodiscard]] inline double gumbelCdfKernel(double z) noexcept {
    return std::exp(-std::exp(-z));
}

/// −log p, evaluated so that the upper tail survives: for p ≥ 1/2 the
/// subtraction p − 1 is exact (Sterbenz) and log1p is accurate for a tiny
/// argument, whereas log(p) alone would round to 0 near p = 1.
[[nodiscard]] inline double gumbelNegLogP(double p) noexcept {
    return (p >= detail::HALF) ? -std::log1p(p - detail::ONE) : -std::log(p);
}

}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

GumbelDistribution::GumbelDistribution(double mu, double beta)
    : DistributionBase(), mu_(mu), beta_(beta) {
    validateParameters(mu, beta);
    updateCacheUnsafe();
}

GumbelDistribution::GumbelDistribution(const GumbelDistribution& other) : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    beta_ = other.beta_;
    inv_beta_ = other.inv_beta_;
    neg_log_beta_ = other.neg_log_beta_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

GumbelDistribution& GumbelDistribution::operator=(const GumbelDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_ = other.mu_;
        beta_ = other.beta_;
        inv_beta_ = other.inv_beta_;
        neg_log_beta_ = other.neg_log_beta_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicBeta_.store(beta_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

GumbelDistribution::GumbelDistribution(GumbelDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mu_ = other.mu_;
    beta_ = other.beta_;
    inv_beta_ = other.inv_beta_;
    neg_log_beta_ = other.neg_log_beta_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.beta_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
}

GumbelDistribution& GumbelDistribution::operator=(GumbelDistribution&& other) noexcept {
    if (this != &other) {
        mu_ = other.mu_;
        beta_ = other.beta_;
        inv_beta_ = other.inv_beta_;
        neg_log_beta_ = other.neg_log_beta_;
        other.mu_ = detail::ZERO_DOUBLE;
        other.beta_ = detail::ONE;
        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicBeta_.store(beta_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        other.atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

GumbelDistribution GumbelDistribution::createUnchecked(double mu, double beta) noexcept {
    return GumbelDistribution(mu, beta, true);
}

GumbelDistribution::GumbelDistribution(double mu, double beta, bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), beta_(beta) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

double GumbelDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicMu_.load(std::memory_order_acquire);
    return getMu();
}

double GumbelDistribution::getBetaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicBeta_.load(std::memory_order_acquire);
    return getBeta();
}

void GumbelDistribution::setMu(double mu) {
    validateParameters(mu, beta_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void GumbelDistribution::setBeta(double beta) {
    validateParameters(mu_, beta);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void GumbelDistribution::setParameters(double mu, double beta) {
    validateParameters(mu, beta);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult GumbelDistribution::trySetMu(double mu) noexcept {
    auto v = validateGumbelParameters(mu, beta_);
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

VoidResult GumbelDistribution::trySetBeta(double beta) noexcept {
    auto v = validateGumbelParameters(mu_, beta);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult GumbelDistribution::trySetParameters(double mu, double beta) noexcept {
    auto v = validateGumbelParameters(mu, beta);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = mu;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult GumbelDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateGumbelParameters(mu_, beta_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double GumbelDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;  // #103: pdf(±inf) = 0

    double m, ib, nlb;
    withCacheSnapshot([&] {
        m = mu_;
        ib = inv_beta_;
        nlb = neg_log_beta_;
    });
    return std::exp(gumbelLogKernel((x - m) * ib) + nlb);
}

double GumbelDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;  // #103: logpdf(±inf) = −inf

    double m, ib, nlb;
    withCacheSnapshot([&] {
        m = mu_;
        ib = inv_beta_;
        nlb = neg_log_beta_;
    });
    return gumbelLogKernel((x - m) * ib) + nlb;
}

double GumbelDistribution::getCumulativeProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return (x > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;  // #103

    double m, ib;
    withCacheSnapshot([&] {
        m = mu_;
        ib = inv_beta_;
    });
    return gumbelCdfKernel((x - m) * ib);
}

double GumbelDistribution::getQuantile(double p) const {
    if (std::isnan(p) || p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return -std::numeric_limits<double>::infinity();
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    double m, b;
    withCacheSnapshot([&] {
        m = mu_;
        b = beta_;
    });
    // #104: for p ∈ (0,1) strictly, −log p is finite and strictly positive, so
    // log(−log p) is finite and the result is never NaN.  The outer logarithm
    // also makes the lower tail extremely well conditioned: q grows only like
    // log(−log p), e.g. q(1e-15) = μ − β·log(34.54) — accurate to a few ulp.
    return m - b * std::log(gumbelNegLogP(p));
}

double GumbelDistribution::sample(std::mt19937& rng) const {
    double m, b;
    withCacheSnapshot([&] {
        m = mu_;
        b = beta_;
    });
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    const double u = uniform(rng);
    return m - b * std::log(gumbelNegLogP(u));
}

std::vector<double> GumbelDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    double m, b;
    withCacheSnapshot([&] {
        m = mu_;
        b = beta_;
    });
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    for (size_t i = 0; i < n; ++i) {
        const double u = uniform(rng);
        samples.push_back(m - b * std::log(gumbelNegLogP(u)));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void GumbelDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    for (double v : values) {
        if (!std::isfinite(v))
            throw std::invalid_argument("All values must be finite for Gumbel fitting");
    }

    const std::size_t n = values.size();
    if (n < 2)
        throw std::invalid_argument("Gumbel moment fitting requires at least two observations");

    // Method of moments (see class documentation — this is NOT the MLE):
    //   Var = π²β²/6  →  β̂ = s·√6/π
    //   E[X] = μ + γβ →  μ̂ = x̄ − γ·β̂
    const double mean =
        std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE) / static_cast<double>(n);
    double ss = detail::ZERO_DOUBLE;
    for (double v : values)
        ss += (v - mean) * (v - mean);
    const double sd = std::sqrt(ss / static_cast<double>(n - 1));

    const double beta_hat = sd * std::sqrt(detail::SIX) / detail::PI;
    if (!std::isfinite(beta_hat) || beta_hat <= detail::ZERO_DOUBLE)
        throw std::invalid_argument(
            "Gumbel moment fit produced a degenerate scale estimate (beta <= 0); data may be "
            "constant");

    const double mu_hat = mean - detail::EULER_MASCHERONI * beta_hat;
    if (!std::isfinite(mu_hat))
        throw std::invalid_argument("Gumbel moment fit produced a non-finite location estimate");

    setParameters(mu_hat, beta_hat);
}

void GumbelDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                          std::vector<GumbelDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void GumbelDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    beta_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string GumbelDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "GumbelDistribution(mu=" << mu_ << ",beta=" << beta_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double GumbelDistribution::getEntropy() const {
    double nlb;
    withCacheSnapshot([&] { nlb = neg_log_beta_; });
    return -nlb + detail::EULER_MASCHERONI + detail::ONE;  // log(β) + γ + 1
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void GumbelDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                        const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        // Scalar element
        [](const GumbelDistribution& d, double x) { return d.getProbability(x); },
        // SIMD vectorised batch
        [](const GumbelDistribution& d, const double* vals, double* res, std::size_t count) {
            double m, ib, nlb;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
                nlb = d.neg_log_beta_;
            });
            d.getProbabilityBatchUnsafeImpl(vals, res, count, m, ib, nlb);
        },
        // Parallel fallback
        [](const GumbelDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double m, ib, nlb;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
                nlb = d.neg_log_beta_;
            });
            const auto kernel = [=](double x) {
                if (std::isnan(x))
                    return x;
                if (!std::isfinite(x))
                    return detail::ZERO_DOUBLE;
                return std::exp(gumbelLogKernel((x - m) * ib) + nlb);
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
        [](const GumbelDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double m, ib, nlb;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
                nlb = d.neg_log_beta_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x))
                    res[i] = x;
                else if (!std::isfinite(x))
                    res[i] = detail::ZERO_DOUBLE;
                else
                    res[i] = std::exp(gumbelLogKernel((x - m) * ib) + nlb);
            });
            pool.waitForAll();
        });
}

void GumbelDistribution::getLogProbability(std::span<const double> values,
                                           std::span<double> results,
                                           const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const GumbelDistribution& d, double x) { return d.getLogProbability(x); },
        [](const GumbelDistribution& d, const double* vals, double* res, std::size_t count) {
            double m, ib, nlb;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
                nlb = d.neg_log_beta_;
            });
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, m, ib, nlb);
        },
        [](const GumbelDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double m, ib, nlb;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
                nlb = d.neg_log_beta_;
            });
            const auto kernel = [=](double x) {
                if (std::isnan(x))
                    return x;
                if (!std::isfinite(x))
                    return detail::NEGATIVE_INFINITY;
                return gumbelLogKernel((x - m) * ib) + nlb;
            };
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count,
                                           [&](std::size_t i) { res[i] = kernel(vals[i]); });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = kernel(vals[i]);
            }
        },
        [](const GumbelDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double m, ib, nlb;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
                nlb = d.neg_log_beta_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x))
                    res[i] = x;
                else if (!std::isfinite(x))
                    res[i] = detail::NEGATIVE_INFINITY;
                else
                    res[i] = gumbelLogKernel((x - m) * ib) + nlb;
            });
            pool.waitForAll();
        });
}

void GumbelDistribution::getCumulativeProbability(std::span<const double> values,
                                                  std::span<double> results,
                                                  const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const GumbelDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const GumbelDistribution& d, const double* vals, double* res, std::size_t count) {
            double m, ib;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
            });
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, m, ib);
        },
        [](const GumbelDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double m, ib;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
            });
            const auto kernel = [=](double x) {
                if (std::isnan(x))
                    return x;
                if (!std::isfinite(x))
                    return (x > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
                return gumbelCdfKernel((x - m) * ib);
            };
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count,
                                           [&](std::size_t i) { res[i] = kernel(vals[i]); });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = kernel(vals[i]);
            }
        },
        [](const GumbelDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double m, ib;
            d.withCacheSnapshot([&] {
                m = d.mu_;
                ib = d.inv_beta_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x))
                    res[i] = x;
                else if (!std::isfinite(x))
                    res[i] = (x > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
                else
                    res[i] = gumbelCdfKernel((x - m) * ib);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool GumbelDistribution::operator==(const GumbelDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) <= detail::DEFAULT_TOLERANCE &&
           std::fabs(beta_ - other.beta_) <= detail::DEFAULT_TOLERANCE;
}

bool GumbelDistribution::operator!=(const GumbelDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const GumbelDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, GumbelDistribution& dist) {
    std::string token;
    double mu, beta;

    is >> token;
    if (!token.starts_with("GumbelDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t mu_pos = token.find("mu=");
    const size_t beta_pos = token.find(",beta=");
    const size_t close = token.find(")", beta_pos != std::string::npos ? beta_pos : 0);

    if (mu_pos == std::string::npos || beta_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        mu = std::stod(token.substr(mu_pos + 3, beta_pos - mu_pos - 3));
        beta = std::stod(token.substr(beta_pos + 6, close - beta_pos - 6));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(mu, beta);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================
//
// #112 discipline for all three kernels: `values` is read exactly once, in
// Step 1, before anything is written to `results`.  Every later step —
// including the NaN/±inf fixups — reads only the local temp buffer `tmp`
// (which holds z, sign preserved) or `results` itself.

void GumbelDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_inv_beta, double cached_neg_log_beta) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    std::vector<double, arch::simd::aligned_allocator<double>> tmp(count);

    // Step 1: tmp = z = (x − μ)/β   (the only read of `values`)
    VectorOps::scalar_add(values, -cached_mu, tmp.data(), count);
    VectorOps::scalar_multiply(tmp.data(), cached_inv_beta, tmp.data(), count);

    // Step 2: results = e^(−z); overflows to +inf for very negative z, which is
    // exactly what drives LogPDF to −inf in Step 3.
    VectorOps::scalar_multiply(tmp.data(), detail::NEG_ONE, results, count);
    VectorOps::vector_exp(results, results, count);

    // Step 3: results = −e^(−z) − z − log β
    VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    VectorOps::vector_subtract(results, tmp.data(), results, count);
    VectorOps::scalar_add(results, cached_neg_log_beta, results, count);

    // Fixup from the local temp (never from `values`).  Required, not defensive:
    // at z = −inf Step 3 forms (−inf) − (−inf) = NaN, and at z = +inf it forms
    // (−0) − (+inf) which is fine but is pinned here for determinism.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(tmp[i]))
            results[i] = std::numeric_limits<double>::quiet_NaN();
        else if (!std::isfinite(tmp[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void GumbelDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                       std::size_t count, double cached_mu,
                                                       double cached_inv_beta,
                                                       double cached_neg_log_beta) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    // LogPDF into results (already fixed up: NaN stays NaN, ±inf → −inf)
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_mu, cached_inv_beta,
                                     cached_neg_log_beta);

    // exp(−inf) = 0 and exp(NaN) = NaN, so both tails underflow to exactly 0 and
    // the LogPDF fixups carry through unchanged — no second fixup pass needed.
    VectorOps::vector_exp(results, results, count);
}

void GumbelDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu,
    double cached_inv_beta) const noexcept {
    using VectorOps = arch::simd::VectorOps;

    std::vector<double, arch::simd::aligned_allocator<double>> tmp(count);

    // Step 1: tmp = z = (x − μ)/β   (the only read of `values`)
    VectorOps::scalar_add(values, -cached_mu, tmp.data(), count);
    VectorOps::scalar_multiply(tmp.data(), cached_inv_beta, tmp.data(), count);

    // Steps 2–4: the double-exponential chain exp(−exp(−z)).
    // z ≪ 0 → e^(−z) = +inf → −inf → exp = 0 exactly;
    // z ≫ 0 → e^(−z) = 0    → −0   → exp = 1 exactly.
    VectorOps::scalar_multiply(tmp.data(), detail::NEG_ONE, results, count);
    VectorOps::vector_exp(results, results, count);
    VectorOps::scalar_multiply(results, detail::NEG_ONE, results, count);
    VectorOps::vector_exp(results, results, count);

    // Fixup from the local temp: pins the ±inf endpoints regardless of how a
    // given SIMD tier's vector_exp treats infinities.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(tmp[i]))
            results[i] = std::numeric_limits<double>::quiet_NaN();
        else if (!std::isfinite(tmp[i]))
            results[i] = (tmp[i] > detail::ZERO_DOUBLE) ? detail::ONE : detail::ZERO_DOUBLE;
    }
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void GumbelDistribution::updateCacheUnsafe() const noexcept {
    inv_beta_ = detail::ONE / beta_;
    neg_log_beta_ = -std::log(beta_);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

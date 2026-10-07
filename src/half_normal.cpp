#include "libstats/distributions/half_normal.h"

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stats {

namespace {
// PARALLEL and WORK_STEALING run the batch kernel over slices of this many elements (#191).
constexpr std::size_t kBatchSlice = 1024;
}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

HalfNormalDistribution::HalfNormalDistribution(double sigma) : DistributionBase(), sigma_(sigma) {
    validateParameters(sigma);
    updateCacheUnsafe();
}

HalfNormalDistribution::HalfNormalDistribution(const HalfNormalDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    sigma_ = other.sigma_;
    logSigma_ = other.logSigma_;
    negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
    normConstant_ = other.normConstant_;
    logNormConst_ = other.logNormConst_;
    invSigmaSqrt2_ = other.invSigmaSqrt2_;
    sigmaSqrt2_ = other.sigmaSqrt2_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    atomicSigma_.store(sigma_, std::memory_order_release);
}

HalfNormalDistribution& HalfNormalDistribution::operator=(const HalfNormalDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        sigma_ = other.sigma_;
        logSigma_ = other.logSigma_;
        negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
        normConstant_ = other.normConstant_;
        logNormConst_ = other.logNormConst_;
        invSigmaSqrt2_ = other.invSigmaSqrt2_;
        sigmaSqrt2_ = other.sigmaSqrt2_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicSigma_.store(sigma_, std::memory_order_release);
    }
    return *this;
}

HalfNormalDistribution::HalfNormalDistribution(HalfNormalDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    sigma_ = other.sigma_;
    logSigma_ = other.logSigma_;
    negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
    normConstant_ = other.normConstant_;
    logNormConst_ = other.logNormConst_;
    invSigmaSqrt2_ = other.invSigmaSqrt2_;
    sigmaSqrt2_ = other.sigmaSqrt2_;
    mean_ = other.mean_;
    variance_ = other.variance_;
    other.sigma_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
}

HalfNormalDistribution& HalfNormalDistribution::operator=(HalfNormalDistribution&& other) noexcept {
    if (this != &other) {
        // Both locks, as copy-assignment takes, in std::lock order; the source is written, so
        // exclusively (#184).
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        sigma_ = other.sigma_;
        logSigma_ = other.logSigma_;
        negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
        normConstant_ = other.normConstant_;
        logNormConst_ = other.logNormConst_;
        invSigmaSqrt2_ = other.invSigmaSqrt2_;
        sigmaSqrt2_ = other.sigmaSqrt2_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        other.sigma_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicSigma_.store(sigma_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

HalfNormalDistribution HalfNormalDistribution::createUnchecked(double sigma) noexcept {
    return HalfNormalDistribution(sigma, true);
}

HalfNormalDistribution::HalfNormalDistribution(double sigma, bool /*bypassValidation*/) noexcept
    : DistributionBase(), sigma_(sigma) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void HalfNormalDistribution::setSigma(double sigma) {
    validateParameters(sigma);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void HalfNormalDistribution::setParameters(double sigma) {
    setSigma(sigma);
}

double HalfNormalDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return mean_;  // snapshot + early return under unique_lock
    }
    return mean_;
}

double HalfNormalDistribution::getVariance() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return variance_;  // snapshot + early return under unique_lock
    }
    return variance_;
}

double HalfNormalDistribution::getSkewness() const {
    // Constant for Half-Normal: √2·(4−π)/(π−2)^(3/2)
    const double pi_minus_two = detail::PI - detail::TWO;
    return detail::SQRT_2 * (detail::FOUR - detail::PI) / (pi_minus_two * std::sqrt(pi_minus_two));
}

double HalfNormalDistribution::getKurtosis() const {
    // Excess kurtosis: 8·(π−3)/(π−2)²
    const double pi_minus_two = detail::PI - detail::TWO;
    return 8.0 * (detail::PI - detail::THREE) / (pi_minus_two * pi_minus_two);
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult HalfNormalDistribution::trySetSigma(double sigma) noexcept {
    auto v = validateHalfNormalParameters(sigma);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = sigma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult HalfNormalDistribution::trySetParameters(double sigma) noexcept {
    return trySetSigma(sigma);
}

VoidResult HalfNormalDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateHalfNormalParameters(sigma_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double HalfNormalDistribution::getProbability(double x) const {
    if (x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    double norm, nhis;
    withCacheSnapshot([&] {
        norm = normConstant_;
        nhis = negHalfInvSigmaSquared_;
    });
    // x = +inf: exp(−inf) = 0 — the correct limit, no special case needed.
    // NaN propagates (NaN < 0 is false, and NaN·NaN → NaN through exp).
    return norm * std::exp(nhis * (x * x));
}

double HalfNormalDistribution::getLogProbability(double x) const {
    if (x < detail::ZERO_DOUBLE)
        return detail::NEGATIVE_INFINITY;

    double lnc, nhis;
    withCacheSnapshot([&] {
        lnc = logNormConst_;
        nhis = negHalfInvSigmaSquared_;
    });
    // x = +inf: lnc + (−inf) = −inf; NaN propagates.
    return lnc + nhis * (x * x);
}

double HalfNormalDistribution::getCumulativeProbability(double x) const {
    if (x < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;

    double inv;
    withCacheSnapshot([&] { inv = invSigmaSqrt2_; });
    // erf argument is non-negative here — no #49-style cancellation branch
    // is needed (that hazard is specific to the 0.5·(1+erf) left-tail form,
    // which this formulation never constructs). erf(+inf) = 1, erf(NaN) = NaN.
    return std::erf(x * inv);
}

double HalfNormalDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1] for Half-Normal distribution");
    }
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return std::numeric_limits<double>::infinity();

    double s_param, ss2;
    withCacheSnapshot([&] {
        s_param = sigma_;
        ss2 = sigmaSqrt2_;
    });

    if (p <= detail::HALF) {
        // Central region: Q(p) = σ√2·erf⁻¹(p), then a Newton polish on the
        // erf residual. The polish removes detail::erf_inv's small-argument
        // relative floor (its Halley loop stops at an ABSOLUTE 1e-12
        // tolerance, ~1.4e-8 relative at p = 1e-10, measured during #57
        // bring-up). g(u) = erf(u) − p is perfectly conditioned here:
        // u ≤ erf⁻¹(½) ≈ 0.477, so exp(u²) ≤ 1.26.
        double u = detail::erf_inv(p);
        for (int i = 0; i < 2; ++i) {
            const double r = std::erf(u) - p;
            u -= r * (detail::SQRT_PI * detail::HALF) * std::exp(u * u);
        }
        return ss2 * u;
    }

    // Upper tail: work in the survival domain. 1−p is EXACT for p ≥ ½
    // (Sterbenz), and F(x) = p ⇔ erfc(x/(σ√2)) = 1−p ⇔ Q(x/σ) = (1−p)/2,
    // so x = σ·Φ⁻¹-complement((1−p)/2) via the erfc-domain solver above.
    // This deliberately bypasses detail::erf_inv's extreme-tail branch (see
    // detail::inv_survival_normal's banner) and is finite and never NaN for every
    // p ∈ (0,1) (#104), accurate to the |ln(1−p)|·2⁻⁵² conditioning law.
    const double s = detail::HALF * (detail::ONE - p);
    return s_param * detail::inv_survival_normal(s);
}

double HalfNormalDistribution::sample(std::mt19937& rng) const {
    double s;
    withCacheSnapshot([&] { s = sigma_; });

    // |Z| with Z ~ Normal(0, σ²) via Box–Muller (exact, no rejection).
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    const double u1 = uniform(rng);
    const double u2 = uniform(rng);
    const double magnitude = std::sqrt(detail::NEG_TWO * std::log(u1));
    const double z = magnitude * std::cos(detail::TWO_PI * u2);
    return s * std::fabs(z);
}

std::vector<double> HalfNormalDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    double s;
    withCacheSnapshot([&] { s = sigma_; });

    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);

    // Pairwise Box–Muller: each (u1, u2) yields two independent normals.
    const size_t pairs = n / 2;
    for (size_t i = 0; i < pairs; ++i) {
        const double u1 = uniform(rng);
        const double u2 = uniform(rng);
        const double magnitude = std::sqrt(detail::NEG_TWO * std::log(u1));
        const double angle = detail::TWO_PI * u2;
        samples.push_back(s * std::fabs(magnitude * std::cos(angle)));
        samples.push_back(s * std::fabs(magnitude * std::sin(angle)));
    }
    if ((n % 2) == 1) {
        const double u1 = uniform(rng);
        const double u2 = uniform(rng);
        const double magnitude = std::sqrt(detail::NEG_TWO * std::log(u1));
        samples.push_back(s * std::fabs(magnitude * std::cos(detail::TWO_PI * u2)));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void HalfNormalDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    double sum_sq = detail::ZERO_DOUBLE;
    for (double v : values) {
        if (v < detail::ZERO_DOUBLE || !std::isfinite(v)) {
            throw std::invalid_argument(
                "Half-Normal distribution requires non-negative finite values");
        }
        sum_sq += v * v;
    }

    // σ̂ = √(Σxᵢ²/n)
    const double sigma_hat = std::sqrt(sum_sq / static_cast<double>(values.size()));
    if (std::isfinite(sigma_hat) && sigma_hat > detail::ZERO_DOUBLE) {
        setSigma(sigma_hat);
    } else {
        throw std::invalid_argument(
            "Half-Normal MLE requires at least one strictly positive value");
    }
}

void HalfNormalDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                              std::vector<HalfNormalDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void HalfNormalDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    sigma_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string HalfNormalDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "HalfNormalDistribution(sigma=" << sigma_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double HalfNormalDistribution::getSigmaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        return atomicSigma_.load(std::memory_order_acquire);
    }
    return getSigma();
}

double HalfNormalDistribution::getMode() const {
    return detail::ZERO_DOUBLE;  // Density is maximal at the origin.
}

double HalfNormalDistribution::getMedian() const {
    double ss2;
    withCacheSnapshot([&] { ss2 = sigmaSqrt2_; });
    // Median = σ√2·erf⁻¹(½) ≈ 0.674490·σ
    return ss2 * detail::erf_inv(detail::HALF);
}

double HalfNormalDistribution::getEntropy() const {
    double ls;
    withCacheSnapshot([&] { ls = logSigma_; });
    // H = ½·log(πσ²/2) + ½ = ½·(log π − log 2) + log σ + ½
    return detail::HALF * (std::log(detail::PI) - detail::LN2) + ls + detail::HALF;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void HalfNormalDistribution::getProbability(std::span<const double> values,
                                            std::span<double> results,
                                            const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const HalfNormalDistribution& d, double x) { return d.getProbability(x); },
        [](const HalfNormalDistribution& d, const double* vals, double* res, size_t count) {
            double nhis, norm;
            d.withCacheSnapshot([&] {
                nhis = d.negHalfInvSigmaSquared_;
                norm = d.normConstant_;
            });
            d.getProbabilityBatchUnsafeImpl(vals, res, count, nhis, norm);
        },
        [](const HalfNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double nhis, norm;
            d.withCacheSnapshot([&] {
                nhis = d.negHalfInvSigmaSquared_;
                norm = d.normConstant_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    d.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                    nhis, norm);
                });
        },
        [](const HalfNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double nhis, norm;
            d.withCacheSnapshot([&] {
                nhis = d.negHalfInvSigmaSquared_;
                norm = d.normConstant_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                d.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len, nhis,
                                                norm);
            });
        });
}

void HalfNormalDistribution::getLogProbability(std::span<const double> values,
                                               std::span<double> results,
                                               const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const HalfNormalDistribution& d, double x) { return d.getLogProbability(x); },
        [](const HalfNormalDistribution& d, const double* vals, double* res, size_t count) {
            double nhis, lnc;
            d.withCacheSnapshot([&] {
                nhis = d.negHalfInvSigmaSquared_;
                lnc = d.logNormConst_;
            });
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, nhis, lnc);
        },
        [](const HalfNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double nhis, lnc;
            d.withCacheSnapshot([&] {
                nhis = d.negHalfInvSigmaSquared_;
                lnc = d.logNormConst_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    d.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                       nhis, lnc);
                });
        },
        [](const HalfNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double nhis, lnc;
            d.withCacheSnapshot([&] {
                nhis = d.negHalfInvSigmaSquared_;
                lnc = d.logNormConst_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                d.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                   nhis, lnc);
            });
        });
}

void HalfNormalDistribution::getCumulativeProbability(std::span<const double> values,
                                                      std::span<double> results,
                                                      const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const HalfNormalDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const HalfNormalDistribution& d, const double* vals, double* res, size_t count) {
            double inv;
            d.withCacheSnapshot([&] { inv = d.invSigmaSqrt2_; });
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, inv);
        },
        [](const HalfNormalDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double inv;
            d.withCacheSnapshot([&] { inv = d.invSigmaSqrt2_; });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    d.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start,
                                                              res.data() + start, len, inv);
                });
        },
        [](const HalfNormalDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double inv;
            d.withCacheSnapshot([&] { inv = d.invSigmaSqrt2_; });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                d.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start,
                                                          len, inv);
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool HalfNormalDistribution::operator==(const HalfNormalDistribution& other) const {
    // d == d would lock the same shared_mutex twice from one thread (undefined).
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(sigma_ - other.sigma_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool HalfNormalDistribution::operator!=(const HalfNormalDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const HalfNormalDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, HalfNormalDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("HalfNormalDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t sg_pos = token.find("sigma=");
    const size_t close = token.find(")", sg_pos);
    if (sg_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double sg = std::stod(token.substr(sg_pos + 6, close - sg_pos - 6));
        auto result = d.trySetSigma(sg);
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
// All three pipelines are the Gaussian batch pipelines with μ = 0 and
// constant offsets folded into the cached values; the CDF pipeline drops the
// 0.5·(1+·) affine step because F(x) = erf(x/(σ√2)) directly. Out-of-support
// fixups follow the bounded-support pattern (B): run the SIMD chain over all
// inputs, then a scalar pass overwrites x < 0 lanes. The fixup pass re-reads
// `values` after `results` is written — legal at the distribution layer
// under the documented no-overlap contract (#112); illegal only inside
// VectorOps kernels.
//
// NaN handling: a NaN input flows through every vector op as NaN, and the
// fixup comparison (NaN < 0) is false, so NaN lanes propagate unchanged —
// matching the scalar path exactly.
//==============================================================================

void HalfNormalDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_neg_half_inv_sigma2,
    double cached_norm_constant) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            results[i] = cached_norm_constant * std::exp(cached_neg_half_inv_sigma2 * (x * x));
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = −x²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_half_inv_sigma2, results, count);
    // Step 3: results = exp(−x²/(2σ²))
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Step 4: results ·= √(2/π)/σ
    arch::simd::VectorOps::scalar_multiply(results, cached_norm_constant, results, count);

    // Fixup: x < 0 is outside support; PDF = 0. (x = 0 is IN support: the mode.)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

void HalfNormalDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_neg_half_inv_sigma2,
    double cached_log_norm_const) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
                continue;
            }
            results[i] = cached_log_norm_const + cached_neg_half_inv_sigma2 * (x * x);
        }
        return;
    }

    // Step 1: results = x²
    arch::simd::VectorOps::vector_multiply(values, values, results, count);
    // Step 2: results = −x²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(results, cached_neg_half_inv_sigma2, results, count);
    // Step 3: results += ½log(2/π) − log σ
    arch::simd::VectorOps::scalar_add(results, cached_log_norm_const, results, count);

    // Fixup: x < 0 is outside support; LogPDF = −∞.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE)
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void HalfNormalDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count,
    double cached_inv_sigma_sqrt2) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            results[i] = std::erf(x * cached_inv_sigma_sqrt2);
        }
        return;
    }

    // Step 1: results = x/(σ√2)
    arch::simd::VectorOps::scalar_multiply(values, cached_inv_sigma_sqrt2, results, count);
    // Step 2: results = erf(x/(σ√2))
    // No per-lane erfc fixup (#49 pattern) is needed here: inside the support
    // the erf argument is non-negative, so the left-tail 1+erf cancellation
    // band that forces the Gaussian batch CDF's fixup never occurs.
    arch::simd::VectorOps::vector_erf(results, results, count);

    // Fixup: x < 0 is outside support; CDF = 0.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE)
            results[i] = detail::ZERO_DOUBLE;
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void HalfNormalDistribution::updateCacheUnsafe() const noexcept {
    logSigma_ = std::log(sigma_);
    const double inv_sigma2 = detail::ONE / (sigma_ * sigma_);
    negHalfInvSigmaSquared_ = -detail::HALF * inv_sigma2;

    // √(2/π)/σ = √2 / (√π·σ)
    normConstant_ = detail::SQRT_2 / (detail::SQRT_PI * sigma_);
    // ½·log(2/π) − log σ
    logNormConst_ = detail::HALF * (detail::LN2 - std::log(detail::PI)) - logSigma_;

    sigmaSqrt2_ = sigma_ * detail::SQRT_2;
    invSigmaSqrt2_ = detail::ONE / sigmaSqrt2_;

    // Mean: σ·√(2/π)
    mean_ = sigma_ * detail::SQRT_2 / detail::SQRT_PI;
    // Variance: σ²·(1 − 2/π)
    variance_ = sigma_ * sigma_ * (detail::ONE - detail::TWO / detail::PI);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicSigma_.store(sigma_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

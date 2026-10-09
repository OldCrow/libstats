#include "libstats/distributions/truncated_normal.h"

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

namespace {

// Standard normal pdf φ(z); 0 at ±∞ (the correct limit — avoids inf·0 NaNs
// in the moment terms below).
inline double phi_std(double z) noexcept {
    if (std::isinf(z))
        return detail::ZERO_DOUBLE;
    return detail::INV_SQRT_2PI * std::exp(-detail::HALF * z * z);
}

// z·φ(z) with the ±∞ / underflow guard: once φ(z) underflows to 0, the
// product is 0 regardless of how large z is (prevents inf·0 and huge·0
// artifacts in the raw-moment recursion for extreme finite bounds).
inline double zphi(double z, double pdf) noexcept {
    if (!std::isfinite(z) || pdf == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    return z * pdf;
}

// Clamp to [0,1] with explicit branches so NaN passes through unchanged
// (std::clamp/min/max would convert NaN to a bound — a #103 clamp escape).
inline double clamp01(double c) noexcept {
    if (c < detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (c > detail::ONE)
        return detail::ONE;
    return c;
}

// Near-lower-bound band: within d = (x−a)/σ of the bound such that
// |α|·d ≤ 1/4, every difference-form regime below subtracts two
// independently rounded normal-CDF pieces whose absolute rounding error
// does not shrink with the vanishing true numerator — relative error
// reaches ~0.45 at a+1ulp (v2.4.0 sweep finding). Inside the band the
// numerator is evaluated directly by truncnorm_cdf_near_lower; at the band
// edge the difference forms are still good to a few ε (their term scale is
// ≥ φ(α)·d/ε there), so the handoff is seamless. Non-finite α or d (a or
// α = −∞) makes the product +∞/NaN and the predicate false, which is the
// intended routing — an infinite lower bound has no near-bound band.
inline bool truncnorm_near_lower_band(double d, double alpha) noexcept {
    return d * std::max(detail::ONE, std::fabs(alpha)) <= 0.25;
}

// Scaled CDF numerator via the probabilists'-Hermite expansion
//   S(d, α) = (Φ(α+d) − Φ(α)) / φ(α) = Σ_{n≥1} Tₙ,  Tₙ = (−1)^{n−1} He_{n−1}(α) dⁿ/n!,
// He_{k+1} = α·He_k − k·He_{k−1}. d is formed from x−a, which is exact for
// x this close to a (Sterbenz), so the sum carries full relative precision:
// S lies in [0.75d, 1.3d] throughout the band, so no term cancels against
// the sum and the rounding error is a few ε.
//
// Stopping rule: two consecutive terms both ≤ τ = 1e-17·|sum|. One term
// alone is not evidence — He_{n−1}(α) = 0 exactly when α is a root of a
// Hermite polynomial (0, ±1, ±√3, ±0.742, ±2.334, …), and the series then
// stopped after the linear term with a relative error of d²/6 (1e-2 at
// α = 0, d = 1/4). Consecutive Hermite polynomials share no root, and the
// recurrence bounds every later term by the two before it:
//   |T_{m+2}| ≤ (|α|d/(m+2))·|T_{m+1}| + (m·d²/((m+1)(m+2)))·|T_m|
//            ≤ 0.094·max(|T_m|, |T_{m+1}|)   for |α|d ≤ 1/4, d ≤ 1/4, m ≥ 1,
// so once two consecutive terms are ≤ τ the whole tail is ≤ 0.21·τ: the
// truncation error is below 2.1e-18 relative, whatever α is. ~20 terms
// reach it at the band edge; the 64-term cap is unreachable slack.
inline double truncnorm_hermite_sum(double d, double alpha) noexcept {
    double hkm1 = detail::ONE;   // He_0(α)
    double hk = alpha;           // He_1(α)
    double dn = d;               // dⁿ/n! at n = 1
    double sum = dn;             // n = 1 term: He_0 · d
    double prev = dn;            // |T_{n−1}|
    double sign = -detail::ONE;  // sign of the n = 2 term
    for (int n = 2; n <= 64; ++n) {
        dn *= d / static_cast<double>(n);
        const double term = sign * hk * dn;
        sum += term;
        const double tau = std::fabs(sum) * 1e-17;
        const double abs_term = std::fabs(term);
        if (abs_term <= tau && prev <= tau)
            break;
        prev = abs_term;
        const double hn = alpha * hk - static_cast<double>(n - 1) * hkm1;  // He_n
        hkm1 = hk;
        hk = hn;
        sign = -sign;
    }
    return sum;
}

// φ(α)/Z: the factor that turns S(d, α) into the CDF. The same expression in
// the CDF and the quantile keeps F(Q(p)) a round trip at the ulp level. φ(α)
// stays representable for every accepted window (the factory rejects windows
// whose Z — of the same exp scale — underflows).
inline double truncnorm_band_scale(double alpha, double inv_z) noexcept {
    return detail::INV_SQRT_2PI * std::exp(-detail::HALF * alpha * alpha) * inv_z;
}

inline double truncnorm_cdf_near_lower(double d, double alpha, double inv_z) noexcept {
    return clamp01(truncnorm_band_scale(alpha, inv_z) * truncnorm_hermite_sum(d, alpha));
}

// d with S(d, α) = t, for a target inside the band: Newton on S, whose
// derivative is S'(d) = e^{−αd − d²/2} in closed form. S(d) = d − αd²/2 + …,
// so d₀ = t·(1 + αt/2) starts within O((αt)²) ≤ 1/16 relative and the
// quadratic iteration reaches an ulp in ≤ 4 steps; the loop stops on a step
// below 2^-53·d (the 8-step cap is slack). Each S evaluation is accurate to a
// few ε and the Newton correction is the exact residual over the exact
// derivative, so the returned d carries the same few-ε relative error as S.
inline double truncnorm_band_invert(double t, double alpha) noexcept {
    double d = t * (detail::ONE + detail::HALF * alpha * t);
    if (!(d > detail::ZERO_DOUBLE))
        d = t;
    for (int i = 0; i < 8; ++i) {
        const double f = truncnorm_hermite_sum(d, alpha) - t;
        const double step = f * std::exp(alpha * d + detail::HALF * d * d);  // f / S'(d)
        d -= step;
        if (!(d > detail::ZERO_DOUBLE)) {
            d = t;  // cannot happen for a target inside the band; keep the iteration finite
            continue;
        }
        if (std::fabs(step) <= 0x1p-53 * d)
            break;
    }
    return d;
}

// Regime-split scalar CDF (single source of truth: the scalar method, the
// batch scalar kernel, the batch per-lane fixups, and the parallel lambdas
// all call this, so every path is expression-identical).
inline double truncnorm_cdf_scalar(double x, double mu, double sigma, double a, double b,
                                   double alpha, double q_alpha, double phi_alpha, double erf_alpha,
                                   double inv_z, double half_inv_z) noexcept {
    if (x <= a)
        return detail::ZERO_DOUBLE;  // exact, also covers x = −∞
    if (x >= b)
        return detail::ONE;  // exact, also covers x = +∞
    {
        const double d = (x - a) / sigma;
        if (truncnorm_near_lower_band(d, alpha))
            return truncnorm_cdf_near_lower(d, alpha, inv_z);
    }
    const double xi = (x - mu) / sigma;
    if (alpha >= detail::ZERO_DOUBLE) {
        // Whole window in the right tail (ξ ≥ α ≥ 0): survival difference
        // Q(α) − Q(ξ) — small same-scale erfc quantities, well-conditioned.
        return clamp01((q_alpha - detail::HALF * std::erfc(xi * detail::INV_SQRT_2)) * inv_z);
    }
    if (xi <= detail::ZERO_DOUBLE) {
        // Left half (α ≤ ξ ≤ 0): reflected erfc difference Φ(ξ) − Φ(α).
        return clamp01((detail::HALF * std::erfc(-xi * detail::INV_SQRT_2) - phi_alpha) * inv_z);
    }
    // Straddling lane (α < 0 < ξ): erf difference — both arguments benign,
    // and the expression matches the batch vector_erf chain term-for-term.
    return clamp01((std::erf(xi * detail::INV_SQRT_2) - erf_alpha) * half_inv_z);
}

// Regime-split quantile core (see the header's Quantile notes). p ∈ (0,1)
// is guaranteed by the callers. Both targets are formed as SUMS of
// non-negative quantities (q_low + s_high = 1 identically), and whichever
// half-target is ≤ ½ is inverted in the erfc/survival domain — no cancelled
// difference is ever reconstructed.
//
// Near either bound the sums below lose what the band expansion keeps: at
// α = 0, Φ(α) + p·Z rounds p·Z = 5e-16 to the half-ulp of ½ and the quantile
// at p = 1e-15 came back 1.6e-3 relative off. So a target inside the
// near-bound band — p ≤ F(a + σ·d_max), decided by the same S(d_max, α) the
// CDF uses — is inverted on the band expansion instead, and a target within
// the mirror-image band at the upper bound (1 − p ≤ the reflected window's
// F, with 1 − p exact for p ≥ ½) likewise, since the window (−b, −a) has
// lower bound −β. S(d) ≤ e^{1/4}·d < 1.3d screens most targets out before
// the series is summed.
inline double truncnorm_quantile_core(double p, double mu, double sigma, double a, double b,
                                      double alpha, double beta, double phi_alpha, double q_beta,
                                      double z, double inv_z) noexcept {
    if (std::isfinite(alpha)) {
        const double scale = truncnorm_band_scale(alpha, inv_z);
        const double d_max = 0.25 / std::max(detail::ONE, std::fabs(alpha));
        if (p <= scale * 1.3 * d_max && p <= scale * truncnorm_hermite_sum(d_max, alpha)) {
            double x = a + sigma * truncnorm_band_invert(p / scale, alpha);
            if (x > b)
                x = b;
            return x;
        }
    }
    if (std::isfinite(beta)) {
        const double q = detail::ONE - p;
        const double scale = truncnorm_band_scale(beta, inv_z);
        const double d_max = 0.25 / std::max(detail::ONE, std::fabs(beta));
        if (q <= scale * 1.3 * d_max && q <= scale * truncnorm_hermite_sum(d_max, -beta)) {
            double x = b - sigma * truncnorm_band_invert(q / scale, -beta);
            if (x < a)
                x = a;
            return x;
        }
    }
    const double q_low = phi_alpha + p * z;                // Φ target from below
    const double s_high = q_beta + (detail::ONE - p) * z;  // survival target from above
    double xi;
    if (q_low <= s_high) {
        xi = -detail::inv_survival_normal(q_low);
    } else {
        xi = detail::inv_survival_normal(s_high);
    }
    double x = mu + sigma * xi;
    if (x < a)
        x = a;
    if (x > b)
        x = b;
    return x;
}

}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

TruncatedNormalDistribution::TruncatedNormalDistribution(double mean, double standardDeviation,
                                                         double lowerBound, double upperBound)
    : DistributionBase(),
      mean_(mean),
      standardDeviation_(standardDeviation),
      lowerBound_(lowerBound),
      upperBound_(upperBound) {
    validateParameters(mean, standardDeviation, lowerBound, upperBound);
    updateCacheUnsafe();
}

TruncatedNormalDistribution::TruncatedNormalDistribution(const TruncatedNormalDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
    lowerBound_ = other.lowerBound_;
    upperBound_ = other.upperBound_;
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    z_ = other.z_;
    logZ_ = other.logZ_;
    phiAlpha_ = other.phiAlpha_;
    qAlpha_ = other.qAlpha_;
    qBeta_ = other.qBeta_;
    erfAlpha_ = other.erfAlpha_;
    invZ_ = other.invZ_;
    halfInvZ_ = other.halfInvZ_;
    logSigma_ = other.logSigma_;
    negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
    invSigmaSqrt2_ = other.invSigmaSqrt2_;
    logPdfNormConst_ = other.logPdfNormConst_;
    distMean_ = other.distMean_;
    distVariance_ = other.distVariance_;
    delta_ = other.delta_;
    eta_ = other.eta_;
    atomicMean_.store(mean_, std::memory_order_release);
    atomicStandardDeviation_.store(standardDeviation_, std::memory_order_release);
    atomicLowerBound_.store(lowerBound_, std::memory_order_release);
    atomicUpperBound_.store(upperBound_, std::memory_order_release);
}

TruncatedNormalDistribution& TruncatedNormalDistribution::operator=(
    const TruncatedNormalDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mean_ = other.mean_;
        standardDeviation_ = other.standardDeviation_;
        lowerBound_ = other.lowerBound_;
        upperBound_ = other.upperBound_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        updateCacheUnsafe();
    }
    return *this;
}

TruncatedNormalDistribution::TruncatedNormalDistribution(
    TruncatedNormalDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mean_ = other.mean_;
    standardDeviation_ = other.standardDeviation_;
    lowerBound_ = other.lowerBound_;
    upperBound_ = other.upperBound_;
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    z_ = other.z_;
    logZ_ = other.logZ_;
    phiAlpha_ = other.phiAlpha_;
    qAlpha_ = other.qAlpha_;
    qBeta_ = other.qBeta_;
    erfAlpha_ = other.erfAlpha_;
    invZ_ = other.invZ_;
    halfInvZ_ = other.halfInvZ_;
    logSigma_ = other.logSigma_;
    negHalfInvSigmaSquared_ = other.negHalfInvSigmaSquared_;
    invSigmaSqrt2_ = other.invSigmaSqrt2_;
    logPdfNormConst_ = other.logPdfNormConst_;
    distMean_ = other.distMean_;
    distVariance_ = other.distVariance_;
    delta_ = other.delta_;
    eta_ = other.eta_;
    other.mean_ = detail::ZERO_DOUBLE;
    other.standardDeviation_ = detail::ONE;
    other.lowerBound_ = -std::numeric_limits<double>::infinity();
    other.upperBound_ = std::numeric_limits<double>::infinity();
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMean_.store(mean_, std::memory_order_release);
    atomicStandardDeviation_.store(standardDeviation_, std::memory_order_release);
    atomicLowerBound_.store(lowerBound_, std::memory_order_release);
    atomicUpperBound_.store(upperBound_, std::memory_order_release);
}

TruncatedNormalDistribution& TruncatedNormalDistribution::operator=(
    TruncatedNormalDistribution&& other) noexcept {
    if (this != &other) {
        // Both locks, as copy-assignment takes, in std::lock order; the source is written, so
        // exclusively (#184).
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mean_ = other.mean_;
        standardDeviation_ = other.standardDeviation_;
        lowerBound_ = other.lowerBound_;
        upperBound_ = other.upperBound_;
        other.mean_ = detail::ZERO_DOUBLE;
        other.standardDeviation_ = detail::ONE;
        other.lowerBound_ = -std::numeric_limits<double>::infinity();
        other.upperBound_ = std::numeric_limits<double>::infinity();

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        updateCacheUnsafe();
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

TruncatedNormalDistribution TruncatedNormalDistribution::createUnchecked(
    double mean, double standardDeviation, double lowerBound, double upperBound) noexcept {
    return TruncatedNormalDistribution(mean, standardDeviation, lowerBound, upperBound, true);
}

TruncatedNormalDistribution::TruncatedNormalDistribution(double mean, double standardDeviation,
                                                         double lowerBound, double upperBound,
                                                         bool /*bypassValidation*/) noexcept
    : DistributionBase(),
      mean_(mean),
      standardDeviation_(standardDeviation),
      lowerBound_(lowerBound),
      upperBound_(upperBound) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

namespace {
// Shared body for the four single-parameter setters: validation must see the
// NEW candidate value alongside the OTHER current values inside the same
// critical section (NEW-TS-3 pattern from gaussian.cpp).
}  // namespace

void TruncatedNormalDistribution::setMu(double mean) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(mean, standardDeviation_, lowerBound_, upperBound_);
    mean_ = mean;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void TruncatedNormalDistribution::setSigma(double standardDeviation) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(mean_, standardDeviation, lowerBound_, upperBound_);
    standardDeviation_ = standardDeviation;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void TruncatedNormalDistribution::setLowerBound(double lowerBound) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(mean_, standardDeviation_, lowerBound, upperBound_);
    lowerBound_ = lowerBound;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void TruncatedNormalDistribution::setUpperBound(double upperBound) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(mean_, standardDeviation_, lowerBound_, upperBound);
    upperBound_ = upperBound;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void TruncatedNormalDistribution::setParameters(double mean, double standardDeviation,
                                                double lowerBound, double upperBound) {
    validateParameters(mean, standardDeviation, lowerBound, upperBound);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = mean;
    standardDeviation_ = standardDeviation;
    lowerBound_ = lowerBound;
    upperBound_ = upperBound;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double TruncatedNormalDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return distMean_;  // snapshot + early return under unique_lock
    }
    return distMean_;
}

double TruncatedNormalDistribution::getVariance() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return distVariance_;  // snapshot + early return under unique_lock
    }
    return distVariance_;
}

double TruncatedNormalDistribution::getSkewness() const {
    double al, be, z;
    withCacheSnapshot([&] {
        al = alpha_;
        be = beta_;
        z = z_;
    });
    // Raw-moment recursion m_k = (k−1)m_{k−2} + (α^{k−1}φα − β^{k−1}φβ)/Z
    // (terms with infinite bound or underflowed φ are exactly 0 — zphi).
    const double pa = phi_std(al), pb = phi_std(be);
    const double m1 = (pa - pb) / z;
    const double m2 = detail::ONE + (zphi(al, pa) - zphi(be, pb)) / z;
    const double m3 = detail::TWO * m1 + (zphi(al, zphi(al, pa)) - zphi(be, zphi(be, pb))) / z;
    const double c2 = m2 - m1 * m1;
    const double c3 = m3 - detail::THREE * m1 * m2 + detail::TWO * m1 * m1 * m1;
    return c3 / (c2 * std::sqrt(c2));
}

double TruncatedNormalDistribution::getKurtosis() const {
    double al, be, z;
    withCacheSnapshot([&] {
        al = alpha_;
        be = beta_;
        z = z_;
    });
    const double pa = phi_std(al), pb = phi_std(be);
    const double m1 = (pa - pb) / z;
    const double m2 = detail::ONE + (zphi(al, pa) - zphi(be, pb)) / z;
    const double m3 = detail::TWO * m1 + (zphi(al, zphi(al, pa)) - zphi(be, zphi(be, pb))) / z;
    const double m4 = detail::THREE * m2 +
                      (zphi(al, zphi(al, zphi(al, pa))) - zphi(be, zphi(be, zphi(be, pb)))) / z;
    const double c2 = m2 - m1 * m1;
    const double c4 = m4 - detail::FOUR * m1 * m3 + detail::SIX * m1 * m1 * m2 -
                      detail::THREE * m1 * m1 * m1 * m1;
    return c4 / (c2 * c2) - detail::THREE;  // excess kurtosis
}

double TruncatedNormalDistribution::getNormalizationConstant() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return z_;
    }
    return z_;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult TruncatedNormalDistribution::trySetMu(double mean) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateTruncatedNormalParameters(mean, standardDeviation_, lowerBound_, upperBound_);
    if (v.isError())
        return v;
    if (!isWindowRepresentable(mean, standardDeviation_, lowerBound_, upperBound_)) {
        return VoidResult::makeError(ValidationError::InvalidRange,
                                     "Truncation window too deep in the tail: Z underflows");
    }
    mean_ = mean;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult TruncatedNormalDistribution::trySetSigma(double standardDeviation) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateTruncatedNormalParameters(mean_, standardDeviation, lowerBound_, upperBound_);
    if (v.isError())
        return v;
    if (!isWindowRepresentable(mean_, standardDeviation, lowerBound_, upperBound_)) {
        return VoidResult::makeError(ValidationError::InvalidRange,
                                     "Truncation window too deep in the tail: Z underflows");
    }
    standardDeviation_ = standardDeviation;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult TruncatedNormalDistribution::trySetLowerBound(double lowerBound) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateTruncatedNormalParameters(mean_, standardDeviation_, lowerBound, upperBound_);
    if (v.isError())
        return v;
    if (!isWindowRepresentable(mean_, standardDeviation_, lowerBound, upperBound_)) {
        return VoidResult::makeError(ValidationError::InvalidRange,
                                     "Truncation window too deep in the tail: Z underflows");
    }
    lowerBound_ = lowerBound;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult TruncatedNormalDistribution::trySetUpperBound(double upperBound) noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto v = validateTruncatedNormalParameters(mean_, standardDeviation_, lowerBound_, upperBound);
    if (v.isError())
        return v;
    if (!isWindowRepresentable(mean_, standardDeviation_, lowerBound_, upperBound)) {
        return VoidResult::makeError(ValidationError::InvalidRange,
                                     "Truncation window too deep in the tail: Z underflows");
    }
    upperBound_ = upperBound;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult TruncatedNormalDistribution::trySetParameters(double mean, double standardDeviation,
                                                         double lowerBound,
                                                         double upperBound) noexcept {
    auto v = validateTruncatedNormalParameters(mean, standardDeviation, lowerBound, upperBound);
    if (v.isError())
        return v;
    if (!isWindowRepresentable(mean, standardDeviation, lowerBound, upperBound)) {
        return VoidResult::makeError(ValidationError::InvalidRange,
                                     "Truncation window too deep in the tail: Z underflows");
    }
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = mean;
    standardDeviation_ = standardDeviation;
    lowerBound_ = lowerBound;
    upperBound_ = upperBound;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult TruncatedNormalDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateTruncatedNormalParameters(mean_, standardDeviation_, lowerBound_, upperBound_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double TruncatedNormalDistribution::getProbability(double x) const {
    double mu, a, b, nhis, logc;
    withCacheSnapshot([&] {
        mu = mean_;
        a = lowerBound_;
        b = upperBound_;
        nhis = negHalfInvSigmaSquared_;
        logc = logPdfNormConst_;
    });
    if (x < a || x > b)
        return detail::ZERO_DOUBLE;
    // x = ±∞ with an infinite bound falls through the comparisons above
    // (−∞ < −∞ is false) but the formula yields the correct limit:
    // diff² = ∞ → exp(−∞) = 0. NaN propagates.
    const double diff = x - mu;
    return std::exp(logc + nhis * (diff * diff));
}

double TruncatedNormalDistribution::getLogProbability(double x) const {
    double mu, a, b, nhis, logc;
    withCacheSnapshot([&] {
        mu = mean_;
        a = lowerBound_;
        b = upperBound_;
        nhis = negHalfInvSigmaSquared_;
        logc = logPdfNormConst_;
    });
    if (x < a || x > b)
        return detail::NEGATIVE_INFINITY;
    // Exact log space; log Z is finite by the supported-window policy, so
    // the result is never NaN and never a clamp constant inside the support.
    const double diff = x - mu;
    return logc + nhis * (diff * diff);
}

double TruncatedNormalDistribution::getCumulativeProbability(double x) const {
    double mu, sigma, a, b, al, qa, pa, ea, iz, hiz;
    withCacheSnapshot([&] {
        mu = mean_;
        sigma = standardDeviation_;
        a = lowerBound_;
        b = upperBound_;
        al = alpha_;
        qa = qAlpha_;
        pa = phiAlpha_;
        ea = erfAlpha_;
        iz = invZ_;
        hiz = halfInvZ_;
    });
    return truncnorm_cdf_scalar(x, mu, sigma, a, b, al, qa, pa, ea, iz, hiz);
}

double TruncatedNormalDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument(
            "Probability must be in [0, 1] for Truncated Normal distribution");
    }

    double mu, sigma, a, b, al, be, pa, qb, z, iz;
    withCacheSnapshot([&] {
        mu = mean_;
        sigma = standardDeviation_;
        a = lowerBound_;
        b = upperBound_;
        al = alpha_;
        be = beta_;
        pa = phiAlpha_;
        qb = qBeta_;
        z = z_;
        iz = invZ_;
    });

    if (p == detail::ZERO_DOUBLE)
        return a;
    if (p == detail::ONE)
        return b;
    return truncnorm_quantile_core(p, mu, sigma, a, b, al, be, pa, qb, z, iz);
}

double TruncatedNormalDistribution::sample(std::mt19937& rng) const {
    double mu, sigma, a, b, al, be, pa, qb, z, iz;
    withCacheSnapshot([&] {
        mu = mean_;
        sigma = standardDeviation_;
        a = lowerBound_;
        b = upperBound_;
        al = alpha_;
        be = beta_;
        pa = phiAlpha_;
        qb = qBeta_;
        z = z_;
        iz = invZ_;
    });
    // Inverse-CDF transform — exact in every regime because the quantile is
    // computed tail-stably in the survival domain (see header). No rejection
    // step, so far-tail windows cost the same as central ones.
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    return truncnorm_quantile_core(uniform(rng), mu, sigma, a, b, al, be, pa, qb, z, iz);
}

std::vector<double> TruncatedNormalDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    double mu, sigma, a, b, al, be, pa, qb, z, iz;
    withCacheSnapshot([&] {
        mu = mean_;
        sigma = standardDeviation_;
        a = lowerBound_;
        b = upperBound_;
        al = alpha_;
        be = beta_;
        pa = phiAlpha_;
        qb = qBeta_;
        z = z_;
        iz = invZ_;
    });
    std::uniform_real_distribution<double> uniform(std::numeric_limits<double>::min(), detail::ONE);
    for (size_t i = 0; i < n; ++i) {
        samples.push_back(
            truncnorm_quantile_core(uniform(rng), mu, sigma, a, b, al, be, pa, qb, z, iz));
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void TruncatedNormalDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }
    if (values.size() < detail::MIN_DATA_POINTS_FOR_FITTING) {
        throw std::invalid_argument(
            "Insufficient data points for reliable Truncated Normal fitting");
    }

    // Bounds are KNOWN (held fixed at their current values) — the standard
    // formulation; see the header's MLE scope note.
    // The estimate is computed from a snapshot of the bounds and committed under the write lock
    // only if the bounds are unchanged; otherwise it is recomputed for the new window. Writing it
    // through setParameters(mu, sig, a, b) undid a concurrent bound setter (#185).
    for (;;) {
        double a, b;
        withCacheSnapshot([&] {
            a = lowerBound_;
            b = upperBound_;
        });

        double sum = detail::ZERO_DOUBLE;
        for (double v : values) {
            if (!std::isfinite(v) || v < a || v > b) {
                throw std::invalid_argument(
                    "Truncated Normal fit requires finite values within the truncation window");
            }
            sum += v;
        }
        const double n = static_cast<double>(values.size());
        const double xbar = sum / n;
        double ss = detail::ZERO_DOUBLE;
        for (double v : values) {
            const double d = v - xbar;
            ss += d * d;
        }
        const double s2 = ss / n;  // MLE (biased) second central moment
        if (!(s2 > detail::ZERO_DOUBLE)) {
            throw std::invalid_argument("Data has zero variance - cannot fit Truncated Normal");
        }

        // Fixed-point iteration on the exponential-family moment equations
        // (Cohen 1959 style):  σ² ← s²/(1 + η − δ²),  μ ← x̄ − σδ.
        double mu = xbar;
        double sig = std::sqrt(s2);
        constexpr int kMaxIter = 500;
        constexpr double kRelTol = 1e-10;
        bool converged = false;
        for (int iter = 0; iter < kMaxIter; ++iter) {
            const auto nc = computeNormalization(mu, sig, a, b);
            if (!nc.valid) {
                throw std::runtime_error(
                    "Truncated Normal MLE failed: normalization constant underflowed during "
                    "iteration (window too deep in the tail for the current iterate)");
            }
            const double pa = phi_std(nc.alpha), pb = phi_std(nc.beta);
            const double delta = (pa - pb) / nc.z;
            const double eta = (zphi(nc.alpha, pa) - zphi(nc.beta, pb)) / nc.z;
            const double denom = detail::ONE + eta - delta * delta;
            if (!(denom > detail::ZERO_DOUBLE)) {
                throw std::runtime_error(
                    "Truncated Normal MLE failed: degenerate variance ratio (1 + η − δ² ≤ 0)");
            }
            const double sig_new = std::sqrt(s2 / denom);
            const double mu_new = xbar - sig_new * delta;
            if (!std::isfinite(sig_new) || !std::isfinite(mu_new) ||
                sig_new <= detail::ZERO_DOUBLE) {
                throw std::runtime_error("Truncated Normal MLE failed: non-finite iterate");
            }
            const bool done = std::fabs(mu_new - mu) <= kRelTol * (detail::ONE + std::fabs(mu)) &&
                              std::fabs(sig_new - sig) <= kRelTol * sig;
            mu = mu_new;
            sig = sig_new;
            if (done) {
                converged = true;
                break;
            }
        }
        if (!converged) {
            throw std::runtime_error("Truncated Normal MLE did not converge within 500 iterations");
        }
        validateParameters(mu, sig, a, b);
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        if (lowerBound_ != a || upperBound_ != b)
            continue;  // a bound setter ran during the iteration
        mean_ = mu;
        standardDeviation_ = sig;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        updateCacheUnsafe();
        return;
    }
}

void TruncatedNormalDistribution::parallelBatchFit(
    const std::vector<std::vector<double>>& datasets,
    std::vector<TruncatedNormalDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void TruncatedNormalDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mean_ = detail::ZERO_DOUBLE;
    standardDeviation_ = detail::ONE;
    lowerBound_ = -std::numeric_limits<double>::infinity();
    upperBound_ = std::numeric_limits<double>::infinity();
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string TruncatedNormalDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "TruncatedNormalDistribution(mu=" << mean_ << ", sigma=" << standardDeviation_
        << ", a=" << lowerBound_ << ", b=" << upperBound_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double TruncatedNormalDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // Density is unimodal: maximal at μ if μ ∈ [a,b], else at the nearer bound.
    if (mean_ < lowerBound_)
        return lowerBound_;
    if (mean_ > upperBound_)
        return upperBound_;
    return mean_;
}

double TruncatedNormalDistribution::getMedian() const {
    return getQuantile(detail::HALF);
}

double TruncatedNormalDistribution::getEntropy() const {
    double ls, lz, eta;
    withCacheSnapshot([&] {
        ls = logSigma_;
        lz = logZ_;
        eta = eta_;
    });
    // H = ½log(2πe) + log σ + log Z + η/2,  η = (αφ(α)−βφ(β))/Z
    return detail::HALF_LN_2PI + detail::HALF + ls + lz + detail::HALF * eta;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void TruncatedNormalDistribution::getProbability(std::span<const double> values,
                                                 std::span<double> results,
                                                 const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const TruncatedNormalDistribution& d, double x) { return d.getProbability(x); },
        [](const TruncatedNormalDistribution& d, const double* vals, double* res, size_t count) {
            double mu, a, b, nhis, logc;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                a = d.lowerBound_;
                b = d.upperBound_;
                nhis = d.negHalfInvSigmaSquared_;
                logc = d.logPdfNormConst_;
            });
            d.getProbabilityBatchUnsafeImpl(vals, res, count, mu, a, b, nhis, logc);
        },
        [](const TruncatedNormalDistribution& d, std::span<const double> vals,
           std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double mu, a, b, nhis, logc;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                a = d.lowerBound_;
                b = d.upperBound_;
                nhis = d.negHalfInvSigmaSquared_;
                logc = d.logPdfNormConst_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    d.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                    mu, a, b, nhis, logc);
                });
        },
        [](const TruncatedNormalDistribution& d, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double mu, a, b, nhis, logc;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                a = d.lowerBound_;
                b = d.upperBound_;
                nhis = d.negHalfInvSigmaSquared_;
                logc = d.logPdfNormConst_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                d.getProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len, mu, a,
                                                b, nhis, logc);
            });
        });
}

void TruncatedNormalDistribution::getLogProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const TruncatedNormalDistribution& d, double x) { return d.getLogProbability(x); },
        [](const TruncatedNormalDistribution& d, const double* vals, double* res, size_t count) {
            double mu, a, b, nhis, logc;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                a = d.lowerBound_;
                b = d.upperBound_;
                nhis = d.negHalfInvSigmaSquared_;
                logc = d.logPdfNormConst_;
            });
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, mu, a, b, nhis, logc);
        },
        [](const TruncatedNormalDistribution& d, std::span<const double> vals,
           std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double mu, a, b, nhis, logc;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                a = d.lowerBound_;
                b = d.upperBound_;
                nhis = d.negHalfInvSigmaSquared_;
                logc = d.logPdfNormConst_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    d.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len,
                                                       mu, a, b, nhis, logc);
                });
        },
        [](const TruncatedNormalDistribution& d, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double mu, a, b, nhis, logc;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                a = d.lowerBound_;
                b = d.upperBound_;
                nhis = d.negHalfInvSigmaSquared_;
                logc = d.logPdfNormConst_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                d.getLogProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start, len, mu,
                                                   a, b, nhis, logc);
            });
        });
}

void TruncatedNormalDistribution::getCumulativeProbability(
    std::span<const double> values, std::span<double> results,
    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const TruncatedNormalDistribution& d, double x) {
            return d.getCumulativeProbability(x);
        },
        [](const TruncatedNormalDistribution& d, const double* vals, double* res, size_t count) {
            double mu, sigma, a, b, al, be, qa, pa, ea, iz, hiz, iss2;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                sigma = d.standardDeviation_;
                a = d.lowerBound_;
                b = d.upperBound_;
                al = d.alpha_;
                be = d.beta_;
                qa = d.qAlpha_;
                pa = d.phiAlpha_;
                ea = d.erfAlpha_;
                iz = d.invZ_;
                hiz = d.halfInvZ_;
                iss2 = d.invSigmaSqrt2_;
            });
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, mu, sigma, a, b, al, be, qa,
                                                      pa, ea, iz, hiz, iss2);
        },
        [](const TruncatedNormalDistribution& d, std::span<const double> vals,
           std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double mu, sigma, a, b, al, be, qa, pa, ea, iz, hiz, iss2;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                sigma = d.standardDeviation_;
                a = d.lowerBound_;
                b = d.upperBound_;
                al = d.alpha_;
                be = d.beta_;
                qa = d.qAlpha_;
                pa = d.phiAlpha_;
                ea = d.erfAlpha_;
                iz = d.invZ_;
                hiz = d.halfInvZ_;
                iss2 = d.invSigmaSqrt2_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    d.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start,
                                                              res.data() + start, len, mu, sigma, a,
                                                              b, al, be, qa, pa, ea, iz, hiz, iss2);
                });
        },
        [](const TruncatedNormalDistribution& d, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double mu, sigma, a, b, al, be, qa, pa, ea, iz, hiz, iss2;
            d.withCacheSnapshot([&] {
                mu = d.mean_;
                sigma = d.standardDeviation_;
                a = d.lowerBound_;
                b = d.upperBound_;
                al = d.alpha_;
                be = d.beta_;
                qa = d.qAlpha_;
                pa = d.phiAlpha_;
                ea = d.erfAlpha_;
                iz = d.invZ_;
                hiz = d.halfInvZ_;
                iss2 = d.invSigmaSqrt2_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                d.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start, res.data() + start,
                                                          len, mu, sigma, a, b, al, be, qa, pa, ea,
                                                          iz, hiz, iss2);
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

namespace {
// Tolerant equality that also treats matching infinities as equal
// (|∞ − ∞| is NaN, which would compare unequal under the plain form).
inline bool param_equal(double x, double y) noexcept {
    return (x == y) || (std::fabs(x - y) < detail::ULTRA_HIGH_PRECISION_TOLERANCE);
}
}  // namespace

bool TruncatedNormalDistribution::operator==(const TruncatedNormalDistribution& other) const {
    // d == d would lock the same shared_mutex twice from one thread (undefined).
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return param_equal(mean_, other.mean_) &&
           param_equal(standardDeviation_, other.standardDeviation_) &&
           param_equal(lowerBound_, other.lowerBound_) &&
           param_equal(upperBound_, other.upperBound_);
}

bool TruncatedNormalDistribution::operator!=(const TruncatedNormalDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const TruncatedNormalDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, TruncatedNormalDistribution& d) {
    std::string line;
    if (!std::getline(is, line)) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t start = line.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    line = line.substr(start);
    if (!line.starts_with("TruncatedNormalDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const auto grab = [&line](const char* key) -> std::string {
        const size_t kp = line.find(key);
        if (kp == std::string::npos)
            return {};
        const size_t vs = kp + std::string(key).size();
        const size_t ve = line.find_first_of(",)", vs);
        if (ve == std::string::npos)
            return {};
        return line.substr(vs, ve - vs);
    };
    try {
        const std::string ms = grab("mu="), ss = grab("sigma="), as = grab("a="), bs = grab("b=");
        if (ms.empty() || ss.empty() || as.empty() || bs.empty()) {
            is.setstate(std::ios::failbit);
            return is;
        }
        const double mu = std::stod(ms), sg = std::stod(ss), a = std::stod(as), b = std::stod(bs);
        auto result = d.trySetParameters(mu, sg, a, b);
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
// PDF/LogPDF are the Gaussian log-space pipeline plus the cached −log Z
// offset, with the bounded-support compute+fixup pattern (B). The fixup
// pass re-reads `values` after `results` is written — legal at the
// distribution layer under the documented no-overlap contract (#112).
//
// CDF: the vector_erf chain computes the straddling-branch expression
// (erf(w) − erf(α/√2))·(0.5/Z). That form is 100% cancelled garbage when
// the whole window sits in one tail (α ≥ 0 or β ≤ 0) — those regimes run
// the regime-split scalar kernel per lane instead (correctness over
// throughput). For straddling windows, per-lane fixups overwrite the lanes
// where the scalar path takes an erfc branch (ξ ≤ 0) with the scalar
// path's exact expression — bit-identical to getCumulativeProbability —
// plus the exact-0/1 bound lanes and the [0,1] clamp on every lane.
// Non-fixed lanes (α < 0 < ξ) may differ from scalar only by vector_erf's
// documented ulp band and the w-rounding difference ((x−μ)·(1/(σ√2)) vs
// ((x−μ)/σ)·(1/√2)) — ≲ 2e-15 absolute, same class as the Gaussian batch
// CDF's non-fixed band (#49 gate).
//==============================================================================

void TruncatedNormalDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double mu, double a, double b,
    double neg_half_inv_sigma2, double log_pdf_norm) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < a || x > b) {
                results[i] = detail::ZERO_DOUBLE;
                continue;
            }
            const double diff = x - mu;
            results[i] = std::exp(log_pdf_norm + neg_half_inv_sigma2 * (diff * diff));
        }
        return;
    }

    // Step 1: results = x − μ
    arch::simd::VectorOps::scalar_add(values, -mu, results, count);
    // Step 2: results = (x − μ)²
    arch::simd::VectorOps::vector_multiply(results, results, results, count);
    // Step 3: results = −(x − μ)²/(2σ²)
    arch::simd::VectorOps::scalar_multiply(results, neg_half_inv_sigma2, results, count);
    // Step 4: results += −log σ − log Z − ½log(2π)
    arch::simd::VectorOps::scalar_add(results, log_pdf_norm, results, count);
    // Step 5: results = exp(·) — underflows cleanly to 0 in the far tail
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: outside [a, b] → 0. NaN lanes compare false and propagate.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < a || values[i] > b)
            results[i] = detail::ZERO_DOUBLE;
    }
}

void TruncatedNormalDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double mu, double a, double b,
    double neg_half_inv_sigma2, double log_pdf_norm) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            const double x = values[i];
            if (x < a || x > b) {
                results[i] = detail::NEGATIVE_INFINITY;
                continue;
            }
            const double diff = x - mu;
            results[i] = log_pdf_norm + neg_half_inv_sigma2 * (diff * diff);
        }
        return;
    }

    // Steps 1–4 of the PDF pipeline (no exp) — exact log space.
    arch::simd::VectorOps::scalar_add(values, -mu, results, count);
    arch::simd::VectorOps::vector_multiply(results, results, results, count);
    arch::simd::VectorOps::scalar_multiply(results, neg_half_inv_sigma2, results, count);
    arch::simd::VectorOps::scalar_add(results, log_pdf_norm, results, count);

    // Fixup: outside [a, b] → −∞.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < a || values[i] > b)
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void TruncatedNormalDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double mu, double sigma, double a,
    double b, double alpha, double beta, double q_alpha, double phi_alpha, double erf_alpha,
    double inv_z, double half_inv_z, double inv_sigma_sqrt2) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    // Same-tail windows (α ≥ 0 or β ≤ 0): every lane needs the erfc-domain
    // regime split — the vector_erf chain below would be pure cancellation.
    // Run the regime-split scalar kernel per lane (see section banner).
    if (!use_simd || alpha >= detail::ZERO_DOUBLE || beta <= detail::ZERO_DOUBLE) {
        for (std::size_t i = 0; i < count; ++i) {
            results[i] = truncnorm_cdf_scalar(values[i], mu, sigma, a, b, alpha, q_alpha, phi_alpha,
                                              erf_alpha, inv_z, half_inv_z);
        }
        return;
    }

    // Straddling window (α < 0 < β): vectorized erf chain.
    // Step 1: results = x − μ
    arch::simd::VectorOps::scalar_add(values, -mu, results, count);
    // Step 2: results = (x − μ)/(σ√2)
    arch::simd::VectorOps::scalar_multiply(results, inv_sigma_sqrt2, results, count);
    // Step 3: results = erf((x − μ)/(σ√2))
    arch::simd::VectorOps::vector_erf(results, results, count);
    // Step 4: results −= erf(α/√2)
    arch::simd::VectorOps::scalar_add(results, -erf_alpha, results, count);
    // Step 5: results ·= 0.5/Z
    arch::simd::VectorOps::scalar_multiply(results, half_inv_z, results, count);

    // Per-lane fixups (#49 pattern): ξ is recomputed with the SCALAR path's
    // exact expressions, so every fixed-up lane is bit-identical to
    // getCumulativeProbability(x). NaN lanes fall through every comparison
    // and keep the (NaN) vectorized result via the pass-through clamp.
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (x <= a) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (x >= b) {
            results[i] = detail::ONE;
        } else if (truncnorm_near_lower_band((x - a) / sigma, alpha)) {
            // Near-lower-bound lanes take the series path in every regime —
            // route through the scalar kernel so the lane is bit-identical
            // to getCumulativeProbability(x).
            results[i] = truncnorm_cdf_scalar(x, mu, sigma, a, b, alpha, q_alpha, phi_alpha,
                                              erf_alpha, inv_z, half_inv_z);
        } else {
            const double xi = (x - mu) / sigma;
            if (xi <= detail::ZERO_DOUBLE) {
                results[i] = clamp01(
                    (detail::HALF * std::erfc(-xi * detail::INV_SQRT_2) - phi_alpha) * inv_z);
            } else {
                results[i] = clamp01(results[i]);
            }
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

TruncatedNormalDistribution::NormalizationConstants
TruncatedNormalDistribution::computeNormalization(double mean, double sigma, double a,
                                                  double b) noexcept {
    NormalizationConstants n{};
    // ±∞ bounds flow through directly: (±∞ − μ)/σ = ±∞ for finite μ, σ > 0.
    n.alpha = (a - mean) / sigma;
    n.beta = (b - mean) / sigma;

    // Tail pieces — each full relative precision on its own small side.
    // erfc(−∞) = 2 and erfc(+∞) = 0 make every formula below collapse
    // correctly for infinite bounds (verified by test, not assumed).
    n.q_alpha = detail::HALF * std::erfc(n.alpha * detail::INV_SQRT_2);
    n.q_beta = detail::HALF * std::erfc(n.beta * detail::INV_SQRT_2);
    n.phi_alpha = detail::HALF * std::erfc(-n.alpha * detail::INV_SQRT_2);
    n.erf_alpha = std::erf(n.alpha * detail::INV_SQRT_2);

    // Regime-split Z (see header): never the cancelling Φ(β) − Φ(α) form.
    if (n.alpha >= detail::ZERO_DOUBLE) {
        n.z = n.q_alpha - n.q_beta;  // right tail: small − smaller
    } else if (n.beta <= detail::ZERO_DOUBLE) {
        const double phi_beta = detail::HALF * std::erfc(-n.beta * detail::INV_SQRT_2);
        n.z = phi_beta - n.phi_alpha;  // left tail: reflection
    } else {
        // Straddling: erf difference — opposite signs, adds magnitudes.
        n.z = detail::HALF * (std::erf(n.beta * detail::INV_SQRT_2) - n.erf_alpha);
    }

    n.valid = std::isfinite(n.z) && n.z > detail::ZERO_DOUBLE;
    n.log_z = n.valid ? std::log(n.z) : detail::NEGATIVE_INFINITY;
    return n;
}

void TruncatedNormalDistribution::updateCacheUnsafe() const noexcept {
    const auto nc = computeNormalization(mean_, standardDeviation_, lowerBound_, upperBound_);
    // nc.valid is guaranteed by the factory/setter window checks; this
    // method only re-derives cached values from already-accepted parameters.
    alpha_ = nc.alpha;
    beta_ = nc.beta;
    z_ = nc.z;
    logZ_ = nc.log_z;
    phiAlpha_ = nc.phi_alpha;
    qAlpha_ = nc.q_alpha;
    qBeta_ = nc.q_beta;
    erfAlpha_ = nc.erf_alpha;
    invZ_ = detail::ONE / z_;
    halfInvZ_ = detail::HALF / z_;

    logSigma_ = std::log(standardDeviation_);
    negHalfInvSigmaSquared_ = -detail::HALF / (standardDeviation_ * standardDeviation_);
    invSigmaSqrt2_ = detail::ONE / (standardDeviation_ * detail::SQRT_2);
    logPdfNormConst_ = -logSigma_ - logZ_ - detail::HALF_LN_2PI;

    // Truncated first/second moment terms (guarded against ∞·0).
    const double pa = phi_std(alpha_), pb = phi_std(beta_);
    delta_ = (pa - pb) / z_;
    eta_ = (zphi(alpha_, pa) - zphi(beta_, pb)) / z_;
    distMean_ = mean_ + standardDeviation_ * delta_;
    double c2 = detail::ONE + eta_ - delta_ * delta_;
    if (c2 < detail::ZERO_DOUBLE)
        c2 = detail::ZERO_DOUBLE;  // deep same-tail windows: cancellation floor
    distVariance_ = standardDeviation_ * standardDeviation_ * c2;

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMean_.store(mean_, std::memory_order_release);
    atomicStandardDeviation_.store(standardDeviation_, std::memory_order_release);
    atomicLowerBound_.store(lowerBound_, std::memory_order_release);
    atomicUpperBound_.store(upperBound_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats

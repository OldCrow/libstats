#include "libstats/distributions/gamma.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/core/parallel_batch_fit.h"

// Core functionality - lightweight headers
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/log_space_ops.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/safety.h"
#include "libstats/core/statistical_constants.h"

// Platform headers - use forward declarations where available
#include "libstats/common/cpu_detection_fwd.h"  // Lightweight CPU detection
// Note: parallel_execution.h is transitively included via dispatch_utils.h
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stats {

namespace {
// PARALLEL and WORK_STEALING run the batch kernel over slices of this many elements (#191).
constexpr std::size_t kBatchSlice = 1024;
}  // namespace

//==========================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==========================================================================

GammaDistribution::GammaDistribution(double alpha, double beta) {
    auto validation = validateGammaParameters(alpha, beta);
    if (validation.isError()) {
        throw std::invalid_argument(validation.message());
    }
    alpha_ = alpha;
    beta_ = beta;
    updateCacheUnsafe();
}

GammaDistribution::GammaDistribution(const GammaDistribution& other) : DistributionBase(other) {
    // A copy is read-only on the source: shared_lock allows concurrent readers
    // while still excluding concurrent writers. The previous unique_lock blocked
    // all concurrent reads of other for the duration of the copy unnecessarily.
    std::shared_lock lock(other.cache_mutex_);
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    // The cache starts invalid and is rebuilt on first read: copying other's validity flag
    // without its cached members made a copy compute with the default cache (#163).
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

GammaDistribution& GammaDistribution::operator=(const GammaDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        updateCacheUnsafe();
    }
    return *this;
}

GammaDistribution::GammaDistribution(GammaDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    // Cache starts invalid, as in the copy constructor (#163).
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

GammaDistribution& GammaDistribution::operator=(GammaDistribution&& other) noexcept {
    if (this != &other) {
        // Both locks, as copy-assignment takes, in std::lock order; the source is written, so
        // exclusively (#184).
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        // The cache is invalidated and rebuilt on the next read.
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

// Destructor is explicitly defaulted in header - no definition needed here

//==========================================================================
// 2. SAFE FACTORY METHODS (Exception-free construction)
//==========================================================================

// Note: Safe factory methods are implemented inline in header for performance
// All create() and createWithScale() methods are header-only implementations

//==========================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==========================================================================

double GammaDistribution::getScale() const {
    double sc;
    withCacheSnapshot([&] { sc = scale_; });
    return sc;
}

double GammaDistribution::getMean() const {
    double m;
    withCacheSnapshot([&] { m = mean_; });
    return m;
}

double GammaDistribution::getVariance() const {
    double v;
    withCacheSnapshot([&] { v = variance_; });
    return v;
}

double GammaDistribution::getSkewness() const {
    double sa;
    withCacheSnapshot([&] { sa = sqrtAlpha_; });
    return detail::TWO / sa;
}

double GammaDistribution::getKurtosis() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return detail::SIX / alpha_;  // Direct computation is safe
}

double GammaDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (alpha_ < detail::ONE) {
        return detail::ZERO_DOUBLE;
    }
    return (alpha_ - detail::ONE) / beta_;
}

void GammaDistribution::setAlpha(double alpha) {
    // Validate against the other parameter under the lock that commits (#183).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(alpha, beta_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::setBeta(double beta) {
    // Validate against the other parameter under the lock that commits (#183).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    validateParameters(alpha_, beta);
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::setParameters(double alpha, double beta) {
    // Validate parameters
    validateParameters(alpha, beta);

    // Update with unique lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==========================================================================
// 4. RESULT-BASED SETTERS
//==========================================================================

VoidResult GammaDistribution::trySetAlpha(double alpha) noexcept {
    // Validate against the other parameter under the lock that commits (#183).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto validation = validateGammaParameters(alpha, beta_);
    if (validation.isError()) {
        return validation;
    }

    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult GammaDistribution::trySetBeta(double beta) noexcept {
    // Validate against the other parameter under the lock that commits (#183).
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    auto validation = validateGammaParameters(alpha_, beta);
    if (validation.isError()) {
        return validation;
    }

    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult GammaDistribution::trySetParameters(double alpha, double beta) noexcept {
    auto validation = validateGammaParameters(alpha, beta);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    beta_ = beta;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

//==========================================================================
// 5. CORE PROBABILITY METHODS
//==========================================================================

namespace {
// Below this shape the direct log density is kept bit for bit; from it, log_gamma_prefactor is
// in Stirling's form.
constexpr double kStirlingDensityShape = detail::STIRLING_PREFACTOR_SHAPE;

// The shape-only constant gammaLogDensity takes: α·log β − lgamma(α) for the direct form, and
// from α = 20 the Stirling form's ½·log(α/2π) − c(α). Batch loops compute it once.
[[nodiscard]] inline double gammaDensityConstant(double alpha, double alpha_log_beta,
                                                 double log_gamma_alpha) noexcept {
    return alpha < kStirlingDensityShape ? alpha_log_beta - log_gamma_alpha
                                         : detail::log_gamma_prefactor_constant(alpha);
}

// F(x) = P(α, βx) for x > 0 finite. Once βx is below DBL_MIN (a subnormal x at a rate below
// 1, or a product that underflowed to 0) the argument is taken in log space: the mass there is
// not small for small shapes (#216: Gamma(0.005, 0.5).cdf(5e-324) = 0.024, which βx = 0 gave
// as 0).
[[nodiscard]] inline double gammaCdfCore(double alpha, double beta, double x) noexcept {
    const double y = beta * x;
    if (y < std::numeric_limits<double>::min())
        return detail::gamma_p_from_log_x(alpha, std::log(beta) + std::log(x));
    return detail::gamma_p(alpha, y);
}

// Entropy at shapes from kEntropyStirlingShape: the direct form α + lgamma(α) + (1 − α)ψ(α)
// cancels at α·log α (DH2 M2: 7.5e-7 relative at 1e10, −64 for 19.84 at 1e16). From the
// Stirling series of lgamma and ψ with Bernoulli numbers B_2k,
//   h = ½·log(2πeα) − 1/(2α) + Σ_k B_2k / ((2k−1) α^(2k−1)) − Σ_k B_2k / (2k α^(2k)),
// whose first terms are −1/(3α) − 1/(12α²) − 1/(90α³) + 1/(120α⁴) + 1/(210α⁵) − 1/(252α⁶)
// − 1/(210α⁷) + 1/(240α⁸). The next term, 5/(594α⁹), is 8e-21 at α = 100; the direct form's
// error there is ~α·log α·ε = 1e-13.
constexpr double kEntropyStirlingShape = 100.0;

[[nodiscard]] inline double gammaEntropyStirling(double alpha) noexcept {
    const double t = detail::ONE / alpha;
    const double tail =
        t * (-1.0 / 3.0 +
             t * (-1.0 / 12.0 +
                  t * (-1.0 / 90.0 +
                       t * (1.0 / 120.0 +
                            t * (1.0 / 210.0 +
                                 t * (-1.0 / 252.0 + t * (-1.0 / 210.0 + t * (1.0 / 240.0))))))));
    return detail::HALF * (detail::LN_2PI + detail::ONE + std::log(alpha)) + tail;
}

// log of the Gamma(α, rate β) density at a finite x > 0. From α = 20 it is log P(α, βx) − log x,
// with P = (βx)^α·e^{−βx}/Γ(α) in Stirling form: the direct α·log β − lgamma(α) + (α − 1)·log x −
// βx cancels terms of size α·log α, 2e-11 relative at α = 1e4. density_constant is
// gammaDensityConstant(α, …).
[[nodiscard]] inline double gammaLogDensity(double x, double alpha, double beta,
                                            double density_constant,
                                            double alpha_minus_one) noexcept {
    if (alpha < kStirlingDensityShape)
        return density_constant + alpha_minus_one * std::log(x) - beta * x;
    return detail::log_gamma_prefactor(alpha, beta * x, density_constant) - std::log(x);
}

// gammaLogDensity's Stirling branch (α ≥ 20) over a batch: α·log1pmx(t) + C − log x with
// t = (βx − α)/α and C = gammaDensityConstant(α, …), in the scalar path's order. The
// branch-free log1pmx_series covers |t| < ½ and vectorizes; the few elements beyond take
// log(βx/α) − t as the scalar path does (log1p(t) from the ratio, not from the rounded t, which
// is −1 exactly at x ≪ α), not vector_log1p: log1p(t) − t still cancels up to threefold there and
// α multiplies it, so its few ulp left 1e-12 in the log at large α.
// Inputs ≤ 0 and non-finite inputs come out meaningless here; the caller's fixup pass overwrites
// them.
void stirlingLogDensityBatch(const double* x, double* out, std::size_t count, double alpha,
                             double beta, double density_constant) noexcept {
    constexpr std::size_t kBlock = 256;
    alignas(64) double log_x[kBlock];
    std::size_t far_index[kBlock];
    for (std::size_t start = 0; start < count; start += kBlock) {
        const std::size_t n = std::min(kBlock, count - start);
        const double* xb = x + start;
        double* ob = out + start;
        arch::simd::VectorOps::vector_log(xb, log_x, n);
        for (std::size_t i = 0; i < n; ++i)
            ob[i] = detail::log1pmx_series((beta * xb[i] - alpha) / alpha);
        // The few lanes past the series, gathered first: written as a guarded log in one loop,
        // AppleClang (no errno on Darwin, so log is pure) calls the log for every lane, 1.35x
        // the whole batch at alpha = 20 (Kaby Lake, 2026-10-04).
        std::size_t n_far = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double t = (beta * xb[i] - alpha) / alpha;
            if (std::fabs(t) >= detail::LOG1PMX_SERIES_LIMIT)
                far_index[n_far++] = i;
        }
        for (std::size_t k = 0; k < n_far; ++k) {
            const std::size_t i = far_index[k];
            const double xp = beta * xb[i];
            const double t = (xp - alpha) / alpha;
            ob[i] = std::log(xp / alpha) - t;  // log1p(t) as the scalar path forms it
        }
        for (std::size_t i = 0; i < n; ++i)
            ob[i] = alpha * ob[i] + density_constant - log_x[i];
    }
}
}  // namespace

double GammaDistribution::getProbability(double x) const {
    if (!std::isfinite(x)) {
        // +inf → density limit is 0 (#103); -inf → outside support → 0
        // NaN  → propagate NaN
        // Guard required: for alpha >= 1 the log-space formula below hits
        // 0*log(inf) (alpha == 1) or inf - inf (alpha > 1) at x = +inf → NaN.
        if (std::isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        return detail::ZERO_DOUBLE;
    }
    if (x < detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    double a, b, alb, lga, am1;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
        alb = alphaLogBeta_;
        lga = logGammaAlpha_;
        am1 = alphaMinusOne_;
    });
    // Handle special case x = 0
    if (x == detail::ZERO_DOUBLE) {
        return (a < detail::ONE)    ? std::numeric_limits<double>::infinity()
               : (a == detail::ONE) ? b
                                    : detail::ZERO_DOUBLE;
    }
    // Inline log-space computation.
    return std::exp(gammaLogDensity(x, a, b, gammaDensityConstant(a, alb, lga), am1));
}

double GammaDistribution::getLogProbability(double x) const {
    if (!std::isfinite(x)) {
        // ±inf → log-density limit is -inf (#103); NaN → propagate NaN.
        // Same 0*log(inf) / inf - inf hazard as getProbability at x = +inf.
        if (std::isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        return detail::NEGATIVE_INFINITY;
    }
    if (x < detail::ZERO_DOUBLE) {
        return detail::NEGATIVE_INFINITY;
    }

    double a, b, lb, alb, lga, am1;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
        lb = logBeta_;
        alb = alphaLogBeta_;
        lga = logGammaAlpha_;
        am1 = alphaMinusOne_;
    });
    // Handle special case x = 0
    if (x == detail::ZERO_DOUBLE) {
        if (a < detail::ONE)
            return std::numeric_limits<double>::infinity();
        else if (a == detail::ONE)
            return lb;
        else
            return detail::NEGATIVE_INFINITY;
    }
    // General case: log(f(x)) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - βx
    return gammaLogDensity(x, a, b, gammaDensityConstant(a, alb, lga), am1);
}

double GammaDistribution::getCumulativeProbability(double x) const {
    if (!std::isfinite(x)) {
        // +inf → all probability mass lies below +∞ → 1.0
        // -inf → no probability mass lies below -∞  → 0.0
        // NaN  → propagate NaN
        if (std::isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        return (x > 0) ? detail::ONE : detail::ZERO_DOUBLE;
    }
    if (x <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    // Use regularized incomplete gamma function P(α, βx)
    return gammaCdfCore(a, b, x);
}

double GammaDistribution::getQuantile(double p) const {
    if (std::isnan(p))
        return std::numeric_limits<double>::quiet_NaN();  // NaN in, NaN out (AR D3)
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    return computeQuantile(p);
}

double GammaDistribution::sample(std::mt19937& rng) const {
    double cached_alpha, cached_beta;
    withCacheSnapshot([&] {
        cached_alpha = alpha_;
        cached_beta = beta_;
    });

    // Choose sampling method based on cached α — lock released.
    // Inline the sampling algorithms using cached parameters to avoid
    // calling private helpers that would read unlocked member variables.
    if (cached_alpha >= detail::ONE) {
        // Marsaglia-Tsang "squeeze" method for α ≥ 1
        std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);
        std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);
        const double d = cached_alpha - detail::ONE / detail::THREE;
        const double c = detail::ONE / std::sqrt(detail::NINE * d);
        while (true) {
            double x, v;
            do {
                x = normal(rng);
                v = detail::ONE + c * x;
            } while (v <= detail::ZERO_DOUBLE);
            v = v * v * v;
            double u = uniform(rng);
            if (u < detail::ONE - 0.0331 * (x * x) * (x * x)) {
                return d * v / cached_beta;
            }
            if (std::log(u) < detail::HALF * x * x + d * (detail::ONE - v + std::log(v))) {
                return d * v / cached_beta;
            }
        }
    } else {
        // Ahrens-Dieter acceptance-rejection method for α < 1
        std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);
        const double b = (detail::E + cached_alpha) / detail::E;
        while (true) {
            double u = uniform(rng);
            double p = b * u;
            if (p <= detail::ONE) {
                double x = std::pow(p, detail::ONE / cached_alpha);
                double u2 = uniform(rng);
                if (u2 <= std::exp(-x)) {
                    return x / cached_beta;
                }
            } else {
                double x = -std::log((b - p) / cached_alpha);
                double u2 = uniform(rng);
                if (u2 <= std::pow(x, cached_alpha - detail::ONE)) {
                    return x / cached_beta;
                }
            }
        }
    }
}

std::vector<double> GammaDistribution::sample(std::mt19937& rng, size_t n) const {
    double cached_alpha, cached_beta;
    withCacheSnapshot([&] {
        cached_alpha = alpha_;
        cached_beta = beta_;
    });

    std::vector<double> samples;
    samples.reserve(n);

    if (cached_alpha >= detail::ONE) {
        // Marsaglia-Tsang method for α ≥ 1
        std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);
        std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);
        const double d = cached_alpha - detail::ONE / detail::THREE;
        const double c = detail::ONE / std::sqrt(detail::NINE * d);
        for (size_t i = 0; i < n; ++i) {
            double x, v;
            do {
                x = normal(rng);
                v = detail::ONE + c * x;
            } while (v <= detail::ZERO_DOUBLE);
            v = v * v * v;
            double u = uniform(rng);
            if (u < detail::ONE - 0.0331 * (x * x) * (x * x)) {
                samples.push_back(d * v / cached_beta);
                continue;
            }
            if (std::log(u) < detail::HALF * x * x + d * (detail::ONE - v + std::log(v))) {
                samples.push_back(d * v / cached_beta);
                continue;
            }
            --i;  // rejection — retry
        }
    } else {
        // Ahrens-Dieter method for α < 1
        std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);
        const double b = (detail::E + cached_alpha) / detail::E;
        for (size_t i = 0; i < n; ++i) {
            double u = uniform(rng);
            double p = b * u;
            if (p <= detail::ONE) {
                double x = std::pow(p, detail::ONE / cached_alpha);
                if (uniform(rng) <= std::exp(-x)) {
                    samples.push_back(x / cached_beta);
                    continue;
                }
            } else {
                double x = -std::log((b - p) / cached_alpha);
                if (uniform(rng) <= std::pow(x, cached_alpha - detail::ONE)) {
                    samples.push_back(x / cached_beta);
                    continue;
                }
            }
            --i;  // rejection — retry
        }
    }

    return samples;
}

//==========================================================================
// 6. DISTRIBUTION MANAGEMENT
//==========================================================================

void GammaDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    // Check for invalid values (FIT-1: NaN passes `<= 0` since NaN comparisons are false)
    for (double value : values) {
        if (!std::isfinite(value) || value <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument(
                "All values must be positive and finite for Gamma distribution");
        }
    }

    // FIT-2: fitMethodOfMoments() was called here as an initial-estimate step but
    // fitMaximumLikelihood() computes its own Choi-Wette initial estimate and
    // overwrites alpha_/beta_ unconditionally.  The MoM call was pure dead work;
    // removed to avoid two wasted lock acquisitions and O(n) compute per fit call.
    fitMaximumLikelihood(values);
}

void GammaDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                         std::vector<GammaDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void GammaDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = detail::ONE;
    beta_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string GammaDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "GammaDistribution(alpha=" << alpha_ << ", beta=" << beta_ << ")";
    return oss.str();
}

//==========================================================================
// 7. ADVANCED STATISTICAL METHODS
//==========================================================================

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
//==========================================================================

//==========================================================================
// 9. CROSS-VALIDATION METHODS
//==========================================================================

//==========================================================================
// 10. INFORMATION CRITERIA
//==========================================================================

//==========================================================================
// 11. BOOTSTRAP METHODS
//==========================================================================

//==========================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==========================================================================

// Moved from inline methods in header for better compilation speed

double GammaDistribution::getAlphaAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicAlpha_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getAlpha();
}

double GammaDistribution::getBetaAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicBeta_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getBeta();
}

int GammaDistribution::getNumParameters() const noexcept {
    return 2;
}

bool GammaDistribution::isDiscrete() const noexcept {
    return false;
}

double GammaDistribution::getSupportLowerBound() const noexcept {
    return 0.0;
}

double GammaDistribution::getSupportUpperBound() const noexcept {
    return std::numeric_limits<double>::infinity();
}

VoidResult GammaDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateGammaParameters(alpha_, beta_);
}

double GammaDistribution::getMedian() const {
    return getQuantile(0.5);
}

bool GammaDistribution::operator!=(const GammaDistribution& other) const {
    return !(*this == other);
}

GammaDistribution GammaDistribution::createUnchecked(double alpha, double beta) noexcept {
    GammaDistribution dist(alpha, beta, true);  // bypass validation
    return dist;
}

GammaDistribution::GammaDistribution(double alpha, double beta, bool /*bypassValidation*/) noexcept
    : DistributionBase(), alpha_(alpha), beta_(beta) {
    // Cache will be updated on first use
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Initialize atomic parameters to invalid state
    atomicAlpha_.store(alpha, std::memory_order_release);
    atomicBeta_.store(beta, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

bool GammaDistribution::isExponentialDistribution() const noexcept {
    bool ie;
    withCacheSnapshot([&] { ie = isExponential_; });
    return ie;
}

bool GammaDistribution::isChiSquaredDistribution() const noexcept {
    bool ics;
    withCacheSnapshot([&] { ics = isChiSquared_; });
    return ics;
}

double GammaDistribution::getDegreesOfFreedom() const {
    if (!isChiSquaredDistribution()) {
        throw std::logic_error(
            "Distribution is not a chi-squared distribution (beta != detail::HALF)");
    }
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return detail::TWO * alpha_;
}

double GammaDistribution::getEntropy() const {
    double a, lb, lga, da;
    withCacheSnapshot([&] {
        a = alpha_;
        lb = logBeta_;
        lga = logGammaAlpha_;
        da = digammaAlpha_;
    });
    // H(X) = α - log(β) + log(Γ(α)) + (1-α)ψ(α); the Stirling form from α = 100 (DH2 M2).
    if (a >= kEntropyStirlingShape)
        return gammaEntropyStirling(a) - lb;
    return a - lb + lga + (detail::ONE - a) * da;
}

bool GammaDistribution::canUseNormalApproximation() const noexcept {
    bool la;
    withCacheSnapshot([&] { la = isLargeAlpha_; });
    return la;
}

Result<GammaDistribution> GammaDistribution::createFromMoments(double mean,
                                                               double variance) noexcept {
    if (mean <= detail::ZERO_DOUBLE) {
        return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter,
                                                    "Mean must be positive");
    }
    if (variance <= detail::ZERO_DOUBLE) {
        return Result<GammaDistribution>::makeError(ValidationError::InvalidParameter,
                                                    "Variance must be positive");
    }

    // Method of moments: α = mean²/variance, β = mean/variance
    double alpha = (mean * mean) / variance;
    double beta = mean / variance;

    return create(alpha, beta);
}

//==========================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS IMPLEMENTATION
//==========================================================================

void GammaDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                       const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const GammaDistribution& dist, double value) { return dist.getProbability(value); },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            double alpha, beta, lga, alb, am1;
            dist.withCacheSnapshot([&] {
                alpha = dist.alpha_;
                beta = dist.beta_;
                lga = dist.logGammaAlpha_;
                alb = dist.alphaLogBeta_;
                am1 = dist.alphaMinusOne_;
            });
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta, lga, alb, am1);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            [[maybe_unused]] double cached_alpha;
            double cached_beta, cached_log_gamma_alpha, cached_alpha_log_beta,
                cached_alpha_minus_one;
            dist.withCacheSnapshot([&] {
                cached_alpha = dist.alpha_;
                cached_beta = dist.beta_;
                cached_log_gamma_alpha = dist.logGammaAlpha_;
                cached_alpha_log_beta = dist.alphaLogBeta_;
                cached_alpha_minus_one = dist.alphaMinusOne_;
            });

            // Slice so each parallel task runs the SIMD log+exp pipeline
            // rather than computing log(x) per element in each task.
            constexpr std::size_t CHUNK = 1024;
            ParallelUtils::parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getProbabilityBatchUnsafeImpl(
                    vals.data() + start, res.data() + start, len, cached_alpha, cached_beta,
                    cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            double cached_alpha, cached_beta, cached_log_gamma_alpha;
            double cached_alpha_log_beta, cached_alpha_minus_one;
            dist.withCacheSnapshot([&] {
                cached_alpha = dist.alpha_;
                cached_beta = dist.beta_;
                cached_log_gamma_alpha = dist.logGammaAlpha_;
                cached_alpha_log_beta = dist.alphaLogBeta_;
                cached_alpha_minus_one = dist.alphaMinusOne_;
            });

            // Slice into SIMD-sized pieces so pool tasks use the SIMD pipeline.
            constexpr std::size_t CHUNK = 1024;
            pool.parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getProbabilityBatchUnsafeImpl(
                    vals.data() + start, res.data() + start, len, cached_alpha, cached_beta,
                    cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
            });
            pool.waitForAll();
        });
}

void GammaDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const GammaDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            double alpha, beta, lga, alb, am1;
            dist.withCacheSnapshot([&] {
                alpha = dist.alpha_;
                beta = dist.beta_;
                lga = dist.logGammaAlpha_;
                alb = dist.alphaLogBeta_;
                am1 = dist.alphaMinusOne_;
            });
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta, lga, alb, am1);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            double cached_alpha, cached_beta, cached_log_gamma_alpha;
            double cached_alpha_log_beta, cached_alpha_minus_one;
            dist.withCacheSnapshot([&] {
                cached_alpha = dist.alpha_;
                cached_beta = dist.beta_;
                cached_log_gamma_alpha = dist.logGammaAlpha_;
                cached_alpha_log_beta = dist.alphaLogBeta_;
                cached_alpha_minus_one = dist.alphaMinusOne_;
            });

            // Slice so each parallel task runs the SIMD log pipeline.
            constexpr std::size_t CHUNK = 1024;
            ParallelUtils::parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getLogProbabilityBatchUnsafeImpl(
                    vals.data() + start, res.data() + start, len, cached_alpha, cached_beta,
                    cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
            });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            // Work-Stealing lambda: should use pool.parallelFor
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            double cached_alpha, cached_beta, cached_log_gamma_alpha;
            double cached_alpha_log_beta, cached_alpha_minus_one;
            dist.withCacheSnapshot([&] {
                cached_alpha = dist.alpha_;
                cached_beta = dist.beta_;
                cached_log_gamma_alpha = dist.logGammaAlpha_;
                cached_alpha_log_beta = dist.alphaLogBeta_;
                cached_alpha_minus_one = dist.alphaMinusOne_;
            });

            constexpr std::size_t CHUNK = 1024;
            pool.parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                dist.getLogProbabilityBatchUnsafeImpl(
                    vals.data() + start, res.data() + start, len, cached_alpha, cached_beta,
                    cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
            });
            pool.waitForAll();
        });
}

void GammaDistribution::getCumulativeProbability(std::span<const double> values,
                                                 std::span<double> results,
                                                 const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const GammaDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const GammaDistribution& dist, const double* vals, double* res, size_t count) {
            double alpha, beta;
            dist.withCacheSnapshot([&] {
                alpha = dist.alpha_;
                beta = dist.beta_;
            });
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double alpha, beta;
            dist.withCacheSnapshot([&] {
                alpha = dist.alpha_;
                beta = dist.beta_;
            });
            ParallelUtils::parallelForSlices(
                count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                    dist.getCumulativeProbabilityBatchUnsafeImpl(
                        vals.data() + start, res.data() + start, len, alpha, beta);
                });
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double alpha, beta;
            dist.withCacheSnapshot([&] {
                alpha = dist.alpha_;
                beta = dist.beta_;
            });
            pool.parallelForSlices(count, kBatchSlice, [&](std::size_t start, std::size_t len) {
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals.data() + start,
                                                             res.data() + start, len, alpha, beta);
            });
        });
}

//==========================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==========================================================================

//==========================================================================
// 15. COMPARISON OPERATORS
//==========================================================================

bool GammaDistribution::operator==(const GammaDistribution& other) const {
    // d == d would lock the same shared_mutex twice from one thread (undefined).
    if (this == &other)
        return true;
    // Use scoped_lock to prevent deadlock when comparing two distributions
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    // Compare parameters within tolerance
    return (std::abs(alpha_ - other.alpha_) <= detail::DEFAULT_TOLERANCE) &&
           (std::abs(beta_ - other.beta_) <= detail::DEFAULT_TOLERANCE);
}

//==========================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==========================================================================

std::ostream& operator<<(std::ostream& os, const GammaDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, GammaDistribution& dist) {
    std::string temp;
    double alpha, beta;

    // Expected format: "GammaDistribution(alpha=value, beta=value)"
    // Read "GammaDistribution(alpha="
    is >> temp;  // "GammaDistribution(alpha=value,"

    if (temp.starts_with("GammaDistribution(alpha=")) {
        // Extract alpha value
        size_t equals_pos = temp.find('=');
        size_t comma_pos = temp.find(',');
        if (equals_pos != std::string::npos && comma_pos != std::string::npos) {
            std::string alpha_str =
                temp.substr(equals_pos + detail::ONE_INT, comma_pos - equals_pos - detail::ONE_INT);
            alpha = detail::parse_double(alpha_str);

            // Read "beta=value)"
            is >> temp;
            if (temp.starts_with("beta=")) {
                size_t beta_equals_pos = temp.find('=');
                size_t close_paren_pos = temp.find(')');
                if (beta_equals_pos != std::string::npos && close_paren_pos != std::string::npos) {
                    std::string beta_str =
                        temp.substr(beta_equals_pos + detail::ONE_INT,
                                    close_paren_pos - beta_equals_pos - detail::ONE_INT);
                    beta = detail::parse_double(beta_str);

                    // Set parameters if valid
                    auto result = dist.trySetParameters(alpha, beta);
                    if (result.isError()) {
                        is.setstate(std::ios::failbit);
                    }
                } else {
                    is.setstate(std::ios::failbit);
                }
            } else {
                is.setstate(std::ios::failbit);
            }
        } else {
            is.setstate(std::ios::failbit);
        }
    } else {
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==============================================================================
// 17. PRIVATE FACTORY IMPLEMENTATION METHODS
//==============================================================================

// Note: Private factory implementation methods are currently inline in the header
// This section exists for standardization and documentation purposes

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// These methods are the computational core of Gamma batch ops. Called after
// the public API validates inputs and extracts cached parameters under a read
// lock. Raw pointers avoid span bounds-checking overhead.
//
// PDF/LogPDF architecture: fully vectorized log-space pipeline
//   The previous implementation had a scalar std::log() prepass that
//   dominated runtime. Both methods now use VectorOps::vector_log to
//   compute log(values) across the whole batch, then apply SIMD arithmetic.
//   Non-positive inputs produce NaN/-Inf from vector_log; a scalar fixup
//   pass at the end corrects them. One aligned temporary holds -beta*x.
//
//   PDF:    results = log(x) → * alpha_minus_one → + log_constant
//             temp = -beta*x → results += temp → vector_exp
//             fixup: x <= 0 → 0
//   LogPDF: same pipeline, no final exp step
//             fixup: x < 0 → -inf; x = 0 → the limit by shape (#161, as the scalar path)
//
// CDF architecture: the regularized incomplete gamma function gamma_p() is
//   evaluated per element via a continued-fraction or series algorithm.
//   The number of iterations required for convergence varies with the input
//   value — elements near the crossover between series and continued-fraction
//   regimes or with slow convergence require significantly more work than
//   "easy" values. This data-dependent iteration count means each SIMD lane
//   would need a different amount of work, which cannot be expressed as a
//   uniform sequence of SIMD operations. Vectorizing correctly would require
//   replacing gamma_p with a fixed-iteration polynomial approximation, which
//   is outside the scope of this library.
//   Only the beta*x pre-scaling uses SIMD; the gamma_p loop is scalar.
//==============================================================================

void GammaDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                      std::size_t count, double alpha, double beta,
                                                      double log_gamma_alpha, double alpha_log_beta,
                                                      double alpha_minus_one) const noexcept {
    // The SIMD pipeline forms the direct log density below α = 20; from there, where the direct
    // form cancels (see gammaLogDensity), it is the Stirling form (stirlingLogDensityBatch).
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    // EDGE-4 helper: write the correct PDF(0) value consistent with scalar path.
    // For alpha < 1: PDF(0) = +inf. For alpha = 1: PDF(0) = beta. For alpha > 1: PDF(0) = 0.
    auto fixup_zero = [&](std::size_t i) {
        if (alpha_minus_one < 0.0)
            results[i] = std::numeric_limits<double>::infinity();
        else if (alpha_minus_one == 0.0)
            results[i] = beta;
        else
            results[i] = detail::ZERO_DOUBLE;
    };

    if (!use_simd) {
        const double density_constant =
            gammaDensityConstant(alpha, alpha_log_beta, log_gamma_alpha);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) {
                // #103: pdf(±inf) = 0, NaN propagates — the formula is NaN at
                // +inf for alpha >= 1 (0*log(inf) / inf - inf), matching scalar.
                results[i] = std::isnan(values[i]) ? values[i] : detail::ZERO_DOUBLE;
            } else if (values[i] < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (values[i] == detail::ZERO_DOUBLE) {
                fixup_zero(i);
            } else {
                results[i] = std::exp(
                    gammaLogDensity(values[i], alpha, beta, density_constant, alpha_minus_one));
            }
        }
        return;
    }

    if (alpha < kStirlingDensityShape) {
        // Fully vectorized log-space pipeline.
        // One aligned temporary; results serves as workspace throughout.
        const double log_constant = alpha_log_beta - log_gamma_alpha;
        std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

        // Step 1: results = log(values)  [NaN/-Inf for x <= 0; corrected by fixup]
        arch::simd::VectorOps::vector_log(values, results, count);
        // Step 2: results = alpha_minus_one * log(values)
        arch::simd::VectorOps::scalar_multiply(results, alpha_minus_one, results, count);
        // Step 3: results += log_constant  (= alpha_log_beta - log_gamma_alpha)
        arch::simd::VectorOps::scalar_add(results, log_constant, results, count);
        // Step 4: temp = -beta * values
        arch::simd::VectorOps::scalar_multiply(values, -beta, temp.data(), count);
        // Step 5: results = log_constant + (alpha-1)*log(x) - beta*x
        arch::simd::VectorOps::vector_add(results, temp.data(), results, count);
    } else {
        stirlingLogDensityBatch(values, results, count, alpha, beta,
                                gammaDensityConstant(alpha, alpha_log_beta, log_gamma_alpha));
    }
    // Step 6: results = exp(log-space result)
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Fixup: non-finite per #103 (pdf(±inf) = 0, NaN propagates — the SIMD
    // pipeline yields NaN at +inf for alpha >= 1); x < 0 → 0; x = 0 → EDGE-4.
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            results[i] = std::isnan(values[i]) ? values[i] : detail::ZERO_DOUBLE;
        } else if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (values[i] == detail::ZERO_DOUBLE) {
            fixup_zero(i);
        }
    }
}

void GammaDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count, double alpha,
                                                         double beta, double log_gamma_alpha,
                                                         double alpha_log_beta,
                                                         double alpha_minus_one) const noexcept {
    // The SIMD pipeline forms the direct log density below α = 20; from there, where the direct
    // form cancels (see gammaLogDensity), it is the Stirling form (stirlingLogDensityBatch).
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    // logpdf(0) is the limit, as on the scalar path (#161): +inf for alpha < 1, log(beta) for
    // alpha = 1 (alpha_log_beta is exactly log(beta) there), -inf for alpha > 1.
    auto fixup_zero = [&](std::size_t i) {
        if (alpha_minus_one < 0.0)
            results[i] = std::numeric_limits<double>::infinity();
        else if (alpha_minus_one == 0.0)
            results[i] = alpha_log_beta;
        else
            results[i] = detail::NEGATIVE_INFINITY;
    };

    if (!use_simd) {
        const double density_constant =
            gammaDensityConstant(alpha, alpha_log_beta, log_gamma_alpha);
        for (std::size_t i = 0; i < count; ++i) {
            if (!std::isfinite(values[i])) {
                // #103: logpdf(±inf) = -inf, NaN propagates — the formula is
                // NaN at +inf for alpha >= 1, matching the scalar guard.
                results[i] = std::isnan(values[i]) ? values[i] : detail::NEGATIVE_INFINITY;
            } else if (values[i] < detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
            } else if (values[i] == detail::ZERO_DOUBLE) {
                fixup_zero(i);
            } else {
                results[i] =
                    gammaLogDensity(values[i], alpha, beta, density_constant, alpha_minus_one);
            }
        }
        return;
    }

    if (alpha < kStirlingDensityShape) {
        // Fully vectorized log-space computation; no exp step needed.
        // One aligned temporary for -beta*x; results is the accumulator.
        const double log_constant = alpha_log_beta - log_gamma_alpha;
        std::vector<double, arch::simd::aligned_allocator<double>> temp(count);

        // Step 1: results = log(values)  [NaN/-Inf for x <= 0; corrected by fixup]
        arch::simd::VectorOps::vector_log(values, results, count);
        // Step 2: results = alpha_minus_one * log(values)
        arch::simd::VectorOps::scalar_multiply(results, alpha_minus_one, results, count);
        // Step 3: results += log_constant
        arch::simd::VectorOps::scalar_add(results, log_constant, results, count);
        // Step 4: temp = -beta * values
        arch::simd::VectorOps::scalar_multiply(values, -beta, temp.data(), count);
        // Step 5: results = log_constant + (alpha-1)*log(x) - beta*x
        arch::simd::VectorOps::vector_add(results, temp.data(), results, count);
    } else {
        stirlingLogDensityBatch(values, results, count, alpha, beta,
                                gammaDensityConstant(alpha, alpha_log_beta, log_gamma_alpha));
    }
    // Fixup: non-finite per #103 (logpdf(±inf) = -inf, NaN propagates); x < 0 → -inf;
    // x = 0 → the limit by shape (#161).
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            results[i] = std::isnan(values[i]) ? values[i] : detail::NEGATIVE_INFINITY;
        } else if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::NEGATIVE_INFINITY;
        } else if (values[i] == detail::ZERO_DOUBLE) {
            fixup_zero(i);
        }
    }
}

void GammaDistribution::getCumulativeProbabilityBatchUnsafeImpl(const double* values,
                                                                double* results, std::size_t count,
                                                                double alpha,
                                                                double beta) const noexcept {
    // Check if vectorization is beneficial and CPU supports it
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or unsupported SIMD.
        // MC-1/MC-2: use the same log-space detail::gamma_p implementation as the
        // scalar CDF. The old private regularizedIncompleteGamma divided by
        // std::tgamma(alpha), overflowing for alpha > ~172 and diverging from scalar.
        for (std::size_t i = 0; i < count; ++i) {
            if (std::isnan(values[i])) {
                results[i] = values[i];
            } else if (values[i] <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (values[i] == std::numeric_limits<double>::infinity()) {
                results[i] = detail::ONE;  // gamma_p(alpha, +inf) is NaN (#103)
            } else {
                results[i] = gammaCdfCore(alpha, beta, values[i]);
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation
    // Create aligned temporary array for beta * values
    std::vector<double, arch::simd::aligned_allocator<double>> scaled_values(count);

    // Step 1: Compute beta * values using SIMD
    arch::simd::VectorOps::scalar_multiply(values, beta, scaled_values.data(), count);

    // Step 2: Evaluate gamma_p per element — inherently scalar.
    // gamma_p uses a continued fraction or series whose iteration count varies
    // per input; no uniform SIMD sequence can express this. See section 18
    // header for the full explanation.
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            results[i] = values[i];
        } else if (values[i] <= detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (values[i] == std::numeric_limits<double>::infinity()) {
            results[i] = detail::ONE;  // gamma_p(alpha, +inf) is NaN (#103)
        } else if (scaled_values[i] < std::numeric_limits<double>::min()) {
            results[i] = gammaCdfCore(alpha, beta, values[i]);  // log-space argument (#216)
        } else {
            results[i] = detail::gamma_p(alpha, scaled_values[i]);
        }
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

double GammaDistribution::incompleteGamma(double a, double x) noexcept {
    // Legacy helper retained for source compatibility inside the class. Prefer
    // detail::gamma_p/detail::gamma_q for numerically stable regularized forms.
    if (x <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (a <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return detail::gamma_p(a, x) * std::exp(detail::lgamma(a));
}

double GammaDistribution::regularizedIncompleteGamma(double a, double x) noexcept {
    // Regularized lower incomplete gamma function P(a,x).
    if (x <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (a <= detail::ZERO_DOUBLE) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return detail::gamma_p(a, x);
}

double GammaDistribution::computeQuantile(double p) const noexcept {
    // Gamma(alpha, rate beta): x with P(alpha, beta x) = p. gamma_p_inv solves against the smaller
    // of p and 1 - p in log x, relative-accurate down to subnormal answers (#160); the
    // Newton/bisection solver this replaced, and its Wilson-Hilferty seed, are gone.
    if (p <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p >= detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }
    double a, b;  // one (α, β) state under a concurrent setter
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    return detail::gamma_p_inv(a, p) / b;
}

void GammaDistribution::fitMethodOfMoments(const std::vector<double>& values) {
    // Method of moments parameter estimation
    if (values.empty()) {
        return;
    }

    // Method-of-moments estimates for Gamma(α, β): α̂ = mean²/var, β̂ = mean/var
    const std::size_t nv = values.size();
    double sum_x = std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE);
    double sum_x2 =
        std::inner_product(values.begin(), values.end(), values.begin(), detail::ZERO_DOUBLE);
    double mean_x = sum_x / static_cast<double>(nv);
    double var_x = sum_x2 / static_cast<double>(nv) - mean_x * mean_x;
    double alpha_hat = (var_x > detail::ZERO) ? (mean_x * mean_x / var_x) : detail::ONE;
    double beta_hat = (var_x > detail::ZERO) ? (mean_x / var_x) : detail::ONE;

    // Update parameters using the estimates
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha_hat;
    beta_ = beta_hat;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::fitMaximumLikelihood(const std::vector<double>& values) {
    // Maximum likelihood estimation using Newton-Raphson iteration
    if (values.empty()) {
        return;
    }

    size_t n = values.size();
    double sum_x = std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE);
    double sum_log_x = detail::ZERO_DOUBLE;
    for (double x : values) {
        sum_log_x += std::log(x);
    }

    double mean_x = sum_x / static_cast<double>(n);
    double mean_log_x = sum_log_x / static_cast<double>(n);

    // Initial guess using method of moments
    double s = std::log(mean_x) - mean_log_x;
    // s = log(mean) − mean(log) ≥ 0 (Jensen), and 0 exactly for one point or all-equal data,
    // where the Choi-Wette estimate is 0/0 and the fit was left at α = β = NaN (DH D13).
    if (!(s > detail::ZERO_DOUBLE) || !std::isfinite(s))
        throw std::invalid_argument(
            "Gamma MLE needs at least two distinct observations (data has zero variance)");
    double alpha_est =
        (detail::THREE - s + std::sqrt((s - detail::THREE) * (s - detail::THREE) + 24.0 * s)) /
        (12.0 * s);

    // Newton-Raphson iteration for α
    const double tolerance = detail::NEWTON_RAPHSON_TOLERANCE;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;

    for (int i = 0; i < max_iterations; ++i) {
        double digamma_alpha = detail::digamma(alpha_est);
        double trigamma_alpha = detail::trigamma(alpha_est);

        double f = std::log(alpha_est) - digamma_alpha - s;
        double df = detail::ONE / alpha_est - trigamma_alpha;

        if (std::abs(f) < tolerance) {
            break;
        }

        alpha_est = alpha_est - f / df;
        alpha_est = std::max(alpha_est, detail::NEWTON_RAPHSON_TOLERANCE);  // Ensure positive
    }

    double beta_est = alpha_est / mean_x;

    // Update parameters
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha_est;
    beta_ = beta_est;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==============================================================================
// 20. PRIVATE UTILITY METHODS
//==============================================================================

// computeDigamma and computeTrigamma removed: use detail::digamma / detail::trigamma
// from math_utils.h. These shared implementations are more accurate (A&S §6.3/6.4)
// and improvements propagate automatically to Gamma, Beta, and StudentT MLE.

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

}  // namespace stats

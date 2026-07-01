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
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace stats {

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
    cache_valid_ = other.cache_valid_;
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
    cache_valid_ = other.cache_valid_;
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
}

GammaDistribution& GammaDistribution::operator=(GammaDistribution&& other) noexcept {
    if (this != &other) {
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        // Preserve noexcept move assignment by avoiding lock acquisition.
        // As with standard containers, callers must not concurrently access
        // an object while it is being move-assigned. Cache is invalidated and
        // rebuilt on next read rather than updated unsafely here.
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
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return scale_;  // snapshot + early return under unique_lock
    }
    return scale_;
}

double GammaDistribution::getMean() const {
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

double GammaDistribution::getVariance() const {
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

double GammaDistribution::getSkewness() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return detail::TWO / sqrtAlpha_;  // snapshot + early return under unique_lock
    }
    return detail::TWO / sqrtAlpha_;
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
    // Copy current beta for validation (thread-safe)
    double currentBeta;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentBeta = beta_;
    }

    // Validate parameters
    validateParameters(alpha, currentBeta);

    // Update with unique lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

void GammaDistribution::setBeta(double beta) {
    // Copy current alpha for validation (thread-safe)
    double currentAlpha;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentAlpha = alpha_;
    }

    // Validate parameters
    validateParameters(currentAlpha, beta);

    // Update with unique lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
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
    // Copy current beta for validation (thread-safe)
    double currentBeta;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentBeta = beta_;
    }

    auto validation = validateGammaParameters(alpha, currentBeta);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    alpha_ = alpha;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult GammaDistribution::trySetBeta(double beta) noexcept {
    // Copy current alpha for validation (thread-safe)
    double currentAlpha;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentAlpha = alpha_;
    }

    auto validation = validateGammaParameters(currentAlpha, beta);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
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

double GammaDistribution::getProbability(double x) const {
    if (x < detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        // Snapshot under unique_lock — eliminates TOCTOU gap.
        // Inline log-space formula instead of calling getLogProbability to
        // avoid re-entrant shared_lock acquisition on the same mutex.
        const double a = alpha_, b = beta_;
        const double alb = alphaLogBeta_, lga = logGammaAlpha_, am1 = alphaMinusOne_;
        if (x == detail::ZERO_DOUBLE) {
            return (a < detail::ONE)    ? std::numeric_limits<double>::infinity()
                   : (a == detail::ONE) ? b
                                        : detail::ZERO_DOUBLE;
        }
        return std::exp(alb - lga + am1 * std::log(x) - b * x);
    }

    // Handle special case x = 0
    if (x == detail::ZERO_DOUBLE) {
        return (alpha_ < detail::ONE)    ? std::numeric_limits<double>::infinity()
               : (alpha_ == detail::ONE) ? beta_
                                         : detail::ZERO_DOUBLE;
    }

    // Inline log-space computation (avoids re-entrant lock via getLogProbability).
    return std::exp(alphaLogBeta_ - logGammaAlpha_ + alphaMinusOne_ * std::log(x) - beta_ * x);
}

double GammaDistribution::getLogProbability(double x) const {
    if (x < detail::ZERO_DOUBLE) {
        return detail::NEGATIVE_INFINITY;
    }

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        // Snapshot under unique_lock — eliminates TOCTOU gap.
        const double a = alpha_, b = beta_;
        const double lb = logBeta_, alb = alphaLogBeta_, lga = logGammaAlpha_, am1 = alphaMinusOne_;
        if (x == detail::ZERO_DOUBLE) {
            if (a < detail::ONE)
                return std::numeric_limits<double>::infinity();
            else if (a == detail::ONE)
                return lb;
            else
                return detail::MIN_LOG_PROBABILITY;
        }
        return alb - lga + am1 * std::log(x) - b * x;
    }

    // Handle special case x = 0
    if (x == detail::ZERO_DOUBLE) {
        if (alpha_ < detail::ONE) {
            return std::numeric_limits<double>::infinity();
        } else if (alpha_ == detail::ONE) {
            return logBeta_;  // log(β)
        } else {
            return detail::MIN_LOG_PROBABILITY;
        }
    }

    // General case: log(f(x)) = α*log(β) - log(Γ(α)) + (α-1)*log(x) - βx
    return alphaLogBeta_ - logGammaAlpha_ + alphaMinusOne_ * std::log(x) - beta_ * x;
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

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        // Snapshot under unique_lock — eliminates TOCTOU gap.
        const double a = alpha_, b = beta_;
        return detail::gamma_p(a, b * x);
    }
    // Use regularized incomplete gamma function P(α, βx)
    return detail::gamma_p(alpha_, beta_ * x);
}

double GammaDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p == detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    // Use Newton-Raphson iteration with bracketing
    return computeQuantile(p);
}

double GammaDistribution::sample(std::mt19937& rng) const {
    // Snapshot alpha, beta under the appropriate lock to avoid TOCTOU.
    double cached_alpha, cached_beta;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
            if (!cache_valid_)
                updateCacheUnsafe();
            cached_alpha = alpha_;
            cached_beta = beta_;
        } else {
            cached_alpha = alpha_;
            cached_beta = beta_;
        }
    }

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
    // Snapshot alpha, beta under the appropriate lock to avoid TOCTOU.
    double cached_alpha, cached_beta;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        if (!cache_valid_) {
            lock.unlock();
            std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
            if (!cache_valid_)
                updateCacheUnsafe();
            cached_alpha = alpha_;
            cached_beta = beta_;  // snapshot under unique_lock
        } else {
            cached_alpha = alpha_;
            cached_beta = beta_;  // snapshot under shared_lock
        }
    }

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
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return isExponential_;  // snapshot + early return under unique_lock
    }
    return isExponential_;
}

bool GammaDistribution::isChiSquaredDistribution() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return isChiSquared_;  // snapshot + early return under unique_lock
    }
    return isChiSquared_;
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
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        // Snapshot under unique_lock — eliminates TOCTOU gap.
        const double a = alpha_, lb = logBeta_, lga = logGammaAlpha_, da = digammaAlpha_;
        return a - lb + lga + (detail::ONE - a) * da;
    }
    // H(X) = α - log(β) + log(Γ(α)) + (1-α)ψ(α)
    return alpha_ - logBeta_ + logGammaAlpha_ + (detail::ONE - alpha_) * digammaAlpha_;
}

bool GammaDistribution::canUseNormalApproximation() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_)
            updateCacheUnsafe();
        return isLargeAlpha_;  // snapshot + early return under unique_lock
    }
    return isLargeAlpha_;
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
            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    dist.updateCacheUnsafe();
                // Snapshot under unique_lock.
                const double alpha = dist.alpha_, beta = dist.beta_;
                const double lga = dist.logGammaAlpha_, alb = dist.alphaLogBeta_;
                const double am1 = dist.alphaMinusOne_;
                dist.getProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta, lga, alb, am1);
                return;  // early return; ulock releases here
            }
            // Cache hit — snapshot under shared_lock, then unlock.
            const double alpha = dist.alpha_, beta = dist.beta_;
            const double lga = dist.logGammaAlpha_, alb = dist.alphaLogBeta_;
            const double am1 = dist.alphaMinusOne_;
            lock.unlock();
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

            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            [[maybe_unused]] double cached_alpha;
            double cached_beta, cached_log_gamma_alpha, cached_alpha_log_beta,
                cached_alpha_minus_one;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                } else {
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                }
            }

            // Chunk so each parallel task runs the SIMD log+exp pipeline
            // rather than computing log(x) per element in each task.
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    dist.getProbabilityBatchUnsafeImpl(
                        vals.data() + start, res.data() + start, len, cached_alpha, cached_beta,
                        cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
                });
            } else {
                dist.getProbabilityBatchUnsafeImpl(vals.data(), res.data(), count, cached_alpha,
                                                   cached_beta, cached_log_gamma_alpha,
                                                   cached_alpha_log_beta, cached_alpha_minus_one);
            }
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

            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            double cached_alpha, cached_beta, cached_log_gamma_alpha;
            double cached_alpha_log_beta, cached_alpha_minus_one;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                } else {
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                }
            }

            // Chunk into SIMD-sized slices so pool tasks use the SIMD pipeline.
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
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
            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    dist.updateCacheUnsafe();
                // Snapshot under unique_lock.
                const double alpha = dist.alpha_, beta = dist.beta_;
                const double lga = dist.logGammaAlpha_, alb = dist.alphaLogBeta_;
                const double am1 = dist.alphaMinusOne_;
                dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta, lga, alb, am1);
                return;  // early return; ulock releases here
            }
            // Cache hit — snapshot under shared_lock, then unlock.
            const double alpha = dist.alpha_, beta = dist.beta_;
            const double lga = dist.logGammaAlpha_, alb = dist.alphaLogBeta_;
            const double am1 = dist.alphaMinusOne_;
            lock.unlock();
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

            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            double cached_alpha, cached_beta, cached_log_gamma_alpha;
            double cached_alpha_log_beta, cached_alpha_minus_one;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                } else {
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                }
            }

            // Chunk so each parallel task runs the SIMD log pipeline.
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    dist.getLogProbabilityBatchUnsafeImpl(
                        vals.data() + start, res.data() + start, len, cached_alpha, cached_beta,
                        cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
                });
            } else {
                dist.getLogProbabilityBatchUnsafeImpl(
                    vals.data(), res.data(), count, cached_alpha, cached_beta,
                    cached_log_gamma_alpha, cached_alpha_log_beta, cached_alpha_minus_one);
            }
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

            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            double cached_alpha, cached_beta, cached_log_gamma_alpha;
            double cached_alpha_log_beta, cached_alpha_minus_one;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                } else {
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                    cached_log_gamma_alpha = dist.logGammaAlpha_;
                    cached_alpha_log_beta = dist.alphaLogBeta_;
                    cached_alpha_minus_one = dist.alphaMinusOne_;
                }
            }

            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
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
            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_)
                    dist.updateCacheUnsafe();
                // Snapshot under unique_lock.
                const double alpha = dist.alpha_, beta = dist.beta_;
                dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta);
                return;  // early return; ulock releases here
            }
            // Cache hit — snapshot under shared_lock, then unlock.
            const double alpha = dist.alpha_, beta = dist.beta_;
            lock.unlock();
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, alpha, beta);
        },
        [](const GammaDistribution& dist, std::span<const double> vals, std::span<double> res) {
            // Parallel-SIMD lambda: should use ParallelUtils
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            double cached_alpha, cached_beta;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                } else {
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                }
            }

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (x <= detail::ZERO_DOUBLE) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else {
                        res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                    }
                }
            }
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

            // Snapshot cache fields under the appropriate lock — no TOCTOU gap.
            double cached_alpha, cached_beta;
            {
                std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    lock.unlock();
                    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                    if (!dist.cache_valid_)
                        dist.updateCacheUnsafe();
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                } else {
                    cached_alpha = dist.alpha_;
                    cached_beta = dist.beta_;
                }
            }

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (x <= detail::ZERO_DOUBLE) {
                    res[i] = detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::gamma_p(cached_alpha, cached_beta * x);
                }
            });
            pool.waitForAll();
        });
}

//==========================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==========================================================================

//==========================================================================
// 15. COMPARISON OPERATORS
//==========================================================================

bool GammaDistribution::operator==(const GammaDistribution& other) const {
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
            alpha = std::stod(alpha_str);

            // Read "beta=value)"
            is >> temp;
            if (temp.starts_with("beta=")) {
                size_t beta_equals_pos = temp.find('=');
                size_t close_paren_pos = temp.find(')');
                if (beta_equals_pos != std::string::npos && close_paren_pos != std::string::npos) {
                    std::string beta_str =
                        temp.substr(beta_equals_pos + detail::ONE_INT,
                                    close_paren_pos - beta_equals_pos - detail::ONE_INT);
                    beta = std::stod(beta_str);

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
//             fixup: x <= 0 → MIN_LOG_PROBABILITY (finite proxy; matches single-value method)
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
                                                      std::size_t count,
                                                      [[maybe_unused]] double alpha, double beta,
                                                      double log_gamma_alpha, double alpha_log_beta,
                                                      double alpha_minus_one) const noexcept {
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
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (values[i] == detail::ZERO_DOUBLE) {
                fixup_zero(i);
            } else {
                results[i] = std::exp(alpha_log_beta - log_gamma_alpha +
                                      alpha_minus_one * std::log(values[i]) - beta * values[i]);
            }
        }
        return;
    }

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
    // Step 6: results = exp(log-space result)
    arch::simd::VectorOps::vector_exp(results, results, count);
    // Fixup: x < 0 → 0; x = 0 → depends on alpha (EDGE-4).
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (values[i] == detail::ZERO_DOUBLE) {
            fixup_zero(i);
        }
    }
}

void GammaDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count,
                                                         [[maybe_unused]] double alpha, double beta,
                                                         double log_gamma_alpha,
                                                         double alpha_log_beta,
                                                         double alpha_minus_one) const noexcept {
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] <= detail::ZERO_DOUBLE) {
                results[i] = detail::NEGATIVE_INFINITY;
            } else {
                results[i] = alpha_log_beta - log_gamma_alpha +
                             alpha_minus_one * std::log(values[i]) - beta * values[i];
            }
        }
        return;
    }

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
    // Fixup: x <= 0 is outside support. Use MIN_LOG_PROBABILITY (finite proxy for -inf)
    // to match the single-value getLogProbability() behaviour for alpha > 1 at x = 0,
    // which avoids -inf propagation in log-probability summation algorithms.
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] <= detail::ZERO_DOUBLE) {
            results[i] = detail::MIN_LOG_PROBABILITY;
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
            if (values[i] <= detail::ZERO_DOUBLE) {
                results[i] = detail::ZERO_DOUBLE;
            } else {
                results[i] = detail::gamma_p(alpha, beta * values[i]);
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
        if (values[i] <= detail::ZERO_DOUBLE) {
            results[i] = detail::ZERO_DOUBLE;
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
    return detail::gamma_p(a, x) * std::exp(std::lgamma(a));
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
    // Quantile function using Newton-Raphson iteration with initial guess
    if (p <= detail::ZERO_DOUBLE) {
        return detail::ZERO_DOUBLE;
    }
    if (p >= detail::ONE) {
        return std::numeric_limits<double>::infinity();
    }

    // Initial guess.
    //
    // Wilson-Hilferty (WH) is reliable for moderate p but can produce a
    // negative value when alpha > 1 and p is very small (z << 0 with
    // small h makes the cube negative).  Clamping a negative WH result to
    // NEWTON_RAPHSON_TOLERANCE (1e-10) then causes the first Newton step
    // to shoot x to ~p/pdf(1e-10) ~ 1e9, after which the PDF underflows
    // and the solver exits without converging.
    //
    // Fix: when WH is negative use the small-x asymptotic expansion of the
    // Gamma CDF: P(alpha,beta*x) ~ (beta*x)^alpha / (alpha * Gamma(alpha))
    // => x ~ (p * Gamma(alpha+1))^(1/alpha) / beta.
    double initial_guess;
    if (alpha_ > detail::ONE) {
        double h = detail::TWO / (detail::NINE * alpha_);
        double z = detail::inverse_normal_cdf(p);
        double wh = detail::ONE - h + z * std::sqrt(h);
        if (wh > detail::ZERO_DOUBLE) {
            initial_guess = alpha_ * std::pow(wh, 3) / beta_;
        } else {
            // WH failed; small-p asymptotic: x ~ (p * Gamma(alpha+1))^(1/alpha) / beta
            initial_guess =
                std::pow(p * std::exp(std::lgamma(alpha_ + detail::ONE)), detail::ONE / alpha_) /
                beta_;
        }
    } else {
        // For alpha <= 1, use exponential approximation
        initial_guess = -std::log(detail::ONE - p) / beta_;
    }

    // Newton-Raphson iteration with positive-x guard.
    double x = std::max(initial_guess, detail::NEWTON_RAPHSON_TOLERANCE);
    const double tolerance = detail::HIGH_PRECISION_TOLERANCE;
    const int max_iterations = detail::MAX_NEWTON_ITERATIONS;

    for (int i = 0; i < max_iterations; ++i) {
        double cdf = getCumulativeProbability(x);
        double pdf = getProbability(x);

        if (std::abs(cdf - p) < tolerance) {
            break;
        }

        if (pdf < detail::ULTRA_SMALL_THRESHOLD) {
            // PDF underflow: fall back to bisection between current x and a
            // simple upper bound so the solver doesn't stall.
            double lo = detail::NEWTON_RAPHSON_TOLERANCE, hi = x;
            if (cdf < p)
                hi = x * 10.0;  // need to go higher
            for (int j = 0; j < 60; ++j) {
                double mid = (lo + hi) * detail::HALF;
                double cmid = getCumulativeProbability(mid);
                if (std::abs(cmid - p) < tolerance) {
                    x = mid;
                    break;
                }
                if (cmid < p)
                    lo = mid;
                else
                    hi = mid;
                if (hi - lo < tolerance) {
                    x = (lo + hi) * detail::HALF;
                    break;
                }
            }
            break;
        }

        double delta = (cdf - p) / pdf;
        x = std::max(x - delta, x * 0.1);  // Ensure x stays positive

        if (std::abs(delta) < tolerance * x) {
            break;
        }
    }

    return x;
}

double GammaDistribution::sampleMarsagliaTsang(std::mt19937& rng) const noexcept {
    // Marsaglia-Tsang "squeeze" method for α ≥ 1
    // This is a fast rejection sampling method

    std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);
    std::normal_distribution<double> normal(detail::ZERO_DOUBLE, detail::ONE);

    const double d = alpha_ - detail::ONE / detail::THREE;
    const double c = detail::ONE / std::sqrt(detail::NINE * d);

    while (true) {
        double x, v;

        do {
            x = normal(rng);
            v = detail::ONE + c * x;
        } while (v <= detail::ZERO_DOUBLE);

        v = v * v * v;
        double u = uniform(rng);

        // Quick accept
        if (u < detail::ONE - 0.0331 * (x * x) * (x * x)) {
            return d * v / beta_;
        }

        // Quick reject
        if (std::log(u) < detail::HALF * x * x + d * (detail::ONE - v + std::log(v))) {
            return d * v / beta_;
        }
    }
}

double GammaDistribution::sampleAhrensDieter(std::mt19937& rng) const noexcept {
    // Ahrens-Dieter acceptance-rejection method for α < 1
    std::uniform_real_distribution<double> uniform(detail::ZERO_DOUBLE, detail::ONE);

    const double b = (detail::E + alpha_) / detail::E;

    while (true) {
        double u = uniform(rng);
        double p = b * u;

        if (p <= detail::ONE) {
            double x = std::pow(p, detail::ONE / alpha_);
            double u2 = uniform(rng);
            if (u2 <= std::exp(-x)) {
                return x / beta_;
            }
        } else {
            double x = -std::log((b - p) / alpha_);
            double u2 = uniform(rng);
            if (u2 <= std::pow(x, alpha_ - detail::ONE)) {
                return x / beta_;
            }
        }
    }
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

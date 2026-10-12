#include "libstats/distributions/inverse_gamma.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace stats {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

/// Widest log-x bracket the quantile solver can return: e^-745 is the smallest
/// positive (subnormal) double, e^709 the largest finite one.
constexpr double kLogXLo = -745.0;
constexpr double kLogXHi = 709.0;

/// Halvings of [kLogXLo, kLogXHi]; 1454 / 2^80 is far below one ulp of any
/// representable log-x, so the bracket collapses onto adjacent doubles.
constexpr int kBisectIterations = 80;

// x is exact, but the argument the incomplete gamma and the density take, β/x, is rounded, and
// at a large shape the result amplifies that rounding: d log f / d log(β/x) = α − β/x, ~3e9 at
// α = 1e16 thirty standard deviations out, so half an ulp of β/x moved the density 1.6e-7 and
// the CDF 7.4e-8 relative (Gamma(1e16, 1) is clean at the mirrored points: there βx is x).
// Each rounding is recovered exactly with an fma and carried to second order: the first-order
// carry left α·(δ/y)²/2 for a relative rounding δ/y ≤ ε/2, 6e-17 at α = 1e16 but 6e-13 at 1e20,
// and its CDF erred 1100 ulp at InverseGamma(1e20, 3), 1.7e5 at 1e22 (Fix7C).

// log f(x) − (what the delegate returns at inv_x = fl(1/x), minus 2·log x). The delegate takes
// y = fl(β·inv_x); with e = 1 − inv_x·x (log(1/x) − log inv_x to first order), exact by fma.
// Below α = STIRLING_PREFACTOR_SHAPE it evaluates (α − 1)·log inv_x − y, which takes y only
// linearly and uncorrected, so the argument's whole error d = β/x − y counts, exact up to the
// final division. From there it evaluates log P(α, Y) − log inv_x at Y = β·inv_x, exact, with
// its own product's rounding e_p = Y − y already carried (gamma.cpp, withRateProductRounding),
// so what remains is log P(α, β/x) − log P(α, Y) + log(inv_x·x) = −(α − 1)·log1p(−e) − d with
// d = β/x − Y = Y·e/(1 − e). Expanded about y, so that the terms of size α·e cancel exactly:
// e·((α − y − 1) + e·((α − 1)/2 − y) − e_p), whose next term is ~α·e³ (4e-29 at α = 1e20). The
// first-order (α − y)·(d/y) − e it replaces missed ~α·e²/2, 6e-13 at α = 1e20 (Fix7C). Coupled
// to that delegate: the β = 3 rows of the H4 guard in test_contract_gates fail if the two
// disagree.
[[nodiscard]] double reciprocalArgumentCorrection(double x, double inv_x, double alpha,
                                                  double beta) noexcept {
    const double y = beta * inv_x;
    if (!(y >= std::numeric_limits<double>::min()) || !std::isfinite(y))
        return detail::ZERO_DOUBLE;  // the delegate's own edge handling stands
    const double e = std::fma(-inv_x, x, detail::ONE);
    if (alpha < detail::STIRLING_PREFACTOR_SHAPE)
        return (alpha - detail::ONE) * e - std::fma(-y, x, beta) / x;
    const double e_p = std::fma(beta, inv_x, -y);
    return e * (((alpha - y) - detail::ONE) + e * (detail::HALF * (alpha - detail::ONE) - y) - e_p);
}

// Q(α, t) or P(α, t) at t = fl(β/x), corrected to t + δ, δ = β/x − t = r/x with the exact
// residual r = β − t·x, so δ/t = r/(t·x) = r/(β − r). P gains and Q loses the density's integral
// over [t, t + δ], f(t)·δ·(expm1(u)/u − b·δ²/3) with u = δ·(α − 1 − t)/t and b = (α − 1)/(2t²),
// t·f(t) = exp(log_gamma_prefactor(α, t)): gamma.cpp's rateProductIncrement, which derives it.
// Carried in the log to first order until Fix7C, which left 1100 ulp at InverseGamma(1e20, 3)
// and 1.7e5 at 1e22. The ratio is formed in the log, so an exp(lgp) below DBL_MIN
// keeps it.
[[nodiscard]] double correctedIncompleteGamma(double value, bool upper, double alpha, double beta,
                                              double x, double t) noexcept {
    const double r = std::fma(-t, x, beta);
    // value > 0, not >= DBL_MIN: a subnormal P or Q keeps ~47 bits near 1e-309, and skipping the
    // correction there left it 1.8e-7 off at InverseGamma(1e16, 3) (M1 survival sweep).
    if (r == detail::ZERO_DOUBLE || !(value > detail::ZERO_DOUBLE))
        return value;
    const double rel = r / (beta - r);
    const double ratio = std::exp(detail::log_gamma_prefactor(alpha, t) - std::log(value));
    if (ratio == detail::ZERO_DOUBLE)
        return value;  // and u may overflow (t ≫ α): see rateProductIncrement
    const double u = rel * ((alpha - t) - detail::ONE);
    const double phi = u == detail::ZERO_DOUBLE ? detail::ONE : std::expm1(u) / u;
    const double increment = rel * ratio * (phi - (alpha - detail::ONE) * rel * rel / 6.0);
    // Clamped as gamma.cpp's callers clamp (the expansion fails from α ~ 1e32).
    return std::clamp(std::fma(upper ? -value : value, increment, value), detail::ZERO_DOUBLE,
                      detail::ONE);
}

// log f(x) where 1/x overflows (0 < x < 1/DBL_MAX, ~5.6e-309), from t = β/x, which is finite
// there for β < 1: log f = log_gamma_prefactor(α, t) − log x, with t's rounding δ = r/x carried
// as α·log1p(δ/t) − δ to second order. Read as −inf, the reciprocal path's limit, until Fix7C,
// though the density need not be small there: InverseGamma(1e10, 1e-300).pdf(1.0003e-310) =
// 1.35e119. Where t overflows too, −t dominates the log density for any α below ~t/709.
[[nodiscard]] double logDensityPastReciprocalRange(double x, double alpha, double beta) noexcept {
    const double t = beta / x;
    if (!(t <= std::numeric_limits<double>::max()))
        return detail::NEGATIVE_INFINITY;
    const double r = std::fma(-t, x, beta);
    const double rel = r / (beta - r);
    const double log_value = detail::log_gamma_prefactor(alpha, t) - std::log(x);
    return std::fma(rel, alpha - t, log_value - detail::HALF * alpha * rel * rel);
}

}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

// Validate-and-return helpers so a bad parameter throws before any member is
// constructed (mirrors ErlangDistribution::requireValidLambda).
static double requireValidShape(double alpha) {
    if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0)
        throw std::invalid_argument("Shape parameter alpha must be a positive finite number");
    return alpha;
}

static double requireValidScale(double beta) {
    if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0)
        throw std::invalid_argument("Scale parameter beta must be a positive finite number");
    return beta;
}

InverseGammaDistribution::InverseGammaDistribution(double alpha, double beta)
    : DistributionBase(),
      alpha_(requireValidShape(alpha)),
      beta_(requireValidScale(beta)),
      // This class's SCALE is the delegate's RATE -- the same number, passed
      // through unchanged. See the parameterization note in inverse_gamma.h.
      gamma_(alpha_, beta_) {}

InverseGammaDistribution::InverseGammaDistribution(const InverseGammaDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    gamma_ = other.gamma_;
}

InverseGammaDistribution& InverseGammaDistribution::operator=(
    const InverseGammaDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        gamma_ = other.gamma_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

InverseGammaDistribution::InverseGammaDistribution(InverseGammaDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    alpha_ = other.alpha_;
    beta_ = other.beta_;
    gamma_ = std::move(other.gamma_);
    other.alpha_ = detail::ONE;
    other.beta_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

InverseGammaDistribution& InverseGammaDistribution::operator=(
    InverseGammaDistribution&& other) noexcept {
    if (this != &other) {
        // Both locks, as copy-assignment takes, in std::lock order; the source is written, so
        // exclusively (#184).
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        gamma_ = std::move(other.gamma_);
        other.alpha_ = detail::ONE;
        other.beta_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        other.atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

InverseGammaDistribution InverseGammaDistribution::createUnchecked(double alpha,
                                                                   double beta) noexcept {
    return InverseGammaDistribution(alpha, beta, true);
}

InverseGammaDistribution::InverseGammaDistribution(double alpha, double beta,
                                                   bool /*bypassValidation*/) noexcept
    : DistributionBase(), alpha_(alpha), beta_(beta), gamma_(alpha, beta) {
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

double InverseGammaDistribution::getAlphaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicAlpha_.load(std::memory_order_acquire);
    return getAlpha();
}

double InverseGammaDistribution::getBetaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicBeta_.load(std::memory_order_acquire);
    return getBeta();
}

void InverseGammaDistribution::setAlpha(double alpha) {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        validateParameters(alpha, beta_);
        alpha_ = alpha;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetAlpha(alpha);
    }
}

void InverseGammaDistribution::setBeta(double beta) {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        validateParameters(alpha_, beta);
        beta_ = beta;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetBeta(beta);  // SCALE here == RATE there
    }
}

void InverseGammaDistribution::setParameters(double alpha, double beta) {
    validateParameters(alpha, beta);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        alpha_ = alpha;
        beta_ = beta;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetParameters(alpha, beta);
    }
}

// The trySet* validate and assign under one lock rather than calling the throwing setters,
// which would end in std::terminate from these noexcept functions; the delegate is synced
// outside the lock, as in the setters.
VoidResult InverseGammaDistribution::trySetAlpha(double alpha) noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto v = validateInverseGammaParameters(alpha, beta_);
        if (v.isError())
            return v;
        alpha_ = alpha;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetAlpha(alpha);
    }
    return VoidResult::ok({});
}

VoidResult InverseGammaDistribution::trySetBeta(double beta) noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        auto v = validateInverseGammaParameters(alpha_, beta);
        if (v.isError())
            return v;
        beta_ = beta;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetBeta(beta);  // SCALE here == RATE there
    }
    return VoidResult::ok({});
}

VoidResult InverseGammaDistribution::trySetParameters(double alpha, double beta) noexcept {
    auto v = validateInverseGammaParameters(alpha, beta);
    if (v.isError())
        return v;
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        alpha_ = alpha;
        beta_ = beta;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetParameters(alpha, beta);
    }
    return VoidResult::ok({});
}

VoidResult InverseGammaDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateInverseGammaParameters(alpha_, beta_);
}

//==============================================================================
// 4. MOMENTS
//==============================================================================

double InverseGammaDistribution::getMean() const {
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    if (a <= detail::ONE)
        return kNaN;  // undefined
    return b / (a - detail::ONE);
}

double InverseGammaDistribution::getVariance() const {
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    if (a <= detail::TWO)
        return kNaN;  // undefined
    const double am1 = a - detail::ONE;
    return (b * b) / (am1 * am1 * (a - detail::TWO));
}

double InverseGammaDistribution::getSkewness() const {
    double a;
    withCacheSnapshot([&] { a = alpha_; });
    if (a <= 3.0)
        return kNaN;  // undefined
    return 4.0 * std::sqrt(a - detail::TWO) / (a - 3.0);
}

double InverseGammaDistribution::getKurtosis() const {
    double a;
    withCacheSnapshot([&] { a = alpha_; });
    if (a <= 4.0)
        return kNaN;  // undefined
    return (30.0 * a - 66.0) / ((a - 3.0) * (a - 4.0));
}

double InverseGammaDistribution::getMode() const {
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    return b / (a + detail::ONE);
}

double InverseGammaDistribution::getEntropy() const {
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    // H = alpha + ln(beta) + lnGamma(alpha) - (1+alpha) psi(alpha)
    return a + std::log(b) + detail::lgamma(a) - (detail::ONE + a) * detail::digamma(a);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

// Classify x for the reciprocal transform. Returns true only for an ordinary
// interior point whose reciprocal is a finite positive double; otherwise
// edge_logpdf holds the correct log-density and the delegate must be skipped.
//
// The edges the reciprocal creates are the whole reason this layer exists:
//   x  = NaN                -> NaN                    (propagate)
//   x <= 0 (incl. -inf)     -> -inf   (outside the support)
//   x  = +inf               -> -inf   (density ~ x^(-alpha-1) -> 0)
//   x < ~5.6e-309           -> -inf   1/x overflows, so +inf must never reach the
//                                     delegate. -inf is not the density there for a
//                                     small beta: callers take a positive finite x
//                                     to logDensityPastReciprocalRange instead.
bool InverseGammaDistribution::reciprocalIsUsable(double x, double& inv_x,
                                                  double& edge_logpdf) noexcept {
    if (std::isnan(x)) {
        edge_logpdf = kNaN;
        return false;
    }
    if (x <= detail::ZERO_DOUBLE || x == kInf) {
        edge_logpdf = detail::NEGATIVE_INFINITY;
        return false;
    }
    inv_x = detail::ONE / x;
    if (!std::isfinite(inv_x)) {
        edge_logpdf = detail::NEGATIVE_INFINITY;
        return false;
    }
    edge_logpdf = detail::ZERO_DOUBLE;  // unused
    return true;
}

// CDF(x) = Q(alpha, beta/x), the regularized UPPER incomplete gamma.
//
// Derivation: Y = 1/X with X ~ Gamma(alpha, rate beta), so
//   P(Y <= x) = P(X >= 1/x) = Q(alpha, beta * (1/x)) = Q(alpha, beta/x).
// detail::gamma_q evaluates its own continued fraction for Q whenever
// beta/x > alpha+1 -- which is exactly the x -> 0 tail where the CDF is tiny
// and `1 - CDF_Gamma(1/x)` would return literally zero. The complement is
// never formed here. See inverse_gamma.h.
double InverseGammaDistribution::cdfImpl(double x, double alpha, double beta) noexcept {
    if (std::isnan(x))
        return kNaN;
    if (x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;  // includes -inf
    if (x == kInf)
        return detail::ONE;

    const double t = beta / x;
    if (!std::isfinite(t))
        return detail::ZERO_DOUBLE;              // x subnormal-tiny: Q(alpha, +inf) = 0
    if (t < std::numeric_limits<double>::min())  // β/x below DBL_MIN: log space (#216)
        return detail::gamma_q_from_log_x(alpha, std::log(beta) - std::log(x));
    return correctedIncompleteGamma(detail::gamma_q(alpha, t), true, alpha, beta, x, t);
}

// Survival 1 - CDF(x) = P(alpha, beta/x), the LOWER regularized incomplete
// gamma -- again computed, not subtracted. In the upper tail (x large) beta/x
// is small, so gamma_p takes its series branch and returns the tiny value with
// full relative precision.
double InverseGammaDistribution::sfImpl(double x, double alpha, double beta) noexcept {
    if (std::isnan(x))
        return kNaN;
    if (x <= detail::ZERO_DOUBLE)
        return detail::ONE;
    if (x == kInf)
        return detail::ZERO_DOUBLE;

    const double t = beta / x;
    if (!std::isfinite(t))
        return detail::ONE;
    if (t < std::numeric_limits<double>::min())  // β/x below DBL_MIN: log space (#216)
        return detail::gamma_p_from_log_x(alpha, std::log(beta) - std::log(x));
    return correctedIncompleteGamma(detail::gamma_p(alpha, t), false, alpha, beta, x, t);
}

double InverseGammaDistribution::getLogProbability(double x) const {
    double inv_x = detail::ZERO_DOUBLE;
    double edge = detail::ZERO_DOUBLE;
    if (!reciprocalIsUsable(x, inv_x, edge)) {
        if (!(x > detail::ZERO_DOUBLE) || x == kInf)
            return edge;
        double a, b;
        withCacheSnapshot([&] {
            a = alpha_;
            b = beta_;
        });
        return logDensityPastReciprocalRange(x, a, b);
    }
    // Jacobian of Y = 1/X is 1/x^2, i.e. -2 ln x in log space. Exact in log
    // space -- no cancellation, and no intermediate density to overflow.
    // The delegate is evaluated inside this object's snapshot: every setter updates it under
    // our unique lock (ours, then the delegate's), so under our lock it holds (alpha_, beta_),
    // and the correction and the delegate's value come from one state.
    double a, b, delegate_logpdf;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
        delegate_logpdf = gamma_.getLogProbability(inv_x);
    });
    return delegate_logpdf + reciprocalArgumentCorrection(x, inv_x, a, b) -
           detail::TWO * std::log(x);
}

double InverseGammaDistribution::getProbability(double x) const {
    const double lp = getLogProbability(x);
    if (std::isnan(lp))
        return kNaN;
    return std::exp(lp);  // exp(-inf) == 0
}

double InverseGammaDistribution::getCumulativeProbability(double x) const {
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    return cdfImpl(x, a, b);
}

double InverseGammaDistribution::getSurvivalProbability(double x) const {
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });
    return sfImpl(x, a, b);
}

double InverseGammaDistribution::getQuantile(double p) const {
    if (std::isnan(p))
        return std::numeric_limits<double>::quiet_NaN();  // NaN in, NaN out (AR D3)
    if (p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability p must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return kInf;

    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
    });

    // Safeguarded bisection on t = ln(x), with the residual always taken on
    // the small side of the probability scale: the CDF below the median, the
    // survival function above it. Preferred over 1/delegate.getQuantile(1-p),
    // which would solve the delegate at a probability near 1 -- precisely
    // where its absolute stopping rule is least informative.
    const bool upper = (p > detail::HALF);
    const double target = upper ? (detail::ONE - p) : p;

    auto residual = [&](double t) {
        const double x = std::exp(t);
        return upper ? (target - sfImpl(x, a, b)) : (cdfImpl(x, a, b) - target);
    };

    double lo = kLogXLo;
    double hi = kLogXHi;
    if (residual(lo) >= detail::ZERO_DOUBLE)
        return std::exp(lo);  // quantile is below the smallest positive double
    if (residual(hi) <= detail::ZERO_DOUBLE)
        return kInf;  // true overflow: the quantile exceeds the double range

    for (int i = 0; i < kBisectIterations; ++i) {
        const double mid = detail::HALF * (lo + hi);
        if (residual(mid) < detail::ZERO_DOUBLE)
            lo = mid;
        else
            hi = mid;
    }
    return std::exp(detail::HALF * (lo + hi));
}

double InverseGammaDistribution::sample(std::mt19937& rng) const {
    const double g = gamma_.sample(rng);
    if (!(g > detail::ZERO_DOUBLE))
        return kInf;  // a Gamma draw of exactly 0 maps to the upper tail
    return detail::ONE / g;
}

std::vector<double> InverseGammaDistribution::sample(std::mt19937& rng, size_t n) const {
    // One state per call: copy the delegate under this object's lock (owner, then delegate) and
    // draw all n from the copy. Looping the scalar sample mixed states under a setter (#186).
    std::optional<GammaDistribution> gamma;
    withCacheSnapshot([&] { gamma.emplace(gamma_); });
    std::vector<double> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const double g = gamma->sample(rng);
        out.push_back(g > detail::ZERO_DOUBLE ? detail::ONE / g : kInf);
    }
    return out;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

// Gamma's MLE structure applied to the reciprocals. If Y_i ~ InvGamma(alpha,
// scale beta) then 1/Y_i ~ Gamma(alpha, rate beta) exactly, so fitting a
// GammaDistribution to the reciprocals and lifting the estimates back
// unchanged is the maximum-likelihood estimate of (alpha, beta) -- the
// transformation is a bijection with a parameter-free Jacobian, so it does not
// perturb the likelihood's argmax.
void InverseGammaDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    std::vector<double> reciprocals;
    reciprocals.reserve(values.size());
    for (double v : values) {
        if (!std::isfinite(v) || v <= detail::ZERO_DOUBLE)
            throw std::invalid_argument(
                "All values must be positive and finite for InverseGamma fitting");
        const double r = detail::ONE / v;
        if (!std::isfinite(r) || r <= detail::ZERO_DOUBLE)
            throw std::invalid_argument(
                "Value too small to invert for InverseGamma fitting (reciprocal overflow)");
        reciprocals.push_back(r);
    }

    auto g = GammaDistribution::create(detail::ONE, detail::ONE).unwrap();
    g.fit(reciprocals);

    // getBeta() on the delegate is a RATE; this class's beta is a SCALE. They
    // are the same number under the reciprocal identity -- no conversion.
    setParameters(g.getAlpha(), g.getBeta());
}

void InverseGammaDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                                std::vector<InverseGammaDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void InverseGammaDistribution::reset() noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        alpha_ = detail::ONE;
        beta_ = detail::ONE;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        // Under our lock (ours, then the delegate's, the order updateCacheUnsafe already uses),
        // so concurrent setters update the delegate in the order they update us.
        (void)gamma_.trySetParameters(detail::ONE, detail::ONE);
    }
}

std::string InverseGammaDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<double>::max_digits10);
    oss << "InverseGammaDistribution(alpha=" << alpha_ << ",beta=" << beta_ << ")";
    return oss.str();
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void InverseGammaDistribution::getLogProbability(std::span<const double> values,
                                                 std::span<double> results,
                                                 const detail::PerformanceHint& hint) const {
    if (values.size() != results.size())
        throw std::invalid_argument("Input and output spans must have the same size");
    const std::size_t count = values.size();
    // The delegate sees the scratch buffer, so its own overlap assert cannot see the caller's
    // spans; the fixup pass below re-reads values after results is written (AR D5).
    LIBSTATS_ASSERT_NO_OVERLAP(values.data(), results.data(), count,
                               "InverseGammaDistribution::getLogProbability");
    if (count == 0)
        return;

    // Scratch buffer of reciprocals. It is a distinct allocation, so the
    // delegate is never called with its input and output aliased (#112).
    // Non-usable inputs get a finite placeholder in the support interior and
    // are overwritten in the fixup pass below.
    std::vector<double> scratch(count);
    for (std::size_t i = 0; i < count; ++i) {
        double inv_x = detail::ZERO_DOUBLE;
        double edge = detail::ZERO_DOUBLE;
        scratch[i] = reciprocalIsUsable(values[i], inv_x, edge) ? inv_x : detail::ONE;
    }

    // The delegate batch runs inside this object's snapshot, as in the scalar path, so (a, b) and
    // every delegate value come from one state; a setter waits for the batch. Its worker threads
    // take only the delegate's lock, in the setters' order (ours, then the delegate's).
    double a, b;
    withCacheSnapshot([&] {
        a = alpha_;
        b = beta_;
        gamma_.getLogProbability(std::span<const double>(scratch), results, hint);
    });

    // Re-reading `values` after `results` has been written is exactly what the
    // documented non-overlap contract permits (#112) -- the same shape as the
    // Gaussian and Gamma CDF tail fixups.
    for (std::size_t i = 0; i < count; ++i) {
        double inv_x = detail::ZERO_DOUBLE;
        double edge = detail::ZERO_DOUBLE;
        if (reciprocalIsUsable(values[i], inv_x, edge))
            results[i] = results[i] + reciprocalArgumentCorrection(values[i], inv_x, a, b) -
                         detail::TWO * std::log(values[i]);
        else if (values[i] > detail::ZERO_DOUBLE && values[i] != kInf)
            results[i] = logDensityPastReciprocalRange(values[i], a, b);
        else
            results[i] = edge;
    }
}

void InverseGammaDistribution::getProbability(std::span<const double> values,
                                              std::span<double> results,
                                              const detail::PerformanceHint& hint) const {
    // Route through the log path so batch and scalar agree bit for bit, and so
    // no intermediate Gamma density (astronomically large at tiny x) is ever
    // materialised.
    getLogProbability(values, results, hint);
    for (std::size_t i = 0; i < results.size(); ++i)
        results[i] = std::isnan(results[i]) ? kNaN : std::exp(results[i]);
}

void InverseGammaDistribution::cdfKernel(const double* values, double* results, std::size_t count,
                                         double alpha, double beta) noexcept {
    for (std::size_t i = 0; i < count; ++i)
        results[i] = cdfImpl(values[i], alpha, beta);
}

void InverseGammaDistribution::getCumulativeProbability(std::span<const double> values,
                                                        std::span<double> results,
                                                        const detail::PerformanceHint& hint) const {
    // No delegation here: detail::gamma_q has no vector form (see the DEFERRED
    // note on vector_gamma_q in math_utils.h), and the delegate's own CDF is
    // the wrong tail. Own scalar kernel under autoDispatch.
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const InverseGammaDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const InverseGammaDistribution& dist, const double* vals, double* res,
           std::size_t count) {
            double a, b;
            dist.withCacheSnapshot([&] {
                a = dist.alpha_;
                b = dist.beta_;
            });
            cdfKernel(vals, res, count, a, b);
        },
        [](const InverseGammaDistribution& dist, std::span<const double> vals,
           std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b;
            dist.withCacheSnapshot([&] {
                a = dist.alpha_;
                b = dist.beta_;
            });
            constexpr std::size_t CHUNK = 1024;
            ParallelUtils::parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                cdfKernel(vals.data() + start, res.data() + start, len, a, b);
            });
        },
        [](const InverseGammaDistribution& dist, std::span<const double> vals,
           std::span<double> res, WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b;
            dist.withCacheSnapshot([&] {
                a = dist.alpha_;
                b = dist.beta_;
            });
            constexpr std::size_t CHUNK = 1024;
            pool.parallelForSlices(count, CHUNK, [&](std::size_t start, std::size_t len) {
                cdfKernel(vals.data() + start, res.data() + start, len, a, b);
            });
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool InverseGammaDistribution::operator==(const InverseGammaDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(alpha_ - other.alpha_) <= detail::DEFAULT_TOLERANCE &&
           std::abs(beta_ - other.beta_) <= detail::DEFAULT_TOLERANCE;
}

bool InverseGammaDistribution::operator!=(const InverseGammaDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const InverseGammaDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, InverseGammaDistribution& dist) {
    std::string token;

    is >> token;
    if (!token.starts_with("InverseGammaDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t a_pos = token.find("alpha=");
    const size_t b_pos = token.find("beta=");
    if (a_pos == std::string::npos || b_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t a_comma = token.find(",", a_pos);
    const size_t b_close = token.find(")", b_pos);
    if (a_comma == std::string::npos || b_close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    double alpha, beta;
    try {
        alpha = detail::parse_double(token.substr(a_pos + 6, a_comma - a_pos - 6));
        beta = detail::parse_double(token.substr(b_pos + 5, b_close - b_pos - 5));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(alpha, beta);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void InverseGammaDistribution::updateCacheUnsafe() const noexcept {
    // Keep the delegate in sync: shape alpha_, RATE beta_ (== our SCALE).
    // gamma_'s mutex is independent of ours and gamma_ is private, so this
    // nested acquisition cannot deadlock.
    (void)gamma_.trySetParameters(alpha_, beta_);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicAlpha_.store(alpha_, std::memory_order_release);
    atomicBeta_.store(beta_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

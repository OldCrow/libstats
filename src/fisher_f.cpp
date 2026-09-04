#include "libstats/distributions/fisher_f.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel

#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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

/// Halvings of [kLogXLo, kLogXHi]. 1454 / 2^80 is far below one ulp of any
/// representable log-x, so the bracket collapses onto adjacent doubles.
constexpr int kBisectIterations = 80;

}  // namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

// Validate-and-return helper so a bad parameter throws before any member is
// constructed (mirrors ErlangDistribution::requireValidLambda).
static double requireValidDF(double v, const char* which) {
    if (std::isnan(v) || std::isinf(v) || v <= 0.0) {
        std::string msg = std::string(which) +
                          " degrees of freedom must be a positive finite number";
        throw std::invalid_argument(msg);
    }
    return v;
}

FDistribution::FDistribution(double d1, double d2)
    : DistributionBase(),
      d1_(requireValidDF(d1, "Numerator")),
      d2_(requireValidDF(d2, "Denominator")),
      beta_(d1_ * detail::HALF, d2_ * detail::HALF) {}

FDistribution::FDistribution(const FDistribution& other) : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    d1_ = other.d1_;
    d2_ = other.d2_;
    beta_ = other.beta_;
}

FDistribution& FDistribution::operator=(const FDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        d1_ = other.d1_;
        d2_ = other.d2_;
        beta_ = other.beta_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

FDistribution::FDistribution(FDistribution&& other) noexcept : DistributionBase(std::move(other)) {
    d1_ = other.d1_;
    d2_ = other.d2_;
    beta_ = std::move(other.beta_);
    other.d1_ = detail::ONE;
    other.d2_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

FDistribution& FDistribution::operator=(FDistribution&& other) noexcept {
    if (this != &other) {
        d1_ = other.d1_;
        d2_ = other.d2_;
        beta_ = std::move(other.beta_);
        other.d1_ = detail::ONE;
        other.d2_ = detail::ONE;

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

FDistribution FDistribution::createUnchecked(double d1, double d2) noexcept {
    return FDistribution(d1, d2, true);
}

FDistribution::FDistribution(double d1, double d2, bool /*bypassValidation*/) noexcept
    : DistributionBase(), d1_(d1), d2_(d2), beta_(d1 * detail::HALF, d2 * detail::HALF) {
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

double FDistribution::getD1Atomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicD1_.load(std::memory_order_acquire);
    return getD1();
}

double FDistribution::getD2Atomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicD2_.load(std::memory_order_acquire);
    return getD2();
}

void FDistribution::setD1(double d1) {
    validateParameters(d1, d2_);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        d1_ = d1;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
}

void FDistribution::setD2(double d2) {
    validateParameters(d1_, d2);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        d2_ = d2;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
}

void FDistribution::setParameters(double d1, double d2) {
    validateParameters(d1, d2);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        d1_ = d1;
        d2_ = d2;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
}

VoidResult FDistribution::trySetD1(double d1) noexcept {
    auto v = validateFisherFParameters(d1, d2_);
    if (v.isError())
        return v;
    setD1(d1);
    return VoidResult::ok({});
}

VoidResult FDistribution::trySetD2(double d2) noexcept {
    auto v = validateFisherFParameters(d1_, d2);
    if (v.isError())
        return v;
    setD2(d2);
    return VoidResult::ok({});
}

VoidResult FDistribution::trySetParameters(double d1, double d2) noexcept {
    auto v = validateFisherFParameters(d1, d2);
    if (v.isError())
        return v;
    setParameters(d1, d2);
    return VoidResult::ok({});
}

VoidResult FDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateFisherFParameters(d1_, d2_);
}

//==============================================================================
// 4. MOMENTS
//==============================================================================

double FDistribution::getMean() const {
    double d2;
    withCacheSnapshot([&] { d2 = d2_; });
    if (d2 <= detail::TWO)
        return kNaN;  // undefined
    return d2 / (d2 - detail::TWO);
}

double FDistribution::getVariance() const {
    double d1, d2;
    withCacheSnapshot([&] {
        d1 = d1_;
        d2 = d2_;
    });
    if (d2 <= 4.0)
        return kNaN;  // undefined
    const double num = detail::TWO * d2 * d2 * (d1 + d2 - detail::TWO);
    const double den = d1 * (d2 - detail::TWO) * (d2 - detail::TWO) * (d2 - 4.0);
    return num / den;
}

double FDistribution::getSkewness() const {
    double d1, d2;
    withCacheSnapshot([&] {
        d1 = d1_;
        d2 = d2_;
    });
    if (d2 <= 6.0)
        return kNaN;  // undefined
    const double num = (detail::TWO * d1 + d2 - detail::TWO) * std::sqrt(8.0 * (d2 - 4.0));
    const double den = (d2 - 6.0) * std::sqrt(d1) * std::sqrt(d1 + d2 - detail::TWO);
    return num / den;
}

double FDistribution::getKurtosis() const {
    double d1, d2;
    withCacheSnapshot([&] {
        d1 = d1_;
        d2 = d2_;
    });
    if (d2 <= 8.0)
        return kNaN;  // undefined
    const double num = 12.0 * (d1 * (5.0 * d2 - 22.0) * (d1 + d2 - detail::TWO) +
                               (d2 - 4.0) * (d2 - detail::TWO) * (d2 - detail::TWO));
    const double den = d1 * (d2 - 6.0) * (d2 - 8.0) * (d1 + d2 - detail::TWO);
    return num / den;
}

double FDistribution::getMode() const {
    double d1, d2;
    withCacheSnapshot([&] {
        d1 = d1_;
        d2 = d2_;
    });
    if (d1 <= detail::TWO)
        return detail::ZERO_DOUBLE;  // density is unbounded (d1<2) or maximal (d1=2) at 0
    return ((d1 - detail::TWO) / d1) * (d2 / (d2 + detail::TWO));
}

double FDistribution::getEntropy() const {
    double d1, d2, a, b;
    withCacheSnapshot([&] {
        d1 = d1_;
        d2 = d2_;
        a = a_;
        b = b_;
    });
    // H = ln B(a,b) + ln(d2/d1) - (a-1)psi(a) - (b+1)psi(b) + (a+b)psi(a+b)
    return detail::lbeta(a, b) + std::log(d2 / d1) - (a - detail::ONE) * detail::digamma(a) -
           (b + detail::ONE) * detail::digamma(b) + (a + b) * detail::digamma(a + b);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

// The log-PDF in closed form:
//
//   log f(x) = a ln(d1) + b ln(d2) - ln B(a,b)      <- logPdfConst
//              + (a-1) ln x - (a+b) ln(d1 x + d2)
//
// No subtraction of near-equal quantities occurs anywhere in it, so it is
// accurate across the whole support without any tail steering (unlike the CDF).
double FDistribution::logPdfImpl(double x, double a, double b, double d1, double d2,
                                 double log_pdf_const) noexcept {
    // #103 contract: non-finite input handled first, before any arithmetic.
    if (std::isnan(x))
        return kNaN;
    if (x < detail::ZERO_DOUBLE || x == kInf)
        return detail::NEGATIVE_INFINITY;

    if (x == detail::ZERO_DOUBLE) {
        // Limit at the support edge. (a-1)*ln(0) would be 0*(-inf) = NaN for
        // a == 1 exactly (d1 == 2), so branch on a instead of evaluating it.
        if (a > detail::ONE)
            return detail::NEGATIVE_INFINITY;  // density vanishes at 0
        if (a < detail::ONE)
            return kInf;  // density is unbounded at 0
        return log_pdf_const - (a + b) * std::log(d2);  // a == 1: finite limit
    }

    const double denom = d1 * x + d2;
    if (!std::isfinite(denom))
        return detail::NEGATIVE_INFINITY;  // x so large the density has underflowed

    return log_pdf_const + (a - detail::ONE) * std::log(x) - (a + b) * std::log(denom);
}

// CDF = I_y(a,b) with y = d1 x/(d1 x + d2), evaluated on the small side.
//
// The complement ybar = d2/(d1 x + d2) is formed by division, never as 1 - y:
// as x grows y -> 1 and `1 - y` in double loses every significant bit of the
// complement (the #49 cancellation class). detail::beta_i's own internal
// series/complement switch is `z < (A+1)/(A+B+2)`; we pick the branch that
// satisfies it, so beta_i always takes its direct continued-fraction path.
//
//   y < (a+1)/(a+b+2)  =>  I_y(a,b)              (direct)
//   otherwise          =>  1 - I_ybar(b,a), and  y > (a+1)/(a+b+2) implies
//                          ybar < (b+1)/(a+b+2), so that call is direct too.
//
// The log-beta prefix lgamma(a+b)-lgamma(a)-lgamma(b) is symmetric in (a,b),
// so one cached value serves both branches.
// Forms y = d1·x/(d1·x + d2) and ybar = d2/(d1·x + d2), both by division
// (#49). For x below ~DBL_TRUE_MIN/d1 the product d1·x is subnormal and
// loses relative precision bit by bit — collapsing to y = 0 at the bottom —
// even though y itself is representable (F(0.01,0.01)'s own quantile(1e-300)
// output lands there, so quantile→cdf did not close). That band switches to
// the algebraically identical
//   y = x/(x + d2/d1),  ybar = (d2/d1)/(x + d2/d1)
// where x enters exactly. If d2/d1 itself overflows, y underflows to 0 in
// every form, so the pre-switch result is restored. Returns false when
// d1·x + d2 overflows: y is 1 to every bit available.
static inline bool f_beta_args(double x, double d1, double d2, double& y,
                               double& ybar) noexcept {
    const double dx = d1 * x;
    if (!std::isfinite(dx + d2))
        return false;
    if (dx >= std::numeric_limits<double>::min()) {
        const double denom = dx + d2;
        y = dx / denom;
        ybar = d2 / denom;
    } else {
        const double c = d2 / d1;
        if (std::isfinite(c)) {
            const double denom = x + c;
            y = x / denom;
            ybar = c / denom;
        } else {
            y = detail::ZERO_DOUBLE;
            ybar = detail::ONE;
        }
    }
    return true;
}

double FDistribution::cdfImpl(double x, double a, double b, double d1, double d2,
                              double log_beta_prefix) noexcept {
    if (std::isnan(x))
        return kNaN;
    if (x <= detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;  // includes -inf
    if (x == kInf)
        return detail::ONE;

    double y, ybar;
    if (!f_beta_args(x, d1, d2, y, ybar))
        return detail::ONE;  // d1*x overflowed: y is 1 to every bit available

    const double switch_point = (a + detail::ONE) / (a + b + detail::TWO);

    if (y < switch_point)
        return detail::beta_i(y, a, b, log_beta_prefix);
    return detail::ONE - detail::beta_i(ybar, b, a, log_beta_prefix);
}

// Survival function, complement-native: I_ybar(b,a) is computed, never
// 1 - CDF(x). This retains full relative precision far below 1e-16, which is
// what the quantile solver's upper-tail branch needs.
double FDistribution::sfImpl(double x, double a, double b, double d1, double d2,
                             double log_beta_prefix) noexcept {
    if (std::isnan(x))
        return kNaN;
    if (x <= detail::ZERO_DOUBLE)
        return detail::ONE;
    if (x == kInf)
        return detail::ZERO_DOUBLE;

    double y, ybar;
    if (!f_beta_args(x, d1, d2, y, ybar))
        return detail::ZERO_DOUBLE;

    const double switch_point = (a + detail::ONE) / (a + b + detail::TWO);

    // Mirror of cdfImpl: whichever argument is on its small side is the one
    // handed to beta_i; only the branch that is already accurate is complemented.
    if (y < switch_point)
        return detail::ONE - detail::beta_i(y, a, b, log_beta_prefix);
    return detail::beta_i(ybar, b, a, log_beta_prefix);
}

double FDistribution::getProbability(double x) const {
    double a, b, d1, d2, c;
    withCacheSnapshot([&] {
        a = a_;
        b = b_;
        d1 = d1_;
        d2 = d2_;
        c = logPdfConst_;
    });
    const double lp = logPdfImpl(x, a, b, d1, d2, c);
    if (std::isnan(lp))
        return kNaN;
    return std::exp(lp);  // exp(-inf) == 0, exp(+inf) == +inf: both correct here
}

double FDistribution::getLogProbability(double x) const {
    double a, b, d1, d2, c;
    withCacheSnapshot([&] {
        a = a_;
        b = b_;
        d1 = d1_;
        d2 = d2_;
        c = logPdfConst_;
    });
    return logPdfImpl(x, a, b, d1, d2, c);
}

double FDistribution::getCumulativeProbability(double x) const {
    double a, b, d1, d2, pfx;
    withCacheSnapshot([&] {
        a = a_;
        b = b_;
        d1 = d1_;
        d2 = d2_;
        pfx = logBetaPrefix_;
    });
    return cdfImpl(x, a, b, d1, d2, pfx);
}

double FDistribution::getSurvivalProbability(double x) const {
    double a, b, d1, d2, pfx;
    withCacheSnapshot([&] {
        a = a_;
        b = b_;
        d1 = d1_;
        d2 = d2_;
        pfx = logBetaPrefix_;
    });
    return sfImpl(x, a, b, d1, d2, pfx);
}

double FDistribution::getQuantile(double p) const {
    if (std::isnan(p) || p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability p must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return detail::ZERO_DOUBLE;
    if (p == detail::ONE)
        return kInf;

    double a, b, d1, d2, pfx;
    withCacheSnapshot([&] {
        a = a_;
        b = b_;
        d1 = d1_;
        d2 = d2_;
        pfx = logBetaPrefix_;
    });

    // Safeguarded bisection on t = ln(x) against the *steered* CDF/survival.
    // For p > 1/2 the residual is taken on the survival function, so the
    // quantity being compared is always the small one -- the same reason
    // cdfImpl picks its branch. See the header note on why
    // detail::inverse_beta_i is not used.
    const bool upper = (p > detail::HALF);
    const double target = upper ? (detail::ONE - p) : p;

    // f(t) is increasing in t for both branches once signed this way.
    auto residual = [&](double t) {
        const double x = std::exp(t);
        return upper ? (target - sfImpl(x, a, b, d1, d2, pfx))
                     : (cdfImpl(x, a, b, d1, d2, pfx) - target);
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

double FDistribution::sample(std::mt19937& rng) const {
    // Y ~ Beta(d1/2, d2/2)  =>  X = (d2/d1) Y/(1-Y) ~ F(d1,d2).
    // The 1-Y here is unavoidable (the delegate hands back Y, not its
    // complement), so a draw in the extreme upper tail is resolution-limited.
    // That is a property of the sampled value, not of a precision contract:
    // the probability functions never go through this transform.
    double d1, d2;
    withCacheSnapshot([&] {
        d1 = d1_;
        d2 = d2_;
    });
    const double y = beta_.sample(rng);
    if (y >= detail::ONE)
        return kInf;
    return (d2 / d1) * y / (detail::ONE - y);
}

std::vector<double> FDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i)
        out.push_back(sample(rng));
    return out;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

// Method-of-moments fit, best effort. There is no closed-form MLE for (d1,d2)
// and in practice both are fixed by the experimental design rather than
// estimated, so this exists to satisfy the DistributionInterface contract with
// a defensible estimate rather than to be a recommended estimator.
//
//   mean  m = d2/(d2-2)                      =>  d2 = 2m/(m-1),  needs m > 1
//   var   v = 2 d2^2 (d1+d2-2)
//             / (d1 (d2-2)^2 (d2-4))         =>  d1 = 2 d2^2 (d2-2)
//                                                    / (v (d2-2)^2 (d2-4) - 2 d2^2)
//
// Documented fallbacks (each keeps the fit total rather than throwing on data
// that is merely uninformative):
//   * m <= 1                    -> d2_hat = 10 (mean undefined for d2 <= 2, so
//                                  the mean equation carries no information)
//   * d2_hat outside [4.5, 1e6] -> clamped there; below 4.5 the variance
//                                  equation has no finite solution
//   * variance-equation denominator <= 0, or a non-finite d1_hat
//                               -> d1_hat = 1
//   * d1_hat outside [1e-3, 1e6]-> clamped there
void FDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    double sum = detail::ZERO_DOUBLE;
    double sum2 = detail::ZERO_DOUBLE;
    for (double v : values) {
        if (!std::isfinite(v) || v <= detail::ZERO_DOUBLE)
            throw std::invalid_argument("All values must be positive and finite for F fitting");
        sum += v;
        sum2 += v * v;
    }

    const double nD = static_cast<double>(values.size());
    const double m = sum / nD;
    const double v = std::max(sum2 / nD - m * m, detail::ZERO_DOUBLE);

    double d2_hat = 10.0;
    if (m > detail::ONE) {
        const double cand = detail::TWO * m / (m - detail::ONE);
        if (std::isfinite(cand))
            d2_hat = cand;
    }
    d2_hat = std::clamp(d2_hat, 4.5, 1.0e6);

    double d1_hat = detail::ONE;
    if (v > detail::ZERO_DOUBLE) {
        const double d2m2 = d2_hat - detail::TWO;
        const double den = v * d2m2 * d2m2 * (d2_hat - 4.0) - detail::TWO * d2_hat * d2_hat;
        if (den > detail::ZERO_DOUBLE) {
            const double cand = detail::TWO * d2_hat * d2_hat * d2m2 / den;
            if (std::isfinite(cand) && cand > detail::ZERO_DOUBLE)
                d1_hat = cand;
        }
    }
    d1_hat = std::clamp(d1_hat, 1.0e-3, 1.0e6);

    setParameters(d1_hat, d2_hat);
}

void FDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                     std::vector<FDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void FDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    d1_ = detail::ONE;
    d2_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

std::string FDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "FDistribution(d1=" << d1_ << ",d2=" << d2_ << ")";
    return oss.str();
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void FDistribution::pdfKernel(const double* values, double* results, std::size_t count, double a,
                              double b, double d1, double d2, double log_pdf_const) noexcept {
    for (std::size_t i = 0; i < count; ++i) {
        const double lp = logPdfImpl(values[i], a, b, d1, d2, log_pdf_const);
        results[i] = std::isnan(lp) ? kNaN : std::exp(lp);
    }
}

void FDistribution::logPdfKernel(const double* values, double* results, std::size_t count, double a,
                                 double b, double d1, double d2, double log_pdf_const) noexcept {
    for (std::size_t i = 0; i < count; ++i)
        results[i] = logPdfImpl(values[i], a, b, d1, d2, log_pdf_const);
}

void FDistribution::cdfKernel(const double* values, double* results, std::size_t count, double a,
                              double b, double d1, double d2, double log_beta_prefix) noexcept {
    for (std::size_t i = 0; i < count; ++i)
        results[i] = cdfImpl(values[i], a, b, d1, d2, log_beta_prefix);
}

void FDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                   const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const FDistribution& dist, double value) { return dist.getProbability(value); },
        [](const FDistribution& dist, const double* vals, double* res, std::size_t count) {
            double a, b, d1, d2, c;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                c = dist.logPdfConst_;
            });
            pdfKernel(vals, res, count, a, b, d1, d2, c);
        },
        [](const FDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b, d1, d2, c;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                c = dist.logPdfConst_;
            });
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    pdfKernel(vals.data() + start, res.data() + start, len, a, b, d1, d2, c);
                });
            } else {
                pdfKernel(vals.data(), res.data(), count, a, b, d1, d2, c);
            }
        },
        [](const FDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b, d1, d2, c;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                c = dist.logPdfConst_;
            });
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
                pdfKernel(vals.data() + start, res.data() + start, len, a, b, d1, d2, c);
            });
            pool.waitForAll();
        });
}

void FDistribution::getLogProbability(std::span<const double> values, std::span<double> results,
                                      const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const FDistribution& dist, double value) { return dist.getLogProbability(value); },
        [](const FDistribution& dist, const double* vals, double* res, std::size_t count) {
            double a, b, d1, d2, c;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                c = dist.logPdfConst_;
            });
            logPdfKernel(vals, res, count, a, b, d1, d2, c);
        },
        [](const FDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b, d1, d2, c;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                c = dist.logPdfConst_;
            });
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    logPdfKernel(vals.data() + start, res.data() + start, len, a, b, d1, d2, c);
                });
            } else {
                logPdfKernel(vals.data(), res.data(), count, a, b, d1, d2, c);
            }
        },
        [](const FDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b, d1, d2, c;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                c = dist.logPdfConst_;
            });
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
                logPdfKernel(vals.data() + start, res.data() + start, len, a, b, d1, d2, c);
            });
            pool.waitForAll();
        });
}

void FDistribution::getCumulativeProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const FDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const FDistribution& dist, const double* vals, double* res, std::size_t count) {
            double a, b, d1, d2, pfx;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                pfx = dist.logBetaPrefix_;
            });
            cdfKernel(vals, res, count, a, b, d1, d2, pfx);
        },
        [](const FDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b, d1, d2, pfx;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                pfx = dist.logBetaPrefix_;
            });
            constexpr std::size_t CHUNK = 1024;
            if (arch::should_use_parallel(count)) {
                const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
                ParallelUtils::parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                    const std::size_t start = ci * CHUNK;
                    const std::size_t len = std::min(CHUNK, count - start);
                    cdfKernel(vals.data() + start, res.data() + start, len, a, b, d1, d2, pfx);
                });
            } else {
                cdfKernel(vals.data(), res.data(), count, a, b, d1, d2, pfx);
            }
        },
        [](const FDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double a, b, d1, d2, pfx;
            dist.withCacheSnapshot([&] {
                a = dist.a_;
                b = dist.b_;
                d1 = dist.d1_;
                d2 = dist.d2_;
                pfx = dist.logBetaPrefix_;
            });
            constexpr std::size_t CHUNK = 1024;
            const std::size_t num_chunks = (count + CHUNK - 1) / CHUNK;
            pool.parallelFor(std::size_t{0}, num_chunks, [&](std::size_t ci) {
                const std::size_t start = ci * CHUNK;
                const std::size_t len = std::min(CHUNK, count - start);
                cdfKernel(vals.data() + start, res.data() + start, len, a, b, d1, d2, pfx);
            });
            pool.waitForAll();
        });
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool FDistribution::operator==(const FDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(d1_ - other.d1_) <= detail::DEFAULT_TOLERANCE &&
           std::abs(d2_ - other.d2_) <= detail::DEFAULT_TOLERANCE;
}

bool FDistribution::operator!=(const FDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const FDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, FDistribution& dist) {
    std::string token;

    is >> token;
    if (!token.starts_with("FDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t d1_pos = token.find("d1=");
    const size_t d2_pos = token.find("d2=");
    if (d1_pos == std::string::npos || d2_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t d1_comma = token.find(",", d1_pos);
    const size_t d2_close = token.find(")", d2_pos);
    if (d1_comma == std::string::npos || d2_close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    double d1, d2;
    try {
        d1 = std::stod(token.substr(d1_pos + 3, d1_comma - d1_pos - 3));
        d2 = std::stod(token.substr(d2_pos + 3, d2_close - d2_pos - 3));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(d1, d2);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void FDistribution::updateCacheUnsafe() const noexcept {
    a_ = d1_ * detail::HALF;
    b_ = d2_ * detail::HALF;

    // lgamma(a+b) - lgamma(a) - lgamma(b) == -ln B(a,b); symmetric in (a,b), so
    // both incomplete-beta branches in cdfImpl/sfImpl reuse this one value.
    logBetaPrefix_ = -detail::lbeta(a_, b_);
    logPdfConst_ = a_ * std::log(d1_) + b_ * std::log(d2_) + logBetaPrefix_;

    // Keep the Beta delegate in sync (used by sample()). beta_ owns its own
    // mutex, independent of ours; it is private so no external thread can be
    // holding it.
    (void)beta_.trySetParameters(a_, b_);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicD1_.store(d1_, std::memory_order_release);
    atomicD2_.store(d2_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

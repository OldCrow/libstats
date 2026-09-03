#include "libstats/distributions/erlang.h"

#include "libstats/common/distribution_impl_common.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

// Helpers: validate each parameter and return it, throwing before any member
// is constructed (mirrors ChiSquaredDistribution's requireValidDOF pattern).
static int requireValidK(int k) {
    if (k < 1)
        throw std::invalid_argument("Shape parameter k must be a positive integer (k >= 1)");
    return k;
}

static double requireValidLambda(double lambda) {
    if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0)
        throw std::invalid_argument("Rate parameter lambda must be a positive finite number");
    return lambda;
}

ErlangDistribution::ErlangDistribution(int k, double lambda)
    : DistributionBase(),
      k_(requireValidK(k)),
      lambda_(requireValidLambda(lambda)),
      gamma_(static_cast<double>(k_), lambda_) {}

ErlangDistribution::ErlangDistribution(const ErlangDistribution& other) : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    k_ = other.k_;
    lambda_ = other.lambda_;
    gamma_ = other.gamma_;
}

ErlangDistribution& ErlangDistribution::operator=(const ErlangDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        k_ = other.k_;
        lambda_ = other.lambda_;
        gamma_ = other.gamma_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

ErlangDistribution::ErlangDistribution(ErlangDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    k_ = other.k_;
    lambda_ = other.lambda_;
    gamma_ = std::move(other.gamma_);
    other.k_ = 1;
    other.lambda_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
}

ErlangDistribution& ErlangDistribution::operator=(ErlangDistribution&& other) noexcept {
    if (this != &other) {
        k_ = other.k_;
        lambda_ = other.lambda_;
        gamma_ = std::move(other.gamma_);
        other.k_ = 1;
        other.lambda_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

ErlangDistribution ErlangDistribution::createUnchecked(int k, double lambda) noexcept {
    return ErlangDistribution(k, lambda, true);
}

ErlangDistribution::ErlangDistribution(int k, double lambda, bool /*bypassValidation*/) noexcept
    : DistributionBase(), k_(k), lambda_(lambda), gamma_(static_cast<double>(k), lambda) {
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

void ErlangDistribution::setK(int k) {
    validateParameters(k, lambda_);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = k;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    // Sync gamma_ outside our lock -- same pattern as ChiSquaredDistribution.
    // gamma_ is private, so no external thread can acquire its lock while we
    // don't hold ours.
    (void)gamma_.trySetAlpha(static_cast<double>(k));
}

void ErlangDistribution::setLambda(double lambda) {
    validateParameters(k_, lambda);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        lambda_ = lambda;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetBeta(lambda);
}

void ErlangDistribution::setParameters(int k, double lambda) {
    validateParameters(k, lambda);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = k;
        lambda_ = lambda;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetParameters(static_cast<double>(k), lambda);
}

VoidResult ErlangDistribution::trySetK(int k) noexcept {
    auto v = validateErlangParameters(k, lambda_);
    if (v.isError())
        return v;
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = k;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetAlpha(static_cast<double>(k));
    return VoidResult::ok({});
}

VoidResult ErlangDistribution::trySetLambda(double lambda) noexcept {
    auto v = validateErlangParameters(k_, lambda);
    if (v.isError())
        return v;
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        lambda_ = lambda;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetBeta(lambda);
    return VoidResult::ok({});
}

VoidResult ErlangDistribution::trySetParameters(int k, double lambda) noexcept {
    auto v = validateErlangParameters(k, lambda);
    if (v.isError())
        return v;
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = k;
        lambda_ = lambda;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetParameters(static_cast<double>(k), lambda);
    return VoidResult::ok({});
}

VoidResult ErlangDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateErlangParameters(k_, lambda_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double ErlangDistribution::getProbability(double x) const {
    // GammaDistribution::getProbability has no !isfinite(x) guard, and its
    // log-space formula evaluates 0*inf / inf-inf (both NaN) at x=+inf for
    // alpha>=1 -- always true for Erlang (alpha=k>=1). See class-level note.
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(x))
        return detail::ZERO_DOUBLE;
    return gamma_.getProbability(x);
}

double ErlangDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(x))
        return detail::NEGATIVE_INFINITY;
    return gamma_.getLogProbability(x);
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void ErlangDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    double sum = detail::ZERO_DOUBLE;
    double sum2 = detail::ZERO_DOUBLE;
    for (double v : values) {
        if (!std::isfinite(v) || v <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive and finite for Erlang MLE");
        }
        sum += v;
        sum2 += v * v;
    }

    const double nD = static_cast<double>(values.size());
    const double mean_x = sum / nD;
    const double var_x = sum2 / nD - mean_x * mean_x;

    // Method-of-moments shape estimate: k_hat = round(mean^2 / var), clamped
    // to >= 1 (var_x <= 0 -- degenerate/near-constant data -- falls back to
    // k_hat = 1 rather than diverging toward infinity).
    //
    // #125-class guard: the round() result is compared against INT_MAX *as a
    // double*, before any narrowing cast. An unguarded static_cast<int> of a
    // double outside int's range is undefined behaviour, not just wrong;
    // values beyond int range saturate to INT_MAX -- the mathematically
    // nearest representable answer -- instead of invoking UB.
    constexpr double kMaxKAsDouble = 2147483647.0;  // INT_MAX, exactly representable as double
    int k_hat = 1;
    if (var_x > detail::ZERO_DOUBLE && std::isfinite(mean_x) && std::isfinite(var_x)) {
        const double raw = std::round((mean_x * mean_x) / var_x);
        if (std::isfinite(raw) && raw > 1.0) {
            k_hat = (raw > kMaxKAsDouble) ? std::numeric_limits<int>::max()
                                          : static_cast<int>(raw);
        }
    }
    const double lambda_hat = static_cast<double>(k_hat) / mean_x;
    setParameters(k_hat, lambda_hat);
}

void ErlangDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                          std::vector<ErlangDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void ErlangDistribution::reset() noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = 1;
        lambda_ = detail::ONE;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetParameters(detail::ONE, detail::ONE);  // Gamma(1,1) = Erlang(1,1)
}

std::string ErlangDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "ErlangDistribution(k=" << k_ << ",lambda=" << lambda_ << ")";
    return oss.str();
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

namespace {

// See ErlangDistribution's class-level "±inf / NaN handling" note: scrub
// non-finite inputs to a safe finite placeholder before delegating a batch
// call to gamma_, so its unguarded log-space kernel never sees +-inf.
void scrubNonFinite(std::span<const double> values, std::vector<double>& scratch) {
    scratch.assign(values.begin(), values.end());
    for (double& v : scratch) {
        if (!std::isfinite(v))
            v = detail::ONE;  // arbitrary finite placeholder in the support interior
    }
}

}  // namespace

void ErlangDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                        const detail::PerformanceHint& hint) const {
    if (values.size() != results.size())
        throw std::invalid_argument("Input and output spans must have the same size");
    const std::size_t count = values.size();
    if (count == 0)
        return;

    bool has_nonfinite = false;
    for (double v : values) {
        if (!std::isfinite(v)) {
            has_nonfinite = true;
            break;
        }
    }
    if (!has_nonfinite) {
        gamma_.getProbability(values, results, hint);
        return;
    }

    std::vector<double> scratch;
    scrubNonFinite(values, scratch);
    gamma_.getProbability(std::span<const double>(scratch), results, hint);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i]))
            results[i] = std::numeric_limits<double>::quiet_NaN();
        else if (std::isinf(values[i]))
            results[i] = detail::ZERO_DOUBLE;
    }
}

void ErlangDistribution::getLogProbability(std::span<const double> values,
                                           std::span<double> results,
                                           const detail::PerformanceHint& hint) const {
    if (values.size() != results.size())
        throw std::invalid_argument("Input and output spans must have the same size");
    const std::size_t count = values.size();
    if (count == 0)
        return;

    bool has_nonfinite = false;
    for (double v : values) {
        if (!std::isfinite(v)) {
            has_nonfinite = true;
            break;
        }
    }
    if (!has_nonfinite) {
        gamma_.getLogProbability(values, results, hint);
        return;
    }

    std::vector<double> scratch;
    scrubNonFinite(values, scratch);
    gamma_.getLogProbability(std::span<const double>(scratch), results, hint);
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i]))
            results[i] = std::numeric_limits<double>::quiet_NaN();
        else if (std::isinf(values[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool ErlangDistribution::operator==(const ErlangDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return k_ == other.k_ && std::abs(lambda_ - other.lambda_) <= detail::DEFAULT_TOLERANCE;
}

bool ErlangDistribution::operator!=(const ErlangDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const ErlangDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, ErlangDistribution& dist) {
    std::string token;

    is >> token;
    if (!token.starts_with("ErlangDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t k_pos = token.find("k=");
    const size_t lambda_pos = token.find("lambda=");
    if (k_pos == std::string::npos || lambda_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t k_comma = token.find(",", k_pos);
    const size_t lambda_close = token.find(")", lambda_pos);
    if (k_comma == std::string::npos || lambda_close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    int k;
    double lambda;
    try {
        k = static_cast<int>(std::stod(token.substr(k_pos + 2, k_comma - k_pos - 2)));
        lambda = std::stod(token.substr(lambda_pos + 7, lambda_close - lambda_pos - 7));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(k, lambda);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void ErlangDistribution::updateCacheUnsafe() const noexcept {
    // Sync gamma_ with current k_/lambda_. gamma_'s own mutex is independent
    // of ours, so this nested lock acquisition is safe.
    (void)gamma_.trySetParameters(static_cast<double>(k_), lambda_);
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
}

}  // namespace stats

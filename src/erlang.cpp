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
    // Pure delegation (ChiSquared pattern): GammaDistribution guards
    // non-finite x itself since #130 (pdf(±inf)=0, NaN propagates).
    return gamma_.getProbability(x);
}

double ErlangDistribution::getLogProbability(double x) const {
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

// Pure batch delegation (ChiSquared pattern). The scrub-delegate-fixup
// machinery that lived here at first landing was a workaround for Gamma's
// missing non-finite guards; #130 moved that handling into Gamma's own
// scalar entry points and batch fixup loops (scalar == batch at specials),
// so the wrapper adds nothing. Size checks and the no-overlap debug assert
// happen inside the delegate's DispatchUtils path.

void ErlangDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                        const detail::PerformanceHint& hint) const {
    gamma_.getProbability(values, results, hint);
}

void ErlangDistribution::getLogProbability(std::span<const double> values,
                                           std::span<double> results,
                                           const detail::PerformanceHint& hint) const {
    gamma_.getLogProbability(values, results, hint);
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
        const double k_raw = std::stod(token.substr(k_pos + 2, k_comma - k_pos - 2));
        lambda = std::stod(token.substr(lambda_pos + 7, lambda_close - lambda_pos - 7));
        // #125-class guard: static_cast<int> of a double outside int's range is
        // UB. A serialized k beyond INT_MAX cannot be a valid round-trip, so
        // reject it (failbit) rather than saturate. The comparison also rejects
        // NaN (all comparisons false) and non-integral k values.
        constexpr double kMaxKAsDouble = 2147483647.0;  // INT_MAX, exact as double
        if (!(k_raw >= 1.0 && k_raw <= kMaxKAsDouble && k_raw == std::floor(k_raw))) {
            is.setstate(std::ios::failbit);
            return is;
        }
        k = static_cast<int>(k_raw);
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

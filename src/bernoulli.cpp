#include "libstats/distributions/bernoulli.h"

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

// Helper: validate p and return it, throwing before any member is constructed.
static double requireValidP(double p) {
    if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0)
        throw std::invalid_argument("Success probability p must be in [0, 1]");
    return p;
}

BernoulliDistribution::BernoulliDistribution(double p)
    : DistributionBase(), p_(requireValidP(p)), binomial_(1, p) {
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

BernoulliDistribution::BernoulliDistribution(const BernoulliDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    p_ = other.p_;
    binomial_ = other.binomial_;
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

BernoulliDistribution& BernoulliDistribution::operator=(const BernoulliDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        p_ = other.p_;
        binomial_ = other.binomial_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

BernoulliDistribution::BernoulliDistribution(BernoulliDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    p_ = other.p_;
    binomial_ = std::move(other.binomial_);
    other.p_ = detail::HALF;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
}

BernoulliDistribution& BernoulliDistribution::operator=(BernoulliDistribution&& other) noexcept {
    if (this != &other) {
        p_ = other.p_;
        binomial_ = std::move(other.binomial_);
        other.p_ = detail::HALF;
        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
        other.atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

BernoulliDistribution BernoulliDistribution::createUnchecked(double p) noexcept {
    return BernoulliDistribution(p, true);
}

BernoulliDistribution::BernoulliDistribution(double p, bool /*bypassValidation*/) noexcept
    : DistributionBase(), p_(p), binomial_(1, p) {
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

double BernoulliDistribution::getPAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicP_.load(std::memory_order_acquire);
    return getP();
}

void BernoulliDistribution::setP(double p) {
    validateParameters(p);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        p_ = p;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    // Update the delegate outside our lock to avoid holding two locks at once.
    // binomial_ is private, so no external thread can reach it while we don't
    // hold our lock.
    (void)binomial_.trySetP(p);
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult BernoulliDistribution::trySetP(double p) noexcept {
    auto v = validateBernoulliParameters(p);
    if (v.isError())
        return v;
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        p_ = p;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    (void)binomial_.trySetP(p);
    return VoidResult::ok({});
}

VoidResult BernoulliDistribution::validateCurrentParameters() const noexcept {
    return validateBernoulliParameters(getP());
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void BernoulliDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    // Validate: every value must be exactly 0.0 or 1.0 (Bernoulli outcomes).
    double sum = detail::ZERO_DOUBLE;
    for (double v : values) {
        if (v != detail::ZERO_DOUBLE && v != detail::ONE)
            throw std::invalid_argument("All values must be 0.0 or 1.0 for Bernoulli MLE");
        sum += v;
    }

    // MLE: p_hat = x_bar (sample mean of 0/1 data)
    const double p_hat = sum / static_cast<double>(values.size());
    setP(p_hat);
}

void BernoulliDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                             std::vector<BernoulliDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void BernoulliDistribution::reset() noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        p_ = detail::HALF;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(detail::HALF, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    (void)binomial_.trySetP(detail::HALF);
}

std::string BernoulliDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "BernoulliDistribution(p=" << p_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double BernoulliDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (p_ > detail::HALF)
        return detail::ONE;
    if (p_ < detail::HALF)
        return detail::ZERO_DOUBLE;
    return detail::ZERO_DOUBLE;  // tie at p=0.5: 0 by convention
}

double BernoulliDistribution::getMedian() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (p_ > detail::HALF)
        return detail::ONE;
    if (p_ < detail::HALF)
        return detail::ZERO_DOUBLE;
    return detail::HALF;  // tie at p=0.5: 0.5 by convention (any value in [0,1] satisfies it)
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool BernoulliDistribution::operator==(const BernoulliDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(p_ - other.p_) <= detail::DEFAULT_TOLERANCE;
}

bool BernoulliDistribution::operator!=(const BernoulliDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const BernoulliDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, BernoulliDistribution& dist) {
    std::string token;
    double p;

    is >> token;
    if (!token.starts_with("BernoulliDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t p_pos = token.find("p=");
    if (p_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t close = token.find(")", p_pos);
    if (close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        p = std::stod(token.substr(p_pos + 2, close - p_pos - 2));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetP(p);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void BernoulliDistribution::updateCacheUnsafe() const noexcept {
    // Sync the delegate with current p_ (n stays fixed at 1). binomial_'s own
    // mutex is independent of ours, so acquiring it here is safe.
    (void)binomial_.trySetP(p_);
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

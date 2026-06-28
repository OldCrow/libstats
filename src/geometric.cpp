#include "libstats/distributions/geometric.h"
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
    if (!std::isfinite(p) || p <= 0.0 || p > 1.0)
        throw std::invalid_argument("Success probability p must be in (0, 1]");
    return p;
}

GeometricDistribution::GeometricDistribution(double p)
    : DistributionBase(), p_(requireValidP(p)), negbinom_(detail::ONE, p) {
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

GeometricDistribution::GeometricDistribution(const GeometricDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    p_       = other.p_;
    negbinom_ = other.negbinom_;
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

GeometricDistribution& GeometricDistribution::operator=(const GeometricDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        p_        = other.p_;
        negbinom_ = other.negbinom_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}

GeometricDistribution::GeometricDistribution(GeometricDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    p_        = other.p_;
    negbinom_ = std::move(other.negbinom_);
    other.p_  = detail::HALF;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    other.atomicParamsValid_.store(false, std::memory_order_release);
}

GeometricDistribution& GeometricDistribution::operator=(GeometricDistribution&& other) noexcept {
    if (this != &other) {
        p_        = other.p_;
        negbinom_ = std::move(other.negbinom_);
        other.p_  = detail::HALF;
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

GeometricDistribution GeometricDistribution::createUnchecked(double p) noexcept {
    return GeometricDistribution(p, true);
}

GeometricDistribution::GeometricDistribution(double p, bool /*bypassValidation*/) noexcept
    : DistributionBase(), p_(p), negbinom_(detail::ONE, p) {
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

double GeometricDistribution::getPAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicP_.load(std::memory_order_acquire);
    return getP();
}

void GeometricDistribution::setP(double p) {
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
    // negbinom_ is private, so no external thread can reach it while we don't
    // hold our lock.
    (void)negbinom_.trySetP(p);
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult GeometricDistribution::trySetP(double p) noexcept {
    auto v = validateGeometricParameters(p);
    if (v.isError()) return v;
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        p_ = p;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(p_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    (void)negbinom_.trySetP(p);
    return VoidResult::ok({});
}

VoidResult GeometricDistribution::validateCurrentParameters() const noexcept {
    return validateGeometricParameters(getP());
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void GeometricDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    // Validate: all values must be non-negative integers (failure counts)
    double sum = detail::ZERO_DOUBLE;
    for (double v : values) {
        if (!std::isfinite(v) || v < detail::ZERO_DOUBLE)
            throw std::invalid_argument(
                "All values must be non-negative and finite for Geometric MLE");
        sum += v;
    }

    // MLE: p̂ = 1 / (1 + x̄)
    const double x_bar = sum / static_cast<double>(values.size());
    const double p_hat = detail::ONE / (detail::ONE + x_bar);

    // Clamp to valid range (x̄ = 0 → p̂ = 1; protect against floating-point
    // edge cases where x̄ is very large)
    setP(std::clamp(p_hat, std::numeric_limits<double>::min(), detail::ONE));
}

void GeometricDistribution::parallelBatchFit(
    const std::vector<std::vector<double>>& datasets,
    std::vector<GeometricDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void GeometricDistribution::reset() noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        p_ = detail::HALF;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicP_.store(detail::HALF, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    }
    (void)negbinom_.trySetP(detail::HALF);
}

std::string GeometricDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "GeometricDistribution(p=" << p_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double GeometricDistribution::getMedian() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // Median of Geometric: ⌈−ln 2 / ln(1−p)⌉ − 1
    // Special case p = 1: degenerate at 0 → median = 0
    if (p_ >= detail::ONE)
        return detail::ZERO_DOUBLE;
    const double log1mp = std::log(detail::ONE - p_);
    // log1mp is negative (since 0 < 1-p < 1 for p in (0,1))
    return std::ceil(-detail::LN2 / log1mp) - detail::ONE;
}

double GeometricDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    // H = [−(1−p)·ln(1−p) − p·ln p] / p  (nats)
    // Special case p = 1: H = 0 (degenerate distribution, no uncertainty)
    if (p_ >= detail::ONE)
        return detail::ZERO_DOUBLE;
    const double q = detail::ONE - p_;
    return (-(q * std::log(q)) - p_ * std::log(p_)) / p_;
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool GeometricDistribution::operator==(const GeometricDistribution& other) const {
    if (this == &other) return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(p_ - other.p_) <= detail::DEFAULT_TOLERANCE;
}

bool GeometricDistribution::operator!=(const GeometricDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const GeometricDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, GeometricDistribution& dist) {
    std::string token;
    double p;

    is >> token;
    if (!token.starts_with("GeometricDistribution(")) {
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

void GeometricDistribution::updateCacheUnsafe() const noexcept {
    // Sync the delegate with current p_. negbinom_'s own mutex is independent
    // of ours, so acquiring it here is safe.
    (void)negbinom_.trySetP(p_);
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicP_.store(p_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

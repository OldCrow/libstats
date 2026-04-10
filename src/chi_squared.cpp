#include "libstats/distributions/chi_squared.h"

#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/validation.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

// Helper: validate k and return it, throwing before any member is constructed.
// Used in the initializer list so that gamma_ is never constructed for invalid k.
static double requireValidDOF(double k) {
    if (k <= 0.0 || !std::isfinite(k)) {
        throw std::invalid_argument("Degrees of freedom k must be a positive finite number");
    }
    return k;
}

ChiSquaredDistribution::ChiSquaredDistribution(double k)
    : DistributionBase(), k_(requireValidDOF(k)), gamma_(k / detail::TWO, detail::HALF) {
    // requireValidDOF throws before k_ or gamma_ are constructed for invalid k.
    // For valid k, gamma_ is initialized as Gamma(k/2, 0.5).
}

ChiSquaredDistribution::ChiSquaredDistribution(const ChiSquaredDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    k_ = other.k_;
    gamma_ = other.gamma_;
}

ChiSquaredDistribution& ChiSquaredDistribution::operator=(const ChiSquaredDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        k_ = other.k_;
        gamma_ = other.gamma_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

ChiSquaredDistribution::ChiSquaredDistribution(ChiSquaredDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    std::unique_lock<std::shared_mutex> lock(other.cache_mutex_);
    k_ = other.k_;
    gamma_ = std::move(other.gamma_);
    other.k_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
}

ChiSquaredDistribution& ChiSquaredDistribution::operator=(ChiSquaredDistribution&& other) noexcept {
    if (this != &other) {
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);

        bool success = false;
        try {
            std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
            std::unique_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
            if (std::try_lock(lock1, lock2) == -1) {
                k_ = other.k_;
                gamma_ = std::move(other.gamma_);
                other.k_ = detail::ONE;
                cache_valid_ = false;
                other.cache_valid_ = false;
                success = true;
            }
        } catch (...) {
        }

        if (!success) {
            k_ = other.k_;
            gamma_ = std::move(other.gamma_);
            other.k_ = detail::ONE;
            cache_valid_ = false;
            other.cache_valid_ = false;
        }
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

ChiSquaredDistribution ChiSquaredDistribution::createUnchecked(double k) noexcept {
    return ChiSquaredDistribution(k, true);
}

ChiSquaredDistribution::ChiSquaredDistribution(double k, bool /*bypassValidation*/) noexcept
    : DistributionBase(), k_(k), gamma_(k / detail::TWO, detail::HALF) {
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

void ChiSquaredDistribution::setK(double k) {
    validateParameters(k);
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = k;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    // Update gamma_ outside our lock to avoid holding two locks at once.
    // gamma_ is private, so no external thread can lock it while we don't hold ours.
    (void)gamma_.trySetAlpha(k / detail::TWO);
}

VoidResult ChiSquaredDistribution::trySetK(double k) noexcept {
    auto validation = validateChiSquaredParameters(k);
    if (validation.isError()) {
        return validation;
    }
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = k;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetAlpha(k / detail::TWO);
    return VoidResult::ok(true);
}

VoidResult ChiSquaredDistribution::validateCurrentParameters() const noexcept {
    return validateChiSquaredParameters(getK());
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void ChiSquaredDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }
    for (double v : values) {
        if (v <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("All values must be positive for chi-squared MLE");
        }
    }

    // MLE for chi-squared: k_hat = sample_mean  (E[X] = k)
    const double sum = std::accumulate(values.begin(), values.end(), detail::ZERO_DOUBLE);
    const double k_hat = sum / static_cast<double>(values.size());
    setK(k_hat);
}

void ChiSquaredDistribution::reset() noexcept {
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        k_ = detail::ONE;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }
    (void)gamma_.trySetAlpha(detail::HALF);  // Gamma(0.5, 0.5) = χ²(1)
}

std::string ChiSquaredDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "ChiSquaredDistribution(k=" << k_ << ")";
    return oss.str();
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool ChiSquaredDistribution::operator==(const ChiSquaredDistribution& other) const {
    if (this == &other)
        return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::abs(k_ - other.k_) <= detail::DEFAULT_TOLERANCE;
}

bool ChiSquaredDistribution::operator!=(const ChiSquaredDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const ChiSquaredDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, ChiSquaredDistribution& dist) {
    std::string token;
    double k;

    is >> token;
    if (token.find("ChiSquaredDistribution(") != 0) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t k_pos = token.find("k=");
    if (k_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t close = token.find(")", k_pos);
    if (close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        k = std::stod(token.substr(k_pos + 2, close - k_pos - 2));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetK(k);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void ChiSquaredDistribution::updateCacheUnsafe() const noexcept {
    // Sync gamma_ with current k_. gamma_'s own mutex is independent of ours,
    // so this nested lock acquisition is safe.
    (void)gamma_.trySetAlpha(k_ / detail::TWO);
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
}

}  // namespace stats

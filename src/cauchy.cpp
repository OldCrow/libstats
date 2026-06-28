#include "libstats/distributions/cauchy.h"
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;
using stats::detail::validateNonNegativeParameter;

#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

// Validate x0 and gamma before constructing; throws before any member is built.
static double requireFiniteX0(double x0) {
    if (!std::isfinite(x0)) {
        throw std::invalid_argument("Location parameter x0 must be a finite number");
    }
    return x0;
}
static double requirePositiveGamma(double gamma) {
    if (std::isnan(gamma) || std::isinf(gamma) || gamma <= 0.0) {
        throw std::invalid_argument("Scale parameter gamma must be a positive finite number");
    }
    return gamma;
}

CauchyDistribution::CauchyDistribution(double x0, double gamma)
    : DistributionBase(),
      x0_(requireFiniteX0(x0)),
      gamma_(requirePositiveGamma(gamma)) {
    updateCacheUnsafe();
}

CauchyDistribution::CauchyDistribution(const CauchyDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    x0_       = other.x0_;
    gamma_    = other.gamma_;
    inv_gamma_ = other.inv_gamma_;
    log_gamma_ = other.log_gamma_;
    atomicX0_.store(x0_, std::memory_order_release);
    atomicGamma_.store(gamma_, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
}

CauchyDistribution& CauchyDistribution::operator=(const CauchyDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        x0_        = other.x0_;
        gamma_     = other.gamma_;
        inv_gamma_ = other.inv_gamma_;
        log_gamma_ = other.log_gamma_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicX0_.store(x0_, std::memory_order_release);
        atomicGamma_.store(gamma_, std::memory_order_release);
    }
    return *this;
}

CauchyDistribution::CauchyDistribution(CauchyDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    x0_        = other.x0_;
    gamma_     = other.gamma_;
    inv_gamma_ = other.inv_gamma_;
    log_gamma_ = other.log_gamma_;
    other.x0_    = detail::ZERO_DOUBLE;
    other.gamma_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicX0_.store(x0_, std::memory_order_release);
    atomicGamma_.store(gamma_, std::memory_order_release);
}

CauchyDistribution& CauchyDistribution::operator=(CauchyDistribution&& other) noexcept {
    if (this != &other) {
        x0_        = other.x0_;
        gamma_     = other.gamma_;
        inv_gamma_ = other.inv_gamma_;
        log_gamma_ = other.log_gamma_;
        other.x0_    = detail::ZERO_DOUBLE;
        other.gamma_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicX0_.store(x0_, std::memory_order_release);
        atomicGamma_.store(gamma_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

CauchyDistribution CauchyDistribution::createUnchecked(double x0, double gamma) noexcept {
    return CauchyDistribution(x0, gamma, true);
}

CauchyDistribution::CauchyDistribution(double x0, double gamma,
                                       bool /*bypassValidation*/) noexcept
    : DistributionBase(), x0_(x0), gamma_(gamma) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER SETTERS
//==============================================================================

void CauchyDistribution::setX0(double x0) {
    validateParameters(x0, gamma_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    x0_ = x0;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void CauchyDistribution::setGamma(double gamma) {
    validateParameters(x0_, gamma);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    gamma_ = gamma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void CauchyDistribution::setParameters(double x0, double gamma) {
    validateParameters(x0, gamma);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    x0_    = x0;
    gamma_ = gamma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult CauchyDistribution::trySetX0(double x0) noexcept {
    auto v = validateCauchyParameters(x0, gamma_);
    if (v.isError()) return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    x0_ = x0;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult CauchyDistribution::trySetGamma(double gamma) noexcept {
    auto v = validateCauchyParameters(x0_, gamma);
    if (v.isError()) return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    gamma_ = gamma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult CauchyDistribution::trySetParameters(double x0, double gamma) noexcept {
    auto v = validateCauchyParameters(x0, gamma);
    if (v.isError()) return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    x0_    = x0;
    gamma_ = gamma;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult CauchyDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateCauchyParameters(x0_, gamma_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double CauchyDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, ig = inv_gamma_;
    lock.unlock();

    // PDF_Cauchy(x) = PDF_StudentT(1)((x-x0)/gamma) / gamma
    const double z = (x - x0) * ig;
    return student_t_.getProbability(z) * ig;
}

double CauchyDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, ig = inv_gamma_, lg = log_gamma_;
    lock.unlock();

    // LogPDF_Cauchy(x) = LogPDF_StudentT(1)((x-x0)/gamma) - log(gamma)
    const double z = (x - x0) * ig;
    return student_t_.getLogProbability(z) - lg;
}

double CauchyDistribution::getCumulativeProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, ig = inv_gamma_;
    lock.unlock();

    // CDF_Cauchy(x) = CDF_StudentT(1)((x-x0)/gamma)  [exact, no output scaling]
    const double z = (x - x0) * ig;
    return student_t_.getCumulativeProbability(z);
}

double CauchyDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE)
        throw std::invalid_argument("Probability must be in [0, 1]");
    if (p == detail::ZERO_DOUBLE)
        return -std::numeric_limits<double>::infinity();
    if (p == detail::ONE)
        return  std::numeric_limits<double>::infinity();

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, g = gamma_;
    lock.unlock();

    // Closed form: x0 + gamma * tan(pi * (p - 0.5))
    return x0 + g * std::tan(detail::PI * (p - detail::HALF));
}

double CauchyDistribution::sample(std::mt19937& rng) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, g = gamma_;
    lock.unlock();

    // X = x0 + gamma * Z  where Z ~ StudentT(1) ~ Cauchy(0,1)
    return x0 + g * student_t_.sample(rng);
}

std::vector<double> CauchyDistribution::sample(std::mt19937& rng, size_t n) const {
    std::vector<double> samples;
    samples.reserve(n);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, g = gamma_;
    lock.unlock();

    auto z_samples = student_t_.sample(rng, n);
    for (double z : z_samples)
        samples.push_back(x0 + g * z);
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void CauchyDistribution::fit(const std::vector<double>& values) {
    if (values.empty())
        throw std::invalid_argument("Data vector cannot be empty");

    for (double v : values) {
        if (!std::isfinite(v))
            throw std::invalid_argument("All values must be finite for Cauchy MLE");
    }

    const std::size_t n = values.size();
    const double nd = static_cast<double>(n);

    // Step 1: Sort for quantile-based initial estimates.
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    // Step 2: Median → initial x₀ seed.
    double x0_hat;
    if (n % 2 == 1) {
        x0_hat = sorted[n / 2];
    } else {
        x0_hat = (sorted[n / 2 - 1] + sorted[n / 2]) * detail::HALF;
    }

    // Step 3: IQR/2 → initial γ seed (robust scale estimate).
    double gamma_hat;
    if (n >= 4) {
        const double q1 = sorted[n / 4];
        const double q3 = sorted[(3 * n) / 4];
        gamma_hat = std::max(detail::HIGH_PRECISION_TOLERANCE, (q3 - q1) / detail::TWO);
    } else {
        gamma_hat = detail::ONE;
    }

    // Step 4: Fisher-scoring iterations.
    // Score equations:
    //   ∂L/∂x₀ = Σᵢ 2dᵢ/sᵢ          where dᵢ = xᵢ-x₀, sᵢ = γ²+dᵢ²
    //   ∂L/∂γ  = Σᵢ (-1/γ + 2dᵢ²/(γ·sᵢ))
    // Expected Fisher info (per obs): I(x₀) = I(γ) = 1/(2γ²)
    // Step: Δθ = 2γ² · (1/n) · ∂L/∂θ
    constexpr int    MAX_ITER = 20;
    constexpr double CONV_TOL = 1e-10;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double score_x0 = detail::ZERO_DOUBLE;
        double score_g  = detail::ZERO_DOUBLE;
        const double g2 = gamma_hat * gamma_hat;
        for (double x : values) {
            const double d = x - x0_hat;
            const double s = g2 + d * d;
            score_x0 += detail::TWO * d / s;
            score_g  += -detail::ONE / gamma_hat + detail::TWO * d * d / (gamma_hat * s);
        }

        const double scale = detail::TWO * g2 / nd;  // = 2γ²/n = I⁻¹
        const double dx0  = scale * score_x0;
        const double dg   = scale * score_g;

        x0_hat    += dx0;
        gamma_hat  = std::max(detail::HIGH_PRECISION_TOLERANCE, gamma_hat + dg);

        if (!std::isfinite(x0_hat) || !std::isfinite(gamma_hat)) break;

        if (std::fabs(dx0) < CONV_TOL * (detail::ONE + std::fabs(x0_hat)) &&
            std::fabs(dg)  < CONV_TOL * gamma_hat)
            break;
    }

    // Guard against non-finite output (fallback to seeds).
    if (!std::isfinite(x0_hat))
        x0_hat = (n % 2 == 1) ? sorted[n / 2]
                               : (sorted[n / 2 - 1] + sorted[n / 2]) * detail::HALF;
    if (!std::isfinite(gamma_hat) || gamma_hat <= detail::ZERO_DOUBLE)
        gamma_hat = detail::ONE;

    setParameters(x0_hat, gamma_hat);
}

void CauchyDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                          std::vector<CauchyDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void CauchyDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    x0_    = detail::ZERO_DOUBLE;
    gamma_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

std::string CauchyDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "CauchyDistribution(x0=" << x0_ << ",gamma=" << gamma_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double CauchyDistribution::getEntropy() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    // H = log(4πγ)
    return std::log(detail::FOUR_PI * gamma_);
}

//==============================================================================
// 12b. LOCK-FREE ATOMIC GETTERS
//==============================================================================

double CauchyDistribution::getX0Atomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicX0_.load(std::memory_order_acquire);
    return getX0();
}

double CauchyDistribution::getGammaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicGamma_.load(std::memory_order_acquire);
    return getGamma();
}

//==============================================================================
// 13. SMART BATCH OPERATIONS
// Input transform: z[i] = (x[i] − x₀) / γ
// Delegate to StudentT(1)'s auto-dispatch batch, then scale output.
//==============================================================================

void CauchyDistribution::getProbability(std::span<const double> values,
                                        std::span<double> results,
                                        const detail::PerformanceHint& hint) const {
    const std::size_t n = values.size();
    if (n == 0) return;
    if (n != results.size())
        throw std::invalid_argument("Input and output spans must have the same size");

    // Read cache under lock
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, ig = inv_gamma_;
    lock.unlock();

    // Transform inputs: z[i] = (x[i] - x0) / gamma
    std::vector<double, arch::simd::aligned_allocator<double>> z(n);
    using VectorOps = arch::simd::VectorOps;
    VectorOps::scalar_add(values.data(), -x0, z.data(), n);
    VectorOps::scalar_multiply(z.data(), ig, z.data(), n);

    // Delegate to StudentT(1) — inherits its full SIMD/parallel dispatch
    student_t_.getProbability(std::span<const double>(z.data(), n), results, hint);

    // Scale output: PDF_Cauchy(x) = PDF_StudentT(1)(z) / gamma = PDF_StudentT(1)(z) * inv_gamma
    VectorOps::scalar_multiply(results.data(), ig, results.data(), n);
}

void CauchyDistribution::getLogProbability(std::span<const double> values,
                                           std::span<double> results,
                                           const detail::PerformanceHint& hint) const {
    const std::size_t n = values.size();
    if (n == 0) return;
    if (n != results.size())
        throw std::invalid_argument("Input and output spans must have the same size");

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, ig = inv_gamma_, lg = log_gamma_;
    lock.unlock();

    std::vector<double, arch::simd::aligned_allocator<double>> z(n);
    using VectorOps = arch::simd::VectorOps;
    VectorOps::scalar_add(values.data(), -x0, z.data(), n);
    VectorOps::scalar_multiply(z.data(), ig, z.data(), n);

    student_t_.getLogProbability(std::span<const double>(z.data(), n), results, hint);

    // LogPDF_Cauchy(x) = LogPDF_StudentT(1)(z) - log(gamma)
    VectorOps::scalar_add(results.data(), -lg, results.data(), n);
}

void CauchyDistribution::getCumulativeProbability(std::span<const double> values,
                                                  std::span<double> results,
                                                  const detail::PerformanceHint& hint) const {
    const std::size_t n = values.size();
    if (n == 0) return;
    if (n != results.size())
        throw std::invalid_argument("Input and output spans must have the same size");

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) updateCacheUnsafe();
        ulock.unlock();
        lock.lock();
    }
    const double x0 = x0_, ig = inv_gamma_;
    lock.unlock();

    std::vector<double, arch::simd::aligned_allocator<double>> z(n);
    using VectorOps = arch::simd::VectorOps;
    VectorOps::scalar_add(values.data(), -x0, z.data(), n);
    VectorOps::scalar_multiply(z.data(), ig, z.data(), n);

    // CDF_Cauchy(x) = CDF_StudentT(1)(z) — no output scaling needed
    student_t_.getCumulativeProbability(std::span<const double>(z.data(), n), results, hint);
}

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool CauchyDistribution::operator==(const CauchyDistribution& other) const {
    if (this == &other) return true;
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(x0_    - other.x0_)    <= detail::DEFAULT_TOLERANCE &&
           std::fabs(gamma_ - other.gamma_) <= detail::DEFAULT_TOLERANCE;
}

bool CauchyDistribution::operator!=(const CauchyDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const CauchyDistribution& dist) {
    return os << dist.toString();
}

std::istream& operator>>(std::istream& is, CauchyDistribution& dist) {
    std::string token;
    double x0, gamma;

    is >> token;
    if (!token.starts_with("CauchyDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    const size_t x0_pos    = token.find("x0=");
    const size_t gamma_pos = token.find(",gamma=");
    const size_t close     = token.find(")", gamma_pos != std::string::npos ? gamma_pos : 0);

    if (x0_pos == std::string::npos || gamma_pos == std::string::npos ||
        close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        x0    = std::stod(token.substr(x0_pos + 3, gamma_pos - x0_pos - 3));
        gamma = std::stod(token.substr(gamma_pos + 7, close - gamma_pos - 7));
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    auto result = dist.trySetParameters(x0, gamma);
    if (result.isError())
        is.setstate(std::ios::failbit);
    return is;
}

//==============================================================================
// 20. PRIVATE CACHE MANAGEMENT
//==============================================================================

void CauchyDistribution::updateCacheUnsafe() const noexcept {
    inv_gamma_ = detail::ONE / gamma_;
    log_gamma_ = std::log(gamma_);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicX0_.store(x0_, std::memory_order_release);
    atomicGamma_.store(gamma_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

}  // namespace stats

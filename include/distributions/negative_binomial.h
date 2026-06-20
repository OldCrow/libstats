#pragma once

#include "libstats/common/distribution_common.h"
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Negative Binomial Distribution.
 *
 * @details Models the number of failures k before the r-th success in a
 * sequence of independent Bernoulli trials, each with success probability p.
 * Supports real-valued r (dispersion parameter) for use as a count-data
 * over-dispersion model.
 *
 * @par Mathematical Definition:
 * - PMF:    P(X = k) = C(k+r−1, k)·p^r·(1−p)^k  for k ∈ {0, 1, 2, …}
 * - LogPMF: lgamma(k+r) − lgamma(k+1) − lgamma(r) + r·log(p) + k·log(1−p)
 * - CDF:    I_p(r, k+1)  via the regularized incomplete beta (detail::beta_i)
 * - Quantile: smallest k ≥ 0 such that CDF(k) ≥ p
 * - Parameters: r > 0 (number of successes, real-valued), p ∈ (0,1] (success probability)
 * - Support: k ∈ {0, 1, 2, …}
 *
 * @par Moments:
 * - Mean:     r(1−p)/p
 * - Variance: r(1−p)/p²
 * - Mode:     floor((r−1)(1−p)/p)  for r > 1; 0 for r ≤ 1
 * - Skewness: (2−p)/√(r(1−p))
 * - Excess kurtosis: 6/r + p²/(r(1−p))
 *
 * @par Batch operations:
 * The VECTORIZED path uses a scalar loop — lgamma per element is not in
 * VectorOps.  Loop-invariants logGammaR_ = lgamma(r), logP_ = log(p), and
 * log1mP_ = log(1−p) are cached.  The PARALLEL strategy provides genuine
 * multi-core throughput for large batches.
 *
 * @par MLE:
 * p̂ = r/(r+k̄); r is estimated by Newton-Raphson on the profile score
 * equation using digamma (ψ) and trigamma (ψ₁):
 *   f(r) = Σ[ψ(kᵢ+r) − ψ(r)] + n·[log(r) − log(r+k̄)] = 0
 * Seeded with method-of-moments: r̂ = k̄²/(s²−k̄) when s² > k̄.
 *
 * @author libstats Development Team
 * @version 1.3.0
 * @since 1.3.0
 */
class NegativeBinomialDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<NegativeBinomialDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::NEGATIVE_BINOMIAL;
    static constexpr bool kIsDiscrete = true;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Negative Binomial distribution.
     * @param r  Number of successes (real-valued, r > 0, default 1.0)
     * @param p  Success probability (p ∈ (0,1], default 0.5)
     * @throws std::invalid_argument if r ≤ 0 or p not in (0,1]
     */
    explicit NegativeBinomialDistribution(double r = detail::ONE, double p = detail::HALF);

    /** @brief Thread-safe copy constructor. */
    NegativeBinomialDistribution(const NegativeBinomialDistribution& other);

    /** @brief Copy assignment operator. */
    NegativeBinomialDistribution& operator=(const NegativeBinomialDistribution& other);

    /** @brief Move constructor. */
    NegativeBinomialDistribution(NegativeBinomialDistribution&& other) noexcept;

    /** @brief Move assignment operator. @warning NOT noexcept. */
    NegativeBinomialDistribution& operator=(NegativeBinomialDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~NegativeBinomialDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    [[nodiscard]] static Result<NegativeBinomialDistribution> create(
        double r = detail::ONE, double p = detail::HALF) noexcept {
        auto v = validateNegativeBinomialParameters(r, p);
        if (v.isError())
            return Result<NegativeBinomialDistribution>::makeError(v.error_code, v.message);
        return Result<NegativeBinomialDistribution>::ok(createUnchecked(r, p));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get number of successes r. */
    [[nodiscard]] double getR() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return r_;
    }

    /** @brief Get success probability p. */
    [[nodiscard]] double getP() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return p_;
    }

    /** @brief Lock-free atomic getter for r. */
    [[nodiscard]] double getRAtomic() const noexcept;

    /** @brief Lock-free atomic getter for p. */
    [[nodiscard]] double getPAtomic() const noexcept;

    /**
     * @brief Set r (number of successes).
     * @throws std::invalid_argument if r ≤ 0
     */
    void setR(double r);

    /**
     * @brief Set success probability p.
     * @throws std::invalid_argument if p not in (0,1]
     */
    void setP(double p);

    /**
     * @brief Set both parameters simultaneously.
     */
    void setParameters(double r, double p);

    /** @brief Mean = r(1−p)/p. */
    [[nodiscard]] double getMean() const noexcept override;

    /** @brief Variance = r(1−p)/p². */
    [[nodiscard]] double getVariance() const noexcept override;

    /** @brief Skewness = (2−p)/√(r(1−p)). */
    [[nodiscard]] double getSkewness() const noexcept override;

    /** @brief Excess kurtosis = 6/r + p²/(r(1−p)). */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string getDistributionName() const override {
        return "NegativeBinomialDistribution";
    }

    /** @brief Negative Binomial is discrete. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return true; }

    /** @brief Support lower bound: 0. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return detail::ZERO_DOUBLE;
    }

    /** @brief Support upper bound: +∞. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetR(double r) noexcept;
    [[nodiscard]] VoidResult trySetP(double p) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double r, double p) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PMF at k (rounded to nearest non-negative integer).
     * Returns 0 for negative or non-finite values.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PMF at k.
     * Returns −∞ for negative or non-finite values.
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief CDF via regularized incomplete beta I_p(r, k+1).
     * O(1) computation using existing detail::beta_i.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Smallest k ≥ 0 such that CDF(k) ≥ p.
     * @throws std::invalid_argument if p not in [0,1]
     */
    [[nodiscard]] double getQuantile(double prob) const override;

    /** @brief Sample via Gamma(r, (1−p)/p)-Poisson mixture (supports real r). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit r and p by MLE.
     *
     * Uses method-of-moments seed r̂ = k̄²/(s²−k̄) and Newton-Raphson on the
     * profile score equation using digamma and trigamma.  Falls back to
     * method-of-moments when the data shows no over-dispersion.
     *
     * @param values Non-negative integer observations
     * @throws std::invalid_argument if values is empty
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<NegativeBinomialDistribution>& results);

    /** @brief Reset to default (r=1.0, p=0.5). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = floor((r−1)(1−p)/p) for r > 1; 0 otherwise. */
    [[nodiscard]] double getMode() const noexcept;

    /** @brief Entropy (nats) — Sterling approximation for large r. */
    [[nodiscard]] double getEntropy() const noexcept override;

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================

    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 14. EXPLICIT STRATEGY BATCH OPERATIONS
    //==========================================================================

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const NegativeBinomialDistribution& other) const;
    bool operator!=(const NegativeBinomialDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::NegativeBinomialDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::NegativeBinomialDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static NegativeBinomialDistribution createUnchecked(double r, double p) noexcept;
    NegativeBinomialDistribution(double r, double p, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief Scalar-loop batch PMF / LogPMF.
     *
     * The VECTORIZED path uses a scalar loop because lgamma per element is
     * not in VectorOps.  Cached loop-invariants:
     *   logGammaR_ = lgamma(r)  — avoids recomputing per element
     *   logP_      = log(p)     — avoids recomputing log per element
     *   log1mP_    = log(1−p)   — avoids recomputing log(1−p) per element
     *
     * When a vector_lgamma primitive is added to VectorOps, the hot path
     * becomes fully SIMD-accelerated (like the Gaussian erf path).
     * Until then, PARALLEL is the recommended strategy for large batches.
     */
    void getLogProbabilityBatchImpl(const double* values, double* results, std::size_t count,
                                    double cached_r, double cached_logGammaR, double cached_logP,
                                    double cached_log1mP) const noexcept;

    void getProbabilityBatchImpl(const double* values, double* results, std::size_t count,
                                 double cached_r, double cached_logGammaR, double cached_logP,
                                 double cached_log1mP) const noexcept;

    void getCumulativeProbabilityBatchImpl(const double* values, double* results,
                                           std::size_t count) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double r, double p) {
        if (!std::isfinite(r) || r <= detail::ZERO_DOUBLE)
            throw std::invalid_argument("Number of successes r must be positive and finite");
        if (!std::isfinite(p) || p <= detail::ZERO_DOUBLE || p > detail::ONE)
            throw std::invalid_argument("Success probability p must be in (0, 1]");
    }

    //==========================================================================
    // 20–21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Number of successes r — real-valued, must be positive. */
    double r_{detail::ONE};

    /** @brief Success probability p — must be in (0, 1]. */
    double p_{detail::HALF};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicR_{detail::ONE};
    mutable std::atomic<double> atomicP_{detail::HALF};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief lgamma(r) — dominant loop-invariant in LogPMF. */
    mutable double logGammaR_{detail::ZERO_DOUBLE};

    /** @brief log(p) — avoids per-element log computation. */
    mutable double logP_{detail::ZERO_DOUBLE};

    /** @brief log(1−p) — avoids per-element log computation. */
    mutable double log1mP_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: NegativeBinomial uses standard caching only.
    // Section maintained for template compliance.
};

}  // namespace stats

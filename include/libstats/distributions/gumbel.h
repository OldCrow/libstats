#pragma once

#include "libstats/common/distribution_common.h"

namespace stats {

/**
 * @brief Thread-safe Gumbel Distribution (Type I extreme value, max-stable).
 *
 * @details The Gumbel distribution is the limiting law of the *maximum* of many
 * i.i.d. light-tailed samples, and is the ξ → 0 member of the generalized
 * extreme-value family.  It is the workhorse of extreme-value analysis: flood
 * and wind-speed return levels, reliability and lifetime maxima, and the Gumbel
 * softmax trick in machine learning.
 *
 * @par Variant:
 * This class implements the **max-stable** variant only — scipy's `gumbel_r`.
 * The min-stable reflection (`gumbel_l`, the law of −X) is deliberately *not*
 * provided: it is a one-line user-side sign flip, and issue #54's kickoff
 * decision (2026-09-02) reserved no constructor flag, class, or enum slot for
 * it.  A follow-up issue will be filed only on demand.
 *
 * @par Mathematical Definition:
 * With z = (x − μ)/β:
 * - PDF:    f(x; μ, β) = (1/β)·exp(−z − e^(−z))
 * - LogPDF: −log β − z − e^(−z)
 * - CDF:    F(x; μ, β) = exp(−e^(−z))
 * - Quantile: μ − β·log(−log p)
 * - Parameters: μ ∈ ℝ (location), β > 0 (scale)
 * - Support: x ∈ (−∞, +∞)
 *
 * @par Stable formulations:
 * - **LogPDF is exact in log space** and needs no guarding for finite x: as
 *   z → −∞ the term e^(−z) overflows to +inf and the sum collapses to −inf,
 *   which is the correct limit.  The one hazard is z = ±inf itself, where
 *   −z and −e^(−z) would form inf − inf = NaN; both the scalar path and the
 *   batch fixup decide that case before the arithmetic runs.
 * - **PDF = exp(LogPDF)**, so both tails underflow cleanly to exactly 0 (the
 *   lower tail through LogPDF = −inf, the upper through LogPDF ≈ −z).
 * - **CDF is the double exponential** exp(−e^(−z)).  At the extremes the chain
 *   lands on the exact endpoints rather than near them: z → −∞ gives
 *   e^(−z) = +inf → exp(−inf) = 0, and z → +∞ gives e^(−z) = 0 → exp(−0) = 1.
 * - **Quantile** uses −log p = −log1p(p − 1) for p ≥ 1/2, where p − 1 is exact
 *   (Sterbenz) and log1p is accurate for a tiny argument; the naive log(p)
 *   would lose the upper tail.  For p < 1/2 the plain −log p is already
 *   well conditioned.  The lower tail is *doubly* well conditioned: the outer
 *   logarithm compresses it, so q(1e-15) = μ − β·log(34.5) is accurate to a few
 *   ulp and the quantile diverges only like log(−log p).
 *
 * @par Moments:
 * - Mean: μ + γβ,  γ = 0.5772156649015329 (Euler–Mascheroni)
 * - Median: μ − β·log(log 2) = μ + 0.3665129205816643·β
 * - Mode: μ
 * - Variance: π²β²/6
 * - Skewness: 12√6·ζ(3)/π³ = 1.1395470994046487 (right-skewed, constant)
 * - Excess kurtosis: 12/5
 * - Entropy: log(β) + γ + 1  (nats)
 *
 * @par Batch SIMD (LogPDF, PDF, CDF):
 * All three paths run through existing `VectorOps` primitives (`scalar_add`,
 * `scalar_multiply`, `vector_subtract`, `vector_exp`); no new SIMD kernel is
 * introduced.  One aligned temp buffer holds z:
 *   LogPDF: tmp = z; res = e^(−z) [scalar_multiply(−1) + vector_exp];
 *           res = −e^(−z) − z − log β [scalar_multiply(−1), vector_subtract, scalar_add]
 *   PDF:    LogPDF then vector_exp
 *   CDF:    tmp = z; res = e^(−z); res = −e^(−z); res = exp(−e^(−z))  (two vector_exp)
 *
 * @par #112 aliasing note:
 * `values` is read exactly once, before the first write to `results`; every
 * later step — including the NaN/±inf fixups — reads only the local temp buffer.
 *
 * @par Fit (method of moments, NOT the MLE):
 * Issue #54 prescribes the closed-form moment estimators
 *   β̂ = s·√6/π,   μ̂ = x̄ − γ·β̂
 * with s the unbiased (n − 1) sample standard deviation.  These are consistent
 * and cost one O(n) pass, but they are **not** the maximum-likelihood estimates:
 * the true MLE has no closed form and requires iterating
 *   β = x̄ − Σxᵢe^(−xᵢ/β) / Σe^(−xᵢ/β)
 * to a fixed point.  For small samples the moment estimator of β is the less
 * efficient of the two; callers needing the MLE should refine from this value.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class GumbelDistribution : public DistributionBase {
   public:
    // Dispatch metadata — must agree with kDistributionMeta[GUMBEL].
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::GUMBEL;
    static constexpr bool kIsDiscrete = false;

    /// Skewness of every Gumbel distribution: 12·√6·ζ(3)/π³ (parameter-free).
    static constexpr double kSkewness = 1.1395470994046487;

    /// −log(log 2); the median is μ + kMedianOffset·β.
    static constexpr double kMedianOffset = 0.36651292058166433;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Gumbel (max-stable) distribution.
     * @param mu   Location parameter μ (must be finite, default 0)
     * @param beta Scale parameter β (must be positive and finite, default 1)
     * @throws std::invalid_argument if μ is not finite or β ≤ 0
     */
    explicit GumbelDistribution(double mu = detail::ZERO_DOUBLE, double beta = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    GumbelDistribution(const GumbelDistribution& other);

    /** @brief Copy assignment operator. */
    GumbelDistribution& operator=(const GumbelDistribution& other);

    /** @brief Move constructor. */
    GumbelDistribution(GumbelDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    GumbelDistribution& operator=(GumbelDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~GumbelDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Gumbel distribution without throwing exceptions.
     * @param mu   Location parameter (must be finite)
     * @param beta Scale parameter (must be positive and finite)
     * @return Result containing a valid GumbelDistribution or error info
     */
    [[nodiscard]] static Result<GumbelDistribution> create(double mu = detail::ZERO_DOUBLE,
                                                           double beta = detail::ONE) {
        auto v = validateGumbelParameters(mu, beta);
        if (v.isError())
            return Result<GumbelDistribution>::makeError(v.errorCode(), v.message());
        return Result<GumbelDistribution>::ok(createUnchecked(mu, beta));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get location parameter μ. */
    [[nodiscard]] double getMu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Get scale parameter β. */
    [[nodiscard]] double getBeta() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return beta_;
    }

    /** @brief Lock-free atomic getter for μ. */
    [[nodiscard]] double getMuAtomic() const noexcept;

    /** @brief Lock-free atomic getter for β. */
    [[nodiscard]] double getBetaAtomic() const noexcept;

    /**
     * @brief Set location parameter μ.
     * @throws std::invalid_argument if μ is not finite
     */
    void setMu(double mu);

    /**
     * @brief Set scale parameter β.
     * @throws std::invalid_argument if β ≤ 0
     */
    void setBeta(double beta);

    /** @brief Set both parameters simultaneously. */
    void setParameters(double mu, double beta);

    /** @brief Mean = μ + γβ (γ = Euler–Mascheroni). */
    [[nodiscard]] double getMean() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_ + detail::EULER_MASCHERONI * beta_;
    }

    /** @brief Variance = π²β²/6. */
    [[nodiscard]] double getVariance() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return detail::PI * detail::PI * beta_ * beta_ / detail::SIX;
    }

    /** @brief Skewness = 12√6·ζ(3)/π³ ≈ 1.13955 — independent of μ and β. */
    [[nodiscard]] double getSkewness() const noexcept override { return kSkewness; }

    /** @brief Excess kurtosis = 12/5 = 2.4. */
    [[nodiscard]] double getKurtosis() const noexcept override {
        return detail::TWELVE / detail::FIVE;
    }

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "Gumbel";
    }

    /** @brief Gumbel is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: −∞. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        return -std::numeric_limits<double>::infinity();
    }

    /** @brief Support upper bound: +∞. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetMu(double mu) noexcept;
    [[nodiscard]] VoidResult trySetBeta(double beta) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double mu, double beta) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF = exp(LogPDF); underflows cleanly to 0 at both tails.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF: −log β − z − e^(−z), exact in log space.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF: exp(−e^(−z)); reaches exactly 0 and 1 at the extremes.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile (inverse CDF): μ − β·log(−log p), with −log p computed as
     * −log1p(p − 1) for p ≥ 1/2 to preserve the upper tail.
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Single random sample via inverse CDF. */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit μ and β by the **method of moments** (not the MLE).
     *
     *   β̂ = s·√6/π,   μ̂ = x̄ − γ·β̂
     *
     * with s the unbiased (n − 1) sample standard deviation.  See the class
     * documentation for why this is not the maximum-likelihood estimate.
     *
     * @param values Observed data (finite, at least two non-identical points)
     * @throws std::invalid_argument if values is empty, has fewer than two
     *         observations, contains non-finite values, or is constant
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting across multiple independent datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<GumbelDistribution>& results);

    /** @brief Reset to default (μ = 0, β = 1). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = μ (the density peaks exactly at the location parameter). */
    [[nodiscard]] double getMode() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Median = μ − β·log(log 2) (closed form). */
    [[nodiscard]] double getMedian() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_ + kMedianOffset * beta_;
    }

    /** @brief Entropy = log(β) + γ + 1  (nats). */
    [[nodiscard]] double getEntropy() const override;

    /** @brief Check if this is the standard Gumbel (μ=0, β=1). */
    [[nodiscard]] bool isStandard() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return std::fabs(mu_) < detail::DEFAULT_TOLERANCE &&
               std::fabs(beta_ - detail::ONE) < detail::DEFAULT_TOLERANCE;
    }

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================
    // For all three overloads below: values and results must have the same
    // size (a mismatch throws std::invalid_argument) and must not overlap (#112).
    // An in-place call silently returns wrong values; overlap is caught only by
    // a debug-mode assert in detail::DispatchUtils. Full contract in
    // core/distribution_interface.h.

    /** @brief Batch PDF — LogPDF pipeline plus a closing vector_exp. */
    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch log-PDF via one vector_exp and three linear ops. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch CDF via the double-exponential chain (two vector_exp calls). */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const GumbelDistribution& other) const;
    bool operator!=(const GumbelDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::GumbelDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::GumbelDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static GumbelDistribution createUnchecked(double mu, double beta) noexcept;
    GumbelDistribution(double mu, double beta, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief SIMD LogPDF pipeline: −log β − z − e^(−z). */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_mu, double cached_inv_beta,
                                          double cached_neg_log_beta) const noexcept;

    /** @brief SIMD PDF pipeline — LogPDF followed by vector_exp. */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_mu, double cached_inv_beta,
                                       double cached_neg_log_beta) const noexcept;

    /** @brief SIMD CDF pipeline — the exp(−exp(−z)) double-exponential chain. */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_mu,
                                                 double cached_inv_beta) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double mu, double beta) {
        if (!std::isfinite(mu))
            throw std::invalid_argument("Location parameter mu must be a finite number");
        if (!std::isfinite(beta) || beta <= detail::ZERO_DOUBLE)
            throw std::invalid_argument("Scale parameter beta must be a positive finite number");
    }

    //==========================================================================
    // 20. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Location parameter μ. */
    double mu_{detail::ZERO_DOUBLE};

    /** @brief Scale parameter β — must be positive. */
    double beta_{detail::ONE};

    /** @brief Atomic copies for lock-free parameter access. */
    mutable std::atomic<double> atomicMu_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicBeta_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 21. PERFORMANCE CACHE
    //==========================================================================

    /** @brief 1/β — converts x − μ into z. */
    mutable double inv_beta_{detail::ONE};

    /** @brief −log(β) — constant term of LogPDF. */
    mutable double neg_log_beta_{detail::ZERO_DOUBLE};
};

}  // namespace stats

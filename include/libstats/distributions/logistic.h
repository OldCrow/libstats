#pragma once

#include "libstats/common/distribution_common.h"

namespace stats {

/**
 * @brief Thread-safe Logistic Distribution.
 *
 * @details The logistic distribution is the sampling distribution behind the
 * logit link: its CDF *is* the standard sigmoid.  It looks like a Gaussian with
 * heavier tails (excess kurtosis 6/5 rather than 0) and appears in logistic
 * regression, Bayesian logistic priors, item-response theory, and growth
 * modelling (the Gompertz/logistic growth limit).
 *
 * @par Mathematical Definition:
 * With z = (x − μ)/s:
 * - PDF:    f(x; μ, s) = e^(−z) / (s·(1 + e^(−z))²)
 * - LogPDF: −z − 2·log(1 + e^(−z)) − log s
 * - CDF:    F(x; μ, s) = 1 / (1 + e^(−z))
 * - Quantile: μ + s·(log p − log(1 − p))
 * - Parameters: μ ∈ ℝ (location), s > 0 (scale)
 * - Support: x ∈ (−∞, +∞)
 *
 * @par Stable formulations (this is the point of the implementation):
 * The textbook forms above overflow at one tail apiece.  Every path here is
 * written so that the only quantity ever exponentiated is −|z| ∈ (−∞, 0], which
 * cannot overflow and underflows cleanly to 0:
 * - PDF is even in z, so  f = e^(−|z|) / (s·(1 + e^(−|z|))²)  for all x.
 * - LogPDF likewise:      −|z| − 2·log1p(e^(−|z|)) − log s.
 * - CDF branches on the sign instead of on the magnitude:
 *       z ≥ 0 →  1 / (1 + e^(−z))          (no overflow, → 1 monotonically)
 *       z < 0 →  e^(z) / (1 + e^(z))       (no overflow, → 0 monotonically)
 *   Both branches agree at z = 0 (= 1/2) and neither ever forms e^(+|z|).
 * - Quantile keeps the two logarithms split: log(p) − log1p(−p).  The naive
 *   log(p/(1−p)) loses the entire upper tail — at p = 1 − 1e-15 the subtraction
 *   1 − p is catastrophic, while log1p(−p) is exact to a few ulp.
 *
 * @par Moments:
 * - Mean = Median = Mode = μ
 * - Variance: s²π²/3
 * - Skewness: 0 (symmetric)
 * - Excess kurtosis: 6/5
 * - Entropy: log(s) + 2  (nats)
 *
 * @par Batch SIMD (LogPDF, PDF, CDF):
 * All three paths are fully vectorised through existing `VectorOps` primitives
 * (`scalar_add`, `scalar_multiply`, `vector_add`, `vector_exp`, `vector_log`);
 * no new SIMD kernel is introduced.  LogPDF pipeline, one aligned temp buffer:
 *   Step 1: tmp = x − μ                     [scalar_add(values, −μ, tmp)]
 *   Step 2: tmp = −|x − μ|/s = −|z|         [fabs loop + scalar_multiply(−1/s)]
 *   Step 3: res = e^(−|z|)                  [vector_exp(tmp, res)]
 *   Step 4: res = 1 + e^(−|z|)              [scalar_add(res, 1)]
 *   Step 5: res = log(1 + e^(−|z|))         [vector_log(res, res)]
 *   Step 6: res = −2·log(1 + e^(−|z|))      [scalar_multiply(res, −2)]
 *   Step 7: res = res + tmp                 [vector_add(tmp, res, res)]
 *   Step 8: res = res − log s               [scalar_add(res, −log s)]
 * PDF appends a `vector_exp`.  The vector path uses log(1 + e) where the scalar
 * path uses log1p(e); since the argument e = e^(−|z|) never exceeds 1, the two
 * differ by at most one ulp of 1 (≈ 2.2e-16 absolute in the log term), far
 * inside the 1e-10 batch-vs-scalar tolerance.
 *
 * @par #112 aliasing note:
 * Every batch step reads only the caller's input *before* the first write to
 * `results`, and every later decision (including the NaN/±inf fixups) is taken
 * from the local temp buffer, never from a re-read of `values`.
 *
 * @par MLE:
 * - μ̂ = median(data)  — consistent, since the logistic median equals μ.
 * - ŝ  solves the conditional score equation Σ zᵢ·tanh(zᵢ/2) = n with μ fixed
 *   at μ̂, by safeguarded Newton–Raphson in log s (the score is monotone there).
 * This is the *conditional* (profile-at-the-median) MLE, not the joint MLE:
 * the joint solution would iterate μ and s together.  Both estimators are
 * consistent; the conditional one is what issue #54 prescribes and it costs one
 * sort plus a handful of O(n) Newton steps.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class LogisticDistribution : public DistributionBase {
   public:
    // Dispatch metadata — must agree with kDistributionMeta[LOGISTIC].
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::LOGISTIC;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Logistic distribution.
     * @param mu Location parameter μ (must be finite, default 0)
     * @param s  Scale parameter s (must be positive and finite, default 1)
     * @throws std::invalid_argument if μ is not finite or s ≤ 0
     */
    explicit LogisticDistribution(double mu = detail::ZERO_DOUBLE, double s = detail::ONE);

    /** @brief Thread-safe copy constructor. */
    LogisticDistribution(const LogisticDistribution& other);

    /** @brief Copy assignment operator. */
    LogisticDistribution& operator=(const LogisticDistribution& other);

    /** @brief Move constructor. */
    LogisticDistribution(LogisticDistribution&& other) noexcept;

    /** @brief Move assignment operator. */
    LogisticDistribution& operator=(LogisticDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~LogisticDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Logistic distribution without throwing exceptions.
     * @param mu Location parameter (must be finite)
     * @param s  Scale parameter (must be positive and finite)
     * @return Result containing a valid LogisticDistribution or error info
     */
    [[nodiscard]] static Result<LogisticDistribution> create(double mu = detail::ZERO_DOUBLE,
                                                             double s = detail::ONE) {
        auto v = validateLogisticParameters(mu, s);
        if (v.isError())
            return Result<LogisticDistribution>::makeError(v.errorCode(), v.message());
        return Result<LogisticDistribution>::ok(createUnchecked(mu, s));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get location parameter μ. */
    [[nodiscard]] double getMu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Get scale parameter s. */
    [[nodiscard]] double getS() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return s_;
    }

    /** @brief Lock-free atomic getter for μ. */
    [[nodiscard]] double getMuAtomic() const noexcept;

    /** @brief Lock-free atomic getter for s. */
    [[nodiscard]] double getSAtomic() const noexcept;

    /**
     * @brief Set location parameter μ.
     * @throws std::invalid_argument if μ is not finite
     */
    void setMu(double mu);

    /**
     * @brief Set scale parameter s.
     * @throws std::invalid_argument if s ≤ 0
     */
    void setS(double s);

    /** @brief Set both parameters simultaneously. */
    void setParameters(double mu, double s);

    /** @brief Mean = Median = Mode = μ. */
    [[nodiscard]] double getMean() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Variance = s²π²/3. */
    [[nodiscard]] double getVariance() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return s_ * s_ * detail::PI * detail::PI / detail::THREE;
    }

    /** @brief Skewness = 0 (symmetric). */
    [[nodiscard]] double getSkewness() const noexcept override { return detail::ZERO_DOUBLE; }

    /** @brief Excess kurtosis = 6/5 = 1.2. */
    [[nodiscard]] double getKurtosis() const noexcept override { return detail::SIX / detail::FIVE; }

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "Logistic";
    }

    /** @brief Logistic is continuous. */
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
    [[nodiscard]] VoidResult trySetS(double s) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double mu, double s) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF via the symmetric stable form e^(−|z|) / (s·(1 + e^(−|z|))²).
     * Exponentiates only −|z|, so neither tail can overflow.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF: −|z| − 2·log1p(e^(−|z|)) − log s.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF via the sign-branched sigmoid (see class doc): monotone to the
     * exact limits 0 and 1, no overflow at either tail.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile (inverse CDF): μ + s·(log p − log1p(−p)).
     * The split logarithm is required — log(p/(1−p)) loses the p→1 tail.
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
     * @brief Fit μ and s (conditional MLE — see class documentation).
     *
     *   μ̂ = median(data)                                  O(n log n)
     *   ŝ  solves Σ zᵢ·tanh(zᵢ/2) = n, zᵢ = (xᵢ − μ̂)/s     O(n) per Newton step
     *
     * @param values Observed data (must be finite, at least two distinct points)
     * @throws std::invalid_argument if values is empty, contains non-finite
     *         values, or is degenerate (all observations identical)
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting across multiple independent datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<LogisticDistribution>& results);

    /** @brief Reset to default (μ = 0, s = 1). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = μ (the density peaks at the location parameter). */
    [[nodiscard]] double getMode() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Median = μ (exact; CDF(μ) = 1/2 by symmetry). */
    [[nodiscard]] double getMedian() const override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /**
     * @brief Entropy = log(s) + 2  (nats).
     * Independent of μ; increasing in s.
     */
    [[nodiscard]] double getEntropy() const override;

    /** @brief Check if this is the standard Logistic (μ=0, s=1). */
    [[nodiscard]] bool isStandard() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return std::fabs(mu_) < detail::DEFAULT_TOLERANCE &&
               std::fabs(s_ - detail::ONE) < detail::DEFAULT_TOLERANCE;
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

    /** @brief Batch log-PDF via the eight-step vector_exp/vector_log pipeline. */
    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    /** @brief Batch CDF via e^(−|z|) followed by a sign-selected reciprocal. */
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const LogisticDistribution& other) const;
    bool operator!=(const LogisticDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::LogisticDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::LogisticDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static LogisticDistribution createUnchecked(double mu, double s) noexcept;
    LogisticDistribution(double mu, double s, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief SIMD LogPDF pipeline (Steps 1–8 in the class documentation). */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_mu, double cached_neg_inv_s,
                                          double cached_neg_log_s) const noexcept;

    /** @brief SIMD PDF pipeline — LogPDF followed by vector_exp. */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_mu, double cached_neg_inv_s,
                                       double cached_neg_log_s) const noexcept;

    /** @brief SIMD CDF pipeline — e^(−|z|) then the sign-selected sigmoid form. */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_mu,
                                                 double cached_inv_s) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double mu, double s) {
        if (!std::isfinite(mu))
            throw std::invalid_argument("Location parameter mu must be a finite number");
        if (!std::isfinite(s) || s <= detail::ZERO_DOUBLE)
            throw std::invalid_argument("Scale parameter s must be a positive finite number");
    }

    //==========================================================================
    // 20. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Location parameter μ. */
    double mu_{detail::ZERO_DOUBLE};

    /** @brief Scale parameter s — must be positive. */
    double s_{detail::ONE};

    /** @brief Atomic copies for lock-free parameter access. */
    mutable std::atomic<double> atomicMu_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicS_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 21. PERFORMANCE CACHE
    //==========================================================================

    /** @brief 1/s — converts x − μ into z in the CDF pipeline. */
    mutable double inv_s_{detail::ONE};

    /** @brief −1/s — turns |x − μ| into −|z| in the LogPDF pipeline. */
    mutable double neg_inv_s_{-detail::ONE};

    /** @brief −log(s) — constant term of LogPDF. */
    mutable double neg_log_s_{detail::ZERO_DOUBLE};
};

}  // namespace stats

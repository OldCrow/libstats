#pragma once

#include "libstats/common/distribution_common.h"

namespace stats {

/**
 * @brief Thread-safe Half-Normal Distribution for positive-constrained Gaussian data.
 *
 * @details The Half-Normal distribution is the distribution of |X| where
 * X ~ Normal(0, σ²). It is the standard weakly-informative prior for scale
 * parameters in Bayesian modelling and arises for magnitudes of zero-mean
 * Gaussian errors.
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; σ) = √(2/π)/σ · exp(−x²/(2σ²))  for x ≥ 0
 * - LogPDF: ½·log(2/π) − log(σ) − x²/(2σ²)  (exact log space)
 * - CDF:    F(x; σ) = erf(x/(σ√2))  for x ≥ 0
 * - Quantile: Q(p) = σ√2·erf⁻¹(p)
 * - Parameters: σ > 0 (scale)
 * - Support:  x ∈ [0, ∞)
 *
 * @par Moments:
 * - Mean:     σ·√(2/π)  ≈ 0.7979·σ
 * - Variance: σ²·(1 − 2/π)  ≈ 0.3634·σ²
 * - Mode:     0
 * - Median:   σ√2·erf⁻¹(½) ≈ 0.6745·σ
 * - Skewness: √2·(4−π)/(π−2)^(3/2)  ≈ 0.9953 (constant)
 * - Excess kurtosis: 8(π−3)/(π−2)²  ≈ 0.8692 (constant)
 * - Entropy:  ½·log(πσ²/2) + ½
 *
 * @par Numerical notes (tail behaviour):
 * - CDF lower tail is benign: erf near 0 keeps full relative precision, so
 *   F(x) is accurate to ~1 ulp for small x — no tail branch is needed
 *   (contrast with the Gaussian CDF's erfc branch for z < 0, #49: that
 *   hazard came from the 1+erf cancellation, which this formulation never
 *   forms).
 * - CDF upper tail: F(x) → 1 is representable and returned at full absolute
 *   precision, but the survival probability 1−F(x) computed from it cancels
 *   catastrophically for x ≳ 6σ (1−F underflows the 1-ulp floor near
 *   x ≈ 8.2σ). This class deliberately provides no survival API; callers
 *   needing tail probabilities should use erfc(x/(σ√2)) directly.
 * - Quantile: central region (p ≤ ½) via detail::erf_inv plus a Newton
 *   polish on the erf residual; upper tail (p > ½) entirely in the
 *   erfc/survival domain (1−p is exact by Sterbenz there), via an AS 26.2.23
 *   seed refined by Newton on the erfc residual. Never NaN for p ∈ (0,1)
 *   (#104 contract); accuracy is limited only by the ~|ln(1−p)|·2⁻⁵²
 *   conditioning law near p = 1 — the deep-tail band where detail::erf_inv's
 *   extreme-tail branch is unreliable (measured during #57 bring-up) is
 *   deliberately bypassed.
 *
 * @par Batch SIMD:
 * PDF (4 steps + fixup): x² → ·(−1/(2σ²)) → exp → ·(√(2/π)/σ); x < 0 → 0.
 * LogPDF (3 steps + fixup): x² → ·(−1/(2σ²)) → +log-norm; x < 0 → −∞.
 * CDF (2 steps + fixup): ·(1/(σ√2)) → vector_erf; x < 0 → 0.
 * The CDF pipeline is the Gaussian batch CDF minus the mean shift and the
 * 0.5·(1+·) affine step — and needs no per-lane erfc fixup because the erf
 * argument is never negative inside the support.
 *
 * @par MLE:
 * Closed-form: σ̂ = √(Σxᵢ²/n). Single pass, no iteration.
 *
 * @par Applications:
 * - Bayesian scale priors (positive-constrained Normal)
 * - Measurement-error magnitude models
 * - Absolute-value transforms of symmetric zero-mean data
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class HalfNormalDistribution : public DistributionBase {
   public:
    // Dispatch metadata — must match the kDistributionMeta[] row
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::HALF_NORMAL;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Half-Normal distribution.
     * @param sigma Scale parameter σ (must be positive, default 1)
     * @throws std::invalid_argument if sigma is not strictly positive or non-finite
     *
     * Default σ = 1 is the standard Half-Normal distribution.
     * Implementation in .cpp.
     */
    explicit HalfNormalDistribution(double sigma = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    HalfNormalDistribution(const HalfNormalDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    HalfNormalDistribution& operator=(const HalfNormalDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    HalfNormalDistribution(HalfNormalDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    HalfNormalDistribution& operator=(HalfNormalDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~HalfNormalDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Half-Normal distribution without throwing exceptions.
     * @param sigma Scale parameter σ (must be positive)
     * @return Result containing a valid HalfNormalDistribution or error info
     */
    [[nodiscard]] static Result<HalfNormalDistribution> create(double sigma = detail::ONE) {
        auto validation = validateHalfNormalParameters(sigma);
        if (validation.isError()) {
            return Result<HalfNormalDistribution>::makeError(validation.errorCode(),
                                                             validation.message());
        }
        return Result<HalfNormalDistribution>::ok(createUnchecked(sigma));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get scale parameter σ. */
    [[nodiscard]] double getSigma() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return sigma_;
    }

    /** @brief Lock-free atomic getter for σ. */
    [[nodiscard]] double getSigmaAtomic() const noexcept;

    /**
     * @brief Set scale parameter σ.
     * @throws std::invalid_argument if sigma <= 0 or non-finite
     */
    void setSigma(double sigma);

    /** @brief Alias for setSigma. */
    void setParameters(double sigma);

    /** @brief Mean = σ·√(2/π). */
    [[nodiscard]] double getMean() const override;

    /** @brief Variance = σ²·(1 − 2/π). */
    [[nodiscard]] double getVariance() const override;

    /** @brief Skewness ≈ 0.9953 (constant, independent of σ). */
    [[nodiscard]] double getSkewness() const override;

    /** @brief Excess kurtosis ≈ 0.8692 (constant, independent of σ). */
    [[nodiscard]] double getKurtosis() const override;

    /** @brief Number of parameters (always 1). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 1; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "HalfNormal";
    }

    /** @brief Half-Normal is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: 0 (inclusive — PDF(0) is the mode). */
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

    [[nodiscard]] VoidResult trySetSigma(double sigma) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double sigma) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: √(2/π)/σ · exp(−x²/(2σ²)) for x ≥ 0; 0 for x < 0.
     * PDF(0) = √(2/π)/σ is the mode (support includes 0).
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: ½·log(2/π) − log(σ) − x²/(2σ²) (exact log space).
     * Returns −∞ for x < 0 and for x = +∞.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF: erf(x/(σ√2)) for x ≥ 0; 0 for x < 0.
     *
     * Lower tail keeps full relative precision (erf near 0 is
     * well-conditioned). Upper tail returns values approaching 1 exactly;
     * the survival probability 1−CDF computed by the caller cancels for
     * x ≳ 6σ — see the class-level numerical notes.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile: σ√2·erf⁻¹(p), tail-branched (see class-level notes).
     * @throws std::invalid_argument if p not in [0, 1]
     *
     * Q(0) = 0, Q(1) = +∞. For p ∈ (0,1) the result is always finite and
     * never NaN (#104). p ≤ ½ uses detail::erf_inv with a Newton polish;
     * p > ½ is solved in the erfc/survival domain, keeping full law-limited
     * accuracy all the way to p = 1 − ulp.
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate one random sample as |Z|, Z ~ Normal(0, σ²) (Box–Muller).
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples (pairwise Box–Muller, absolute value). */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit σ to data by closed-form MLE.
     *
     * σ̂ = √(Σxᵢ²/n). Single-pass O(n) computation, no iteration.
     *
     * @param values Observed data (must all be non-negative and finite,
     *               with at least one strictly positive value)
     * @throws std::invalid_argument on empty data, negative or non-finite
     *         values, or all-zero data
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple independent datasets.
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<HalfNormalDistribution>& results);

    /** @brief Reset to default (σ = 1 — standard Half-Normal). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = 0 (density is maximal at the origin). */
    [[nodiscard]] double getMode() const;

    /** @brief Median = σ√2·erf⁻¹(½) ≈ 0.6745·σ. */
    [[nodiscard]] double getMedian() const override;

    /** @brief Entropy = ½·log(πσ²/2) + ½. */
    [[nodiscard]] double getEntropy() const override;

    //==========================================================================
    // 13. SMART AUTO-DISPATCH BATCH OPERATIONS
    //==========================================================================
    // For all three overloads below: values and results must have the same
    // size (a mismatch throws std::invalid_argument) and must not overlap (#112).
    // An in-place call silently returns wrong values; overlap is caught only by
    // a debug-mode assert in detail::DispatchUtils. Full contract in
    // core/distribution_interface.h.

    void getProbability(std::span<const double> values, std::span<double> results,
                        const detail::PerformanceHint& hint = {}) const;

    void getLogProbability(std::span<const double> values, std::span<double> results,
                           const detail::PerformanceHint& hint = {}) const;

    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                  const detail::PerformanceHint& hint = {}) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const HalfNormalDistribution& other) const;
    bool operator!=(const HalfNormalDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::HalfNormalDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::HalfNormalDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static HalfNormalDistribution createUnchecked(double sigma) noexcept;
    HalfNormalDistribution(double sigma, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD PDF pipeline (Gaussian PDF pipeline with constant offsets):
     *   Step 1: results = x²                       [vector_multiply(values, values)]
     *   Step 2: results = −x²/(2σ²)               [scalar_multiply(negHalfInvSigmaSquared_)]
     *   Step 3: results = exp(−x²/(2σ²))          [vector_exp]
     *   Step 4: results ·= √(2/π)/σ               [scalar_multiply(normConstant_)]
     * Scalar fixup: x < 0 → 0 (NaN lanes compare false and propagate NaN).
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_neg_half_inv_sigma2,
                                       double cached_norm_constant) const noexcept;

    /**
     * @brief SIMD LogPDF pipeline:
     *   Step 1: results = x²                       [vector_multiply(values, values)]
     *   Step 2: results = −x²/(2σ²)               [scalar_multiply(negHalfInvSigmaSquared_)]
     *   Step 3: results += ½log(2/π) − log σ      [scalar_add(logNormConst_)]
     * Scalar fixup: x < 0 → −∞.
     */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_neg_half_inv_sigma2,
                                          double cached_log_norm_const) const noexcept;

    /**
     * @brief SIMD CDF pipeline (the Gaussian CDF pipeline minus the shift):
     *   Step 1: results = x/(σ√2)                  [scalar_multiply(invSigmaSqrt2_)]
     *   Step 2: results = erf(x/(σ√2))             [vector_erf]
     * Scalar fixup: x < 0 → 0. No per-lane erfc fixup is needed: the erf
     * argument is non-negative inside the support, so the #49 left-tail
     * cancellation band never occurs here.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count,
                                                 double cached_inv_sigma_sqrt2) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double sigma) {
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Sigma (σ) must be a positive finite number");
        }
    }

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Scale parameter σ — must be positive. */
    double sigma_{detail::ONE};

    /** @brief Atomic copy for lock-free access. */
    mutable std::atomic<double> atomicSigma_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief log(σ) — used in logNormConst_ and entropy. */
    mutable double logSigma_{detail::ZERO_DOUBLE};

    /** @brief −1/(2σ²) — exponent coefficient in PDF/LogPDF pipelines. */
    mutable double negHalfInvSigmaSquared_{-detail::HALF};

    /** @brief √(2/π)/σ — PDF normalisation constant. */
    mutable double normConstant_{detail::ZERO_DOUBLE};

    /** @brief ½·log(2/π) − log(σ) — additive LogPDF normalisation. */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief 1/(σ√2) — CDF erf-argument scale. */
    mutable double invSigmaSqrt2_{detail::ZERO_DOUBLE};

    /** @brief σ√2 — quantile scale. */
    mutable double sigmaSqrt2_{detail::ZERO_DOUBLE};

    /** @brief Cached mean = σ·√(2/π). */
    mutable double mean_{detail::ZERO_DOUBLE};

    /** @brief Cached variance = σ²·(1 − 2/π). */
    mutable double variance_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 23–24. OPTIMIZATION FLAGS / SPECIALIZED CACHES
    //==========================================================================

    // Note: Half-Normal uses standard caching only.
    // Sections maintained for template compliance.
};

}  // namespace stats

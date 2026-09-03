#pragma once

#include "libstats/common/distribution_common.h"

namespace stats {

/**
 * @brief Thread-safe Truncated Normal Distribution — a Gaussian renormalized
 *        to the window [a, b].
 *
 * @details X ~ N(μ, σ²) conditioned on a ≤ X ≤ b. Either bound may be ±∞;
 * a = −∞ with b = +∞ degenerates to the plain Gaussian and is deliberately
 * ALLOWED (Z = 1 exactly, every formula collapses to the Gaussian one).
 *
 * @par Mathematical Definition:
 * - α = (a−μ)/σ, β = (b−μ)/σ, ξ = (x−μ)/σ
 * - Z = Φ(β) − Φ(α)  (normalization constant, cached)
 * - PDF:    f(x) = φ(ξ)/(σZ)  for x ∈ [a, b], else 0
 * - LogPDF: −log σ − log Z − ½ξ² − ½log(2π)  (exact log space), else −∞
 * - CDF:    F(x) = (Φ(ξ) − Φ(α))/Z, clamped to [0,1]; exactly 0 at x ≤ a
 *           and exactly 1 at x ≥ b
 * - Quantile: μ + σ·Φ⁻¹(Φ(α) + p·Z), computed regime-split (see below)
 * - Parameters: μ finite, σ > 0, a < b (a may be −∞, b may be +∞)
 * - Support:  x ∈ [a, b]
 *
 * @par THE core numerical hazard — Z in same-tail windows (#57):
 * When both bounds sit in the same Gaussian tail (e.g. a = 8σ, b = 9σ), the
 * textbook Z = Φ(β) − Φ(α) subtracts two numbers that are both ≈ 1 and
 * differ by ~10⁻¹⁶ of their magnitude — catastrophic cancellation that
 * destroys every significant digit. Z, the CDF numerator, and the quantile
 * targets are therefore computed regime-split, never through the cancelling
 * form:
 *   - both α, β ≥ 0 (right tail):  Z = ½·(erfc(α/√2) − erfc(β/√2))
 *     — a difference of SMALL same-scale quantities, each carried at full
 *     relative precision by erfc; well-conditioned because erfc(β) ≪ erfc(α)
 *     for any window wider than a few ulps.
 *   - both α, β ≤ 0 (left tail):   Z = ½·(erfc(−β/√2) − erfc(−α/√2))
 *     — the reflection of the same identity.
 *   - straddling (α < 0 < β):      Z = ½·(erf(β/√2) − erf(α/√2))
 *     — both erf arguments benign; the two terms have OPPOSITE signs, so the
 *     subtraction is an addition of magnitudes (this branch is mandatory:
 *     the Φ-difference form cancels for narrow windows around μ, e.g.
 *     a = −10⁻⁹, b = 10⁻⁹).
 *   ±∞ bounds collapse correctly through erfc(−∞) = 2, erfc(+∞) = 0,
 *   erf(±∞) = ±1 (verified by test, not assumed).
 * α, β, Z, log Z and the erf/erfc tail pieces are cached at construction and
 * invalidated whenever ANY of μ, σ, a, b changes.
 *
 * @par Supported truncation window (log Z underflow policy — DECIDED):
 * The factory and every setter REJECT parameter sets whose regime-split Z
 * computes to exactly 0 in double precision. That happens when (i) the
 * window lies entirely beyond ≈ ±37.5σ, where erfc((x−μ)/(σ√2)) underflows
 * double (erfc underflows near argument 26.6, i.e. |ξ| ≈ 37.6), or (ii) the
 * window is so narrow that the tail-piece difference rounds to zero (window
 * mass below ~ulp of the tail values). Within the accepted window Z > 0 is
 * guaranteed, log Z is finite, and LogPDF never returns NaN or a clamp
 * constant inside the support. A log-domain asymptotic (log-erfc) extension
 * beyond ±37.5σ was considered and deliberately not implemented: it would
 * add an asymptotic-series code path exercised only by windows with total
 * mass below 10⁻³⁰⁰. Verified at the edge: (a=37σ, b=38σ) is accepted
 * (Z ≈ 5.7e-300); (a=40σ, b=41σ) is rejected.
 *
 * @par CDF numerator:
 * F(x) needs Φ(ξ) − Φ(α), which cancels exactly like Z does. The same
 * regime split is applied per evaluation (α ≥ 0 → erfc-difference;
 * ξ ≤ 0 → reflected erfc-difference; α < 0 < ξ → erf-difference), then the
 * ratio is clamped to [0,1]. x ≤ a returns exactly 0, x ≥ b exactly 1.
 *
 * @par Quantile (regime-split, #104 contract):
 * The target Φ(α) + p·Z is formed only as a SUM of non-negative
 * quantities: q_low = Φ(α) + p·Z for the lower half and
 * s_high = (1−Φ(β)) + (1−p)·Z for the upper half (q_low + s_high = 1
 * identically), then whichever is ≤ ½ is inverted entirely in the
 * erfc/survival domain (AS 26.2.23 seed + Newton on the erfc residual —
 * same solver as HalfNormal's upper tail). No cancelled difference is ever
 * reconstructed, so same-tail windows keep full law-limited accuracy:
 * relative error ~|ln F|·2⁻⁵² (#49 law), never NaN for p ∈ (0,1), p = 0 → a,
 * p = 1 → b.
 *
 * @par Sampling:
 * Inverse-CDF transform: x = quantile(U), U ~ Uniform(0,1). Because the
 * quantile is computed tail-stably in the survival domain, this is exact
 * and efficient in EVERY regime — including far-tail one-sided and
 * two-sided windows where naive accept-reject against the parent Gaussian
 * stalls (acceptance probability = Z → 0). Robert (1995, Statistics and
 * Computing 5:121–125) exponential-proposal rejection was considered and is
 * unnecessary given the tail-stable quantile.
 *
 * @par Moments:
 * With δ = (φ(α)−φ(β))/Z and the raw-moment recursion
 * m_k = (k−1)·m_{k−2} + (α^{k−1}φ(α) − β^{k−1}φ(β))/Z (terms with an
 * infinite bound are 0):
 * - Mean:     μ + σδ
 * - Variance: σ²·(1 + (αφ(α)−βφ(β))/Z − δ²)
 * - Skewness/kurtosis from central moments of the recursion.
 * - Entropy:  ½log(2πe) + log σ + log Z + (αφ(α)−βφ(β))/(2Z)
 * CONDITIONING NOTE: variance/skewness/kurtosis are differences of
 * same-scale moment terms; in deep same-tail windows they lose relative
 * digits proportionally to how far the window sits in the tail (measured:
 * variance relative error ~1e-12 at (a,b) = (8σ,9σ), ~1e-9 at (37σ,38σ)).
 * This is a property of the moment formulas, not of Z.
 *
 * @par MLE (fit):
 * SCOPE DECISION (standard formulation): the truncation bounds a, b are
 * treated as KNOWN and kept at their current values; fit() estimates μ and
 * σ only. With fixed bounds the truncated normal is a two-parameter
 * exponential family in (Σx, Σx²), so the MLE equations coincide with
 * matching the first two truncated moments. fit() solves them by the
 * standard fixed-point iteration σ² ← s²/(1 + η − δ²), μ ← x̄ − σδ
 * (Cohen 1959 style), convergence-guarded (relative tolerance 1e-12,
 * max 500 iterations, degenerate-ratio and Z-underflow guards throw).
 *
 * @par Batch SIMD:
 * PDF/LogPDF: the Gaussian log-space pipeline plus the −log Z offset;
 * bounded-support compute+fixup (x outside [a,b] → 0/−∞). CDF: for
 * straddling windows, the vector_erf chain (shift, scale, erf, affine by
 * cached −erf(α/√2) and 0.5/Z) with per-lane scalar fixups for ξ ≤ 0 lanes
 * (reflected-erfc form, bit-identical to the scalar path) and bound lanes;
 * for whole-window tail regimes (α ≥ 0 or β ≤ 0) every lane needs the erfc
 * form, so the batch runs the scalar kernel per lane (correctness over
 * throughput — the erf chain would be 100% cancelled garbage there).
 * Non-fixed (α < 0 < ξ) lanes may differ from scalar by the documented
 * vector_erf ulp band (≲ 2e-15 absolute); fixed-up lanes are bit-identical.
 *
 * @author libstats Development Team
 * @version 2.4.0
 * @since 2.4.0
 */
class TruncatedNormalDistribution : public DistributionBase {
   public:
    // Dispatch metadata — must match the kDistributionMeta[] row
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::TRUNCATED_NORMAL;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Truncated Normal distribution.
     * @param mean Location μ of the parent Gaussian (default 0)
     * @param standardDeviation Scale σ of the parent Gaussian (default 1)
     * @param lowerBound Truncation bound a (default −∞)
     * @param upperBound Truncation bound b (default +∞; must exceed a)
     * @throws std::invalid_argument if parameters are invalid or the window
     *         is outside the supported range (Z underflows — see class notes)
     */
    explicit TruncatedNormalDistribution(
        double mean = detail::ZERO_DOUBLE, double standardDeviation = detail::ONE,
        double lowerBound = -std::numeric_limits<double>::infinity(),
        double upperBound = std::numeric_limits<double>::infinity());

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    TruncatedNormalDistribution(const TruncatedNormalDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    TruncatedNormalDistribution& operator=(const TruncatedNormalDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    TruncatedNormalDistribution(TruncatedNormalDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    TruncatedNormalDistribution& operator=(TruncatedNormalDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~TruncatedNormalDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Truncated Normal distribution without throwing.
     *
     * Rejects both basic parameter violations and windows whose regime-split
     * normalization Z underflows double precision (supported-window policy,
     * see class notes).
     */
    [[nodiscard]] static Result<TruncatedNormalDistribution> create(
        double mean = detail::ZERO_DOUBLE, double standardDeviation = detail::ONE,
        double lowerBound = -std::numeric_limits<double>::infinity(),
        double upperBound = std::numeric_limits<double>::infinity()) {
        auto validation =
            validateTruncatedNormalParameters(mean, standardDeviation, lowerBound, upperBound);
        if (validation.isError()) {
            return Result<TruncatedNormalDistribution>::makeError(validation.errorCode(),
                                                                  validation.message());
        }
        if (!isWindowRepresentable(mean, standardDeviation, lowerBound, upperBound)) {
            return Result<TruncatedNormalDistribution>::makeError(
                ValidationError::InvalidRange,
                "Truncation window lies too deep in the Gaussian tail: the normalization "
                "constant Z underflows double precision (supported roughly within +/-37.5 "
                "sigma of the mean, with window mass above ~1e-308)");
        }
        return Result<TruncatedNormalDistribution>::ok(
            createUnchecked(mean, standardDeviation, lowerBound, upperBound));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get location parameter μ of the parent Gaussian (NOT the
     *  distribution mean — see getMean()). */
    [[nodiscard]] double getMu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mean_;
    }

    /** @brief Get scale parameter σ of the parent Gaussian. */
    [[nodiscard]] double getSigma() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return standardDeviation_;
    }

    /** @brief Get lower truncation bound a (may be −∞). */
    [[nodiscard]] double getLowerBound() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return lowerBound_;
    }

    /** @brief Get upper truncation bound b (may be +∞). */
    [[nodiscard]] double getUpperBound() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return upperBound_;
    }

    /** @brief Get the cached normalization constant Z = Φ(β) − Φ(α). */
    [[nodiscard]] double getNormalizationConstant() const noexcept;

    /** @brief Set location μ. @throws std::invalid_argument (incl. window policy) */
    void setMu(double mean);
    /** @brief Set scale σ. @throws std::invalid_argument (incl. window policy) */
    void setSigma(double standardDeviation);
    /** @brief Set lower bound a. @throws std::invalid_argument (incl. window policy) */
    void setLowerBound(double lowerBound);
    /** @brief Set upper bound b. @throws std::invalid_argument (incl. window policy) */
    void setUpperBound(double upperBound);
    /** @brief Set all four parameters atomically. @throws std::invalid_argument */
    void setParameters(double mean, double standardDeviation, double lowerBound,
                       double upperBound);

    /** @brief Distribution mean = μ + σ·(φ(α)−φ(β))/Z (truncated moment). */
    [[nodiscard]] double getMean() const override;

    /** @brief Distribution variance = σ²(1 + (αφ(α)−βφ(β))/Z − δ²). */
    [[nodiscard]] double getVariance() const override;

    /** @brief Skewness from the truncated raw-moment recursion (see class notes). */
    [[nodiscard]] double getSkewness() const override;

    /** @brief Excess kurtosis from the truncated raw-moment recursion. */
    [[nodiscard]] double getKurtosis() const override;

    /** @brief Number of parameters (4: μ, σ, a, b). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 4; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "TruncatedNormal";
    }

    /** @brief Truncated Normal is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: a. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return lowerBound_;
    }

    /** @brief Support upper bound: b. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return upperBound_;
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetMu(double mean) noexcept;
    [[nodiscard]] VoidResult trySetSigma(double standardDeviation) noexcept;
    [[nodiscard]] VoidResult trySetLowerBound(double lowerBound) noexcept;
    [[nodiscard]] VoidResult trySetUpperBound(double upperBound) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double mean, double standardDeviation,
                                              double lowerBound, double upperBound) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /** @brief PDF: φ((x−μ)/σ)/(σZ) for x ∈ [a,b]; 0 outside (and at ±∞). */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF: −log σ − log Z − ½ξ² − ½log(2π) for x ∈ [a,b], −∞ outside.
     * Exact log space: never NaN and never a clamp constant inside the
     * support (guaranteed by the supported-window policy: log Z is finite).
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF: (Φ(ξ)−Φ(α))/Z with the regime-split numerator (class notes),
     * clamped to [0,1]. Exactly 0 for x ≤ a, exactly 1 for x ≥ b.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile via the survival-domain regime split (class notes).
     * @throws std::invalid_argument if p not in [0, 1]
     * Q(0) = a, Q(1) = b; never NaN for p ∈ (0,1) (#104).
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief One sample by inverse-CDF transform (exact in every regime). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief n samples by inverse-CDF transform. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit μ and σ by MLE with the truncation bounds a, b held FIXED
     * at their current values (standard known-bounds formulation — see the
     * class-level MLE note for the scope decision and algorithm).
     *
     * @param values Observed data; every value must be finite and within
     *               [a, b]
     * @throws std::invalid_argument on empty/too-small data or values
     *         outside the window
     * @throws std::runtime_error if the fixed-point iteration fails to
     *         converge or degenerates
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting for multiple independent datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<TruncatedNormalDistribution>& results);

    /** @brief Reset to default (μ=0, σ=1, a=−∞, b=+∞ — the plain Gaussian). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = clamp(μ, a, b). */
    [[nodiscard]] double getMode() const;

    /** @brief Median = quantile(½). */
    [[nodiscard]] double getMedian() const override;

    /** @brief Entropy = ½log(2πe) + log σ + log Z + (αφ(α)−βφ(β))/(2Z). */
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

    bool operator==(const TruncatedNormalDistribution& other) const;
    bool operator!=(const TruncatedNormalDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::TruncatedNormalDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::TruncatedNormalDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static TruncatedNormalDistribution createUnchecked(double mean, double standardDeviation,
                                                       double lowerBound,
                                                       double upperBound) noexcept;
    TruncatedNormalDistribution(double mean, double standardDeviation, double lowerBound,
                                double upperBound, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /** @brief SIMD PDF pipeline: shift, square, scale, +logC, exp; fixup
     *  outside [a,b] → 0 (see class-level Batch SIMD notes). */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double mu, double a, double b,
                                       double neg_half_inv_sigma2,
                                       double log_pdf_norm) const noexcept;

    /** @brief SIMD LogPDF pipeline: shift, square, scale, +logC; fixup −∞. */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double mu, double a, double b,
                                          double neg_half_inv_sigma2,
                                          double log_pdf_norm) const noexcept;

    /** @brief SIMD CDF: vector_erf chain for straddling windows with
     *  per-lane erfc fixups; whole-scalar for same-tail windows (class notes). */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double mu, double sigma,
                                                 double a, double b, double alpha,
                                                 double beta, double q_alpha, double phi_alpha,
                                                 double erf_alpha, double inv_z,
                                                 double half_inv_z,
                                                 double inv_sigma_sqrt2) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    /** @brief Regime-split Z and tail pieces; the single source of truth for
     *  the normalization (used by validation, cache update, and fit). */
    struct NormalizationConstants {
        double alpha, beta;      // standardized bounds (may be ±∞)
        double z;                // Φ(β) − Φ(α), regime-split
        double log_z;            // log(z)
        double phi_alpha;        // Φ(α)  = ½ erfc(−α/√2)
        double q_alpha;          // 1−Φ(α) = ½ erfc(α/√2)
        double q_beta;           // 1−Φ(β)
        double erf_alpha;        // erf(α/√2)
        bool valid;              // z > 0 and finite
    };
    static NormalizationConstants computeNormalization(double mean, double sigma, double a,
                                                       double b) noexcept;

    /** @brief True iff the regime-split Z is representable (> 0) in double. */
    static bool isWindowRepresentable(double mean, double sigma, double a, double b) noexcept {
        return computeNormalization(mean, sigma, a, b).valid;
    }

    static void validateParameters(double mean, double sigma, double a, double b) {
        if (!std::isfinite(mean)) {
            throw std::invalid_argument("Mean must be finite");
        }
        if (!std::isfinite(sigma) || sigma <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Standard deviation must be positive and finite");
        }
        if (std::isnan(a) || std::isnan(b)) {
            throw std::invalid_argument("Truncation bounds must not be NaN");
        }
        if (!(a < b)) {
            throw std::invalid_argument(
                "Upper truncation bound (b) must be strictly greater than lower bound (a)");
        }
        if (!isWindowRepresentable(mean, sigma, a, b)) {
            throw std::invalid_argument(
                "Truncation window lies too deep in the Gaussian tail: the normalization "
                "constant Z underflows double precision (supported roughly within +/-37.5 "
                "sigma of the mean, with window mass above ~1e-308)");
        }
    }

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Location μ of the parent Gaussian. */
    double mean_{detail::ZERO_DOUBLE};
    /** @brief Scale σ of the parent Gaussian — must be positive. */
    double standardDeviation_{detail::ONE};
    /** @brief Lower truncation bound a (may be −∞). */
    double lowerBound_{-std::numeric_limits<double>::infinity()};
    /** @brief Upper truncation bound b (may be +∞). */
    double upperBound_{std::numeric_limits<double>::infinity()};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicMean_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicStandardDeviation_{detail::ONE};
    mutable std::atomic<double> atomicLowerBound_{-std::numeric_limits<double>::infinity()};
    mutable std::atomic<double> atomicUpperBound_{std::numeric_limits<double>::infinity()};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE (all invalidated when ANY of μ/σ/a/b changes)
    //==========================================================================

    /** @brief α = (a−μ)/σ (−∞ when a = −∞). */
    mutable double alpha_{-std::numeric_limits<double>::infinity()};
    /** @brief β = (b−μ)/σ (+∞ when b = +∞). */
    mutable double beta_{std::numeric_limits<double>::infinity()};
    /** @brief Z = Φ(β) − Φ(α), regime-split (class notes). */
    mutable double z_{detail::ONE};
    /** @brief log Z (finite by the supported-window policy). */
    mutable double logZ_{detail::ZERO_DOUBLE};
    /** @brief Φ(α) = ½erfc(−α/√2). */
    mutable double phiAlpha_{detail::ZERO_DOUBLE};
    /** @brief 1−Φ(α) = ½erfc(α/√2). */
    mutable double qAlpha_{detail::ONE};
    /** @brief 1−Φ(β) = ½erfc(β/√2). */
    mutable double qBeta_{detail::ZERO_DOUBLE};
    /** @brief erf(α/√2) — the batch CDF chain's affine offset. */
    mutable double erfAlpha_{-detail::ONE};
    /** @brief 1/Z. */
    mutable double invZ_{detail::ONE};
    /** @brief 0.5/Z — the batch CDF chain's affine scale. */
    mutable double halfInvZ_{detail::HALF};
    /** @brief log σ. */
    mutable double logSigma_{detail::ZERO_DOUBLE};
    /** @brief −1/(2σ²). */
    mutable double negHalfInvSigmaSquared_{-detail::HALF};
    /** @brief 1/(σ√2). */
    mutable double invSigmaSqrt2_{detail::ZERO_DOUBLE};
    /** @brief −log σ − log Z − ½log(2π) — LogPDF additive constant. */
    mutable double logPdfNormConst_{detail::ZERO_DOUBLE};
    /** @brief Cached distribution mean μ + σδ. */
    mutable double distMean_{detail::ZERO_DOUBLE};
    /** @brief Cached distribution variance σ²(1 + η − δ²). */
    mutable double distVariance_{detail::ONE};
    /** @brief δ = (φ(α)−φ(β))/Z (standardized truncated mean). */
    mutable double delta_{detail::ZERO_DOUBLE};
    /** @brief η = (αφ(α)−βφ(β))/Z (entropy/variance term). */
    mutable double eta_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 23–24. OPTIMIZATION FLAGS / SPECIALIZED CACHES
    //==========================================================================

    // Note: Truncated Normal uses standard caching only.
    // Sections maintained for template compliance.
};

}  // namespace stats

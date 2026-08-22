#pragma once

#include "libstats/common/distribution_common.h"

namespace stats {

/**
 * @brief Thread-safe Von Mises Distribution — the canonical distribution for
 * circular/directional data.
 *
 * @details The Von Mises distribution is the circular analogue of the normal
 * distribution. It is defined on the circle (−π, π] and parameterised by a
 * mean direction μ and a concentration parameter κ ≥ 0.
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; μ, κ) = exp(κ·cos(x−μ)) / (2π·I₀(κ))
 * - LogPDF: κ·cos(x−μ) − logNormaliser_
 *           where logNormaliser_ = log(2π) + log I₀(κ)
 * - CDF:    for 0 < κ ≤ 1000, a Bessel-series expansion (issue #51):
 *             F(x) = (t+π)/(2π) + Σⱼ bⱼ·sin(j·t),  t = wrap(x−μ) ∈ [−π,π]
 *           with bⱼ = Iⱼ(κ)/(j·π·I₀(κ)) from a per-instance Miller backward
 *           recurrence (cdfSeriesCoeffs_, §24), j = 1..j_max,
 *           j_max = ⌈10 + 8.5√κ⌉. κ = 0 uses the exact linear form
 *           (t+π)/(2π). κ > 1000 (unvalidated range for the series) falls
 *           back to the wrapped-normal approximation, ≈ 0.04/κ absolute error.
 * - Quantile: bisection on the CDF grid in (−π, π]; O(log N) via cdfGrid*
 *           (§24), still built from the O(512) trapezoidal rule (issue #51
 *           left this as-is: the full test suite round-trips cleanly against
 *           the more accurate series CDF at the grid's existing resolution;
 *           rebuilding the grid from the series is a tracked follow-up, not
 *           a correctness requirement).
 * - Parameters: μ ∈ ℝ (wrapped to (−π, π]), κ ≥ 0
 * - Support: x ∈ (−π, π]
 *
 * @par Moments (circular):
 * - Circular mean: μ
 * - Circular variance: 1 − I₁(κ)/I₀(κ)  ∈ [0, 1]
 *   (0 = fully concentrated at μ, 1 = uniform)
 * - getVariance() returns the circular variance by convention.
 * - getSkewness() returns 0 (symmetric distribution).
 * - getKurtosis() returns 0 (not meaningfully defined for circular data).
 *
 * @par Special Cases:
 * - κ = 0: uniform distribution on (−π, π]; PDF = 1/(2π) everywhere.
 * - κ → ∞: approaches a Dirac delta at μ.
 *
 * @par Bessel Functions:
 * The implementation requires modified Bessel functions I₀ and I₁.
 * These are provided by include/core/bessel.h via two tiers:
 *   Tier 1 (LIBSTATS_HAS_CXX17_BESSEL): std::cyl_bessel_i (GCC, MSVC).
 *   Tier 2 (fallback): A&S §9.8.1–9.8.4 polynomial, error < 1.6×10⁻⁷.
 *   AppleClang / macOS libc++ always uses Tier 2.
 *
 * @par Batch operations and SIMD:
 * The VECTORIZED strategy uses `VectorOps::vector_cos` via the 4-step pipeline:
 *   1. scalar_add(−μ)         — shift to zero-centred angle
 *   2. vector_cos(results)    — SIMD cosine across all backends (AVX/AVX2/NEON/AVX-512)
 *   3. scalar_multiply(κ)     — scale by concentration
 *   4. scalar_add(−ln Z)      — subtract log-normaliser
 * The CDF batch path (issue #51) evaluates the same Bessel series per lane:
 * `t = wrap(x−μ)` per element, then `VectorOps::vector_sin` once per series
 * term (j = j_max down to 1) accumulated against the per-instance bⱼ. The
 * PARALLEL strategy provides multi-core throughput for very large batches.
 *
 * @par MLE:
 * - μ̂ = atan2(Σsin(xᵢ), Σcos(xᵢ))  (one-pass circular mean)
 * - κ̂ from mean resultant length R̄ = √(S²+C²)/n via the Mardia–Jupp
 *   approximation (Mardia & Jupp 2000, §A.2), refined by Newton–Raphson
 *   on A(κ) = I₁(κ)/I₀(κ) = R̄.
 *
 * @par Applications:
 * - Wind direction / wave direction analysis
 * - Protein bond-angle modelling (bioinformatics)
 * - Phase estimation in signal processing
 * - Robot orientation estimation
 * - HMM observation models for circular data (libhmm)
 *
 * @author libstats Development Team
 * @version 2.0.0
 * @since 2.0.0
 */
class VonMisesDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<VonMisesDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::VON_MISES;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Von Mises distribution.
     * @param mu    Mean direction μ — any finite real, wrapped to (−π, π].
     *              Default 0 (pointing right).
     * @param kappa Concentration κ ≥ 0 (default 1).
     *              κ = 0 gives the uniform circular distribution.
     * @throws std::invalid_argument if mu is non-finite or kappa < 0
     *
     * Implementation in .cpp.
     */
    explicit VonMisesDistribution(double mu = detail::ZERO_DOUBLE, double kappa = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    VonMisesDistribution(const VonMisesDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    VonMisesDistribution& operator=(const VonMisesDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    VonMisesDistribution(VonMisesDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    VonMisesDistribution& operator=(VonMisesDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~VonMisesDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Von Mises distribution without throwing.
     * @param mu    Mean direction (any finite real)
     * @param kappa Concentration κ ≥ 0
     * @return Result containing a valid VonMisesDistribution or error info
     */
    [[nodiscard]] static Result<VonMisesDistribution> create(double mu = detail::ZERO_DOUBLE,
                                                             double kappa = detail::ONE) {
        auto validation = validateVonMisesParameters(mu, kappa);
        if (validation.isError()) {
            return Result<VonMisesDistribution>::makeError(validation.errorCode(),
                                                           validation.message());
        }
        return Result<VonMisesDistribution>::ok(createUnchecked(mu, kappa));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get mean direction μ (always in (−π, π]). */
    [[nodiscard]] double getMu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Get concentration parameter κ (≥ 0). */
    [[nodiscard]] double getKappa() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return kappa_;
    }

    /** @brief Lock-free atomic getter for μ. */
    [[nodiscard]] double getMuAtomic() const noexcept;

    /** @brief Lock-free atomic getter for κ. */
    [[nodiscard]] double getKappaAtomic() const noexcept;

    /**
     * @brief Set mean direction μ (will be wrapped to (−π, π]).
     * @throws std::invalid_argument if mu is non-finite
     */
    void setMu(double mu);

    /**
     * @brief Set concentration parameter κ.
     * @throws std::invalid_argument if kappa < 0
     */
    void setKappa(double kappa);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if mu is non-finite or kappa < 0
     */
    void setParameters(double mu, double kappa);

    /**
     * @brief getMean() returns the circular mean direction μ.
     * (Not a linear mean; meaningful only in circular statistics context.)
     */
    [[nodiscard]] double getMean() const override;

    /**
     * @brief getVariance() returns the circular variance = 1 − I₁(κ)/I₀(κ).
     * Range [0, 1]: 0 = fully concentrated, 1 = uniform.
     */
    [[nodiscard]] double getVariance() const override;

    /**
     * @brief Skewness = 0 (Von Mises is symmetric about μ).
     */
    [[nodiscard]] double getSkewness() const noexcept override { return detail::ZERO_DOUBLE; }

    /**
     * @brief Kurtosis = 0 (not meaningfully defined for circular distributions).
     */
    [[nodiscard]] double getKurtosis() const noexcept override { return detail::ZERO_DOUBLE; }

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "VonMises";
    }

    /** @brief Von Mises is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: −π. */
    [[nodiscard]] double getSupportLowerBound() const noexcept override { return -detail::PI; }

    /** @brief Support upper bound: π. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override { return detail::PI; }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetMu(double mu) noexcept;
    [[nodiscard]] VoidResult trySetKappa(double kappa) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double mu, double kappa) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF: exp(κ·cos(x−μ)) / (2π·I₀(κ)).
     * Returns 0 for non-finite x.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF: κ·cos(x−μ) − logNormaliser_.
     * Returns −∞ for non-finite x.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF via the Bessel-series expansion for 0 < κ ≤ 1000 (issue #51);
     * exact linear form at κ = 0; wrapped-normal approximation for κ > 1000
     * (unvalidated range for the series). O(j_max) per call, j_max = ⌈10+8.5√κ⌉.
     * @note x−μ is wrapped to [−π, π] before series evaluation.
     * PROVISIONAL accuracy bound pending the mpmath-oracle accuracy gate
     * (tests/test_vonmises_cdf_accuracy.cpp).
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile via bisection on CDF in (−π, π].
     * O(512 · 50) = ~25 000 cos calls per query. Use sparingly in hot paths.
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Sample via the Best (1979) rejection sampler for Von Mises.
     * Falls back to uniform sampling when κ < 1e-9.
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit μ and κ by MLE.
     *
     * One-pass circular MLE:
     *   S = Σsin(xᵢ),  C = Σcos(xᵢ)
     *   μ̂ = atan2(S/n, C/n)  (wrapped to (−π, π])
     *   R̄ = √(S²+C²)/n
     *   κ̂ = kappa_from_R_bar(R̄)  via Mardia–Jupp + Newton–Raphson
     *
     * @param values Angles in radians (any real value, not pre-wrapped)
     * @throws std::invalid_argument if values is empty
     */
    void fit(const std::vector<double>& values) override;

    /** @brief Parallel batch fitting for multiple independent datasets. */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<VonMisesDistribution>& results);

    /** @brief Reset to default (μ = 0, κ = 1). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Circular variance = 1 − I₁(κ)/I₀(κ). Same as getVariance(). */
    [[nodiscard]] double getCircularVariance() const;

    /** @brief Median = μ (Von Mises is symmetric about μ). */
    [[nodiscard]] double getMedian() const noexcept override { return getMu(); }

    /** @brief Mode = μ (always at the mean direction). */
    [[nodiscard]] double getMode() const;

    /**
     * @brief Entropy = log(2π) − log I₀(κ) + κ·I₁(κ)/I₀(κ).
     * Matches the uniform entropy log(2π) at κ = 0.
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief True if κ = 0 within tolerance (uniform circular distribution).
     * When true, PDF = 1/(2π) everywhere.
     */
    [[nodiscard]] bool isUniform() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isUniform_;
    }

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
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const VonMisesDistribution& other) const;
    bool operator!=(const VonMisesDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::VonMisesDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::VonMisesDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static VonMisesDistribution createUnchecked(double mu, double kappa) noexcept;
    VonMisesDistribution(double mu, double kappa, bool /*bypassValidation*/) noexcept;

    /** @brief Wrap angle to (−π, π]. */
    static double wrapAngle(double x) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD-accelerated batch LogPDF / PDF via `VectorOps::vector_cos`.
     *
     * Pipeline: scalar_add(−μ) → vector_cos → scalar_multiply(κ) → scalar_add(−ln Z).
     * logNormaliser_ is cached — avoids one Bessel call per element.
     * κ and μ are loop-invariant scalars.
     * These methods are "Unsafe" because they skip parameter validation;
     * callers must hold the cache lock or guarantee parameter stability.
     */
    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_kappa, double cached_mu,
                                          double cached_log_normaliser) const noexcept;

    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_kappa, double cached_mu,
                                       double cached_log_normaliser) const noexcept;

    /**
     * @brief CDF batch — Bessel series evaluated batch-wise (issue #51).
     *
     * `cached_mu`/`cached_coeffs` are a snapshot taken by the caller under
     * the cache lock (see the CDF autoDispatch lambda in .cpp). When
     * `cached_coeffs` is empty (κ = 0 or κ > 1000 — series not applicable,
     * see updateCacheUnsafe()) this falls back to the per-element scalar
     * getCumulativeProbability() loop. Unsafe: no parameter validation.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_mu,
                                                 const std::vector<double>& cached_coeffs) const
        noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double mu, double kappa) {
        if (std::isnan(mu) || std::isinf(mu)) {
            throw std::invalid_argument("Mu (mean direction) must be a finite real number");
        }
        if (std::isnan(kappa) || std::isinf(kappa) || kappa < detail::ZERO_DOUBLE) {
            throw std::invalid_argument(
                "Kappa (concentration) must be a non-negative finite number");
        }
    }

    //==========================================================================
    // 20. PRIVATE UTILITY METHODS
    //==========================================================================

    // Note: No private utility methods required beyond wrapAngle (section 17).
    // Section maintained for template compliance.

    //==========================================================================
    // 21. DISTRIBUTION PARAMETERS
    //==========================================================================

    /** @brief Mean direction μ — always stored in (−π, π]. */
    double mu_{detail::ZERO_DOUBLE};

    /** @brief Concentration κ — must be ≥ 0. */
    double kappa_{detail::ONE};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicMu_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicKappa_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief log(2π) + log I₀(κ) — the normalisation log-constant.
     *  Computing this requires a Bessel function call; caching it once
     *  is the primary performance gain in batch LogPDF evaluation. */
    mutable double logNormaliser_{detail::ZERO_DOUBLE};

    /** @brief 1 − I₁(κ)/I₀(κ) — circular variance ∈ [0, 1]. */
    mutable double circularVariance_{detail::ONE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if κ < 1e-10 (uniform circular distribution). */
    mutable bool isUniform_{false};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    /**
     * @brief Lazily-initialized CDF grid for O(log N) quantile lookup.
     *
     * Built on first getQuantile() call with a given κ. 2049 uniformly-spaced
     * angles in [−π, π] and their CDF values. Binary search + linear interpolation
     * replaces the 60-step bisection that called the 512-point trapezoidal CDF.
     * Thread safety: guarded by cache_mutex_; rebuilt when κ changes.
     */
    mutable std::vector<double> cdfGridAngles_;  ///< angles[i] = -PI + i*h, i in [0,2048]
    mutable std::vector<double> cdfGridValues_;  ///< CDF(angles[i]; mu_=0, current kappa_)
    mutable double cdfGridKappa_{-1.0};          ///< kappa for which grid was built; -1=unbuilt

    /** @brief Build cdfGrid* for the current kappa_. Called under cache_mutex_. */
    void buildCdfGrid() const noexcept;

    /**
     * @brief CDF Bessel-series coefficients bⱼ = Iⱼ(κ)/(j·π·I₀(κ)), j = 1..j_max
     * (issue #51). cdfSeriesCoeffs_[j-1] holds bⱼ. Computed via a Miller
     * backward recurrence in updateCacheUnsafe() — no forward recurrence, no
     * Bessel anchor (the recurrence's own f_j/f_0 ratio already equals
     * Iⱼ/I₀; see updateCacheUnsafe() for the derivation). Empty when the
     * series is not applicable: κ = 0 (isUniform_; exact linear CDF) or
     * κ > 1000 (unvalidated range; wrapped-normal fallback). Invalidated
     * automatically on κ/μ change via the normal cache mechanism.
     */
    mutable std::vector<double> cdfSeriesCoeffs_;
};

}  // namespace stats

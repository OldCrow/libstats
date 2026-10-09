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
 * - CDF:    for 0 < κ ≤ 1000, at t = wrap(x−μ), with w = κ(1 − cos θ) as the
 *           integration variable: where F is within 1/4 of the median,
 *           F = 1/2 ∓ C(|t|)/(2π·I₀(κ)e^−κ), C the central mass by a short
 *           Gauss–Legendre rule; beyond, the tail mass G(t) = g(t)·J(t), J by
 *           a Watson-lemma series where κ(1 + cos t) ≥ 45 (and, within a quarter
 *           turn of μ, κ(1 − cos t) ≥ 45) and by Gauss–Legendre otherwise —
 *           relative to the accuracy law in both tails. κ = 0 uses
 *           the exact linear form (t+π)/(2π). κ > 1000 (unvalidated range)
 *           falls back to the wrapped-normal approximation, ≈ 0.04/κ absolute
 *           error.
 * - Quantile: on the small side of the probability scale (the CDF below the
 *           median, the survival above it): Halley on the central mass within
 *           1/4 of the median, on the logarithm of the tail mass beyond,
 *           bracketed. The left end of the support maps to the smallest double
 *           above −π + μ, wrapped.
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
 * The CDF batch path evaluates the scalar CDF per element on one cache
 * snapshot. The PARALLEL strategy provides multi-core throughput for very
 * large batches.
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
     * @brief CDF for 0 < κ ≤ 1000: the central mass within 1/4 of the median,
     * the tail mass beyond, so both tails are accurate relative to the accuracy
     * law; at most one short Gauss–Legendre rule (≤ 36 nodes) per call, none
     * where the tail's series applies. Exact linear form at κ = 0; wrapped-normal
     * approximation for κ > 1000 (unvalidated range).
     * @note x−μ is wrapped to [−π, π] first.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile in (−π, π]: bracketed Halley on the smaller of p and
     * 1 − p, on the central mass within 1/4 of the median and on the log of
     * the tail mass beyond. A few CDF evaluations per query. NaN propagates;
     * p = 0 gives the smallest double above −π + μ (wrapped), p = 1 gives
     * π + μ (wrapped).
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
     * @brief True if κ = 0 exactly (uniform circular distribution).
     * When true, PDF = 1/(2π) everywhere.
     */
    [[nodiscard]] bool isUniform() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isUniform_;
    }

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
     * @brief CDF batch — the scalar CDF per element on one cache snapshot.
     *
     * `cached_mu`/`cached_kappa`/`cached_log_scaled_norm`/`cached_inv_norm`/
     * `cached_half_j`/`cached_uniform` are a snapshot taken by the caller under the cache lock
     * (see the CDF autoDispatch lambda in .cpp). For κ = 0 or κ > 1000 this is
     * the per-element scalar getCumulativeProbability() loop. Unsafe: no
     * parameter validation.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_mu,
                                                 double cached_kappa, double cached_log_scaled_norm,
                                                 double cached_inv_norm, double cached_half_j,
                                                 bool cached_uniform) const noexcept;

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

    /** @brief log(2π·I₀(κ)·e^−κ) — logNormaliser_ − κ formed without the
     *  cancellation; scales the directly integrated CDF tails and the quantile. */
    mutable double logScaledNormaliser_{detail::ZERO_DOUBLE};

    /** @brief 1/(2π·I₀(κ)·e^−κ), formed directly (not as exp of the log above,
     *  which carries that log's rounding): scales the CDF's central mass. */
    mutable double invScaledNormaliser_{detail::ONE};

    /** @brief 1 − I₁(κ)/I₀(κ) — circular variance ∈ [0, 1]. */
    mutable double circularVariance_{detail::ONE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if κ == 0 (uniform circular distribution); exact equality only. */
    mutable bool isUniform_{false};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    /**
     * @brief J(−π/2) = G(−π/2)/g(−π/2) for μ = 0: the tail mass at a quarter
     * turn over the density there. The CDF tail and the quantile scale it by
     * e^(−κ cos t) for the part of the tail mass beyond a quarter turn.
     */
    mutable double tailHalfJ_{detail::ZERO_DOUBLE};
};

}  // namespace stats

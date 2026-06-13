#pragma once

#include "libstats/common/distribution_common.h"
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Rayleigh Distribution for modelling signal magnitudes and speeds.
 *
 * @details The Rayleigh distribution arises naturally as the magnitude of a 2D vector
 * whose components are independent zero-mean Gaussian random variables.
 *
 * Rayleigh(σ) is mathematically equivalent to Weibull(k=2, λ=σ√2), but is implemented
 * as a standalone distribution rather than a delegation wrapper. The reason is SIMD
 * efficiency: the Rayleigh formulas reduce to quadratic expressions in x (PDF = x·σ⁻²·exp(−x²/2σ²),
 * CDF = 1−exp(−x²/2σ²)) that need only `vector_multiply` and `vector_exp`, giving
 * 5-step LogPDF and 5-step CDF pipelines. Delegating to WeibullDistribution would
 * invoke its 8-step LogPDF pipeline (which computes log(x/λ) via two transcendentals
 * and intermediate z² via a temp buffer) for no arithmetic benefit. The standalone
 * implementation is also more readable and avoids the fixed k=2 / λ=σ√2 bookkeeping.
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; σ) = (x/σ²)·exp(−x²/(2σ²))  for x ≥ 0
 * - LogPDF: log(x) − 2·log(σ) − x²/(2σ²)
 * - CDF:    1 − exp(−x²/(2σ²))  for x ≥ 0
 * - Quantile: σ·√(−2·log(1−p))
 * - Parameters: σ > 0 (scale)
 * - Support:  x ∈ [0, ∞)
 *
 * @par Moments:
 * - Mean:     σ·√(π/2)  ≈ 1.2533·σ
 * - Variance: σ²·(4−π)/2  ≈ 0.4292·σ²
 * - Mode:     σ
 * - Median:   σ·√(2·ln 2) ≈ 1.1774·σ
 * - Skewness: 2√π·(π−3)/(4−π)^(3/2)  ≈ 0.6311 (constant)
 * - Excess kurtosis: −(6π²−24π+16)/(4−π)²  ≈ 0.2451 (constant)
 *
 * @par Batch SIMD:
 * LogPDF uses one aligned temp buffer. Five-step pipeline:
 *   temp    = x²                           [vector_multiply(values, values)]
 *   temp    = −x²/(2σ²)                    [scalar_multiply(neg_half_inv_sigma2_)]
 *   results = log(x)                       [vector_log]
 *   results += temp                        [vector_add]
 *   results += log_norm_const_             [scalar_add(−2·log(σ))]
 * PDF: append vector_exp.
 * CDF uses five steps (no temp buffer):
 *   x² → ·(−½σ²) → exp → negate → +1.
 * Expected speedup: LogPDF ~12–18×, CDF ~10–15×.
 *
 * @par MLE:
 * Closed-form: σ̂ = √(Σxᵢ²/(2n)). Single pass, no iteration.
 *
 * @par Applications:
 * - Wireless communications: fading channel modelling
 * - Signal processing: magnitude of complex Gaussian noise
 * - Wind engineering: wind speed modelling
 * - Acoustics: reverberation tail amplitude
 * - Materials science: fibre strength distribution
 *
 * @author libstats Development Team
 * @version 1.2.0
 * @since 1.2.0
 */
class RayleighDistribution : public DistributionBase {
   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Rayleigh distribution.
     * @param sigma Scale parameter σ (must be positive, default 1)
     * @throws std::invalid_argument if sigma is not strictly positive or non-finite
     *
     * Default σ = 1 is the standard Rayleigh distribution.
     * Implementation in .cpp.
     */
    explicit RayleighDistribution(double sigma = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    RayleighDistribution(const RayleighDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    RayleighDistribution& operator=(const RayleighDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    RayleighDistribution(RayleighDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    RayleighDistribution& operator=(RayleighDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~RayleighDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Rayleigh distribution without throwing exceptions.
     * @param sigma Scale parameter σ (must be positive)
     * @return Result containing a valid RayleighDistribution or error info
     */
    [[nodiscard]] static Result<RayleighDistribution> create(
        double sigma = detail::ONE) noexcept {
        auto validation = validateRayleighParameters(sigma);
        if (validation.isError()) {
            return Result<RayleighDistribution>::makeError(validation.error_code,
                                                           validation.message);
        }
        return Result<RayleighDistribution>::ok(createUnchecked(sigma));
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
     * @throws std::invalid_argument if sigma <= 0
     */
    void setSigma(double sigma);

    /** @brief Alias for setSigma. */
    void setParameters(double sigma);

    /** @brief Mean = σ·√(π/2). */
    [[nodiscard]] double getMean() const noexcept override;

    /** @brief Variance = σ²·(4−π)/2. */
    [[nodiscard]] double getVariance() const noexcept override;

    /** @brief Skewness ≈ 0.6311 (constant, independent of σ). */
    [[nodiscard]] double getSkewness() const noexcept override;

    /** @brief Excess kurtosis ≈ 0.2451 (constant, independent of σ). */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /** @brief Number of parameters (always 1). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 1; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string getDistributionName() const override {
        return "RayleighDistribution";
    }

    /** @brief Rayleigh is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

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

    [[nodiscard]] VoidResult trySetSigma(double sigma) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double sigma) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: (x/σ²)·exp(−x²/(2σ²)) for x > 0; 0 for x ≤ 0.
     * Computed via exp(LogPDF) for x > 0.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: log(x) − 2·log(σ) − x²/(2σ²).
     * Returns −∞ for x ≤ 0.
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief CDF: 1 − exp(−x²/(2σ²)) for x ≥ 0; 0 for x < 0.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile: σ·√(−2·log(1−p)).
     * @throws std::invalid_argument if p not in [0, 1)
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate one random sample via inverse CDF: σ·√(−2·log(U)).
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit σ to data by closed-form MLE.
     *
     * σ̂ = √(Σxᵢ²/(2n)). Single-pass O(n) computation, no iteration.
     *
     * @param values Observed data (must all be strictly positive)
     * @throws std::invalid_argument if values is empty or contains non-positive values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple independent datasets.
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<RayleighDistribution>& results);

    /** @brief Reset to default (σ = 1 — standard Rayleigh). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = σ. */
    [[nodiscard]] double getMode() const noexcept;

    /** @brief Median = σ·√(2·ln 2). */
    [[nodiscard]] double getMedian() const noexcept;

    /** @brief Entropy = 1 + log(σ/√2) + γ/2 (γ = Euler–Mascheroni constant). */
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

    void getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                    detail::Strategy strategy) const;

    void getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                       detail::Strategy strategy) const;

    void getCumulativeProbabilityWithStrategy(std::span<const double> values,
                                              std::span<double> results,
                                              detail::Strategy strategy) const;

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const RayleighDistribution& other) const;
    bool operator!=(const RayleighDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::RayleighDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::RayleighDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static RayleighDistribution createUnchecked(double sigma) noexcept;
    RayleighDistribution(double sigma, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD LogPDF pipeline.
     *
     * Five steps, one aligned temp buffer:
     *   Step 1: temp    = x²                        [vector_multiply(values, values)]
     *   Step 2: temp    = −x²/(2σ²)                 [scalar_multiply(neg_half_inv_sigma2_)]
     *   Step 3: results = log(x)                    [vector_log]
     *   Step 4: results += −x²/(2σ²)               [vector_add(temp, results)]
     *   Step 5: results += log_norm_const_           [scalar_add(−2·log(σ))]
     *
     * PDF: append vector_exp.
     * Scalar fixup: x ≤ 0 → −∞ (LogPDF) or 0 (PDF).
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_neg_half_inv_sigma2,
                                       double cached_log_norm_const) const noexcept;

    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_neg_half_inv_sigma2,
                                          double cached_log_norm_const) const noexcept;

    /**
     * @brief CDF batch pipeline.
     *
     * Five steps, no temp buffer:
     *   Step 1: results = x²                        [vector_multiply(values, values)]
     *   Step 2: results = −x²/(2σ²)                [scalar_multiply(neg_half_inv_sigma2_)]
     *   Step 3: results = exp(−x²/(2σ²))            [vector_exp]
     *   Step 4: results = −exp(...)                 [scalar_multiply(−1)]
     *   Step 5: results = 1 − exp(−x²/(2σ²))        [scalar_add(1)]
     *
     * Scalar fixup: x ≤ 0 → 0.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count,
                                                 double cached_neg_half_inv_sigma2) const noexcept;

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
    // 20. PRIVATE UTILITY METHODS
    //==========================================================================

    // Note: No private utility methods required for this distribution.
    // Section maintained for template compliance.

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

    /** @brief −1/(2σ²) — used in LogPDF and CDF SIMD pipelines. */
    mutable double negHalfInvSigmaSquared_{-detail::HALF};

    /** @brief −2·log(σ) — additive normalisation constant in LogPDF. */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief Cached mean = σ·√(π/2). */
    mutable double mean_{detail::ONE};

    /** @brief Cached variance = σ²·(4−π)/2. */
    mutable double variance_{detail::ONE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief Atomic cache validity flag for lock-free fast path. */
    mutable std::atomic<bool> cacheValidAtomic_{false};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Rayleigh uses standard caching only.
    // Section maintained for template compliance.
};

}  // namespace stats

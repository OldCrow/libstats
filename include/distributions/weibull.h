#pragma once

#include "libstats/common/distribution_common.h"
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Weibull Distribution for reliability analysis and survival modelling.
 *
 * @details Parameterised by shape k > 0 and scale λ > 0. The distribution is used
 * throughout reliability engineering, materials science, and survival analysis.
 * When k = 1 it reduces to Exponential(rate = 1/λ).
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; k, λ) = (k/λ)·(x/λ)^(k−1)·exp(−(x/λ)^k)  for x ≥ 0
 * - LogPDF: log(k) − k·log(λ) + (k−1)·log(x) − (x/λ)^k
 *           = logNormConst_ + (k−1)·log(x) − exp(k·log(x/λ))
 * - CDF:    1 − exp(−(x/λ)^k)  for x ≥ 0
 * - Quantile: λ·(−log(1−p))^(1/k)
 * - Parameters: k > 0 (shape), λ > 0 (scale)
 * - Support:  x ∈ [0, ∞)
 *
 * @par Moments:
 * - Mean:     λ·Γ(1 + 1/k)
 * - Variance: λ²·[Γ(1 + 2/k) − Γ(1 + 1/k)²]
 * - Mode:     λ·((k−1)/k)^(1/k) for k > 1;  0 for k ≤ 1
 * - Median:   λ·(ln 2)^(1/k)
 *
 * @par Special Cases:
 * - k = 1: Exponential distribution with rate 1/λ
 * - k = 2: Rayleigh distribution with σ = λ/√2
 * - k < 1: Decreasing failure rate (infant mortality)
 * - k > 1: Increasing failure rate (wear-out)
 *
 * @par Batch SIMD:
 * LogPDF uses one aligned temp buffer. Eight-step pipeline:
 *   temp    = log(x)                        [vector_log]
 *   results = log(x/λ) = temp − log(λ)      [scalar_add(temp, −logScale_)]
 *   results = k·log(x/λ)                    [scalar_multiply(k_)]
 *   results = exp(k·log(x/λ)) = (x/λ)^k    [vector_exp]
 *   results = −(x/λ)^k                      [scalar_multiply(−1)]
 *   temp    = (k−1)·log(x/λ)               [scalar_multiply(temp, shapeMinus1_)]
 *   results += (k−1)·log(x/λ)              [vector_add]
 *   results += logNormConst_                 [scalar_add]
 * PDF: append vector_exp.
 * CDF uses eight steps (no temp buffer):
 *   log(x/λ) → ·k → exp → negate → exp → negate → +1.
 * Expected speedup: LogPDF ~8–14×, CDF ~6–10×.
 *
 * @par MLE:
 * Newton–Raphson on the Weibull profile score for k (always converges:
 * derivative g'(k) = Var_k[log x] + 1/k² > 0). Initial estimate from
 * method-of-moments using coefficient-of-variation approximation.
 * After convergence: λ̂ = (Σxᵢ^k/n)^(1/k).
 *
 * @par Applications:
 * - Reliability engineering: component lifetime modelling
 * - Wind energy: wind speed distribution
 * - Materials science: tensile strength, fracture mechanics
 * - Hydrology: flood frequency analysis
 * - Finance: extreme event modelling
 *
 * @author libstats Development Team
 * @version 1.2.0
 * @since 1.2.0
 */
class WeibullDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<WeibullDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::WEIBULL;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Weibull distribution.
     * @param shape Shape parameter k (must be positive, default 1)
     * @param scale Scale parameter λ (must be positive, default 1)
     * @throws std::invalid_argument if either parameter is not strictly positive or non-finite
     *
     * Default (1, 1) is the standard Exponential(rate=1) distribution.
     * Implementation in .cpp.
     */
    explicit WeibullDistribution(double shape = detail::ONE, double scale = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    WeibullDistribution(const WeibullDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    WeibullDistribution& operator=(const WeibullDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    WeibullDistribution(WeibullDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    WeibullDistribution& operator=(WeibullDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~WeibullDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Weibull distribution without throwing exceptions.
     * @param shape Shape parameter k (must be positive)
     * @param scale Scale parameter λ (must be positive)
     * @return Result containing a valid WeibullDistribution or error info
     */
    [[nodiscard]] static Result<WeibullDistribution> create(double shape = detail::ONE,
                                                            double scale = detail::ONE) noexcept {
        auto validation = validateWeibullParameters(shape, scale);
        if (validation.isError()) {
            return Result<WeibullDistribution>::makeError(validation.error_code,
                                                          validation.message);
        }
        return Result<WeibullDistribution>::ok(createUnchecked(shape, scale));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get shape parameter k. */
    [[nodiscard]] double getShape() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return shape_;
    }

    /** @brief Get scale parameter λ. */
    [[nodiscard]] double getScale() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return scale_;
    }

    /** @brief Lock-free atomic getter for k. */
    [[nodiscard]] double getShapeAtomic() const noexcept;

    /** @brief Lock-free atomic getter for λ. */
    [[nodiscard]] double getScaleAtomic() const noexcept;

    /**
     * @brief Set shape parameter k.
     * @throws std::invalid_argument if shape <= 0
     */
    void setShape(double shape);

    /**
     * @brief Set scale parameter λ.
     * @throws std::invalid_argument if scale <= 0
     */
    void setScale(double scale);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if either is not strictly positive or non-finite
     */
    void setParameters(double shape, double scale);

    /** @brief Mean = λ·Γ(1 + 1/k). */
    [[nodiscard]] double getMean() const noexcept override;

    /** @brief Variance = λ²·[Γ(1 + 2/k) − Γ(1 + 1/k)²]. */
    [[nodiscard]] double getVariance() const noexcept override;

    /** @brief Skewness (complex gamma expression, computed on demand). */
    [[nodiscard]] double getSkewness() const noexcept override;

    /** @brief Excess kurtosis (complex gamma expression, computed on demand). */
    [[nodiscard]] double getKurtosis() const noexcept override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string getDistributionName() const override { return "WeibullDistribution"; }

    /** @brief Weibull is continuous. */
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

    [[nodiscard]] VoidResult trySetShape(double shape) noexcept;
    [[nodiscard]] VoidResult trySetScale(double scale) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double shape, double scale) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: exp(LogPDF(x)) for x > 0; handled at x = 0 by shape;
     *        0 for x < 0.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: log(k) − k·log(λ) + (k−1)·log(x) − (x/λ)^k.
     * Returns −∞ for x < 0; special case at x = 0 depends on shape.
     */
    [[nodiscard]] double getLogProbability(double x) const noexcept override;

    /**
     * @brief CDF: 1 − exp(−(x/λ)^k) for x ≥ 0; 0 for x < 0.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile: λ·(−log(1−p))^(1/k).
     * @throws std::invalid_argument if p not in [0, 1)
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /**
     * @brief Generate one random sample via std::weibull_distribution<double>(k, λ).
     */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit shape and scale by MLE.
     *
     * Uses a method-of-moments initial estimate then refines with
     * Newton–Raphson on the Weibull profile score equation:
     *   g(k) = E_k[log x] − 1/k − s̄ = 0
     * where E_k[·] weights by xᵢ^k. Derivative g'(k) = Var_k[log x] + 1/k²
     * is always positive, guaranteeing convergence.
     * After k converges: λ̂ = (Σxᵢ^k / n)^(1/k).
     *
     * @param values Observed data (must all be strictly positive)
     * @throws std::invalid_argument if values is empty or contains non-positive values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple independent datasets.
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<WeibullDistribution>& results);

    /** @brief Reset to default (shape = 1, scale = 1 — standard Exponential). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = λ·((k−1)/k)^(1/k) for k > 1; 0 for k ≤ 1. */
    [[nodiscard]] double getMode() const noexcept;

    /** @brief Median = λ·(ln 2)^(1/k). */
    [[nodiscard]] double getMedian() const noexcept;

    /** @brief Entropy = γ·(1 − 1/k) + log(λ/k) + 1 (γ = Euler-Mascheroni). */
    [[nodiscard]] double getEntropy() const noexcept override;

    /**
     * @brief True if k = 1 within tolerance (reduces to Exponential(rate = 1/λ)).
     */
    [[nodiscard]] bool isExponential() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isExponential_;
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
    // 14. EXPLICIT STRATEGY BATCH OPERATIONS
    //==========================================================================

    //==========================================================================
    // 15. COMPARISON OPERATORS
    //==========================================================================

    bool operator==(const WeibullDistribution& other) const;
    bool operator!=(const WeibullDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::WeibullDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::WeibullDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static WeibullDistribution createUnchecked(double shape, double scale) noexcept;
    WeibullDistribution(double shape, double scale, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD LogPDF pipeline.
     *
     * Eight steps, one aligned temp buffer:
     *   Step 1: temp    = log(x)                        [vector_log]
     *   Step 2: results = log(x) − log(λ)  (= z)        [scalar_add(temp, −logScale_)]
     *   Step 3: results = k·z                            [scalar_multiply(k_)]
     *   Step 4: results = exp(k·z) = (x/λ)^k            [vector_exp]
     *   Step 5: results = −(x/λ)^k                      [scalar_multiply(−1)]
     *   Step 6: temp    = (k−1)·z                        [scalar_multiply(temp, shapeMinus1_)]
     *   Step 7: results += (k−1)·z                      [vector_add(temp, results)]
     *   Step 8: results += logNormConst_                 [scalar_add]
     *
     * PDF: append vector_exp.
     * Scalar fixup: x < 0 → −∞ (LogPDF) or 0 (PDF); x = 0 dispatched to scalar.
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_shape, double cached_log_scale,
                                       double cached_shape_minus1,
                                       double cached_log_norm_const) const noexcept;

    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_shape, double cached_log_scale,
                                          double cached_shape_minus1,
                                          double cached_log_norm_const) const noexcept;

    /**
     * @brief CDF batch pipeline.
     *
     * Eight steps, no temp buffer:
     *   Step 1: results = log(x)                   [vector_log]
     *   Step 2: results = log(x/λ)                 [scalar_add(−logScale_)]
     *   Step 3: results = k·log(x/λ)               [scalar_multiply(k_)]
     *   Step 4: results = (x/λ)^k                  [vector_exp]
     *   Step 5: results = −(x/λ)^k                 [scalar_multiply(−1)]
     *   Step 6: results = exp(−(x/λ)^k)            [vector_exp]
     *   Step 7: results = −exp(−(x/λ)^k)           [scalar_multiply(−1)]
     *   Step 8: results = 1 − exp(−(x/λ)^k)        [scalar_add(1)]
     *
     * Scalar fixup: x < 0 → 0; x = 0 → 0.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_log_scale,
                                                 double cached_shape) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double shape, double scale) {
        if (std::isnan(shape) || std::isinf(shape) || shape <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Shape (k) must be a positive finite number");
        }
        if (std::isnan(scale) || std::isinf(scale) || scale <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Scale (λ) must be a positive finite number");
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

    /** @brief Shape parameter k — must be positive. */
    double shape_{detail::ONE};

    /** @brief Scale parameter λ — must be positive. */
    double scale_{detail::ONE};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicShape_{detail::ONE};
    mutable std::atomic<double> atomicScale_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief log(k) — component of logNormConst_ and entropy. */
    mutable double logShape_{detail::ZERO_DOUBLE};

    /** @brief log(λ) — used in the LogPDF SIMD pipeline (step 2). */
    mutable double logScale_{detail::ZERO_DOUBLE};

    /** @brief k − 1 — used in the LogPDF SIMD pipeline (step 6). */
    mutable double shapeMinus1_{detail::ZERO_DOUBLE};

    /** @brief log(k) − k·log(λ) — additive normalisation in LogPDF. */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief Cached mean = λ·Γ(1 + 1/k). */
    mutable double mean_{detail::ONE};

    /** @brief Cached variance = λ²·[Γ(1 + 2/k) − Γ(1 + 1/k)²]. */
    mutable double variance_{detail::ONE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if k = 1 within tolerance (Exponential(rate = 1/λ)). */
    mutable bool isExponential_{true};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Weibull uses standard caching only.
    // Section maintained for template compliance.
};

}  // namespace stats

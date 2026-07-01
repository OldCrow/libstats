#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools,
// adaptive caching, etc.)

namespace stats {

/**
 * @brief Thread-safe Pareto Distribution for modelling heavy-tailed power-law phenomena.
 *
 * @details The Pareto distribution is a continuous power-law distribution with support
 * [x_m, ∞). It arises naturally in income distributions, city sizes, insurance claims,
 * network traffic, and any process that exhibits the "80–20" rule.
 *
 * @par Mathematical Definition:
 * - PDF:     f(x; x_m, α) = α·x_m^α / x^(α+1)  for x ≥ x_m
 * - LogPDF:  log(α) + α·log(x_m) − (α+1)·log(x) = logNormConst_ + neg_alpha_p1·log(x)
 * - CDF:     1 − (x_m/x)^α  for x ≥ x_m; 0 otherwise
 * - Quantile: x_m·(1−p)^(−1/α) = x_m/(1−p)^(1/α)
 * - Parameters: x_m > 0 (scale, minimum value), α > 0 (shape)
 * - Support:  x ∈ [x_m, ∞)
 *
 * @par Moments (conditional on α):
 * - Mean:     α·x_m/(α−1)                           for α > 1; ∞ otherwise
 * - Variance: x_m²·α / ((α−1)²·(α−2))               for α > 2; ∞ otherwise
 * - Mode:     x_m (always at lower boundary)
 * - Median:   x_m·2^(1/α)
 * - Skewness: 2(1+α)/(α−3)·√((α−2)/α)              for α > 3
 * - Excess kurtosis: 6(α³+α²−6α−2)/(α(α−3)(α−4))   for α > 4
 *
 * @par Batch SIMD:
 * LogPDF reduces to a single vector_log followed by two scalar operations — the
 * simplest SIMD pipeline of any distribution in the library:
 *   results = log(x)              [vector_log]
 *   results *= −(α+1)             [scalar_multiply(neg_alpha_p1_)]
 *   results += log_norm_const     [scalar_add(log_norm_const_)]
 * PDF: append vector_exp.
 * CDF uses vector_log + 2 scalars + vector_exp + 2 scalars (6 steps, no temp buffer).
 * Expected speedup: LogPDF ~20–30×, PDF ~15–20×, CDF ~10–15×.
 *
 * @par Applications:
 * - Economics: income/wealth distributions (Pareto principle)
 * - Insurance: large-loss modelling, reinsurance
 * - Network science: degree distributions, file-size distributions
 * - Finance: tail-risk modelling, extreme-value analysis
 * - Physics: earthquake magnitudes, particle sizes
 *
 * @author libstats Development Team
 * @version 2.0.0
 * @since 2.0.0
 */
class ParetoDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<ParetoDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType = detail::DistributionType::PARETO;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Pareto distribution.
     * @param scale Scale parameter x_m — minimum value (must be positive, default 1)
     * @param alpha Shape parameter α (must be positive, default 1)
     * @throws std::invalid_argument if either parameter is not strictly positive or non-finite
     *
     * Default (1, 1) is the standard unit Pareto distribution.
     * Implementation in .cpp.
     */
    explicit ParetoDistribution(double scale = detail::ONE, double alpha = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    ParetoDistribution(const ParetoDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    ParetoDistribution& operator=(const ParetoDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    ParetoDistribution(ParetoDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    ParetoDistribution& operator=(ParetoDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~ParetoDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Pareto distribution without throwing exceptions.
     * @param scale Scale parameter x_m (must be positive)
     * @param alpha Shape parameter α (must be positive)
     * @return Result containing a valid ParetoDistribution or error info
     */
    [[nodiscard]] static Result<ParetoDistribution> create(double scale = detail::ONE,
                                                           double alpha = detail::ONE) {
        auto validation = validateParetoParameters(scale, alpha);
        if (validation.isError()) {
            return Result<ParetoDistribution>::makeError(validation.errorCode(),
                                                         validation.message());
        }
        return Result<ParetoDistribution>::ok(createUnchecked(scale, alpha));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get scale parameter x_m (minimum value). */
    [[nodiscard]] double getScale() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return scale_;
    }

    /** @brief Get shape parameter α. */
    [[nodiscard]] double getAlpha() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return alpha_;
    }

    /**
     * @brief Lock-free atomic getter for scale.
     * Uses acquire-load from atomicScale_ for high-frequency access.
     */
    [[nodiscard]] double getScaleAtomic() const noexcept;

    /**
     * @brief Lock-free atomic getter for α.
     * Uses acquire-load from atomicAlpha_ for high-frequency access.
     */
    [[nodiscard]] double getAlphaAtomic() const noexcept;

    /**
     * @brief Set scale parameter x_m.
     * @throws std::invalid_argument if scale <= 0
     */
    void setScale(double scale);

    /**
     * @brief Set shape parameter α.
     * @throws std::invalid_argument if alpha <= 0
     */
    void setAlpha(double alpha);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if either is not strictly positive or non-finite
     */
    void setParameters(double scale, double alpha);

    /**
     * @brief Mean = α·x_m/(α−1) for α > 1; returns +∞ for α ≤ 1.
     */
    [[nodiscard]] double getMean() const override;

    /**
     * @brief Variance = x_m²·α/((α−1)²·(α−2)) for α > 2; returns +∞ otherwise.
     */
    [[nodiscard]] double getVariance() const override;

    /**
     * @brief Skewness = 2(1+α)/(α−3)·√((α−2)/α) for α > 3; returns +∞ otherwise.
     */
    [[nodiscard]] double getSkewness() const override;

    /**
     * @brief Excess kurtosis for α > 4; returns +∞ otherwise.
     */
    [[nodiscard]] double getKurtosis() const override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "Pareto";
    }

    /** @brief Pareto is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: scale (x_m). */
    [[nodiscard]] double getSupportLowerBound() const noexcept override {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return scale_;
    }

    /** @brief Support upper bound: +∞. */
    [[nodiscard]] double getSupportUpperBound() const noexcept override {
        return std::numeric_limits<double>::infinity();
    }

    //==========================================================================
    // 4. RESULT-BASED SETTERS
    //==========================================================================

    [[nodiscard]] VoidResult trySetScale(double scale) noexcept;
    [[nodiscard]] VoidResult trySetAlpha(double alpha) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double scale, double alpha) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: exp(LogPDF(x)) for x ≥ x_m; 0 otherwise.
     * Computed via exp(LogPDF) for numerical stability.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: log(α) + α·log(x_m) − (α+1)·log(x) for x ≥ x_m.
     * Returns −∞ for x < x_m.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF: 1 − (x_m/x)^α for x ≥ x_m; 0 for x < x_m.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile: x_m·(1−p)^(−1/α).
     * @throws std::invalid_argument if p not in [0, 1)
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Generate one random sample via inverse CDF: x_m/U^(1/α) where U ~ Uniform(0,1). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit scale and α to data by MLE.
     *
     * Closed-form MLE: x̂_m = min(xᵢ), α̂ = n / Σ log(xᵢ/x̂_m).
     * Both estimators are exactly computable — no iteration required.
     *
     * @param values Observed data (must all be strictly positive)
     * @throws std::invalid_argument if values is empty or contains non-positive values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple independent datasets.
     * @param datasets Vector of datasets, each with independent observations
     * @param results  Vector to store fitted ParetoDistribution objects
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<ParetoDistribution>& results);

    /** @brief Reset to default (scale = 1, α = 1 — unit Pareto). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = scale (x_m) — always at the lower boundary. */
    [[nodiscard]] double getMode() const;

    /** @brief Median = x_m·2^(1/α) = x_m·exp(log(2)/α). */
    [[nodiscard]] double getMedian() const override;

    /**
     * @brief Entropy = log(x_m/α) + 1 + 1/α.
     * Differential entropy for Pareto(x_m, α).
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief True if α > 1 (mean is finite).
     */
    [[nodiscard]] bool hasFiniteMean() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return alpha_ > detail::ONE;
    }

    /**
     * @brief True if α > 2 (variance is finite).
     */
    [[nodiscard]] bool hasFiniteVariance() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return alpha_ > detail::TWO;
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

    bool operator==(const ParetoDistribution& other) const;
    bool operator!=(const ParetoDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::ParetoDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::ParetoDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static ParetoDistribution createUnchecked(double scale, double alpha) noexcept;
    ParetoDistribution(double scale, double alpha, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD pipeline for LogPDF — the simplest in the library.
     *
     * Three steps, no temp buffer:
     *   Step 1: results = log(x)              [vector_log]
     *   Step 2: results *= −(α+1)             [scalar_multiply(neg_alpha_p1_)]
     *   Step 3: results += log_norm_const      [scalar_add(log_norm_const_)]
     *
     * PDF: append vector_exp.
     * Scalar fixup: x < scale_ → −∞ (LogPDF) or 0 (PDF).
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_scale, double cached_neg_alpha_p1,
                                       double cached_log_norm_const) const noexcept;

    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_scale, double cached_neg_alpha_p1,
                                          double cached_log_norm_const) const noexcept;

    /**
     * @brief CDF batch pipeline — six steps, no temp buffer.
     *
     *   Step 1: results = log(x)              [vector_log]
     *   Step 2: results = log(x) − log(x_m)   [scalar_add(−log_scale)]
     *   Step 3: results = −α·(log(x)−log(xm)) [scalar_multiply(neg_alpha_)]
     *             = α·(log(x_m) − log(x)) = α·log(x_m/x)
     *   Step 4: results = exp(α·log(x_m/x))   [vector_exp]   → (x_m/x)^α
     *   Step 5: results = −(x_m/x)^α          [scalar_multiply(−1)]
     *   Step 6: results = 1 − (x_m/x)^α       [scalar_add(1)]
     *
     * Scalar fixup: x < scale_ → 0.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_scale,
                                                 double cached_log_scale,
                                                 double cached_neg_alpha) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double scale, double alpha) {
        if (std::isnan(scale) || std::isinf(scale) || scale <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Scale (minimum value) must be a positive finite number");
        }
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Alpha (shape) must be a positive finite number");
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

    /** @brief Scale parameter x_m (minimum value) — must be positive. */
    double scale_{detail::ONE};

    /** @brief Shape parameter α — must be positive. */
    double alpha_{detail::ONE};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicScale_{detail::ONE};
    mutable std::atomic<double> atomicAlpha_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief log(x_m) — used in LogPDF normalisation and CDF pipeline. */
    mutable double logScale_{detail::ZERO_DOUBLE};

    /** @brief log(α) — component of log_norm_const_. */
    mutable double logAlpha_{detail::ZERO_DOUBLE};

    /** @brief −(α+1) — multiplier of log(x) in the LogPDF SIMD pipeline. */
    mutable double negAlphaPlusOne_{-detail::TWO};

    /** @brief log(α) + α·log(x_m) — the additive normalisation in LogPDF. */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief −α — multiplier in the CDF SIMD pipeline (step 3). */
    mutable double negAlpha_{-detail::ONE};

    /** @brief 1/α — for quantile and sampling. */
    mutable double invAlpha_{detail::ONE};

    /** @brief Cached mean (finite only when α > 1). */
    mutable double mean_{std::numeric_limits<double>::infinity()};

    /** @brief Cached variance (finite only when α > 2). */
    mutable double variance_{std::numeric_limits<double>::infinity()};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Pareto uses standard caching only.
    // Section maintained for template compliance.
};

}  // namespace stats

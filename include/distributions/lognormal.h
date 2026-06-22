#pragma once

// Common distribution includes (consolidates std library and core headers)
#include "libstats/common/distribution_common.h"

// Consolidated distribution platform headers (SIMD, parallel execution, thread pools,
// adaptive caching, etc.)
#include "libstats/common/distribution_platform_common.h"

namespace stats {

/**
 * @brief Thread-safe Log-Normal Distribution for modelling positive-valued data.
 *
 * @details If X ~ LogNormal(μ, σ), then log(X) ~ N(μ, σ²). The distribution
 * arises naturally whenever a quantity is the product of many small independent
 * factors (incomes, file sizes, service times, failure lifetimes).
 *
 * @par Mathematical Definition:
 * - PDF:    f(x; μ, σ) = exp(−(log x − μ)²/(2σ²)) / (x·σ·√(2π))  for x > 0
 * - LogPDF: −(log x − μ)²/(2σ²) − log x − log σ − ½·log(2π)
 * - CDF:    Φ((log x − μ)/σ)  where Φ is the standard normal CDF
 * - Quantile: exp(μ + σ·Φ⁻¹(p))
 * - Parameters: μ ∈ ℝ (log-mean), σ > 0 (log-stddev)
 * - Support:  x ∈ (0, ∞)
 *
 * @par Moments:
 * - Mean:     exp(μ + σ²/2)
 * - Variance: (exp(σ²) − 1)·exp(2μ + σ²)
 * - Mode:     exp(μ − σ²)
 * - Median:   exp(μ)
 * - Skewness: (exp(σ²) + 2)·√(exp(σ²) − 1)
 * - Excess kurtosis: exp(4σ²) + 2exp(3σ²) + 3exp(2σ²) − 6
 *
 * @par Batch SIMD:
 * LogPDF uses one vector_log and one vector_multiply (element-wise square) on a
 * single aligned temp buffer. Six-step pipeline:
 *   temp    = log(x)                 [vector_log]
 *   results = log(x) − μ  (= z)      [scalar_add(temp, −mu_)]
 *   results = z²                      [vector_multiply(results, results)]
 *   results = −z²/(2σ²)              [scalar_multiply(neg_inv_2sigma2_)]
 *   results −= log(x)                 [vector_subtract(results, temp)]
 *   results += log_norm_const         [scalar_add(log_norm_const_)]
 * PDF: append vector_exp.
 * CDF uses vector_log + scalar_add + scalar_multiply + vector_erf.
 * Expected speedup: LogPDF ~10–15×, PDF ~8–12×, CDF ~6–10×.
 *
 * @par Applications:
 * - Finance: stock returns, asset prices, option pricing
 * - Reliability: failure times, component lifetimes
 * - Biology: cell growth, organ weights
 * - Network science: degree distributions, file sizes
 * - Environmental science: pollutant concentrations
 *
 * @author libstats Development Team
 * @version 2.0.0
 * @since 2.0.0
 */
class LogNormalDistribution : public DistributionBase {
   public:
    // Dispatch metadata — replaces DistributionTraits<LogNormalDistribution> (v2.0.0)
    static constexpr detail::DistributionType kDistributionType =
        detail::DistributionType::LOG_NORMAL;
    static constexpr bool kIsDiscrete = false;

   public:
    //==========================================================================
    // 1. CONSTRUCTORS AND DESTRUCTOR
    //==========================================================================

    /**
     * @brief Construct a Log-Normal distribution.
     * @param mu    Location parameter μ — log-mean (any finite real, default 0)
     * @param sigma Scale parameter σ — log-stddev (must be positive, default 1)
     * @throws std::invalid_argument if sigma <= 0 or either parameter is non-finite
     *
     * Default (0, 1) is the standard log-normal distribution.
     * Implementation in .cpp.
     */
    explicit LogNormalDistribution(double mu = detail::ZERO_DOUBLE, double sigma = detail::ONE);

    /** @brief Thread-safe copy constructor. Implementation in .cpp. */
    LogNormalDistribution(const LogNormalDistribution& other);

    /** @brief Copy assignment operator. Implementation in .cpp. */
    LogNormalDistribution& operator=(const LogNormalDistribution& other);

    /** @brief Move constructor. Implementation in .cpp. */
    LogNormalDistribution(LogNormalDistribution&& other) noexcept;

    /** @brief Move assignment operator. Implementation in .cpp. */
    LogNormalDistribution& operator=(LogNormalDistribution&& other) noexcept;

    /** @brief Destructor — defaulted. */
    ~LogNormalDistribution() override = default;

    //==========================================================================
    // 2. SAFE FACTORY METHODS (Exception-free construction)
    //==========================================================================

    /**
     * @brief Safely create a Log-Normal distribution without throwing exceptions.
     * @param mu    Location parameter μ (any finite real)
     * @param sigma Scale parameter σ (must be positive)
     * @return Result containing a valid LogNormalDistribution or error info
     */
    [[nodiscard]] static Result<LogNormalDistribution> create(double mu = detail::ZERO_DOUBLE,
                                                              double sigma = detail::ONE) {
        auto validation = validateLogNormalParameters(mu, sigma);
        if (validation.isError()) {
            return Result<LogNormalDistribution>::makeError(validation.error_code,
                                                            validation.message);
        }
        return Result<LogNormalDistribution>::ok(createUnchecked(mu, sigma));
    }

    //==========================================================================
    // 3. PARAMETER GETTERS AND SETTERS
    //==========================================================================

    /** @brief Get location parameter μ (log-mean). */
    [[nodiscard]] double getMu() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return mu_;
    }

    /** @brief Get scale parameter σ (log-stddev). */
    [[nodiscard]] double getSigma() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return sigma_;
    }

    /**
     * @brief Lock-free atomic getter for μ.
     * Uses acquire-load from atomicMu_ for high-frequency access.
     */
    [[nodiscard]] double getMuAtomic() const noexcept;

    /**
     * @brief Lock-free atomic getter for σ.
     * Uses acquire-load from atomicSigma_ for high-frequency access.
     */
    [[nodiscard]] double getSigmaAtomic() const noexcept;

    /**
     * @brief Set location parameter μ.
     * @throws std::invalid_argument if mu is non-finite
     */
    void setMu(double mu);

    /**
     * @brief Set scale parameter σ.
     * @throws std::invalid_argument if sigma <= 0
     */
    void setSigma(double sigma);

    /**
     * @brief Set both parameters simultaneously.
     * @throws std::invalid_argument if sigma <= 0 or mu is non-finite
     */
    void setParameters(double mu, double sigma);

    /** @brief Mean = exp(μ + σ²/2). */
    [[nodiscard]] double getMean() const override;

    /** @brief Variance = (exp(σ²) − 1)·exp(2μ + σ²). */
    [[nodiscard]] double getVariance() const override;

    /** @brief Skewness = (exp(σ²) + 2)·√(exp(σ²) − 1). */
    [[nodiscard]] double getSkewness() const override;

    /** @brief Excess kurtosis = exp(4σ²) + 2exp(3σ²) + 3exp(2σ²) − 6. */
    [[nodiscard]] double getKurtosis() const override;

    /** @brief Number of parameters (always 2). */
    [[nodiscard]] int getNumParameters() const noexcept override { return 2; }

    /** @brief Distribution name. */
    [[nodiscard]] std::string_view getDistributionName() const noexcept override {
        return "LogNormalDistribution";
    }

    /** @brief Log-normal is continuous. */
    [[nodiscard]] bool isDiscrete() const noexcept override { return false; }

    /** @brief Support lower bound: 0 (exclusive). */
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

    [[nodiscard]] VoidResult trySetMu(double mu) noexcept;
    [[nodiscard]] VoidResult trySetSigma(double sigma) noexcept;
    [[nodiscard]] VoidResult trySetParameters(double mu, double sigma) noexcept;
    [[nodiscard]] VoidResult validateCurrentParameters() const noexcept;

    //==========================================================================
    // 5. CORE PROBABILITY METHODS
    //==========================================================================

    /**
     * @brief PDF at x: exp(LogPDF(x)) for x > 0; 0 otherwise.
     * Computed via exp(LogPDF) for numerical stability.
     */
    [[nodiscard]] double getProbability(double x) const override;

    /**
     * @brief Log-PDF at x: −(log x − μ)²/(2σ²) − log x − log σ − ½ log(2π).
     * Returns −∞ for x ≤ 0.
     */
    [[nodiscard]] double getLogProbability(double x) const override;

    /**
     * @brief CDF via Φ((log x − μ)/σ) = 0.5·(1 + erf((log x − μ)/(σ√2))).
     * Returns 0 for x ≤ 0.
     */
    [[nodiscard]] double getCumulativeProbability(double x) const override;

    /**
     * @brief Quantile: exp(μ + σ·Φ⁻¹(p)).
     * @throws std::invalid_argument if p not in [0, 1]
     */
    [[nodiscard]] double getQuantile(double p) const override;

    /** @brief Generate one random sample via exp(Normal(μ, σ)). */
    [[nodiscard]] double sample(std::mt19937& rng) const override;

    /** @brief Generate n random samples. */
    [[nodiscard]] std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    //==========================================================================
    // 6. DISTRIBUTION MANAGEMENT
    //==========================================================================

    /**
     * @brief Fit μ and σ to data by MLE.
     *
     * Closed-form MLE for log-normal: transform data to log-space and apply
     * Gaussian MLE. μ̂ = mean(log xᵢ), σ̂ = std(log xᵢ).
     *
     * @param values Observed data (must all be strictly positive)
     * @throws std::invalid_argument if values is empty or contains non-positive values
     */
    void fit(const std::vector<double>& values) override;

    /**
     * @brief Parallel batch fitting for multiple independent datasets.
     * @param datasets Vector of datasets, each with independent observations
     * @param results  Vector to store fitted LogNormalDistribution objects
     */
    static void parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                 std::vector<LogNormalDistribution>& results);

    /** @brief Reset to default (μ = 0, σ = 1 — standard log-normal). */
    void reset() noexcept override;

    /** @brief String representation. */
    std::string toString() const override;

    //==========================================================================
    // 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
    //==========================================================================

    /** @brief Mode = exp(μ − σ²). */
    [[nodiscard]] double getMode() const;

    /** @brief Median = exp(μ). */
    [[nodiscard]] double getMedian() const;

    /**
     * @brief Entropy = log(σ·√(2πe)) + μ = log σ + μ + ½(1 + log(2π)).
     */
    [[nodiscard]] double getEntropy() const override;

    /**
     * @brief True if μ = 0 and σ = 1 (standard log-normal) within tolerance.
     */
    [[nodiscard]] bool isStandard() const noexcept {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return isStandard_;
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

    bool operator==(const LogNormalDistribution& other) const;
    bool operator!=(const LogNormalDistribution& other) const;

    //==========================================================================
    // 16. FRIEND FUNCTION STREAM OPERATORS
    //==========================================================================

    friend std::istream& operator>>(std::istream& is, stats::LogNormalDistribution&);
    friend std::ostream& operator<<(std::ostream& os, const stats::LogNormalDistribution&);

   private:
    //==========================================================================
    // 17. PRIVATE FACTORY METHODS
    //==========================================================================

    static LogNormalDistribution createUnchecked(double mu, double sigma) noexcept;
    LogNormalDistribution(double mu, double sigma, bool /*bypassValidation*/) noexcept;

    //==========================================================================
    // 18. PRIVATE BATCH IMPLEMENTATION METHODS
    //==========================================================================

    /**
     * @brief SIMD log-space pipeline for LogPDF.
     *
     * Six-step pipeline using one aligned temp buffer:
     *   Step 1: temp    = log(x)                 [vector_log]
     *   Step 2: results = log(x) − μ  (= z)       [scalar_add(temp, −mu_)]
     *   Step 3: results = z²                       [vector_multiply(results, results)]
     *   Step 4: results = −z²/(2σ²)               [scalar_multiply(neg_inv_2sigma2_)]
     *   Step 5: results −= log(x)                  [vector_subtract(results, temp)]
     *   Step 6: results += log_norm_const           [scalar_add(log_norm_const_)]
     *
     * PDF variant: append vector_exp after step 6.
     * Scalar fixup: x ≤ 0 → −∞ (LogPDF) or 0 (PDF).
     */
    void getProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                       double cached_mu, double cached_neg_inv_2sigma2,
                                       double cached_log_norm_const) const noexcept;

    void getLogProbabilityBatchUnsafeImpl(const double* values, double* results, std::size_t count,
                                          double cached_mu, double cached_neg_inv_2sigma2,
                                          double cached_log_norm_const) const noexcept;

    /**
     * @brief CDF batch pipeline.
     *
     * Four-step pipeline:
     *   Step 1: temp    = log(x)                         [vector_log]
     *   Step 2: results = (log(x) − μ) / (σ√2)           [scalar_add + scalar_multiply]
     *   Step 3: results = erf(results)                    [VectorOps::vector_erf]
     *   Step 4: results = 0.5·(1 + erf(...))              [scalar_add(1); scalar_multiply(0.5)]
     *
     * Scalar fixup: x ≤ 0 → 0.
     */
    void getCumulativeProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                 std::size_t count, double cached_mu,
                                                 double cached_inv_sigma_sqrt2) const noexcept;

    //==========================================================================
    // 19. PRIVATE COMPUTATIONAL METHODS
    //==========================================================================

    void updateCacheUnsafe() const noexcept override;

    static void validateParameters(double mu, double sigma) {
        if (std::isnan(mu) || std::isinf(mu)) {
            throw std::invalid_argument("Mu (log-mean) must be a finite real number");
        }
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= detail::ZERO_DOUBLE) {
            throw std::invalid_argument("Sigma (log-stddev) must be a positive finite number");
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

    /** @brief Location parameter μ (log-mean) — any finite real. */
    double mu_{detail::ZERO_DOUBLE};

    /** @brief Scale parameter σ (log-stddev) — must be positive. */
    double sigma_{detail::ONE};

    /** @brief Atomic copies for lock-free access. */
    mutable std::atomic<double> atomicMu_{detail::ZERO_DOUBLE};
    mutable std::atomic<double> atomicSigma_{detail::ONE};
    mutable std::atomic<bool> atomicParamsValid_{false};

    //==========================================================================
    // 22. PERFORMANCE CACHE
    //==========================================================================

    /** @brief log(σ) — avoids repeated transcendental evaluations. */
    mutable double logSigma_{detail::ZERO_DOUBLE};

    /** @brief −1/(2σ²) — the quadratic coefficient in the log-space exponent. */
    mutable double negInv2SigmaSquared_{-detail::HALF};

    /** @brief 1/(σ√2) — used by the CDF erf argument. */
    mutable double invSigmaSqrt2_{detail::ONE};

    /** @brief −log(σ) − ½ log(2π) — additive normalisation constant in LogPDF. */
    mutable double logNormConst_{detail::ZERO_DOUBLE};

    /** @brief Cached mean = exp(μ + σ²/2). */
    mutable double mean_{detail::ONE};

    /** @brief Cached variance = (exp(σ²) − 1)·exp(2μ + σ²). */
    mutable double variance_{detail::ZERO_DOUBLE};

    //==========================================================================
    // 23. OPTIMIZATION FLAGS
    //==========================================================================

    /** @brief True if μ = 0 and σ = 1 (standard log-normal) within tolerance. */
    mutable bool isStandard_{true};

    //==========================================================================
    // 24. SPECIALIZED CACHES
    //==========================================================================

    // Note: Log-normal uses standard caching only.
    // Section maintained for template compliance.
};

}  // namespace stats
